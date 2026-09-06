"""Phase 7b: per-tier measurement harness.

Storage footprint, peak RAM/VRAM (ingest + query), query latency (text/image/
point-seeded KNN/tile-seeded KNN/bbox filter - median+p95, warm), and the
incremental +1000-tile byte-identical proof. Reuses the exact same query
paths as scripts/bench_spatial_index.py (Phase 7a) and the production
SearchEngine/discovery.knn - no new search/index code, only measurement.

Writes data/eval_retrieval/scale_report_<tier>.json.

Usage:
    python scripts/measure_tier.py --tier baseline
    python scripts/measure_tier.py --tier tier1 --incremental-region dehradun
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from shapely import wkt as shapely_wkt

from geoseek.config import get_settings
from geoseek.discovery.knn import find_more_like_this
from geoseek.ingest.store import TileStore
from geoseek.search.engine import SearchEngine
from geoseek.vectorindex import FaissFlatIPIndex
from ingest_diverse_scene import ingest_diverse_scene
from run_diverse_ingest import LEDGER_PATH, MemSampler
from stage_diverse_aois import stage_one

OUT_DIR = get_settings().data_dir / "eval_retrieval"
N_POINTS = 30
REPEATS = 8
SEED = 7042026
TEXT_QUERIES = ["a water body", "dense urban buildings", "agricultural fields",
                 "open bare ground", "a river with sandbars"]


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def _summ(xs: list[float]) -> dict:
    return {"n": len(xs), "median_ms": round(statistics.median(xs), 3) if xs else None,
            "p95_ms": round(_percentile(xs, 0.95), 3) if xs else None,
            "max_ms": round(max(xs), 3) if xs else None}


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def measure_storage() -> dict:
    settings = get_settings()
    faiss_path, sqlite_path = settings.index_dir / "tiles.faiss", settings.index_dir / "tiles.sqlite"
    faiss_mb = faiss_path.stat().st_size / 1e6 if faiss_path.is_file() else 0.0
    sqlite_mb = sqlite_path.stat().st_size / 1e6 if sqlite_path.is_file() else 0.0
    raw_datasets_mb = _dir_size_bytes(settings.datasets_dir) / 1e6
    derived_mb = sum(_dir_size_bytes(settings.data_dir / d)
                      for d in ("discovery", "eval_retrieval", "change_model")) / 1e6
    models_mb = _dir_size_bytes(settings.models_dir) / 1e6
    total_mb = _dir_size_bytes(settings.data_dir) / 1e6
    t0 = time.time()
    FaissFlatIPIndex(faiss_path).persist()
    persist_ms = round((time.time() - t0) * 1000, 2)
    return {
        "faiss_index_mb": round(faiss_mb, 2), "sqlite_db_mb": round(sqlite_mb, 2),
        "raw_datasets_mb": round(raw_datasets_mb, 2), "derived_products_mb": round(derived_mb, 2),
        "models_mb": round(models_mb, 2), "total_data_dir_mb": round(total_mb, 2),
        "faiss_full_rewrite_persist_ms": persist_ms,
    }


def measure_query_latency_and_memory() -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with MemSampler() as mem:
        eng = SearchEngine()
        repo = eng.repo
        n = eng.count()
        recs = repo.query_tiles(collection="sentinel-2-l2a")
        rng = random.Random(SEED)
        sample = rng.sample(recs, k=min(N_POINTS, len(recs)))
        centroids = [shapely_wkt.loads(r.geom_wkt_4326).centroid for r in sample]
        points = [(c.x, c.y) for c in centroids]
        tile_ids = [r.tile_id for r in sample]

        for _ in range(3):  # warm-up, discarded
            eng.search_text(TEXT_QUERIES[0], k=20)

        text_ms: list[float] = []
        for q in TEXT_QUERIES:
            for _ in range(REPEATS):
                _, ms = eng.search_text(q, k=20)
                text_ms.append(ms)

        image_ms: list[float] = []
        for tid in tile_ids[:15]:
            for _ in range(REPEATS):
                _, ms = eng.search_image(tile_id=tid, k=20)
                image_ms.append(ms)

        point_knn_ms: list[float] = []
        for lon, lat in points:
            for _ in range(5):
                try:
                    out = find_more_like_this(eng, lon=lon, lat=lat, k=8)
                    point_knn_ms.append(out["latency_ms"])
                except KeyError:
                    pass

        tile_knn_ms: list[float] = []
        for tid in tile_ids:
            for _ in range(5):
                out = find_more_like_this(eng, tile_id=tid, k=8)
                tile_knn_ms.append(out["latency_ms"])

        bbox_ms: list[float] = []
        eps = 0.01
        for lon, lat in points:
            box = (lon - eps, lat - eps, lon + eps, lat + eps)
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                repo.query_tiles(bbox=box)
                bbox_ms.append((time.perf_counter() - t0) * 1000.0)

        eng.close()
    vram_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None

    return {
        "n_vectors": n,
        "latency": {
            "text_search_k20": _summ(text_ms), "image_search_k20": _summ(image_ms),
            "point_seeded_knn": _summ(point_knn_ms), "tile_seeded_knn": _summ(tile_knn_ms),
            "bbox_filter": _summ(bbox_ms),
        },
        "peak_memory_during_query": {
            "peak_rss_mb": round(mem.peak_rss / 1e6, 1),
            "peak_vram_mb": round(vram_mb, 1) if vram_mb is not None else None,
        },
    }


def incremental_proof(region: str, size_km: float = 81.0, n_sample: int = 25) -> dict:
    """Stage + ingest one small (~1000-tile) new scene; prove a sample of
    PRE-EXISTING vectors is byte-identical before/after (no rebuild)."""
    settings = get_settings()
    idx_path = settings.index_dir / "tiles.faiss"
    before_count = FaissFlatIPIndex(idx_path).count()
    rng = random.Random(SEED)
    sample_ids = rng.sample(range(before_count), k=min(n_sample, before_count))

    store = TileStore()
    before_vecs = {i: store.reconstruct(i).copy() for i in sample_ids}
    store.close()

    staged = stage_one(region, size_km=size_km)
    t0 = time.time()
    report = ingest_diverse_scene(staged["scene_id"])
    elapsed_s = time.time() - t0

    store2 = TileStore()
    after_count = store2.count()
    after_vecs = {i: store2.reconstruct(i) for i in sample_ids}
    store2.close()

    byte_identical = all(before_vecs[i].tobytes() == after_vecs[i].tobytes() for i in sample_ids)
    max_abs_diff = max(float(np.max(np.abs(before_vecs[i] - after_vecs[i]))) for i in sample_ids)

    return {
        "region": region, "scene_id": staged["scene_id"], "size_km": size_km,
        "tiles_added": report["tiles_added"], "before_count": before_count, "after_count": after_count,
        "elapsed_s": round(elapsed_s, 2),
        "tiles_per_sec": round(report["tiles_added"] / elapsed_s, 2) if elapsed_s > 0 else None,
        "n_sampled_preexisting_vectors": len(sample_ids),
        "byte_identical": byte_identical, "max_abs_diff_over_sample": max_abs_diff,
        "no_rebuild": after_count == before_count + report["tiles_added"],
    }


def ingest_peak_memory_for_tier(scene_ids: set[str] | None) -> dict:
    """Max peak RSS/VRAM observed across this tier's ledger rows (written by
    run_diverse_ingest.py). scene_ids=None -> use the whole ledger."""
    if not LEDGER_PATH.is_file():
        return {"peak_rss_mb": None, "peak_vram_mb": None, "n_scenes": 0}
    rows = json.loads(LEDGER_PATH.read_text())
    if scene_ids is not None:
        rows = [r for r in rows if r["scene_id"] in scene_ids]
    if not rows:
        return {"peak_rss_mb": None, "peak_vram_mb": None, "n_scenes": 0}
    rss = [r["peak_rss_mb"] for r in rows if r.get("peak_rss_mb") is not None]
    vram = [r["peak_vram_mb"] for r in rows if r.get("peak_vram_mb") is not None]
    embed_tiles_per_sec = [1000.0 / r["mean_embed_latency_ms"] for r in rows
                            if r.get("mean_embed_latency_ms")]
    return {
        "peak_rss_mb": max(rss) if rss else None,
        "peak_vram_mb": max(vram) if vram else None,
        "n_scenes": len(rows),
        "total_tiles_added": sum(r["tiles_added"] for r in rows),
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "mean_end_to_end_tiles_per_sec": round(
            sum(r["tiles_added"] for r in rows) / sum(r["wall_s"] for r in rows), 2
        ) if rows else None,
        "mean_pure_embed_tiles_per_sec": round(statistics.mean(embed_tiles_per_sec), 2) if embed_tiles_per_sec else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--incremental-region", default=None,
                     help="region slug to use for the +1000-tile incremental proof; omit to skip")
    ap.add_argument("--ledger-since-index", type=int, default=0,
                     help="only include ledger rows at/after this row index when computing ingest peak memory")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"tier": args.tier, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("[measure] storage footprint ...")
    report["storage"] = measure_storage()
    print(json.dumps(report["storage"], indent=2))

    print("[measure] query latency + peak memory during query ...")
    report["query"] = measure_query_latency_and_memory()
    print(json.dumps(report["query"], indent=2))

    all_rows = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.is_file() else []
    tier_rows = all_rows[args.ledger_since_index:]
    tier_scene_ids = {r["scene_id"] for r in tier_rows} if tier_rows else None
    print(f"[measure] ingest peak memory across {len(tier_rows)} ledger rows since index {args.ledger_since_index} ...")
    report["ingest"] = ingest_peak_memory_for_tier(tier_scene_ids)
    print(json.dumps(report["ingest"], indent=2))
    report["ledger_row_count_at_measurement"] = len(all_rows)

    if args.incremental_region:
        print(f"[measure] incremental +1000 proof via region '{args.incremental_region}' ...")
        report["incremental_proof"] = incremental_proof(args.incremental_region)
        print(json.dumps(report["incremental_proof"], indent=2))

    out_path = OUT_DIR / f"scale_report_{args.tier}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[measure] wrote {out_path}")


if __name__ == "__main__":
    main()

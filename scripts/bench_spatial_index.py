"""Phase 7a Step A: before/after latency for the tile-geometry spatial index.

Point-seeded KNN ("find more like this" from a lon/lat) has to first resolve
the seed tile at that point, which goes through ``MetadataRepository.query_tiles(
bbox=...)``. Before Phase 7a that was an O(N) shapely scan over every tile
footprint; Phase 7a adds a SQLite R*Tree bbox prefilter behind the repository
seam so it becomes an O(log N + k) index probe. Tile-*id*-seeded KNN never
touched that path and is the <1 ms reference point.

Run it twice against the SAME production ``data/index/tiles.sqlite``:

    python scripts/bench_spatial_index.py --label before   # with the R*Tree code git-stashed
    python scripts/bench_spatial_index.py --label after    # with it applied
    python scripts/bench_spatial_index.py --report         # print the comparison

Each ``--label`` run writes ``data/eval_retrieval/spatial_index_bench_<label>.json``.
Not a pytest test - a one-off measurement kept for reproducibility. The
correctness of the prefilter (identical result sets) is guarded by
``tests/test_phase7.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from shapely import wkt as shapely_wkt

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.discovery.knn import find_more_like_this
from geoseek.search.engine import SearchEngine

OUT_DIR = get_settings().data_dir / "eval_retrieval"
N_POINTS = 30
BBOX_REPEATS = 8
KNN_REPEATS = 5
SEED = 7


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def _summ(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "median_ms": round(statistics.median(xs), 3),
        "mean_ms": round(statistics.mean(xs), 3),
        "p95_ms": round(_percentile(xs, 0.95), 3),
        "p99_ms": round(_percentile(xs, 0.99), 3),
        "max_ms": round(max(xs), 3),
    }


def _sample_points(repo: SQLiteMetadataRepository) -> list[tuple[float, float]]:
    recs = repo.query_tiles(collection="sentinel-2-l2a")
    rng = random.Random(SEED)
    sample = rng.sample(recs, k=min(N_POINTS, len(recs)))
    pts = []
    for r in sample:
        c = shapely_wkt.loads(r.geom_wkt_4326).centroid
        pts.append((c.x, c.y))
    return pts


def _bench(label: str) -> None:
    repo = SQLiteMetadataRepository()
    has_rtree = bool(getattr(repo, "_has_rtree", False))
    n_tiles = repo.count_tiles()
    points = _sample_points(repo)
    eps = 0.002  # ~200 m half-box

    # (a) bbox filtering through query_tiles
    bbox_ms: list[float] = []
    for lon, lat in points:
        box = (lon - eps, lat - eps, lon + eps, lat + eps)
        for _ in range(BBOX_REPEATS):
            t0 = time.perf_counter()
            repo.query_tiles(bbox=box)
            bbox_ms.append((time.perf_counter() - t0) * 1000.0)

    # (b) point-seeded KNN (find_more_like_this from lon/lat) - total + the
    #     query_tiles point-lookup component that Step A targets
    engine = SearchEngine()
    knn_total_ms: list[float] = []
    knn_lookup_ms: list[float] = []
    for lon, lat in points:
        for _ in range(KNN_REPEATS):
            out = find_more_like_this(engine, lon=lon, lat=lat, k=8)
            knn_total_ms.append(out["latency_ms"])
            knn_lookup_ms.append(out["point_lookup_ms"])

    # (c) tile-id-seeded KNN - the reference path that never hits query_tiles
    seed_tiles = [
        r.tile_id
        for r in repo.query_tiles(collection="sentinel-2-l2a")
    ]
    rng = random.Random(SEED)
    seed_tiles = rng.sample(seed_tiles, k=min(N_POINTS, len(seed_tiles)))
    tile_seeded_ms: list[float] = []
    for tid in seed_tiles:
        for _ in range(KNN_REPEATS):
            out = find_more_like_this(engine, tile_id=tid, k=8)
            tile_seeded_ms.append(out["latency_ms"])

    engine.close()
    repo.close()

    result = {
        "label": label,
        "has_rtree": has_rtree,
        "n_tiles": n_tiles,
        "n_points": len(points),
        "params": {"eps_deg": eps, "bbox_repeats": BBOX_REPEATS, "knn_repeats": KNN_REPEATS, "seed": SEED},
        "bbox_query_tiles": _summ(bbox_ms),
        "point_seeded_knn_total": _summ(knn_total_ms),
        "point_seeded_knn_lookup_component": _summ(knn_lookup_ms),
        "tile_seeded_knn_total": _summ(tile_seeded_ms),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"spatial_index_bench_{label}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[bench] wrote {out_path}")


def _report() -> None:
    before = json.loads((OUT_DIR / "spatial_index_bench_before.json").read_text())
    after = json.loads((OUT_DIR / "spatial_index_bench_after.json").read_text())
    print(f"tiles in catalog: {after['n_tiles']}   "
          f"has_rtree: before={before['has_rtree']} after={after['has_rtree']}")
    rows = [
        ("bbox filter (query_tiles)", "bbox_query_tiles"),
        ("point-seeded KNN (total)", "point_seeded_knn_total"),
        ("  -> point-lookup component", "point_seeded_knn_lookup_component"),
        ("tile-seeded KNN (reference)", "tile_seeded_knn_total"),
    ]
    hdr = f"{'operation':<32} | {'before median/p95 ms':>22} | {'after median/p95 ms':>22} | {'speedup(median)':>15}"
    print(hdr)
    print("-" * len(hdr))
    for name, key in rows:
        b, a = before[key], after[key]
        sp = (b["median_ms"] / a["median_ms"]) if a["median_ms"] > 0 else float("inf")
        print(f"{name:<32} | {b['median_ms']:>10.3f} / {b['p95_ms']:>8.3f} | "
              f"{a['median_ms']:>10.3f} / {a['p95_ms']:>8.3f} | {sp:>13.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["before", "after"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        _report()
    elif args.label:
        _bench(args.label)
    else:
        ap.error("pass --label before|after or --report")


if __name__ == "__main__":
    main()

"""Phase 7a Step B (PS 2.3): retrieval evaluation - prep phase.

Builds a held-out evaluation set for semantic retrieval. For each of a fixed
set of natural-language queries it ranks the ENTIRE production catalog (3267
Sentinel-2 tiles) with:

  * RemoteCLIP - the real search engine (SearchEngine / FAISS), and
  * a vanilla OpenCLIP control - force_quick_gelu=True (Phase 2 lesson), so the
    control is not handicapped by the QuickGELU/GELU mismatch,

pools each system's top-POOL_DEPTH plus a fixed random sample of the corpus,
and writes to data/eval_retrieval/:

  pools.json                  per-query pool + both rankings + the random draw
  contact_sheets/*.png        one labelled thumbnail grid per query (visual QA)
  judgments_template.json     {query: {tile_id: null}} (reference only; the
                              real judgements come from eval_retrieval_judge.py)
  latency.json                cold (fresh process, no pre-warm) + warm
                              (pre-warmed, steady state) query latency
  queries.json                the query list
  vanilla_corpus_*.{npy,json} cached vanilla embeddings for the whole corpus

Pool = union(RemoteCLIP top-20, vanilla top-20, random sample). POOL_DEPTH is
the largest reported K, so every item either system needs judged at any K in
{1,5,10,20} is in the pool; the random draw keeps the pool from being defined
purely by the two systems. Judged by scripts/eval_retrieval_judge.py using an
independent spectral signal (never the embedding models).

Usage:
  python scripts/eval_retrieval_prepare.py            # full prep
  python scripts/eval_retrieval_prepare.py --skip-latency
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geoseek.config import get_settings
from geoseek.ingest.embed import (
    boa_offset_dn_for_scene,
    embed_text_vanilla,
    embed_tiles_batch_vanilla,
    make_true_color_uint8,
)
from geoseek.ingest.reader import read_scene
from geoseek.ingest.tiler import tile_scene
from geoseek.search.engine import SearchEngine

RGB_BANDS = ("B04", "B03", "B02")
SCL_BAND = "SCL"
ALL_BANDS = (*RGB_BANDS, SCL_BAND)
SCENE_DIRS = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)

POOL_DEPTH = 20            # == max reported K
RANDOM_SAMPLE_N = 15       # random tiles added to every query's judged pool
RANDOM_SEED = 20260905
N_COLD_PROCS = 12          # fresh processes for the cold-latency distribution
WARM_PASSES = 6
OUT_DIR_NAME = "eval_retrieval"

# 16 queries spanning the AOI's real content. #1-2 are the PS's own example
# phrasings. No vehicle-scale queries - 10 m GSD cannot resolve them.
QUERIES = [
    "newly built structures near a river",       # PS example phrasing
    "settlement along a riverbank",               # PS example phrasing
    "a river with sandbars",
    "dense urban buildings",
    "agricultural fields",
    "open bare ground",
    "a water body",
    "cropland with visible field boundaries",
    "a paved road",
    "riverside construction",
    "a bridge crossing a river",
    "dense vegetation along a riverbank",
    "an urban residential neighborhood",
    "irrigated farmland",
    "a braided river channel with sandbars",
    "a dirt track or unpaved road",
]

THUMB_PX = 110
GRID_COLS = 8


def _slug(q: str) -> str:
    return q.replace(" ", "_").replace("/", "-")


# --------------------------------------------------------------------------
# vanilla CLIP full-corpus embedding (cached to disk - one-time cost)
# --------------------------------------------------------------------------

def build_or_load_vanilla_corpus(out_dir: Path) -> tuple[np.ndarray, list[str]]:
    vec_path = out_dir / "vanilla_corpus_vectors.npy"
    ids_path = out_dir / "vanilla_corpus_tile_ids.json"
    if vec_path.is_file() and ids_path.is_file():
        vectors = np.load(vec_path)
        tile_ids = json.loads(ids_path.read_text())
        print(f"[prep] loaded cached vanilla corpus: {vectors.shape}")
        return vectors, tile_ids

    settings = get_settings()
    vector_chunks: list[np.ndarray] = []
    tile_ids: list[str] = []
    for scene_dir_name in SCENE_DIRS:
        scene_dir = settings.datasets_dir / scene_dir_name
        print(f"[prep] (vanilla) reading + tiling {scene_dir_name} ...")
        scene = read_scene(scene_dir, list(ALL_BANDS))
        tiles = tile_scene(scene)
        boa_offset_dn = boa_offset_dn_for_scene(scene.scene_id)
        print(f"[prep] (vanilla) {len(tiles)} tiles - stretching + batch-embedding ...")
        rgb_list = [
            make_true_color_uint8({b: t.bands[b] for b in RGB_BANDS}, nodata=t.nodata, boa_offset_dn=boa_offset_dn)
            for t in tiles
        ]
        t0 = time.time()
        vectors = embed_tiles_batch_vanilla(rgb_list)
        print(f"[prep] (vanilla) embedded {len(tiles)} tiles in {time.time() - t0:.1f}s")
        vector_chunks.append(vectors)
        tile_ids.extend(t.tile_id for t in tiles)

    vectors = np.concatenate(vector_chunks, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, vectors)
    ids_path.write_text(json.dumps(tile_ids))
    print(f"[prep] cached vanilla corpus: {vectors.shape} -> {vec_path}")
    return vectors, tile_ids


# --------------------------------------------------------------------------
# ranking + pooling
# --------------------------------------------------------------------------

def rank_vanilla(query: str, vectors: np.ndarray, tile_ids: list[str], k: int) -> list[tuple[str, float]]:
    qvec = embed_text_vanilla(query)
    scores = vectors @ qvec
    order = np.argsort(-scores)[:k]
    return [(tile_ids[i], float(scores[i])) for i in order]


def rank_remoteclip(engine: SearchEngine, query: str, k: int) -> list[tuple[str, float]]:
    results, _ = engine.search_text(query, k=k)
    return [(r.tile_id, float(r.score)) for r in results]


def random_sample_for_query(query: str, all_tile_ids: list[str], n: int) -> list[str]:
    rng = random.Random(f"{RANDOM_SEED}:{query}")
    return rng.sample(all_tile_ids, k=min(n, len(all_tile_ids)))


def build_pool(query: str, rc_ranked, va_ranked, random_ids: list[str]) -> dict:
    rc_rank = {tid: i + 1 for i, (tid, _) in enumerate(rc_ranked)}
    va_rank = {tid: i + 1 for i, (tid, _) in enumerate(va_ranked)}
    pool_ids = list(dict.fromkeys(
        [tid for tid, _ in rc_ranked] + [tid for tid, _ in va_ranked] + random_ids
    ))
    return {
        "pool_tile_ids": pool_ids,
        "remoteclip_ranked": [tid for tid, _ in rc_ranked],
        "vanilla_ranked": [tid for tid, _ in va_ranked],
        "random_sample_tile_ids": random_ids,
        "remoteclip_rank": rc_rank,
        "vanilla_rank": va_rank,
        "remoteclip_score": dict(rc_ranked),
        "vanilla_score": dict(va_ranked),
        "pool_composition": {
            "n_total": len(pool_ids),
            "n_remoteclip_top": len(rc_ranked),
            "n_vanilla_top": len(va_ranked),
            "n_random": len(random_ids),
            "n_overlap_rc_va": len(set(t for t, _ in rc_ranked) & set(t for t, _ in va_ranked)),
        },
    }


# --------------------------------------------------------------------------
# contact sheet for visual QA
# --------------------------------------------------------------------------

def make_pool_contact_sheet(engine: SearchEngine, query: str, pool: dict, out_path: Path) -> dict[str, str]:
    ids = pool["pool_tile_ids"]
    labels = [f"{i + 1:02d}" for i in range(len(ids))]
    n = len(ids)
    cols = min(GRID_COLS, max(n, 1))
    rows = (n + cols - 1) // cols
    header_h, cell_label_h = 30, 16
    sheet = Image.new("RGB", (THUMB_PX * cols, header_h + rows * (THUMB_PX + cell_label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), f"query: '{query}'  ({n} pooled tiles)  R=RemoteCLIP rank  V=vanilla rank  *=random", fill="black")
    rnd = set(pool["random_sample_tile_ids"])
    for i, tid in enumerate(ids):
        r, c = divmod(i, cols)
        x, y = c * THUMB_PX, header_h + r * (THUMB_PX + cell_label_h)
        try:
            png = engine.get_tile_thumbnail_png(tid)
            img = Image.open(io.BytesIO(png)).convert("RGB").resize((THUMB_PX, THUMB_PX))
        except Exception as e:
            img = Image.new("RGB", (THUMB_PX, THUMB_PX), "gray")
            print(f"[prep] WARN thumbnail failed for {tid}: {e}")
        sheet.paste(img, (x, y))
        rc = pool["remoteclip_rank"].get(tid)
        va = pool["vanilla_rank"].get(tid)
        tag = f"{labels[i]} R{rc or '-'}/V{va or '-'}{'*' if tid in rnd else ''}"
        draw.text((x + 2, y + THUMB_PX + 1), tag, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return dict(zip(labels, ids))


# --------------------------------------------------------------------------
# latency: cold (fresh process, pre-warm suppressed) vs warm (pre-warmed)
# --------------------------------------------------------------------------

def _latency_worker(mode: str, rotate: int) -> None:
    """One measurement job, run in its own process; prints a JSON line."""
    import numpy as _np  # noqa

    queries = QUERIES
    if mode == "cold":
        orig = SearchEngine._prewarm
        SearchEngine._prewarm = lambda self: None
        try:
            eng = SearchEngine()
            q = queries[rotate % len(queries)]
            _, ms = eng.search_text(q, k=POOL_DEPTH)   # genuine first query, cold encoder
            eng.close()
        finally:
            SearchEngine._prewarm = orig
        print(json.dumps({"mode": "cold", "query": q, "ms": ms}))
    else:  # warm
        eng = SearchEngine()                            # pre-warm + keepwarm as in production
        for q in queries[:3]:
            eng.search_text(q, k=POOL_DEPTH)            # discard warm-up
        out = []
        for _ in range(WARM_PASSES):
            for q in queries:
                _, ms = eng.search_text(q, k=POOL_DEPTH)
                out.append({"query": q, "ms": ms})
        eng.close()
        print(json.dumps({"mode": "warm", "samples": out}))


def measure_latency() -> dict:
    cold_ms: list[float] = []
    for i in range(N_COLD_PROCS):
        proc = subprocess.run(
            [sys.executable, __file__, "--latency-worker", "cold", str(i)],
            capture_output=True, text=True, check=True,
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        cold_ms.append(json.loads(line)["ms"])
        print(f"[prep] cold proc {i + 1}/{N_COLD_PROCS}: {cold_ms[-1]:.1f} ms")

    proc = subprocess.run(
        [sys.executable, __file__, "--latency-worker", "warm", "0"],
        capture_output=True, text=True, check=True,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    warm_ms = [s["ms"] for s in json.loads(line)["samples"]]
    print(f"[prep] warm: {len(warm_ms)} samples, median {sorted(warm_ms)[len(warm_ms) // 2]:.1f} ms")
    return {
        "cold_ms": cold_ms,
        "warm_ms": warm_ms,
        "method": {
            "cold": f"{N_COLD_PROCS} fresh Python processes, SearchEngine._prewarm suppressed, "
                    f"first search_text() call timed (rotating query)",
            "warm": f"1 pre-warmed process (prewarm + keepwarm as in production), 3 discarded warm-ups, "
                    f"then {WARM_PASSES} passes x {len(QUERIES)} queries",
        },
    }


# --------------------------------------------------------------------------

def main() -> None:
    out_dir = get_settings().data_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-worker", nargs=2, metavar=("MODE", "ROTATE"))
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    if args.latency_worker:
        _latency_worker(args.latency_worker[0], int(args.latency_worker[1]))
        return

    vectors, tile_ids = build_or_load_vanilla_corpus(out_dir)

    engine = SearchEngine()
    print(f"[prep] RemoteCLIP corpus: {engine.count()} vectors; vanilla corpus: {len(tile_ids)} vectors")

    pools: dict[str, dict] = {}
    labels_by_query: dict[str, dict[str, str]] = {}
    judgments_template: dict[str, dict[str, None]] = {}

    for query in QUERIES:
        print(f"[prep] pooling query: {query!r}")
        rc_ranked = rank_remoteclip(engine, query, POOL_DEPTH)
        va_ranked = rank_vanilla(query, vectors, tile_ids, POOL_DEPTH)
        random_ids = random_sample_for_query(query, tile_ids, RANDOM_SAMPLE_N)
        pool = build_pool(query, rc_ranked, va_ranked, random_ids)
        pools[query] = pool

        sheet_path = out_dir / "contact_sheets" / f"{_slug(query)}.png"
        labels_by_query[query] = make_pool_contact_sheet(engine, query, pool, sheet_path)
        judgments_template[query] = {tid: None for tid in pool["pool_tile_ids"]}
        print(f"[prep]   pool {pool['pool_composition']}")

    (out_dir / "pools.json").write_text(json.dumps(pools, indent=2))
    (out_dir / "pool_labels.json").write_text(json.dumps(labels_by_query, indent=2))
    (out_dir / "judgments_template.json").write_text(json.dumps(judgments_template, indent=2))
    (out_dir / "queries.json").write_text(json.dumps(QUERIES, indent=2))
    engine.close()

    if not args.skip_latency:
        latency = measure_latency()
        (out_dir / "latency.json").write_text(json.dumps(latency, indent=2))

    print(f"\n[prep] done -> {out_dir}")
    print("[prep] NEXT: python scripts/eval_retrieval_features.py && "
          "python scripts/eval_retrieval_judge.py && python scripts/eval_retrieval_score.py")


if __name__ == "__main__":
    main()

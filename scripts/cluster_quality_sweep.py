"""PS 2.2.4 follow-up: improve discovery-cluster quality/balance.

The default at-scale config (eom, min_cluster_size=40, no PCA, 20k sample)
gives 6 clusters where one ("bare dry open ground", c4) holds 86% of the
104,089-tile corpus, and two different cluster ids carry the exact same
top concept label ("bare dry open ground" for both c4 and c5) - a weak,
hard-to-read result for a discovery feature. This script sweeps a small,
documented set of alternatives on the SAME stratified sample (fixed seed,
comparable region mix) and reports cluster count / size balance / duplicate
labels / runtime for each, so the choice actually kept is evidence-based.

Usage: python scripts/cluster_quality_sweep.py [--sample-size 12000]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from geoseek.config import get_settings
from geoseek.discovery.cluster import cluster_sample_and_assign, load_all_vectors, stratified_sample_indices
from geoseek.search.engine import SearchEngine
from geoseek.search.rerank import region_key

OUT_DIR = get_settings().data_dir / "discovery"


def _region_map(engine: SearchEngine) -> dict[str, str]:
    aoi_by_obs = {obs.observation_id: region_key(obs.aoi_name) for obs in engine.repo.list_observations()}
    return {rec.tile_id: aoi_by_obs.get(rec.observation_id, "unknown")
            for rec in engine.repo.iter_tile_records() if rec.faiss_id is not None}


def _summarize(name, res, t_s) -> dict:
    sizes = res.sizes
    total = sum(sizes.values())
    top_frac = max(sizes.values()) / total if sizes else 0.0
    labels_used = [res.cluster_concepts[c][0][0] for c in sizes]
    dup_labels = [lbl for lbl, n in Counter(labels_used).items() if n > 1]
    row = {
        "config": name,
        "n_clusters": res.n_clusters,
        "noise_pct": round(100 * res.noise_count / total, 1) if total else None,
        "largest_cluster_pct": round(100 * top_frac, 1),
        "sizes": sizes,
        "top_labels": {c: res.cluster_concepts[c][0][0] for c in sizes},
        "duplicate_top_labels": dup_labels,
        "runtime_s": round(t_s, 1),
        "params": res.params,
    }
    print(f"\n--- {name} ---  ({t_s:.1f}s)")
    print(f"  clusters={res.n_clusters}  largest={top_frac*100:.1f}%  "
          f"noise={row['noise_pct']}%  dup_labels={dup_labels or 'none'}")
    for cid in sorted(sizes, key=lambda k: -sizes[k]):
        print(f"    c{cid:>3} n={sizes[cid]:>7} ({100*sizes[cid]/total:4.1f}%)  "
              f"{res.cluster_concepts[cid][0][0]}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=12_000,
                    help="sample size for the sweep (smaller than production for speed)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated substrings - only run configs whose name contains one of these")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eng = SearchEngine()
    try:
        recs = [r for r in eng.repo.iter_tile_records() if r.faiss_id is not None]
        recs.sort(key=lambda r: int(r.faiss_id))
        tile_ids = [r.tile_id for r in recs]
        region_map = _region_map(eng)
        groups = [region_map.get(t, "unknown") for t in tile_ids]
        print(f"corpus: {len(tile_ids)} tiles, sweep sample_size={args.sample_size}")

        vectors = load_all_vectors(eng.vector_index)

        configs = [
            ("baseline eom mcs=40", dict(min_cluster_size=40, cluster_selection_method="eom")),
            ("leaf mcs=40", dict(min_cluster_size=40, cluster_selection_method="leaf")),
            ("eom mcs=100", dict(min_cluster_size=100, cluster_selection_method="eom")),
            ("eom mcs=15 (finer)", dict(min_cluster_size=15, cluster_selection_method="eom")),
            ("leaf mcs=15 (finer)", dict(min_cluster_size=15, cluster_selection_method="leaf")),
            ("eom mcs=40 + PCA50", dict(min_cluster_size=40, cluster_selection_method="eom", pca_dim=50)),
            ("leaf mcs=40 + PCA50", dict(min_cluster_size=40, cluster_selection_method="leaf", pca_dim=50)),
        ]

        if args.only:
            keep = [s.strip() for s in args.only.split(",") if s.strip()]
            configs = [c for c in configs if any(k in c[0] for k in keep)]

        rows = []
        for name, kw in configs:
            t0 = time.time()
            res = cluster_sample_and_assign(
                vectors, tile_ids, groups, eng.embedding_model,
                sample_size=args.sample_size, core_dist_n_jobs=args.n_jobs, seed=args.seed, **kw)
            rows.append(_summarize(name, res, time.time() - t0))

        (OUT_DIR / "cluster_quality_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n-> {OUT_DIR / 'cluster_quality_sweep.json'}")

        print("\n" + "=" * 100)
        print(f"{'config':28s} {'clusters':>9} {'largest%':>9} {'noise%':>7} {'dup labels':>12} {'runtime':>8}")
        for r in rows:
            print(f"{r['config']:28s} {r['n_clusters']:>9} {r['largest_cluster_pct']:>9} "
                  f"{r['noise_pct']:>7} {len(r['duplicate_top_labels']):>12} {r['runtime_s']:>7}s")
    finally:
        eng.close()


if __name__ == "__main__":
    main()

"""Phase 5 Step E - batch: HDBSCAN over all tile embeddings + a KNN demo.

    python scripts/cluster_tiles.py [--min-cluster-size 40]

Offline. Loads the production SearchEngine (RemoteCLIP + FAISS + catalog),
clusters every embedded tile, labels each cluster with its nearest RemoteCLIP
text concepts, saves a spatial cluster map, and demonstrates the interactive
"find more like this" KNN with its latency. This is a periodic/batch job, not a
per-query path.
"""

from __future__ import annotations

import argparse
import json
import time

from geoseek.config import get_settings
from geoseek.discovery.cluster import cluster_embeddings, load_all_vectors, save_cluster_map
from geoseek.discovery.knn import find_more_like_this
from geoseek.search.engine import SearchEngine

OUT_DIR = get_settings().data_dir / "discovery"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-cluster-tiles")
    p.add_argument("--min-cluster-size", type=int, default=40)
    p.add_argument("--min-samples", type=int, default=None)
    p.add_argument("--knn-seed-lon", type=float, default=82.1998)
    p.add_argument("--knn-seed-lat", type=float, default=26.7922)
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("geoseek Phase 5 Step E - discovery (batch)")
    print("=" * 78)
    eng = SearchEngine()
    try:
        recs = [r for r in eng.repo.iter_tile_records() if r.faiss_id is not None]
        recs.sort(key=lambda r: int(r.faiss_id))
        tile_ids = [r.tile_id for r in recs]
        print(f"  embedded tiles: {len(tile_ids)}  (SAR tiles are not embedded, excluded)")

        t0 = time.time()
        vectors = load_all_vectors(eng.vector_index)
        print(f"  loaded {vectors.shape} vectors in {time.time()-t0:.1f}s")

        t0 = time.time()
        res = cluster_embeddings(vectors, tile_ids, eng.embedding_model,
                                 min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
        print(f"\n  HDBSCAN (min_cluster_size={args.min_cluster_size}) in {time.time()-t0:.1f}s")
        print(f"  clusters: {res.n_clusters}   noise: {res.noise_count} "
              f"({100*res.noise_count/len(tile_ids):.1f}%)")
        print(f"  {'cluster':>8} {'size':>6}   nearest text concepts")
        for cid in sorted(res.sizes, key=lambda k: res.sizes[k], reverse=True):
            concepts = ", ".join(f"{c} ({s:.2f})" for c, s in res.cluster_concepts[cid])
            print(f"  {cid:>8} {res.sizes[cid]:>6}   {concepts}")

        map_path = save_cluster_map(res, recs, OUT_DIR / "cluster_map.png")
        print(f"\n  cluster map -> {map_path}")

        payload = res.as_dict()
        payload["tile_cluster"] = {tid: int(lab) for tid, lab in zip(tile_ids, res.labels.tolist())}
        payload["cluster_map_png"] = str(map_path)
        (get_settings().index_dir / "tile_clusters.json").write_text(json.dumps(payload, indent=2),
                                                                    encoding="utf-8")
        print(f"  tile_clusters.json -> {get_settings().index_dir / 'tile_clusters.json'}")

        # -- KNN "find more like this" demo (interactive path) --
        print(f"\n  KNN 'find more like this' @ ({args.knn_seed_lon}, {args.knn_seed_lat}):")
        knn = find_more_like_this(eng, lon=args.knn_seed_lon, lat=args.knn_seed_lat, k=8)
        print(f"    seed tile: {knn['seed_tile_id']}")
        for r in knn["results"]:
            print(f"      {r['tile_id']:42s} score {r['score']:.3f}  {r['acq_date']}  "
                  f"({r['centroid_lonlat'][0]:.4f}, {r['centroid_lonlat'][1]:.4f})")
        print(f"    latency: {knn['latency_ms']:.1f} ms  (point lookup {knn['point_lookup_ms']:.1f} ms)")

        from geoseek.staging.manifest import record_analysis_section
        record_analysis_section("tile_clustering", {**res.as_dict(),
                                "cluster_map_png": str(map_path),
                                "tile_clusters_json": str(get_settings().index_dir / "tile_clusters.json"),
                                "knn_demo": {"seed": knn["seed_tile_id"], "k": knn["k"],
                                             "latency_ms": knn["latency_ms"],
                                             "top": knn["results"][:5]}})
        print("\n  -> manifest section 'tile_clustering'")
    finally:
        eng.close()


if __name__ == "__main__":
    main()

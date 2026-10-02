"""Offline 3-D projection of the tile embeddings, for the console's vector-space visualiser.

PCA to 50 dimensions (the same preprocessing the clustering uses), then UMAP to 3 dimensions with the cosine metric
(embeddings are unit-norm). Everything runs locally; nothing is downloaded. The vectors come through the ``VectorIndex``
seam and the tile ids / regions / coordinates through the ``MetadataRepository`` seam (no faiss / sqlite import here).

Artifacts under ``data/discovery/``:
  projection_3d.npz        tile_ids, xyz (float32, centred and scaled to a unit sphere), lon, lat, region (index), cluster
  projection_3d.meta.json  parameters, library versions, wall times, power source, the index it was computed from
and a provenance record in ``data/provenance_manifest.json``.

The projection is a picture, not a measurement: UMAP preserves local neighbourhoods, not distances, so distances between
distant clusters in the picture mean nothing. The console says so next to the plot.

UMAP is an OPTIONAL extra (``pip install umap-learn``; BSD-3-Clause) - it is only needed to *recompute* the projection.
The console reads the finished artifact and needs neither umap nor numba.

Per the project's standing rule, benchmark on AC power: the run records the power source and warns if it is not AC.

Usage:
  python scripts/compute_projection.py [--n-neighbors 15] [--min-dist 0.1] [--pca 50] [--seed 42] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.eval.env import power_source
from geoseek.search.rerank import region_key
from geoseek.staging.manifest import build_record, load_manifest, write_manifest
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

OUT_NAME = "projection_3d.npz"
META_NAME = "projection_3d.meta.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="project only the first N vectors (smoke test; NOT for the shipped artifact)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args()

    try:
        import umap
    except ImportError:
        print("umap-learn is not installed (it is only needed to recompute the projection):  pip install umap-learn", file=sys.stderr)
        return 2
    from sklearn.decomposition import PCA
    import sklearn

    settings = get_settings()
    out_dir = args.out_dir or (settings.data_dir / "discovery")
    out_dir.mkdir(parents=True, exist_ok=True)
    power = power_source()
    if power["source"] != "AC":
        print(f"WARNING: power source is {power['source']!r}; wall times recorded here are not AC benchmarks (the project's standing AC-power benchmark rule).")

    index = FaissFlatIPIndex(settings.faiss_index_path)
    repo = SQLiteMetadataRepository(settings.database_path)
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    vectors = index.reconstruct_all()
    recs = [r for r in repo.iter_tile_records() if r.faiss_id is not None]
    recs.sort(key=lambda r: r.faiss_id)
    assert [r.faiss_id for r in recs] == list(range(len(recs))) and len(recs) == vectors.shape[0], (
        f"catalog ({len(recs)} embedded tiles) and vector index ({vectors.shape[0]}) disagree - run the integrity check first")
    if args.limit:
        vectors, recs = vectors[: args.limit], recs[: args.limit]
    load_s = time.perf_counter() - t0
    print(f"loaded {vectors.shape} vectors + {len(recs)} tile records in {load_s:.1f}s   power={power['source']}")

    aoi = {o.observation_id: region_key(o.aoi_name) for o in repo.list_observations()}
    regions = sorted({aoi.get(r.observation_id, "unknown") for r in recs})
    region_idx = np.array([regions.index(aoi.get(r.observation_id, "unknown")) for r in recs], dtype=np.int16)
    from shapely import wkt as shapely_wkt

    ll = np.array([(lambda c: (c.x, c.y))(shapely_wkt.loads(r.geom_wkt_4326).centroid) for r in recs], dtype=np.float64)
    clusters_path = settings.index_dir / "tile_clusters.json"
    cl_map = (json.loads(clusters_path.read_text(encoding="utf-8")).get("tile_cluster") or {}) if clusters_path.is_file() else {}
    cluster = np.array([int(cl_map.get(r.tile_id, -1)) for r in recs], dtype=np.int16)

    t0 = time.perf_counter()
    reduced = PCA(n_components=args.pca, random_state=args.seed).fit_transform(vectors).astype(np.float32)
    pca_s = time.perf_counter() - t0
    print(f"PCA-{args.pca}: {pca_s:.1f}s")

    t0 = time.perf_counter()
    xyz = umap.UMAP(n_components=3, n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="cosine",
                    random_state=args.seed, verbose=True).fit_transform(reduced).astype(np.float32)
    umap_s = time.perf_counter() - t0
    print(f"UMAP-3D: {umap_s:.1f}s ({umap_s / 60:.1f} min)")

    xyz -= np.median(xyz, axis=0)                              # centre, then scale so 99.5% of points sit inside the unit sphere
    xyz /= max(float(np.percentile(np.linalg.norm(xyz, axis=1), 99.5)), 1e-9)
    wall_s = time.perf_counter() - t_all

    out = out_dir / (OUT_NAME if not args.limit else f"projection_3d.limit{args.limit}.npz")
    np.savez_compressed(out, tile_ids=np.array([r.tile_id for r in recs]), xyz=xyz, lon=ll[:, 0].astype(np.float32),
                        lat=ll[:, 1].astype(np.float32), region=region_idx, cluster=cluster, regions=np.array(regions))
    meta = {
        "n_points": int(len(recs)), "dims": 3, "method": f"PCA-{args.pca} -> UMAP-3D",
        "umap": {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist, "metric": "cosine", "random_state": args.seed},
        "pca_components": args.pca,
        "libraries": {"umap-learn": umap.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                      "python": sys.version.split()[0]},
        "wall_seconds": {"total": round(wall_s, 1), "load": round(load_s, 1), "pca": round(pca_s, 1), "umap": round(umap_s, 1)},
        "power_source": power["source"], "battery_percent": power.get("battery_percent"),
        "source_index": {"path": str(settings.faiss_index_path.name), "vectors": int(index.count()),
                         "sha256": _sha256(settings.faiss_index_path), "size_bytes": settings.faiss_index_path.stat().st_size},
        "regions": regions, "n_clusters": int(len(set(cluster.tolist()) - {-1})),
        "n_tiles_without_cluster": int((cluster == -1).sum()),
        "partial": bool(args.limit),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "caveat": "UMAP preserves local neighbourhoods, not distances or cluster sizes; read it as 'which tiles are neighbours', never as a measurement.",
    }
    (out_dir / (META_NAME if not args.limit else f"projection_3d.limit{args.limit}.meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_points", "wall_seconds", "power_source")}))

    if not args.no_manifest and not args.limit:
        m = load_manifest()
        rec = build_record(name="projection-3d-umap", local_path=out, license=(
            "derived locally from the tile embeddings; inherits the imagery licences (Copernicus Sentinel open data; "
            "Maxar Open Data CC-BY-NC-4.0, non-commercial)"), source_url="local:scripts/compute_projection.py")
        m["artifacts"] = [a for a in m.get("artifacts", []) if a.get("name") != rec.name] + [rec.__dict__]
        write_manifest(m)
        print("provenance record written (sha256", rec.sha256[:16] + "…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 8F-2 Step 7: run the trained detector over the staged Maxar Open Data tiles.

Reads the ``maxar-opendata`` observations from the catalog (Chungthang + Van Nuys, and the lake quadkey for contrast),
re-reads each 1024x1024 true-colour tile straight from the staged GeoTIFFs (the same window the embedding pipeline
uses), runs :class:`geoseek.models.yolo_obb.YoloObbDetectionModel` (the concrete ``ObjectDetectionModel``), and writes per
observation

  data/detections/<observation_id>/detections.geojson   every detection as an oriented polygon in EPSG:4326
  data/detections/<observation_id>/summary.json          per-tile counts by class + score statistics
  data/detections/samples/*.png                          annotated tiles (top-count tiles + seeded random ones)

and registers the GeoJSON as ``DerivedProduct(kind="detection")`` in the catalog (FW-5), so it inherits the provenance
chain and shows up next to the other derived rasters.

There is NO ground truth on these tiles, so no accuracy number is produced here; what is reported is what the detector
outputs (counts, score distribution, class mix) and what a human sees on the samples. DOTA (mostly Google Earth,
0.1-4.5 m GSD, median 0.26 m) and Maxar (WorldView, 0.305 m, different sensor / radiometry / off-nadir geometry) are
different domains - expect degradation versus the DOTA val figure, and read the samples to see where.

    python scripts/detect_maxar.py [--weights data/models/detector/geoseek_obb_v15_yolo26s.pt] [--max-tiles N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from rasterio.windows import transform as window_transform  # noqa: E402

from geoseek.catalog.entities import DerivedProduct  # noqa: E402
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.config import get_settings, print_startup_banner  # noqa: E402
from geoseek.models import TileGeoRef, YoloObbDetectionModel  # noqa: E402
from geoseek.staging.download_maxar import COLLECTION_ID, MAXAR_TILE_SIZE  # noqa: E402
from geoseek.staging.manifest import record_analysis_section  # noqa: E402

SEED = 20260920
CLASS_BGR = {"small-vehicle": (0, 255, 0), "large-vehicle": (0, 200, 255), "ship": (255, 200, 0), "plane": (255, 0, 255),
             "helicopter": (0, 0, 255), "storage-tank": (255, 255, 0), "harbor": (255, 128, 0), "bridge": (128, 0, 255)}
BAND_ORDER = ("R", "G", "B")


def read_rgb_tile(scene_dir: Path, row: int, col: int) -> tuple[np.ndarray, TileGeoRef]:
    """One tile's RGB window (uint8 HxWx3) plus its pixel -> CRS affine, read off the staged per-band GeoTIFFs."""
    col0, row0 = col * MAXAR_TILE_SIZE, row * MAXAR_TILE_SIZE
    bands = []
    with rasterio.open(scene_dir / "R.tif") as ref:
        w = min(MAXAR_TILE_SIZE, ref.width - col0)
        h = min(MAXAR_TILE_SIZE, ref.height - row0)
        win = Window(col0, row0, w, h)
        aff = window_transform(win, ref.transform)
        crs = ref.crs.to_string()
    for b in BAND_ORDER:
        with rasterio.open(scene_dir / f"{b}.tif") as ds:
            bands.append(ds.read(1, window=win))
    rgb = np.stack(bands, axis=-1)
    if rgb.dtype != np.uint8:                               # the visual product is 8-bit; refuse silent rescaling
        raise ValueError(f"expected uint8 visual tiles, got {rgb.dtype}")
    return rgb, TileGeoRef(transform=(aff.a, aff.b, aff.c, aff.d, aff.e, aff.f), crs=crs)


def draw(rgb: np.ndarray, dets) -> np.ndarray:
    img = np.ascontiguousarray(rgb[..., ::-1])
    for d in dets:
        poly = np.array(d.polygon_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, CLASS_BGR[d.class_name], 2, cv2.LINE_AA)
    counts = Counter(d.class_name for d in dets)
    cv2.rectangle(img, (0, 0), (1024, 28), (0, 0, 0), -1)
    cv2.putText(img, "  ".join(f"{k}:{v}" for k, v in counts.most_common(5)) or "no detections", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--max-tiles", type=int, default=None, help="per observation (smoke tests)")
    ap.add_argument("--min-score", type=float, default=None, help="default: the operating point in the model card")
    ap.add_argument("--observations", default=None, help="comma-separated substrings; default: every maxar-opendata observation")
    args = ap.parse_args()

    settings = get_settings()
    print_startup_banner(settings)
    weights = args.weights or settings.models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt"
    model = YoloObbDetectionModel(weights, min_score=args.min_score, batch_size=4)
    model.load()
    print(f"[maxar] model: {json.dumps(model.info)}", flush=True)

    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")
    obs_list = repo.list_observations(collection=COLLECTION_ID)
    if args.observations:
        keys = args.observations.split(",")
        obs_list = [o for o in obs_list if any(k in o.observation_id for k in keys)]
    out_root = settings.data_dir / "detections"
    (out_root / "samples").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = {"model": model.info, "min_score": model.min_score, "observations": {}}

    for obs in obs_list:
        scene_dir = Path(obs.dataset_dir or (settings.datasets_dir / obs.observation_id))
        tiles = sorted(repo.list_tiles(observation_id=obs.observation_id), key=lambda t: (t.row, t.col))
        if args.max_tiles:
            tiles = tiles[: args.max_tiles]
        label = obs.aoi_name or obs.observation_id
        print(f"\n[maxar] === {obs.observation_id} ({label}): {len(tiles)} tiles ===", flush=True)
        feats, per_tile, kept_imgs = [], [], {}
        t0 = time.time()
        for k, t in enumerate(tiles):
            row, col = t.row, t.col
            rgb, geo = read_rgb_tile(scene_dir, row, col)
            dets = model.detect(rgb, geo=geo)
            counts = Counter(d.class_name for d in dets)
            per_tile.append({"tile_id": t.tile_id, "row": row, "col": col, "n_detections": len(dets),
                             "by_class": dict(counts), "mean_score": round(float(np.mean([d.score for d in dets])), 3) if dets else None,
                             "cloud_fraction": t.cloud_fraction})
            for d in dets:
                feats.append({"type": "Feature", "geometry": None, "properties": {
                    "tile_id": t.tile_id, "class": d.class_name, "score": round(d.score, 3), "long_side_px": round(d.long_side_px, 1),
                    "heading_deg": round(d.heading_deg, 1), "wkt": d.geom_wkt_4326}})
            kept_imgs[t.tile_id] = (len(dets), rgb, dets)
            if (k + 1) % 50 == 0:
                print(f"[maxar]   {k + 1}/{len(tiles)} tiles  {time.time() - t0:.0f}s  ({len(feats)} detections)", flush=True)
                # keep memory bounded: only retain the current top-6 by count and a small random reservoir for samples
                keep = set(sorted(kept_imgs, key=lambda i: -kept_imgs[i][0])[:6]) | set(rng.sample(list(kept_imgs), min(4, len(kept_imgs))))
                kept_imgs = {i: v for i, v in kept_imgs.items() if i in keep}
        secs = time.time() - t0

        # GeoJSON with real geometries (shapely from the WKT the model produced)
        from shapely import wkt as shapely_wkt
        from shapely.geometry import mapping
        for f in feats:
            f["geometry"] = mapping(shapely_wkt.loads(f["properties"].pop("wkt")))
        obs_dir = out_root / obs.observation_id
        obs_dir.mkdir(parents=True, exist_ok=True)
        gj = obs_dir / "detections.geojson"
        gj.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                                  "properties": {"model": model.info, "observation_id": obs.observation_id}}), encoding="utf-8")

        totals = Counter()
        for r in per_tile:
            totals.update(r["by_class"])
        scores = [f["properties"]["score"] for f in feats]
        n_clear = sum(1 for r in per_tile if (r["cloud_fraction"] or 0) < 0.05)
        summary = {
            "observation_id": obs.observation_id, "n_tiles": len(per_tile), "n_tiles_with_detections": sum(1 for r in per_tile if r["n_detections"]),
            "n_detections": len(feats), "by_class": dict(totals.most_common()), "tile_seconds_mean": round(secs / max(len(tiles), 1), 3),
            "score_quantiles": {q: round(float(np.quantile(scores, q)), 3) for q in (0.1, 0.5, 0.9)} if scores else None,
            "detections_per_tile": {"mean": round(float(np.mean([r["n_detections"] for r in per_tile])), 2),
                                    "max": max(r["n_detections"] for r in per_tile), "n_clear_tiles": n_clear},
            "per_tile": per_tile,
        }
        (obs_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

        # annotated samples: the 3 highest-count tiles + 2 seeded random tiles among those retained
        top = sorted(kept_imgs, key=lambda i: -kept_imgs[i][0])[:3]
        rest = [i for i in kept_imgs if i not in top]
        extra = rng.sample(rest, min(2, len(rest)))
        sample_paths = []
        for tag, tid in [("top", i) for i in top] + [("random", i) for i in extra]:
            n, rgb, dets = kept_imgs[tid]
            p = out_root / "samples" / f"{obs.observation_id}__{tag}__{tid.split('_')[-2]}_{tid.split('_')[-1]}.png"
            cv2.imwrite(str(p), draw(rgb, dets))
            sample_paths.append(str(p))
        print(f"[maxar]   {len(feats)} detections in {len(per_tile)} tiles ({secs:.0f}s, {secs / max(len(tiles), 1):.2f}s/tile): "
              f"{dict(totals.most_common())}", flush=True)

        repo.upsert_derived(DerivedProduct(
            derived_id=f"detection:{obs.observation_id}", kind="detection", path=str(gj), observation_id=obs.observation_id,
            params={"model_sha256": model.info.get("weights_sha256"), "min_score": model.min_score, "n_detections": len(feats),
                    "classes": list(model.class_names)}, created_at=datetime.now(timezone.utc).isoformat()))
        report["observations"][obs.observation_id] = {"aoi": label, "role": (obs.metadata or {}).get("role"),
                                                        **{k: v for k, v in summary.items() if k != "per_tile"},
                                                        "geojson": str(gj), "samples": sample_paths}

    record_analysis_section("detector_maxar_inference", report)
    (out_root / "maxar_inference_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n[maxar] done. Report: {out_root / 'maxar_inference_report.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Ingest the staged Maxar Open Data pre/post observations into the tile
index + catalog, as the new ``maxar-opendata`` collection (Phase 8 Step A).

Reuses the existing generic pipeline pieces UNCHANGED:
  * ``geoseek.ingest.reader.read_scene``  - reads ``<dir>/<BAND>.tif`` files
  * ``geoseek.ingest.tiler.tile_scene``   - fixed-size tiling with a lon/lat
    footprint per tile; resolution-aware here via ``tile_size=MAXAR_TILE_SIZE``
  * ``geoseek.ingest.embed.embed_tiles_batch`` / ``save_sample_png`` - the
    Maxar "visual" product is already an 8-bit natural-color RGB (unlike
    Sentinel-2's uint16 reflectance), so it is fed to RemoteCLIP directly,
    with no reflectance stretch step (``make_true_color_uint8`` is
    Sentinel-2-specific and does not apply here).
  * ``geoseek.ingest.store.TileStore.add_tiles`` - now generalized (Phase 8
    Step A) to accept a distinct collection_id/sensor/native_gsd_m/license
    instead of assuming Sentinel-2 L2A, plus per-observation metadata
    (``obs_metadata``) carrying the ACTUAL delivered GSD so
    ``TemporalObservationMatcher`` gates correctly.

STAGING AND INGESTION ONLY - no object detector is built here (Phase 8 Step A
scope). Run after ``python -m geoseek.staging.download_maxar``.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings, print_startup_banner  # noqa: E402
from geoseek.ingest.embed import save_sample_png  # noqa: E402
from geoseek.ingest.quality import maxar_cloud_fraction  # noqa: E402
from geoseek.ingest.reader import read_scene  # noqa: E402
from geoseek.ingest.store import TileStore  # noqa: E402
from geoseek.ingest.tiler import tile_scene  # noqa: E402
from geoseek.models import RemoteCLIPEmbeddingModel  # noqa: E402
from geoseek.staging.download_maxar import (  # noqa: E402
    ATTRIBUTION,
    COLLECTION_DESCRIPTION,
    COLLECTION_ID,
    DATA_LICENSE,
    EVENT_ID,
    MAXAR_TILE_SIZE,
    QUADKEY,
    maxar_root,
)
from geoseek.staging.manifest import append_ingest_run  # noqa: E402

BAND_ORDER = ("R", "G", "B", "CLOUDMASK")

OBSERVATIONS = (
    {"role": "pre", "catalog_id": "10300100E34B4D00", "platform": "WV02", "acq_date": "2023-02-07"},
    {"role": "post", "catalog_id": "1050010036A0EC00", "platform": "GE01", "acq_date": "2023-10-06"},
)

AOI_NAME = f"maxar_{EVENT_ID.lower()}_q{QUADKEY}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ingest_one(obs_meta: dict, embedding_model: RemoteCLIPEmbeddingModel, sample_png_path: Path) -> dict:
    catalog_id = obs_meta["catalog_id"]
    observation_id = f"{catalog_id}_{QUADKEY}"
    scene_dir = maxar_root() / observation_id

    t0 = time.time()
    scene = read_scene(scene_dir, list(BAND_ORDER))
    print(f"[maxar-ingest] Read '{observation_id}': {scene.width}x{scene.height} px, CRS={scene.crs}")

    tiles = tile_scene(scene, tile_size=MAXAR_TILE_SIZE)
    print(f"[maxar-ingest] Tiled into {len(tiles)} tiles ({MAXAR_TILE_SIZE}x{MAXAR_TILE_SIZE} px, "
          f"~{MAXAR_TILE_SIZE * scene.transform.a:.0f} m/side ground footprint)")

    cloud_fractions: list[float] = []
    rgb_list: list[np.ndarray] = []
    best_std, best_idx = -1.0, None
    for i, tile in enumerate(tiles):
        cf = maxar_cloud_fraction(tile.bands["CLOUDMASK"])
        cloud_fractions.append(cf)
        rgb = np.stack([tile.bands["R"], tile.bands["G"], tile.bands["B"]], axis=-1)
        rgb_list.append(rgb)
        # pick the most VISUALLY INFORMATIVE clear tile (highest RGB std) for the
        # deliverable sample PNG, rather than an arbitrary raster-scan-order tile
        # (e.g. tile r000_c000 here is a near-uniform snowfield - no information).
        if cf < 0.05:
            std = float(np.std(rgb))
            if std > best_std:
                best_std, best_idx = std, i

    if best_idx is not None:
        save_sample_png(rgb_list[best_idx], sample_png_path)
        print(f"[maxar-ingest] Saved sample true-color tile PNG ({tiles[best_idx].tile_id}, "
              f"rgb_std={best_std:.1f}) -> {sample_png_path}")

    vectors = embedding_model.encode_images(rgb_list, batch_size=64) if rgb_list else np.zeros((0, 512), dtype=np.float32)
    print(f"[maxar-ingest] Embedded {len(rgb_list)} tiles")

    records = [
        {
            "tile_id": tile.tile_id,
            "scene_id": observation_id,
            "sensor": obs_meta["platform"],
            "acq_date": obs_meta["acq_date"],
            "geom_wkt_4326": tile.footprint_wkt_4326,
            "cloud_fraction": cf,
            "embedding": vec,
            "processing_history": [
                {"step": "read", "at": _now_iso()},
                {"step": "tiled", "at": _now_iso()},
                {"step": "quality_scored", "at": _now_iso()},
                {"step": "embedded", "at": _now_iso()},
                {"step": "indexed", "at": _now_iso()},
            ],
        }
        for tile, cf, vec in zip(tiles, cloud_fractions, vectors)
    ]

    store = TileStore()
    native_gsd_m = round(float(scene.transform.a), 8)
    store.add_tiles(
        records,
        dataset_dir=str(scene_dir),
        crs=str(scene.crs),
        bands=BAND_ORDER,
        source_url=f"https://maxar-opendata.s3.amazonaws.com/events/{EVENT_ID}/ard/45/{QUADKEY}/"
                   f"{obs_meta['acq_date']}/{catalog_id}-visual.tif",
        aoi_name=AOI_NAME,
        collection_id=COLLECTION_ID,
        collection_sensor="VHR Optical (multi-sensor)",
        collection_platform="Maxar Open Data Program",
        collection_native_gsd_m=native_gsd_m,
        collection_description=COLLECTION_DESCRIPTION,
        collection_metadata={"license": DATA_LICENSE, "attribution": ATTRIBUTION},
        scene_license=DATA_LICENSE,
        obs_metadata={
            "native_gsd_m": native_gsd_m,
            "event_id": EVENT_ID,
            "quadkey": QUADKEY,
            "catalog_id": catalog_id,
            "role": obs_meta["role"],
            "tile_size_px": MAXAR_TILE_SIZE,
            "tile_ground_footprint_m": round(MAXAR_TILE_SIZE * native_gsd_m, 1),
        },
    )
    store.save()
    n_clear = sum(1 for cf in cloud_fractions if cf < 0.05)
    report = {
        "observation_id": observation_id, "role": obs_meta["role"], "platform": obs_meta["platform"],
        "acq_date": obs_meta["acq_date"], "tiles_added": len(records), "n_clear_tiles": n_clear,
        "native_gsd_m": native_gsd_m,
        "tile_ground_footprint_m": round(MAXAR_TILE_SIZE * native_gsd_m, 1),
        "index_total_vectors": store.count(), "build_time_s": round(time.time() - t0, 3),
        "ingested_at": _now_iso(),
    }
    store.close()
    return report


def main() -> None:
    settings = get_settings()
    print_startup_banner(settings)

    embedding_model = RemoteCLIPEmbeddingModel()
    embedding_model.load()

    reports = []
    for obs_meta in OBSERVATIONS:
        sample_png_path = settings.data_dir / f"sample_maxar_tile_true_color_{obs_meta['role']}.png"
        report = _ingest_one(obs_meta, embedding_model, sample_png_path)
        reports.append(report)
        print(f"[maxar-ingest] {report}")

    run_report = {
        "collection_id": COLLECTION_ID, "event_id": EVENT_ID, "quadkey": QUADKEY,
        "observations": reports, "ingested_at": _now_iso(),
    }
    append_ingest_run(run_report)
    print(f"\n[maxar-ingest] Done. {sum(r['tiles_added'] for r in reports)} tiles total across "
          f"{len(reports)} observations. Manifest updated: {settings.provenance_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[maxar-ingest] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

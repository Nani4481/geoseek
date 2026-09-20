"""Ingest the staged Van Nuys quadkey (see
``geoseek.staging.download_maxar_vannuys``) into the existing
``maxar-opendata`` collection, at the SAME 1024x1024 tiling as the other two
quadkeys - a third, geographically distinct observation, not a new
collection.

Reuses the exact same generic pipeline pieces as
``scripts/ingest_maxar_event.py`` / ``scripts/ingest_maxar_chungthang.py``
(``read_scene`` / ``tile_scene`` / ``embed_tiles_batch`` / ``save_sample_png``
/ ``TileStore.add_tiles``); this is again a SINGLE observation (no pre/post
pair here either), indexed under its own ``aoi_name``.
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
    MAXAR_TILE_SIZE,
)
from geoseek.staging.download_maxar_vannuys import (  # noqa: E402
    ACQ_DATE,
    CATALOG_ID,
    EVENT_ID,
    LOCATION_NOTE,
    PLATFORM,
    QUADKEY,
    ROLE,
    UTM_ZONE_FOLDER,
    maxar_root,
)
from geoseek.staging.manifest import append_ingest_run  # noqa: E402

BAND_ORDER = ("R", "G", "B", "CLOUDMASK")
AOI_NAME = f"maxar_{EVENT_ID.lower()}_q{QUADKEY}_vannuys"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    settings = get_settings()
    print_startup_banner(settings)

    observation_id = f"{CATALOG_ID}_{QUADKEY}"
    scene_dir = maxar_root() / observation_id

    embedding_model = RemoteCLIPEmbeddingModel()
    embedding_model.load()

    t0 = time.time()
    scene = read_scene(scene_dir, list(BAND_ORDER))
    print(f"[maxar-ingest-vannuys] Read '{observation_id}': {scene.width}x{scene.height} px, CRS={scene.crs}")

    tiles = tile_scene(scene, tile_size=MAXAR_TILE_SIZE)
    print(f"[maxar-ingest-vannuys] Tiled into {len(tiles)} tiles ({MAXAR_TILE_SIZE}x{MAXAR_TILE_SIZE} px, "
          f"~{MAXAR_TILE_SIZE * scene.transform.a:.0f} m/side ground footprint)")

    cloud_fractions: list[float] = []
    rgb_list: list[np.ndarray] = []
    # keep the top-3 highest-RGB-std CLEAR tiles as sample PNGs - dense
    # clutter (the vehicle lot, the aircraft ramp) reliably scores highest,
    # same selection heuristic used for the Chungthang built-up samples.
    scored: list[tuple[float, int]] = []
    for i, tile in enumerate(tiles):
        cf = maxar_cloud_fraction(tile.bands["CLOUDMASK"])
        cloud_fractions.append(cf)
        rgb = np.stack([tile.bands["R"], tile.bands["G"], tile.bands["B"]], axis=-1)
        rgb_list.append(rgb)
        if cf < 0.05:
            scored.append((float(np.std(rgb)), i))

    scored.sort(reverse=True)
    top_indices = [i for _, i in scored[:3]]
    for rank, idx in enumerate(top_indices, start=1):
        out_path = settings.data_dir / f"sample_maxar_tile_true_color_vannuys_{rank}_{tiles[idx].tile_id}.png"
        save_sample_png(rgb_list[idx], out_path)
        print(f"[maxar-ingest-vannuys] Saved sample true-color tile PNG #{rank} "
              f"({tiles[idx].tile_id}, rgb_std={scored[rank - 1][0]:.1f}) -> {out_path}")

    vectors = embedding_model.encode_images(rgb_list, batch_size=64) if rgb_list else np.zeros((0, 512), dtype=np.float32)
    print(f"[maxar-ingest-vannuys] Embedded {len(rgb_list)} tiles")

    records = [
        {
            "tile_id": tile.tile_id,
            "scene_id": observation_id,
            "sensor": PLATFORM,
            "acq_date": ACQ_DATE,
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
        source_url=f"https://maxar-opendata.s3.amazonaws.com/events/{EVENT_ID}/ard/{UTM_ZONE_FOLDER}/{QUADKEY}/"
                   f"{ACQ_DATE}/{CATALOG_ID}-visual.tif",
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
            "catalog_id": CATALOG_ID,
            "role": ROLE,
            "location": LOCATION_NOTE,
            "tile_size_px": MAXAR_TILE_SIZE,
            "tile_ground_footprint_m": round(MAXAR_TILE_SIZE * native_gsd_m, 1),
        },
    )
    store.save()
    n_clear = sum(1 for cf in cloud_fractions if cf < 0.05)
    report = {
        "observation_id": observation_id, "role": ROLE, "platform": PLATFORM,
        "acq_date": ACQ_DATE, "tiles_added": len(records), "n_clear_tiles": n_clear,
        "native_gsd_m": native_gsd_m,
        "tile_ground_footprint_m": round(MAXAR_TILE_SIZE * native_gsd_m, 1),
        "index_total_vectors": store.count(), "build_time_s": round(time.time() - t0, 3),
        "ingested_at": _now_iso(),
    }
    store.close()
    print(f"[maxar-ingest-vannuys] {report}")

    run_report = {
        "collection_id": COLLECTION_ID, "event_id": EVENT_ID, "quadkey": QUADKEY,
        "location": LOCATION_NOTE, "observations": [report], "ingested_at": _now_iso(),
    }
    append_ingest_run(run_report)
    print(f"\n[maxar-ingest-vannuys] Done. {report['tiles_added']} tiles added. "
          f"Manifest updated: {settings.provenance_manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[maxar-ingest-vannuys] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

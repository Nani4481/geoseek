"""Phase 7b: ingest a staged diverse-AOI scene through the UNMODIFIED geoseek
ingest pipeline (tiler / quality / embedding model / vector index / catalog).

This is deliberately NOT a copy of the pipeline logic with changes - every
substantive call (``read_scene``, ``tile_scene``, ``cloud_fraction``,
``make_true_color_uint8``, ``RemoteCLIPEmbeddingModel.encode_images``,
``TileStore.add_tiles``) is the exact same function ``geoseek.ingest.pipeline.
ingest_scene`` calls, imported unmodified from the same modules, with the same
weights and the same fixed radiometry bounds. Zero retraining, zero changes to
tiling/embedding/index/catalog-schema code - that invariance is the
generalization claim (PHASE7B.md).

The only thing this wrapper does differently from calling ``ingest_scene()``
directly is pass the CORRECT per-region provenance metadata into
``store.add_tiles``: ``ingest_scene`` hardcodes ``aoi_name="ayodhya_..."`` for
every scene (a cosmetic label, fine when every scene really is Ayodhya) and
never passes ``source_url``/``checksums`` at all, which would silently
reconstruct a wrong (still-44RPQ) URL for any other MGRS tile via
``catalog.naming.scene_source_url``. Passing the real values through is a
metadata-only concern, orthogonal to the modeling/indexing path under test.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geoseek.config import EMBEDDING_DIM, get_settings
from geoseek.ingest.embed import (
    RADIOMETRY_CONFIG,
    boa_offset_dn_for_scene,
    make_true_color_uint8,
)
from geoseek.ingest.quality import cloud_fraction
from geoseek.ingest.reader import read_scene
from geoseek.ingest.store import TileStore
from geoseek.ingest.tiler import tile_scene
from geoseek.models import RemoteCLIPEmbeddingModel
from geoseek.staging.manifest import append_ingest_run, load_manifest

RGB_BANDS = ("B04", "B03", "B02")
SCL_BAND = "SCL"
ALL_BANDS = (*RGB_BANDS, SCL_BAND)
SCENE_ID_RE = re.compile(r"^(S2[AB])_\w+_(\d{8})_")
DEFAULT_EMBED_BATCH_SIZE = 64
MANIFEST_KEY = "diverse_aois"


def _sensor_and_date(scene_id: str) -> tuple[str, str]:
    m = SCENE_ID_RE.match(scene_id)
    if not m:
        return "unknown", "unknown"
    sat, date_token = m.groups()
    sensor = "Sentinel-2A" if sat == "S2A" else "Sentinel-2B"
    acq_date = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:]}"
    return sensor, acq_date


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _staging_record(scene_id: str) -> dict:
    manifest = load_manifest()
    for rec in manifest.get(MANIFEST_KEY, []):
        if rec["scene_id"] == scene_id:
            return rec
    raise KeyError(f"'{scene_id}' not found in manifest['{MANIFEST_KEY}'] - stage it first "
                    f"(scripts/stage_diverse_aois.py)")


def ingest_diverse_scene(
    scene_id: str,
    embedding_model: RemoteCLIPEmbeddingModel | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    record_manifest: bool = True,
) -> dict:
    staging = _staging_record(scene_id)
    settings = get_settings()
    scene_path = settings.datasets_dir / scene_id
    sensor, acq_date = _sensor_and_date(scene_id)
    aoi_name = f"{staging['region']}_{staging['mgrs_tile']}_diverse"

    embedding_model = embedding_model or RemoteCLIPEmbeddingModel()
    embedding_model.load()

    t_start = time.time()
    scene = read_scene(scene_path, list(ALL_BANDS))
    read_ts = _now_iso()
    print(f"[ingest-diverse] Read scene '{scene_id}' ({staging['display_name']}): "
          f"{scene.width}x{scene.height} px, CRS={scene.crs}")

    tiles = tile_scene(scene)
    tiled_ts = _now_iso()
    print(f"[ingest-diverse] Tiled into {len(tiles)} tiles (256x256, nodata tiles skipped)")

    store = TileStore()
    boa_offset_dn = boa_offset_dn_for_scene(scene_id)  # unknown scene -> DEFAULT_BOA_OFFSET_DN (0.0)

    cloud_fractions: list[float] = []
    rgb_list: list = []
    for tile in tiles:
        cloud_fractions.append(cloud_fraction(tile.bands[SCL_BAND]))
        rgb_list.append(make_true_color_uint8(
            {b: tile.bands[b] for b in RGB_BANDS}, nodata=tile.nodata, boa_offset_dn=boa_offset_dn
        ))
    quality_ts = _now_iso()

    if rgb_list:
        t_embed = time.time()
        vectors = embedding_model.encode_images(rgb_list, batch_size=embed_batch_size)
        embed_elapsed_ms = (time.time() - t_embed) * 1000.0
        embed_latencies_ms = [embed_elapsed_ms / len(rgb_list)] * len(rgb_list)
        n_batches = -(-len(rgb_list) // embed_batch_size)
        print(f"[ingest-diverse] Embedded {len(rgb_list)} tiles in {n_batches} batch(es) "
              f"of up to {embed_batch_size} ({embed_elapsed_ms:.0f} ms total)")
    else:
        vectors, embed_latencies_ms = np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    embedded_ts = _now_iso()

    records = [
        {
            "tile_id": tile.tile_id, "scene_id": scene_id, "sensor": sensor, "acq_date": acq_date,
            "geom_wkt_4326": tile.footprint_wkt_4326, "cloud_fraction": cf, "embedding": vec,
            "processing_history": [
                {"step": "read", "at": read_ts}, {"step": "tiled", "at": tiled_ts},
                {"step": "quality_scored", "at": quality_ts}, {"step": "embedded", "at": embedded_ts},
                {"step": "indexed", "at": _now_iso()},
            ],
        }
        for tile, cf, vec in zip(tiles, cloud_fractions, vectors)
    ]

    store.add_tiles(
        records,
        dataset_dir=scene_id,
        crs=str(scene.crs) if scene.crs else None,
        bands=tuple(ALL_BANDS),
        source_url=staging["source_url"],
        checksums=staging["band_sha256"],
        aoi_name=aoi_name,
    )
    store.save()
    store.close()

    build_time_s = time.time() - t_start
    index_size_mb = store.index_path.stat().st_size / (1024 * 1024) if store.index_path.is_file() else 0.0
    mean_latency_ms = float(np.mean(embed_latencies_ms)) if embed_latencies_ms else 0.0

    report = {
        "scene_id": scene_id, "region": staging["region"], "category": staging["category"],
        "mgrs_tile": staging["mgrs_tile"], "acq_date": acq_date, "season": staging["season"],
        "cloud_cover_pct": staging["cloud_cover_pct"],
        "tiles_added": len(records), "index_total_vectors": store.count(),
        "build_time_s": round(build_time_s, 3), "tiles_per_sec": round(len(records) / build_time_s, 2) if build_time_s > 0 else None,
        "index_size_mb": round(index_size_mb, 3), "mean_embed_latency_ms": round(mean_latency_ms, 3),
        "ingested_at": _now_iso(),
    }
    if record_manifest:
        append_ingest_run(report)

    print("[ingest-diverse] Run report: " + json.dumps(report))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="geoseek-ingest-diverse")
    ap.add_argument("scene_id")
    args = ap.parse_args(argv)
    ingest_diverse_scene(args.scene_id)


if __name__ == "__main__":
    main()

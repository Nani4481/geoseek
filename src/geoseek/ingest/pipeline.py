"""CLI: `python -m geoseek.ingest.pipeline ingest <scene_dir>`

read -> tile -> quality -> embed -> store, appended incrementally to the
FAISS index + SQLite DB, with the provenance manifest updated with a run
report (scenes, tiles added, build time, index size, mean embed latency).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geoseek.config import EMBEDDING_DIM, get_settings
from geoseek.ingest.embed import (
    RADIOMETRY_CONFIG,
    TRUE_COLOR_DISPLAY_STRETCH_CONFIG,
    boa_offset_dn_for_scene,
    make_true_color_uint8,
    save_sample_png,
)
from geoseek.ingest.quality import cloud_fraction
from geoseek.ingest.reader import read_scene
from geoseek.ingest.store import TileStore
from geoseek.ingest.tiler import tile_scene
from geoseek.models import EmbeddingModel, RemoteCLIPEmbeddingModel
from geoseek.staging.manifest import (
    append_ingest_run,
    record_analysis_section,
    record_radiometry_config,
)

DEFAULT_EMBED_BATCH_SIZE = 64

RGB_BANDS = ("B04", "B03", "B02")
SCL_BAND = "SCL"
ALL_BANDS = (*RGB_BANDS, SCL_BAND)

# S2A/S2B were the only satellites flying when this was first written; Sentinel-2C
# launched 2024-09 and Phase 8 stages a 2026 scene from it - S2[A-Z] covers C/D too.
SCENE_ID_RE = re.compile(r"^(S2[A-Z])_\w+_(\d{8})_")
_SENSOR_NAMES = {"S2A": "Sentinel-2A", "S2B": "Sentinel-2B", "S2C": "Sentinel-2C", "S2D": "Sentinel-2D"}


def _sensor_and_date(scene_id: str) -> tuple[str, str]:
    m = SCENE_ID_RE.match(scene_id)
    if not m:
        return "unknown", "unknown"
    sat, date_token = m.groups()
    sensor = _SENSOR_NAMES.get(sat, sat)
    acq_date = f"{date_token[:4]}-{date_token[4:6]}-{date_token[6:]}"
    return sensor, acq_date


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate_watch_areas_on_ingest(repo, observation_id: str) -> int:
    """Real, additional automation trigger for standing watch areas (see
    ``geoseek.watch.evaluator``): besides the existing trigger (the end of a
    full ``python -m geoseek.change.analyze`` run), watch areas are now ALSO
    evaluated right here, at the end of every ingest - using whatever change
    candidates already exist for this observation in the current change
    report. This means a newly-created watch area (or re-ingesting a scene
    that a prior analyze run already produced candidates for) fires without
    waiting for the next manual pipeline re-run. It is a genuine no-op (fast,
    no torch/rasterio) when there is no change report yet, no active watch
    areas, or no candidates mentioning this observation - i.e. the common
    case of ingesting a brand-new date that has not been through change
    detection yet."""
    if not repo.list_watch_areas(active_only=True):
        return 0
    from geoseek.change.analyze import OUT_DIR

    detail_path = OUT_DIR / "ayodhya_change_ranked_detail.json"
    if not detail_path.is_file():
        return 0
    import json as _json

    details = _json.loads(detail_path.read_text(encoding="utf-8"))
    relevant = [c for c in details if observation_id in (c.get("pair") or "").split("->")]
    if not relevant:
        return 0
    from geoseek.watch.evaluator import evaluate_and_notify

    fired = evaluate_and_notify(repo, relevant, observation_id=observation_id)
    return len(fired)


def ingest_scene(
    scene_path: Path,
    save_sample: bool = True,
    index_dir: Path | None = None,
    sample_png_path: Path | None = None,
    record_manifest: bool = True,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    embedding_model: EmbeddingModel | None = None,
) -> dict:
    scene_path = Path(scene_path)
    scene_id = scene_path.name
    sensor, acq_date = _sensor_and_date(scene_id)

    embedding_model = embedding_model or RemoteCLIPEmbeddingModel()
    # Loaded once, up front, before any tile touches it - never reloaded per tile.
    embedding_model.load()

    t_start = time.time()

    scene = read_scene(scene_path, list(ALL_BANDS))
    read_ts = _now_iso()
    print(f"[pipeline] Read scene '{scene_id}': {scene.width}x{scene.height} px, CRS={scene.crs}")

    tiles = tile_scene(scene)
    tiled_ts = _now_iso()
    print(f"[pipeline] Tiled into {len(tiles)} tiles (256x256, edge tiles partial, nodata tiles skipped)")

    store = TileStore(index_dir=index_dir)

    # Pass 1 (CPU-only, cheap): per-tile quality score + reflectance -> FIXED true-color stretch.
    # Same reflectance bounds for every tile and every date (RADIOMETRY_CONFIG); the per-scene
    # BOA additive offset is 0 for both currently staged dates (already offset-applied upstream).
    boa_offset_dn = boa_offset_dn_for_scene(scene_id)
    print(f"[pipeline] True-color: fixed bounds {RADIOMETRY_CONFIG['clip_reflectance']} reflectance "
          f"(DN {RADIOMETRY_CONFIG['clip_dn_equivalent']}), gamma={RADIOMETRY_CONFIG['gamma']}, "
          f"BOA offset subtracted={boa_offset_dn} DN")
    cloud_fractions: list[float] = []
    rgb_list: list = []
    sample_saved = False
    for tile in tiles:
        cloud_fractions.append(cloud_fraction(tile.bands[SCL_BAND]))
        rgb_uint8 = make_true_color_uint8(
            {b: tile.bands[b] for b in RGB_BANDS}, nodata=tile.nodata, boa_offset_dn=boa_offset_dn
        )
        rgb_list.append(rgb_uint8)
        if save_sample and not sample_saved:
            sample_path = sample_png_path or (get_settings().data_dir / "sample_tile_true_color.png")
            save_sample_png(rgb_uint8, sample_path)
            print(f"[pipeline] Saved sample stretched true-color tile PNG ({tile.tile_id}) -> {sample_path}")
            sample_saved = True
    quality_ts = _now_iso()

    # Pass 2 (GPU): batch-embed every tile through the EmbeddingModel - a handful of
    # forward passes instead of one per tile, which matters once scenes run into the
    # thousands of tiles. batch_size keeps VRAM bounded regardless of scene size
    # (ViT-B-32 batches are tiny - well under the 8GB budget even at generous sizes).
    if rgb_list:
        t_embed = time.time()
        vectors = embedding_model.encode_images(rgb_list, batch_size=embed_batch_size)
        embed_elapsed_ms = (time.time() - t_embed) * 1000.0
        embed_latencies_ms = [embed_elapsed_ms / len(rgb_list)] * len(rgb_list)
        n_batches = -(-len(rgb_list) // embed_batch_size)
        print(f"[pipeline] Embedded {len(rgb_list)} tiles in {n_batches} batch(es) of up to {embed_batch_size}")
    else:
        vectors, embed_latencies_ms = np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    embedded_ts = _now_iso()

    records = [
        {
            "tile_id": tile.tile_id,
            "scene_id": scene_id,
            "sensor": sensor,
            "acq_date": acq_date,
            "geom_wkt_4326": tile.footprint_wkt_4326,
            "cloud_fraction": cf,
            "embedding": vec,
            "processing_history": [
                {"step": "read", "at": read_ts},
                {"step": "tiled", "at": tiled_ts},
                {"step": "quality_scored", "at": quality_ts},
                {"step": "embedded", "at": embedded_ts},
                {"step": "indexed", "at": _now_iso()},
            ],
        }
        for tile, cf, vec in zip(tiles, cloud_fractions, vectors)
    ]

    # bands actually present on disk for this observation (not just the RGB+SCL we read)
    bands_on_disk = tuple(sorted(
        p.stem for p in scene_path.glob("*.tif")
        if p.stem in {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"}
    )) or ALL_BANDS
    idx_csv = get_settings().index_dir / "spectral_indices_per_tile.csv"
    store.add_tiles(
        records,
        dataset_dir=scene_id,
        crs=str(scene.crs) if scene.crs else None,
        bands=bands_on_disk,
        aoi_name="ayodhya_44RPQ_scaled_82km" if scene_id.endswith("_scaled") else "ayodhya_44RPQ_demo",
        indices_ref=str(idx_csv) if idx_csv.is_file() else None,
    )
    try:
        n_fired = _evaluate_watch_areas_on_ingest(store.repo, scene_id)
        if n_fired:
            print(f"[pipeline] watch areas: {n_fired} notification(s) fired for observation '{scene_id}'")
    except Exception as e:  # watch evaluation must never fail an ingest
        print(f"[pipeline] watch-area evaluation skipped: {type(e).__name__}: {e}")

    store.save()
    store.close()

    build_time_s = time.time() - t_start
    index_size_mb = store.index_path.stat().st_size / (1024 * 1024) if store.index_path.is_file() else 0.0
    mean_latency_ms = float(np.mean(embed_latencies_ms)) if embed_latencies_ms else 0.0

    report = {
        "scene_id": scene_id,
        "tiles_added": len(records),
        "index_total_vectors": store.count(),
        "build_time_s": round(build_time_s, 3),
        "index_size_mb": round(index_size_mb, 3),
        "mean_embed_latency_ms": round(mean_latency_ms, 3),
        "ingested_at": _now_iso(),
    }
    if record_manifest:
        record_radiometry_config(RADIOMETRY_CONFIG)
        # separate manifest entry: the display-only per-observation true-color
        # stretch (analyst UI thumbnails/detail), NOT part of analysis radiometry
        record_analysis_section("true_color_display_stretch", TRUE_COLOR_DISPLAY_STRETCH_CONFIG)
        append_ingest_run(report)

    print("\n[pipeline] Run report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="geoseek-ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest one staged scene directory into the tile index+DB")
    ingest_p.add_argument("scene_path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        ingest_scene(args.scene_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[pipeline] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

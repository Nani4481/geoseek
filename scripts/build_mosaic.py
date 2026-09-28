"""Build the analyst UI's offline satellite basemap.

We hold 74 staged Sentinel-2 L2A observations across 15 AOIs scattered over
India (Ayodhya, Delhi NCR, Dehradun, Jaisalmer, Deccan, Kutch, Kerala
backwaters, Sundarbans, Kanha...) plus 3 staged Maxar Open Data events
(Sikkim/Chungthang floods, LA wildfires). This reads the most recent
observation for each AOI straight off its native rasters (per-band GeoTIFFs
for Sentinel-2, the pre-composited visual.tif for Maxar), reprojects it (with
GDAL's averaging resampler, so it is also the downsample step) from its
native UTM zone to plain EPSG:4326, and writes one small true-colour WebP per
AOI plus an index of their bboxes. CoordMap loads that index once and draws
whichever regional tiles intersect the current view straight from these
bundled files - no tile service, no COG reads, no network call once the app
is running.

Each AOI is sized to the same TARGET_LONG_SIDE_PX regardless of its ground
footprint (a fixed deg/px would render an 80km Sentinel-2 AOI at a usable
size but a 5km Maxar chip as a handful of pixels) - so every staged region
reads as imagery, not just the biggest ones.

    python scripts/build_mosaic.py

Output: src/geoseek/analyst/web/mosaic/<aoi>.webp (one per region)
         + src/geoseek/analyst/web/mosaic_index.json (name, bbox, file, date)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.ingest.embed import make_true_color_uint8, true_color_bounds_for_scene, true_color_offsets_for_scene
from geoseek.staging.manifest import load_manifest, write_manifest

S2_BANDS = ("B04", "B03", "B02")
DST_CRS = "EPSG:4326"
TARGET_LONG_SIDE_PX = 900  # every AOI is sized to this, regardless of ground footprint
OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "web" / "mosaic"
INDEX_PATH = OUT_DIR.parent / "mosaic_index.json"


def _resolution_for(src_crs, src_bounds) -> float:
    w, s, e, n = transform_bounds(src_crs, DST_CRS, *src_bounds)
    return max(e - w, n - s) / TARGET_LONG_SIDE_PX


def _reproject_band(src, band_index: int, transform, width: int, height: int, src_nodata=None) -> np.ndarray:
    dst = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=rasterio.band(src, band_index), destination=dst,
        src_transform=src.transform, src_crs=src.crs, src_nodata=src_nodata,
        dst_transform=transform, dst_crs=DST_CRS,
        resampling=Resampling.average, dst_nodata=0,
    )
    return dst


def build_sentinel2(scene_key: str, dataset_dir: Path):
    """-> (rgb_uint8 HxWx3, alpha_mask HxW uint8, (w, s, e, n))"""
    b0 = dataset_dir / f"{S2_BANDS[0]}.tif"
    with rasterio.open(b0) as src:
        res = _resolution_for(src.crs, src.bounds)
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds, resolution=res)
    bands: dict[str, np.ndarray] = {}
    valid = None
    for b in S2_BANDS:
        with rasterio.open(dataset_dir / f"{b}.tif") as src:
            arr = _reproject_band(src, 1, transform, width, height)
        bands[b] = arr
        v = arr > 0
        valid = v if valid is None else (valid & v)
    rgb = make_true_color_uint8(
        bands, nodata=0,
        per_band_offset_dn=true_color_offsets_for_scene(scene_key),
        per_band_bounds_dn=true_color_bounds_for_scene(scene_key),
    )
    w, n = transform.c, transform.f
    e, s = w + transform.a * width, n + transform.e * height
    return rgb, (valid.astype(np.uint8) * 255), (w, s, e, n)


def build_maxar(dataset_dir: Path):
    """-> (rgb_uint8 HxWx3, alpha_mask HxW uint8, (w, s, e, n)) from visual.tif -
    already a display-ready RGB composite, so no true-colour stretch needed."""
    with rasterio.open(dataset_dir / "visual.tif") as src:
        res = _resolution_for(src.crs, src.bounds)
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds, resolution=res)
        bands = [_reproject_band(src, i, transform, width, height, src_nodata=0) for i in (1, 2, 3)]
    rgb = np.clip(np.stack(bands, axis=-1), 0, 255).astype(np.uint8)
    valid = (bands[0] > 0) & (bands[1] > 0) & (bands[2] > 0)
    w, n = transform.c, transform.f
    e, s = w + transform.a * width, n + transform.e * height
    return rgb, (valid.astype(np.uint8) * 255), (w, s, e, n)


def main() -> None:
    settings = get_settings()
    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")

    s2_groups: dict[str, list] = {}
    maxar_groups: dict[str, list] = {}
    for obs in repo.list_observations():
        scene = repo.get_scene(obs.scene_id)
        if not scene:
            continue
        aoi = obs.aoi_name or obs.observation_id
        if scene.collection_id == "sentinel-2-l2a":
            s2_groups.setdefault(aoi, []).append((obs, scene))
        elif scene.collection_id == "maxar-opendata":
            maxar_groups.setdefault(aoi, []).append((obs, scene))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    t0 = time.time()

    for aoi, items in sorted(s2_groups.items()):
        items.sort(key=lambda x: x[0].acquired_at)
        obs, scene = items[-1]  # most recent observation for this AOI
        dataset_dir = settings.datasets_dir / (obs.dataset_dir or obs.observation_id)
        rgb, alpha, bbox = build_sentinel2(scene.scene_id, dataset_dir)
        entries.append(_save_tile(aoi, rgb, alpha, bbox, obs))
        print(f"  [s2]    {aoi}: {entries[-1]['width']}x{entries[-1]['height']} from {obs.observation_id} ({obs.acquired_at})")

    for aoi, items in sorted(maxar_groups.items()):
        items.sort(key=lambda x: x[0].acquired_at)
        obs, scene = items[-1]
        dataset_dir = settings.datasets_dir / (obs.dataset_dir or obs.observation_id)
        rgb, alpha, bbox = build_maxar(dataset_dir)
        entries.append(_save_tile(aoi, rgb, alpha, bbox, obs))
        print(f"  [maxar] {aoi}: {entries[-1]['width']}x{entries[-1]['height']} from {obs.observation_id} ({obs.acquired_at})")

    INDEX_PATH.write_text(json.dumps({"tiles": entries}, indent=2), encoding="utf-8")
    total_bytes = sum((OUT_DIR / Path(e["file"]).name).stat().st_size for e in entries)
    elapsed = time.time() - t0
    print(f"\n{len(entries)} regional mosaic tiles, {total_bytes / 1e6:.2f} MB total, built in {elapsed:.1f}s")

    manifest = load_manifest()
    artifacts = [a for a in manifest.setdefault("artifacts", []) if a.get("name") != "analyst_ui_satellite_mosaic"]
    artifacts.append({
        "name": "analyst_ui_satellite_mosaic",
        "source_url": "rendered internally from our own staged Sentinel-2 L2A + Maxar Open Data scenes (not re-fetched)",
        "local_path": str(OUT_DIR.resolve()),
        "byte_size": total_bytes,
        "n_tiles": len(entries),
        "target_long_side_px": TARGET_LONG_SIDE_PX,
        "license": "derived from Copernicus Sentinel-2 (ESA) + Maxar Open Data (CC-BY-NC-4.0); mosaic composite is project-internal",
        "build_seconds": round(elapsed, 1),
    })
    manifest["artifacts"] = artifacts
    write_manifest(manifest)
    print("recorded provenance for analyst_ui_satellite_mosaic in the manifest")


def _feather_edges(alpha: np.ndarray, margin_frac: float = 0.07) -> np.ndarray:
    """Fade the alpha mask smoothly to 0 over the outer margin_frac of each
    edge. The nodata mask alone is a crisp 0/255 cut at the tile's own bbox,
    which is what made every AOI patch read as a hard rectangle sitting on
    top of the globe/2D basemap rather than blending into it."""
    h, w = alpha.shape
    mx, my = max(1, int(w * margin_frac)), max(1, int(h * margin_frac))
    ramp_x = np.ones(w, dtype=np.float32)
    ramp_x[:mx] = np.linspace(0, 1, mx, dtype=np.float32)
    ramp_x[-mx:] = np.minimum(ramp_x[-mx:], np.linspace(1, 0, mx, dtype=np.float32))
    ramp_y = np.ones(h, dtype=np.float32)
    ramp_y[:my] = np.linspace(0, 1, my, dtype=np.float32)
    ramp_y[-my:] = np.minimum(ramp_y[-my:], np.linspace(1, 0, my, dtype=np.float32))
    feather = np.outer(ramp_y, ramp_x)
    return (alpha.astype(np.float32) * feather).astype(np.uint8)


def _save_tile(aoi: str, rgb: np.ndarray, alpha: np.ndarray, bbox: tuple, obs) -> dict:
    alpha = _feather_edges(alpha)
    img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    img.putalpha(Image.fromarray(alpha, mode="L"))
    # WebP over PNG: ~9x smaller for this photographic content at quality 80
    # (measured: 3.5 MB PNG -> 0.39 MB WebP for one tile) with no visible loss
    # at basemap scale, and every target browser decodes it natively in
    # <img>/canvas - matters because "under 1s" + "no network" means every
    # byte here is paid for by the local disk read.
    fname = f"{aoi}.webp"
    img.save(OUT_DIR / fname, format="WEBP", quality=80, method=6)
    return {
        "name": aoi, "bbox": [round(x, 5) for x in bbox], "file": f"mosaic/{fname}",
        "date": obs.acquired_at, "observation_id": obs.observation_id,
        "width": img.width, "height": img.height,
    }


if __name__ == "__main__":
    main()

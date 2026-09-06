"""Phase 7b: stage geographically DIVERSE Sentinel-2 L2A AOIs for the
scalability + generalization test (see PHASE7B.md).

Same network source as ``geoseek.staging.download_datasets`` (Earth Search
STAC v1 -> the public AWS Sentinel-2 COG bucket), same windowed-COG-read
mechanic (rasterio range requests, WarpedVRT for the 20m SCL band onto the
10m grid) and the same band set (``B04,B03,B02,SCL`` - all ``ingest_scene``
ever reads). The one generalization: band URLs come from the STAC item's own
``assets`` hrefs instead of a hardcoded single-MGRS-tile ``COG_BASE`` path, so
any tile works, not just 44RPQ.

This module is staging-only (network + local file writes under
data/datasets/) - it does not touch the catalog/index. See
``ingest_diverse_scene.py`` for the ingest half.

Run:
    python scripts/stage_diverse_aois.py --list                 # show region plan
    python scripts/stage_diverse_aois.py --region dehradun       # stage 1 more date
    python scripts/stage_diverse_aois.py --region dehradun --n 3 # stage 3 dates total
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import load_manifest, record_artifact, write_manifest

STAC_ROOT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"
BANDS = ("B04", "B03", "B02", "SCL")
BAND_ASSET_KEY = {"B04": "red", "B03": "green", "B02": "blue", "SCL": "scl"}
DATA_LICENSE = "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)"

# Buffer shrunk inward from each item's own STAC bbox so the AOI window sits
# safely inside the product's raster bounds (mirrors the 1km buffer rationale
# in geoseek.staging.download_datasets).
EDGE_BUFFER_DEG = 0.014  # ~1.5 km

# Reflectance sanity bounds copied from download_datasets._confirm_extra_band_
# reflectance_scale (same physical check: BOA offset already applied, no raw
# -1000 pedestal). Phase 7b spans much more varied surfaces (bright desert
# sand, salt crust) than the Ayodhya check was written for, so out-of-range
# here is recorded as a WARNING, not a hard abort.
MEDIAN_REFLECTANCE_SANE = (0.02, 0.65)

SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "summer/pre-monsoon", 4: "summer/pre-monsoon", 5: "summer/pre-monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post-monsoon", 11: "post-monsoon",
}

# Non-overlapping search windows tried in order for the 2nd/3rd/... date of a
# region - each a full Oct-Apr low-cloud season in a different year, with one
# monsoon window (higher cloud tolerance) thrown in deliberately for realism.
DATE_WINDOWS: list[tuple[str, str, float]] = [
    ("2023-10-01T00:00:00Z", "2024-04-30T00:00:00Z", 20.0),
    ("2022-10-01T00:00:00Z", "2023-04-30T00:00:00Z", 20.0),
    ("2021-10-01T00:00:00Z", "2022-04-30T00:00:00Z", 20.0),
    ("2020-10-01T00:00:00Z", "2021-04-30T00:00:00Z", 20.0),
    ("2024-05-01T00:00:00Z", "2024-09-30T00:00:00Z", 40.0),  # monsoon: cloudier, expected
    ("2019-10-01T00:00:00Z", "2020-04-30T00:00:00Z", 20.0),
    ("2023-05-01T00:00:00Z", "2023-09-30T00:00:00Z", 40.0),
]

# --- Region registry --------------------------------------------------------
# probe_bbox: small box used to locate the covering MGRS item via STAC search.
# category / display_name: for the provenance manifest + tier reports.
REGIONS: dict[str, dict] = {
    "dehradun": {
        "display_name": "Dehradun / Rishikesh (Himalayan foothills)",
        "category": "Himalayan foothills",
        "probe_bbox": (77.95, 29.95, 78.35, 30.35),
    },
    "jaisalmer": {
        "display_name": "Jaisalmer (Thar desert edge)",
        "category": "Thar desert edge",
        "probe_bbox": (70.75, 26.75, 71.05, 27.05),
    },
    "sundarbans": {
        "display_name": "Sundarbans (Bengal coastal delta)",
        "category": "coastal delta",
        "probe_bbox": (88.70, 21.80, 89.00, 22.10),
    },
    "delhi_ncr": {
        "display_name": "Delhi NCR (dense urban)",
        "category": "dense urban",
        "probe_bbox": (76.95, 28.45, 77.25, 28.75),
    },
    "kanha": {
        "display_name": "Kanha (Central India forest/plateau)",
        "category": "forest/plateau",
        "probe_bbox": (80.45, 22.15, 80.75, 22.45),
    },
    "kerala_backwaters": {
        "display_name": "Alappuzha (Kerala backwaters, Western Ghats coast)",
        "category": "coastal wetland",
        "probe_bbox": (76.15, 9.75, 76.45, 10.05),
    },
    "kutch": {
        "display_name": "Rann of Kutch (salt marsh desert)",
        "category": "salt desert",
        "probe_bbox": (70.15, 23.70, 70.45, 24.00),
    },
    "deccan": {
        "display_name": "Kurnool (Deccan plateau)",
        "category": "Deccan plateau",
        "probe_bbox": (77.35, 16.35, 77.65, 16.65),
    },
}

MANIFEST_KEY = "diverse_aois"


def _season(acq_date: str) -> str:
    month = int(acq_date.split("-")[1])
    return SEASON_BY_MONTH[month]


def _mgrs_tile(scene_id: str) -> str:
    parts = scene_id.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def search_items(bbox: tuple[float, float, float, float], dt_range: tuple[str, str], max_cloud: float,
                  limit: int = 20) -> list[dict]:
    body = {
        "collections": [STAC_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{dt_range[0]}/{dt_range[1]}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    r = requests.post(f"{STAC_ROOT}/search", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["features"]


def already_staged_scene_ids(region_slug: str) -> set[str]:
    manifest = load_manifest()
    return {rec["scene_id"] for rec in manifest.get(MANIFEST_KEY, []) if rec["region"] == region_slug}


def pick_new_item(region: dict, exclude_ids: set[str]) -> dict | None:
    for start, end, max_cloud in DATE_WINDOWS:
        try:
            feats = search_items(region["probe_bbox"], (start, end), max_cloud=max_cloud)
        except Exception as e:
            print(f"[stage-diverse]   search failed for window {start}/{end}: {e}")
            continue
        for f in feats:
            if f["id"] not in exclude_ids:
                return f
    return None


def _download_full_file(url: str, dest: Path) -> float:
    """Plain sequential HTTP GET of the whole remote file to a local path.

    Measured on this connection: a naive rasterio/GDAL windowed VSICURL read
    of a near-full-tile AOI (thousands of small ranged requests) ran at
    ~300-400 KB/s effective throughput (9 min for one ~180 MB band); a single
    bulk sequential GET of the same remote file sustains ~1 MB/s (a real
    last-mile bandwidth limit here, confirmed by testing 2 concurrent GETs
    against this bucket - aggregate throughput did not scale with concurrency,
    so this is not a request-pattern fix, just the least-overhead way to move
    the bytes). Since every Phase 7b AOI is a near-full MGRS tile anyway (see
    EDGE_BUFFER_DEG), windowing the remote read saves almost nothing - it is
    strictly better to pull the whole file once, then crop locally (rasterio
    against a local file is fast: no network round trips at all).
    """
    t0 = time.time()
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return time.time() - t0


def _fetch_window_from_item(item: dict, aoi_bounds_4326: tuple[float, float, float, float],
                             out_dir: Path, bands: tuple[str, ...] = BANDS) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="geoseek_stage_"))
    try:
        full_paths: dict[str, Path] = {}
        urls = {band: item["assets"][BAND_ASSET_KEY[band]]["href"] for band in bands}

        with ThreadPoolExecutor(max_workers=len(bands)) as pool:
            futures = {
                pool.submit(_download_full_file, url, tmp_dir / f"{band}_full.tif"): band
                for band, url in urls.items()
            }
            for fut in as_completed(futures):
                band = futures[fut]
                dt = fut.result()
                dest = tmp_dir / f"{band}_full.tif"
                full_paths[band] = dest
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"[stage-diverse]   {band}: downloaded {size_mb:.1f} MB in {dt:.1f}s "
                      f"({size_mb / dt:.2f} MB/s) -> {dest.name}")

        ref_band = bands[0]
        with rasterio.open(full_paths[ref_band]) as ref_ds:
            ref_crs = ref_ds.crs
            aoi_bounds_native = transform_bounds("EPSG:4326", ref_crs, *aoi_bounds_4326)
            window = from_bounds(*aoi_bounds_native, transform=ref_ds.transform).round_lengths().round_offsets()
            window = window.intersection(Window(0, 0, ref_ds.width, ref_ds.height))
            out_transform = ref_ds.window_transform(window)
            out_height, out_width = int(window.height), int(window.width)
        print(f"[stage-diverse]   CRS={ref_crs}  AOI window={out_width}x{out_height} px (10m grid, local crop)")

        records = []
        for band in bands:
            url = urls[band]
            needs_resample = band.upper() == "SCL"
            t0 = time.time()
            with rasterio.open(full_paths[band]) as src:
                if needs_resample:
                    with WarpedVRT(
                        src, crs=ref_crs, transform=out_transform,
                        width=out_width, height=out_height, resampling=Resampling.nearest,
                    ) as vrt:
                        data = vrt.read(1)
                        nodata = vrt.nodata
                else:
                    data = src.read(1, window=window, out_shape=(out_height, out_width))
                    nodata = src.nodata
            dt = time.time() - t0

            out_path = out_dir / f"{band}.tif"
            profile = {
                "driver": "GTiff", "height": out_height, "width": out_width, "count": 1,
                "dtype": data.dtype, "crs": ref_crs, "transform": out_transform,
                "nodata": nodata, "compress": "deflate",
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            size_kb = out_path.stat().st_size / 1024
            print(f"[stage-diverse]   {band}: local crop {size_kb:.1f} KB in {dt:.2f}s -> {out_path}")
            records.append({"band": band, "path": out_path, "url": url, "array": data, "nodata": nodata})
        return records
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _reflectance_check(records: list[dict]) -> dict:
    ref = {r["band"]: r["array"] for r in records if r["band"] in ("B04", "B03", "B02")}
    valid = np.ones_like(ref["B04"], dtype=bool)
    for arr in ref.values():
        valid &= arr != 0
    stats = {}
    n_valid = int(valid.sum())
    if n_valid > 0:
        for b, arr in ref.items():
            v = arr[valid].astype(np.float64)
            stats[b] = {"median_reflectance": round(float(np.median(v)) / 10000.0, 4),
                        "min": float(v.min())}
    sane = n_valid > 0 and all(
        MEDIAN_REFLECTANCE_SANE[0] <= s["median_reflectance"] <= MEDIAN_REFLECTANCE_SANE[1] for s in stats.values()
    )
    return {"n_valid_px_sampled": n_valid, "per_band": stats, "sane": sane}


def _centered_box(tile_bbox: list[float], size_km: float) -> tuple[float, float, float, float]:
    """A size_km x size_km box centered in the item's tile bbox (clipped to it).

    Used only for the deliberately-small incremental-ingest proof scenes -
    the bulk scale-up ingestion always uses the (near-)full tile via
    EDGE_BUFFER_DEG instead."""
    import math
    w, s, e, n = tile_bbox
    clon, clat = (w + e) / 2.0, (s + n) / 2.0
    half_lat = (size_km / 2.0) / 111.32
    half_lon = (size_km / 2.0) / (111.32 * max(math.cos(math.radians(clat)), 0.1))
    box = (clon - half_lon, clat - half_lat, clon + half_lon, clat + half_lat)
    return (max(box[0], w + EDGE_BUFFER_DEG), max(box[1], s + EDGE_BUFFER_DEG),
            min(box[2], e - EDGE_BUFFER_DEG), min(box[3], n - EDGE_BUFFER_DEG))


def stage_one(region_slug: str, size_km: float | None = None) -> dict:
    """Stage exactly one NEW date for a region (skips scene_ids already staged
    for it). Returns a record dict; raises if no new low-cloud scene is found.

    ``size_km``: override the default (near-full-MGRS-tile) AOI with a smaller
    box of this physical size, centered in the tile - used for the
    incremental-ingest proof scenes, which are deliberately sized to add close
    to +1000 tiles rather than a full +1500-1800 tile scene."""
    settings = get_settings()
    region = REGIONS[region_slug]
    exclude = already_staged_scene_ids(region_slug)

    print(f"\n=== {region['display_name']} ({region_slug}) - date #{len(exclude) + 1} "
          f"{'(size_km=' + str(size_km) + ')' if size_km else ''} ===")
    item = pick_new_item(region, exclude)
    if item is None:
        raise RuntimeError(f"No new low-cloud scene found for region '{region_slug}' "
                            f"(already staged: {sorted(exclude)})")

    scene_id = item["id"]
    props = item["properties"]
    acq_date = props["datetime"][:10]
    cloud_pct = float(props.get("eo:cloud_cover", -1))
    nodata_pct_scene = float(props.get("s2:nodata_pixel_percentage", -1))
    tile_bbox = item["bbox"]
    if size_km:
        aoi_bounds = _centered_box(tile_bbox, size_km)
    else:
        aoi_bounds = (
            tile_bbox[0] + EDGE_BUFFER_DEG, tile_bbox[1] + EDGE_BUFFER_DEG,
            tile_bbox[2] - EDGE_BUFFER_DEG, tile_bbox[3] - EDGE_BUFFER_DEG,
        )
    print(f"[stage-diverse] scene={scene_id}  acq={acq_date}  cloud={cloud_pct:.4f}%  "
          f"scene_nodata={nodata_pct_scene:.2f}%  mgrs={_mgrs_tile(scene_id)}")
    print(f"[stage-diverse] AOI bounds (tile bbox - buffer): {aoi_bounds}")

    out_dir = settings.datasets_dir / scene_id
    if all((out_dir / f"{b}.tif").is_file() for b in BANDS):
        print(f"[stage-diverse]   all bands already on disk at {out_dir} - skipping network fetch.")
        records = []
        for b in BANDS:
            with rasterio.open(out_dir / f"{b}.tif") as ds:
                records.append({"band": b, "path": out_dir / f"{b}.tif",
                                 "url": item["assets"][BAND_ASSET_KEY[b]]["href"],
                                 "array": ds.read(1), "nodata": ds.nodata})
    else:
        records = _fetch_window_from_item(item, aoi_bounds, out_dir)

    # local nodata fraction over the fetched AOI window (may differ from the
    # whole-tile s2:nodata_pixel_percentage above, since we clip to the tile
    # bbox minus a small buffer, not the full original raster).
    ref_arr = next(r["array"] for r in records if r["band"] == "B04")
    ref_nodata = next(r["nodata"] for r in records if r["band"] == "B04")
    if ref_nodata is not None:
        local_nodata_frac = float(np.mean(ref_arr == ref_nodata))
    else:
        local_nodata_frac = float(np.mean(ref_arr == 0))

    reflectance = _reflectance_check(records)
    if not reflectance["sane"]:
        print(f"[stage-diverse]   WARNING: reflectance sanity check out of "
              f"[{MEDIAN_REFLECTANCE_SANE[0]},{MEDIAN_REFLECTANCE_SANE[1]}] for {scene_id}: "
              f"{reflectance['per_band']} - proceeding anyway (Phase 7b spans surfaces the "
              f"Ayodhya-derived thresholds weren't written for; recorded, not fatal).")

    band_shas = {}
    for rec in records:
        artifact_name = f"sentinel2-diverse-{scene_id}-{rec['band']}"
        arec = record_artifact(name=artifact_name, source_url=rec["url"], local_path=rec["path"],
                                license=DATA_LICENSE)
        band_shas[rec["band"]] = arec.sha256
        print(f"[stage-diverse]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

    source_url = records[0]["url"].rsplit("/", 1)[0] + "/"  # COG folder URL, generic per-scene

    record = {
        "region": region_slug,
        "display_name": region["display_name"],
        "category": region["category"],
        "scene_id": scene_id,
        "dataset_dir": scene_id,
        "mgrs_tile": _mgrs_tile(scene_id),
        "acq_date": acq_date,
        "season": _season(acq_date),
        "cloud_cover_pct": cloud_pct,
        "scene_nodata_pct": nodata_pct_scene,
        "local_window_nodata_frac": round(local_nodata_frac, 4),
        "aoi_bounds_4326": list(aoi_bounds),
        "source_url": source_url,
        "band_sha256": band_shas,
        "bands": list(BANDS),
        "reflectance_check": reflectance,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest = load_manifest()
    section = manifest.setdefault(MANIFEST_KEY, [])
    section.append(record)
    write_manifest(manifest)

    print(f"[stage-diverse] staged + recorded -> {out_dir}")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="geoseek-stage-diverse-aois")
    ap.add_argument("--list", action="store_true", help="list the region registry and exit")
    ap.add_argument("--region", choices=sorted(REGIONS), help="region slug to stage one more date for")
    ap.add_argument("--n", type=int, default=1, help="how many additional dates to stage for --region")
    args = ap.parse_args(argv)

    if args.list:
        for slug, r in REGIONS.items():
            staged = already_staged_scene_ids(slug)
            print(f"  {slug:20s} {r['category']:18s} {r['display_name']:50s} staged_dates={len(staged)}")
        return

    if not args.region:
        ap.error("pass --region <slug> (--list to see options)")

    print_startup_banner()
    for i in range(args.n):
        stage_one(args.region)


if __name__ == "__main__":
    main()

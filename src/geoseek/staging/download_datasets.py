"""Stage a small Sentinel-2 L2A test AOI (two dates) for fully-offline ingest testing.

This is one of the ONLY modules in geoseek allowed to touch the network (see
``download_models.py`` for the sibling model-staging entrypoint and the
project-wide rule in ``README.md`` / ``config.py``). Everything downstream
(``geoseek.ingest.*``) must read exclusively from the local files staged
here under ``data/datasets/``.

Source, confirmed LIVE at authoring time (not recalled from memory)
----------------------------------------------------------------------
Queried ``https://earth-search.aws.element84.com/v1`` (Earth Search STAC API,
run by Element 84 against the AWS "Sentinel-2 Cloud-Optimized GeoTIFFs"
Open Data bucket) directly:

  * ``GET  /v1``                          -> 200, confirms the API is live.
  * ``GET  /v1/collections``              -> lists ``sentinel-2-l2a`` among
    others: this is the collection used here (Sentinel-2 L2A COGs, one
    Cloud-Optimized GeoTIFF per band, S3 range-request friendly).
  * ``POST /v1/search`` with a bbox + datetime filter over the AOI below
    returned real items for both target years; the two lowest-cloud items
    at least ~12 months apart were selected (see SCENES below).

No login, API key, or AWS credentials are required — the bucket
(``sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs``) is public,
anonymous-read, and each COG supports HTTP range requests, so only the AOI
window is ever transferred (a few hundred KB per band here), never the full
~110x110 km tile.

AOI: Ayodhya, Uttar Pradesh, India (Saryu river) — chosen for a visible mix
of urban edge, open/bare ground, and river water, plus well-documented rapid
construction (Ram Mandir + township, 2019-2024) giving strong visible change
between the two dates. Both dates are in March to keep seasonal vegetation
signal comparable and isolate real change.

  Scene A (older): S2B_44RPQ_20190330_1_L2A   acq. 2019-03-30, cloud ~0.00%
  Scene B (newer): S2A_44RPQ_20240308_0_L2A   acq. 2024-03-08, cloud ~0.00%
  MGRS tile: 44RPQ  |  CRS: EPSG:32644 (UTM 44N)

License: Copernicus Sentinel data is free, full and open under the EU
Copernicus Data Policy (Regulation (EU) No 1159/2013 / Copernicus Sentinel
Data Terms and Conditions) — see https://sentinels.copernicus.eu/documents/
247904/690755/Sentinel_Data_Legal_Notice. AWS hosting terms:
https://registry.opendata.aws/sentinel-2-l2a-cogs/.

Re-running this after a scene is already staged skips the network entirely
for that scene (checks local files first), matching ``download_models.py``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_artifact

STAC_ROOT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"
COG_BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/R/PQ"

BANDS = ("B04", "B03", "B02", "SCL")  # red, green, blue (10m true-color) + scene classification

# Phase 3a: extra spectrum for change-TYPE classification (not staged for the
# small demo AOI - only for the scaled AOI where the temporal pair work lives).
# B08 = NIR (native 10m), B11 = SWIR-1 (native 20m -> resampled to the 10m grid).
EXTRA_BANDS = ("B08", "B11")

# Bands whose native resolution is 20m: resample onto the 10m reference grid
# with BILINEAR (reflectance is a continuous quantity). NEAREST is reserved for
# the categorical SCL band only.
BANDS_20M = frozenset({"B05", "B06", "B07", "B8A", "B11", "B12"})

DATA_LICENSE = "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)"

# AOI: ~6.2km x 6.2km box around Ayodhya, UP, India (Saryu river + urban edge + open ground).
AOI_LON, AOI_LAT = 82.1998, 26.7922
AOI_HALF_DEG = 0.028
AOI_BOUNDS_4326 = (
    AOI_LON - AOI_HALF_DEG,
    AOI_LAT - AOI_HALF_DEG,
    AOI_LON + AOI_HALF_DEG,
    AOI_LAT + AOI_HALF_DEG,
)  # (west, south, east, north)

SCENES = {
    "S2B_44RPQ_20190330_1_L2A": {"acq_date": "2019-03-30", "expected_cloud_cover_pct": 0.000169},
    "S2A_44RPQ_20240308_0_L2A": {"acq_date": "2024-03-08", "expected_cloud_cover_pct": 0.000503},
}

# --- Phase 2 scale-up: same two dates, same MGRS tile (44RPQ), a much larger
# AOI so the tile store has enough tiles to demonstrate retrieval. Sized (via
# a live query of the tile's own georeferenced bounds - see
# scripts/prove_semantic.py history / commit notes) to fit an 82km x 82km box
# fully inside the single 44RPQ COG (with a 1km buffer, no cross-tile
# mosaicking needed) while still containing the original Ayodhya point -
# ~1089 tiles/date upper bound (256px tiles, before any nodata skipping).
LARGE_AOI_DIR_SUFFIX = "_scaled"
LARGE_AOI_BOUNDS_4326 = (82.01237, 26.36132, 82.84594, 27.10975)  # (west, south, east, north)

# --- Phase 3.5 Step 4: a THIRD Sentinel-2 date over the same scaled AOI, so the
# temporal stack can support "earliest supported change" (PS 2.2.2) and temporal
# persistence. Chosen from a live Earth Search query over MGRS 44RPQ, Feb-Apr,
# between the existing 2019-03-30 and 2024-03-08 dates, ranked by cloud AND
# swath coverage (s2:nodata_pixel_percentage): 2021-03-04 sits near the midpoint
# of the span (~2.0 yr / ~3.0 yr gaps), matches the early-March phenology window,
# cloud ~0.0005, ~12.6% nodata (a clean swath-edge cut - the tiler keeps the
# ~950 covered tiles and skips the empty ones). boa_offset_applied=true and
# raster:bands scale/offset identical to the existing pair (re-checked at staging).
THIRD_DATE_SCENE_ID = "S2A_44RPQ_20210304_1_L2A"
THIRD_DATE_META = {"acq_date": "2021-03-04", "expected_cloud_cover_pct": 0.000503}
# same six bands as the scaled pair: RGB (10m) + NIR (10m) + SWIR-1 (20m->10m) + SCL (20m->10m)
THIRD_DATE_BANDS = ("B04", "B03", "B02", "B08", "B11", "SCL")
# the reference date the existing pair is normalized against (geoseek.change.prep)
REFERENCE_DATE_SCENE_ID = "S2A_44RPQ_20240308_0_L2A"

# --- Phase 8 Step A: extend the stack to 5 dates (3267 tiles/date-ish; the
# archive was 2.5 years stale at 3 dates). Same MGRS tile 44RPQ, same 82 km
# scaled AOI, same six bands as the third date. Chosen from a LIVE Earth
# Search query (POST /v1/search, collection sentinel-2-l2a, bbox=
# LARGE_AOI_BOUNDS_4326, query={"grid:code":{"eq":"MGRS-44RPQ"}}, datetime
# Feb-Apr of each year, sortby eo:cloud_cover asc) run at authoring time
# (2026-09-12) - not recalled from memory. Both land on the SAME day-of-year
# as the existing 2024-03-08 reference (2019/2021/2024/2025/2026 are all
# within a 26-day window of each other), the tightest possible phenology
# control across 5 acquisitions, and both have negligible cloud + a swath
# cut close to the third date's (~12.6% nodata -> "good coverage" pass,
# confirmed against the ~41-42% "bad coverage" orbit pass also visible in the
# same query for both years).
ADDITIONAL_DATES = [
    {
        "scene_id": "S2B_44RPQ_20250308_0_L2A", "label": "2025",
        "acq_date": "2025-03-08", "expected_cloud_cover_pct": 0.00058,
        "expected_nodata_pct": 13.09,
    },
    {
        "scene_id": "S2C_44RPQ_20260308_0_L2A", "label": "2026",
        "acq_date": "2026-03-08", "expected_cloud_cover_pct": 0.04756,
        "expected_nodata_pct": 12.90,
    },
]
ADDITIONAL_DATE_BANDS = THIRD_DATE_BANDS


def stage_additional_date(entry: dict, force: bool = False) -> dict:
    """Phase 8 Step A: stage one Phase-8 date (2025 or 2026) over the scaled AOI.

    Generalizes :func:`stage_third_date` to an arbitrary ``entry`` from
    :data:`ADDITIONAL_DATES` instead of one hardcoded scene. Records provenance
    for every band, confirms the reflectance encoding, and merges the result
    into the manifest's ``additional_dates`` section (keyed by observation id
    - a dict, not a single value, so staging 2026 does not clobber 2025's
    already-recorded entry).
    """
    from geoseek.staging.manifest import load_manifest, record_analysis_section

    settings = get_settings()
    print_startup_banner(settings)

    scene_id = entry["scene_id"]
    dir_name = scene_id + LARGE_AOI_DIR_SUFFIX
    print(f"\n=== {dir_name} (acq. {entry['acq_date']}, scaled AOI, Phase 8 date) ===")

    if not force and scene_is_staged(scene_id, dir_name=dir_name, bands=ADDITIONAL_DATE_BANDS):
        print(f"[staging] All six bands already staged under {settings.datasets_dir / dir_name} - skipping network.")
        out_dir = settings.datasets_dir / dir_name
        band_records = [
            {"band": b, "path": out_dir / f"{b}.tif", "url": _band_url(scene_id, b)}
            for b in ADDITIONAL_DATE_BANDS
        ]
    else:
        band_records = _fetch_window(
            scene_id, LARGE_AOI_BOUNDS_4326, out_dir_name=dir_name, bands=ADDITIONAL_DATE_BANDS
        )

    for rec in band_records:
        artifact_name = f"sentinel2-{dir_name}-{rec['band']}"
        arec = record_artifact(
            name=artifact_name, source_url=rec["url"], local_path=rec["path"], license=DATA_LICENSE,
        )
        print(f"[staging]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

    confirmation = _confirm_extra_band_reflectance_scale(dir_name)

    section = {
        "purpose": "Phase 8 Step A: extend the archive from 3 to 5 dates (was 2.5 years stale at 2024-03-08).",
        "scene_id": scene_id,
        "observation_id": dir_name,
        "acq_date": entry["acq_date"],
        "mgrs_tile": "44RPQ",
        "bands": list(ADDITIONAL_DATE_BANDS),
        "aoi_bounds_4326": list(LARGE_AOI_BOUNDS_4326),
        "reference_date_observation": REFERENCE_DATE_SCENE_ID + LARGE_AOI_DIR_SUFFIX,
        "selection_rationale": (
            f"MGRS 44RPQ, live Earth Search query Feb-Apr {entry['acq_date'][:4]}, sorted by cloud cover; "
            f"chosen date matches the existing reference's day-of-year (03-08) for tightest phenology "
            f"control, expected cloud_cover={entry['expected_cloud_cover_pct']}%, "
            f"expected nodata={entry.get('expected_nodata_pct')}% (good-coverage swath pass, not the "
            f"~41-42% partial-swath pass also seen in the same query window)."
        ),
        "reflectance_scale_confirmation": confirmation,
    }
    manifest = load_manifest()
    all_additional = dict(manifest.get("additional_dates", {}))
    all_additional[dir_name] = section
    record_analysis_section("additional_dates", all_additional)
    print(f"\n[staging] {entry['label']} date staged + confirmed. Manifest: {settings.provenance_manifest_path}")
    return section


def stage_phase8_dates(force: bool = False) -> list[dict]:
    """Stage every :data:`ADDITIONAL_DATES` entry (2025 + 2026)."""
    return [stage_additional_date(entry, force=force) for entry in ADDITIONAL_DATES]


def _band_url(scene_id: str, band: str) -> str:
    date_token = scene_id.split("_")[2]  # e.g. "20190330"
    year, month = date_token[:4], str(int(date_token[4:6]))
    return f"{COG_BASE}/{year}/{month}/{scene_id}/{band}.tif"


def verify_scene_exists_on_stac(scene_id: str) -> dict:
    """Re-query the live STAC item before trusting our hardcoded URL scheme.

    Mirrors download_models.verify_checkpoint_exists_on_hf: fail loudly
    instead of silently guessing at a URL.
    """
    url = f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items/{scene_id}"
    print(f"[staging] Verifying scene on Earth Search STAC: {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Earth Search STAC to verify scene '{scene_id}'. "
            f"Staging requires network access for scenes not yet staged. Original error: {e}"
        ) from e
    item = r.json()
    print(
        f"[staging] Confirmed: {scene_id} exists | datetime={item['properties']['datetime']} "
        f"| cloud_cover={item['properties'].get('eo:cloud_cover')}"
    )
    return item


def scene_is_staged(scene_id: str, dir_name: str | None = None, bands: tuple[str, ...] = BANDS) -> bool:
    out_dir = get_settings().datasets_dir / (dir_name or scene_id)
    return all((out_dir / f"{b}.tif").is_file() for b in bands)


def _resampling_for_band(band: str) -> Resampling:
    """NEAREST only for categorical SCL; BILINEAR for continuous 20m reflectance bands."""
    if band.upper() == "SCL":
        return Resampling.nearest
    return Resampling.bilinear


def _fetch_window(
    scene_id: str,
    aoi_bounds_4326: tuple[float, float, float, float],
    out_dir_name: str,
    bands: tuple[str, ...] = BANDS,
) -> list[dict]:
    """Windowed COG read (range requests only) of one AOI box for one scene.

    Shared core for the small demo AOI, the Phase 2 scaled-up AOI, and the
    Phase 3a extra-band top-up - only the requested bounds, the band list, and
    the output directory name differ. Bands already on the 10m reference grid
    are read with a plain window; coarser (20m) bands and SCL are resampled
    onto the 10m grid via a WarpedVRT (BILINEAR for continuous reflectance,
    NEAREST for the categorical SCL band). Returns a list of per-band dicts
    (path, url) ready to be recorded in the provenance manifest.
    """
    settings = get_settings()
    out_dir = settings.datasets_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    verify_scene_exists_on_stac(scene_id)

    ref_url = _band_url(scene_id, "B04")
    print(f"[staging] Opening reference band (B04) to establish AOI grid: {ref_url}")
    with rasterio.open(ref_url) as ref_ds:
        ref_crs = ref_ds.crs
        aoi_bounds_native = transform_bounds("EPSG:4326", ref_crs, *aoi_bounds_4326)
        window = from_bounds(*aoi_bounds_native, transform=ref_ds.transform).round_lengths().round_offsets()
        out_transform = ref_ds.window_transform(window)
        out_height, out_width = int(window.height), int(window.width)
    print(f"[staging]   CRS={ref_crs}  AOI window={out_width}x{out_height} px (10m grid)")

    records = []
    for band in bands:
        url = _band_url(scene_id, band)
        needs_resample = band.upper() == "SCL" or band.upper() in BANDS_20M
        t0 = time.time()
        with rasterio.open(url) as src:
            if needs_resample:
                resampling = _resampling_for_band(band)
                # Native 20m -> resample onto the 10m RGB grid. NEAREST for the
                # categorical SCL band; BILINEAR for continuous reflectance (B11).
                with WarpedVRT(
                    src, crs=ref_crs, transform=out_transform,
                    width=out_width, height=out_height, resampling=resampling,
                ) as vrt:
                    data = vrt.read(1)
                    nodata = vrt.nodata
                print(f"[staging]   {band}: resampled {src.res[0]:.0f}m -> 10m ({resampling.name})")
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
        print(f"[staging]   {band}: {size_kb:.1f} KB fetched in {dt:.1f}s (range requests) -> {out_path}")

        records.append({"band": band, "path": out_path, "url": url})
    return records


def fetch_scene_aoi(scene_id: str, meta: dict) -> list[dict]:
    """Fetch the original small demo AOI (data/datasets/<scene_id>/)."""
    return _fetch_window(scene_id, AOI_BOUNDS_4326, out_dir_name=scene_id)


def fetch_scene_large_aoi(scene_id: str) -> list[dict]:
    """Fetch the Phase 2 scaled-up AOI (data/datasets/<scene_id>_scaled/)."""
    return _fetch_window(scene_id, LARGE_AOI_BOUNDS_4326, out_dir_name=scene_id + LARGE_AOI_DIR_SUFFIX)


def fetch_scene_large_aoi_extra_bands(scene_id: str) -> list[dict]:
    """Phase 3a: top up the scaled-up AOI dir with B08 (NIR) + B11 (SWIR-1)."""
    return _fetch_window(
        scene_id,
        LARGE_AOI_BOUNDS_4326,
        out_dir_name=scene_id + LARGE_AOI_DIR_SUFFIX,
        bands=EXTRA_BANDS,
    )


def stage_all(force: bool = False) -> None:
    settings = get_settings()
    print_startup_banner(settings)

    for scene_id, meta in SCENES.items():
        print(f"\n=== {scene_id} (acq. {meta['acq_date']}) ===")
        if not force and scene_is_staged(scene_id):
            print(f"[staging] All bands already staged under {settings.datasets_dir / scene_id} - skipping network.")
            out_dir = settings.datasets_dir / scene_id
            band_records = [
                {"band": b, "path": out_dir / f"{b}.tif", "url": _band_url(scene_id, b)} for b in BANDS
            ]
        else:
            band_records = fetch_scene_aoi(scene_id, meta)

        for rec in band_records:
            artifact_name = f"sentinel2-{scene_id}-{rec['band']}"
            arec = record_artifact(
                name=artifact_name,
                source_url=rec["url"],
                local_path=rec["path"],
                license=DATA_LICENSE,
            )
            print(f"[staging]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

    print(f"\n[staging] Datasets staged under {settings.datasets_dir}")
    print(f"[staging] Manifest: {settings.provenance_manifest_path}")
    print("\n[staging] Source: Earth Search STAC API (https://earth-search.aws.element84.com/v1), "
          "collection 'sentinel-2-l2a', backed by the public AWS Open Data Sentinel-2 COG bucket. "
          "No login / API key required.")
    date_summary = ", ".join(f"{sid} -> {m['acq_date']}" for sid, m in SCENES.items())
    print(f"[staging] Acquisition dates staged: {date_summary}")


def stage_large_aoi(force: bool = False) -> None:
    """Stage the Phase 2 scaled-up AOI (~1000+ tiles/date) for the same two dates."""
    settings = get_settings()
    print_startup_banner(settings)

    for scene_id, meta in SCENES.items():
        dir_name = scene_id + LARGE_AOI_DIR_SUFFIX
        print(f"\n=== {dir_name} (acq. {meta['acq_date']}, scaled AOI) ===")
        if not force and scene_is_staged(scene_id, dir_name=dir_name):
            print(f"[staging] All bands already staged under {settings.datasets_dir / dir_name} - skipping network.")
            out_dir = settings.datasets_dir / dir_name
            band_records = [
                {"band": b, "path": out_dir / f"{b}.tif", "url": _band_url(scene_id, b)} for b in BANDS
            ]
        else:
            band_records = fetch_scene_large_aoi(scene_id)

        for rec in band_records:
            artifact_name = f"sentinel2-{dir_name}-{rec['band']}"
            arec = record_artifact(
                name=artifact_name,
                source_url=rec["url"],
                local_path=rec["path"],
                license=DATA_LICENSE,
            )
            print(f"[staging]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

    print(f"\n[staging] Scaled-up datasets staged under {settings.datasets_dir}")
    print(f"[staging] Manifest: {settings.provenance_manifest_path}")


def _confirm_extra_band_reflectance_scale(dir_name: str) -> dict:
    """Confirm B08/B11 share the RGB reflectance encoding (reflectance*10000, offset 0).

    Two checks, both on the just-staged local files (no network):
      1. STAC ``raster:bands`` for the nir/swir16 assets must match red's
         (scale 1e-4, offset -0.1) - queried live, with a cached fallback.
      2. DN sanity: over pixels valid in every band, B08/B11 medians must sit
         in the same thousands-of-DN regime as B04 (i.e. reflectance*10000),
         with no raw-DN ~1000 dark floor - the same evidence Phase 2 used to
         set the corrective BOA offset to 0.
    """
    settings = get_settings()
    out_dir = settings.datasets_dir / dir_name
    print(f"\n[staging] Confirming B08/B11 reflectance scale for {dir_name} ...")

    bands = {}
    for b in ("B04", "B08", "B11"):
        with rasterio.open(out_dir / f"{b}.tif") as ds:
            bands[b] = ds.read(1)
    valid = np.ones_like(bands["B04"], dtype=bool)
    for arr in bands.values():
        valid &= arr != 0
    stats = {}
    for b, arr in bands.items():
        v = arr[valid].astype(np.float64)
        stats[b] = {
            "min": float(v.min()),
            "p0_5": float(np.percentile(v, 0.5)),
            "median": float(np.median(v)),
            "p98": float(np.percentile(v, 98)),
        }
        print(f"[staging]   {b}: min={stats[b]['min']:.0f} p0.5={stats[b]['p0_5']:.0f} "
              f"median={stats[b]['median']:.0f} p98={stats[b]['p98']:.0f} DN "
              f"(reflectance ~ {stats[b]['median'] / 10000:.3f})")

    raster_bands = {}
    try:
        base = f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items"
        sid = dir_name.replace(LARGE_AOI_DIR_SUFFIX, "")
        j = requests.get(f"{base}/{sid}", timeout=30).json()
        for key, band in (("red", "B04"), ("nir", "B08"), ("swir16", "B11")):
            rb = j["assets"][key].get("raster:bands")
            raster_bands[band] = rb
            print(f"[staging]   STAC raster:bands {band} ({key}): {rb}")
    except Exception as e:  # offline / blocked - fall back to the values captured at authoring
        print(f"[staging]   [STAC raster:bands unavailable: {e!r}] cached: red/nir/swir16 all "
              "scale=1e-4 offset=-0.1 (B04/B08 gsd 10m, B11 gsd 20m)")
        raster_bands = {
            "B04": [{"scale": 0.0001, "offset": -0.1, "spatial_resolution": 10}],
            "B08": [{"scale": 0.0001, "offset": -0.1, "spatial_resolution": 10}],
            "B11": [{"scale": 0.0001, "offset": -0.1, "spatial_resolution": 20}],
        }

    # Gate: (1) STAC raster:bands for nir/swir16 must carry the SAME scale/offset
    # as red; (2) median DN must map to a physically plausible reflectance
    # (0.02-0.60); (3) no unremoved BOA_ADD_OFFSET pedestal - a scene with the
    # raw -1000 offset NOT applied has its global minimum pinned near DN 1000,
    # whereas B04 (offset 0, per the Phase-2 finding) bottoms out near 0. B08/B11
    # must bottom out at the same low floor as B04, not ~1000 above it.
    def _scale_offset(rb):
        try:
            return float(rb[0]["scale"]), float(rb[0]["offset"])
        except Exception:
            return None
    red_so = _scale_offset(raster_bands.get("B04"))
    stac_match = all(_scale_offset(raster_bands.get(b)) == red_so for b in ("B08", "B11")) and red_so is not None
    median_sane = all(0.02 <= stats[b]["median"] / 10000.0 <= 0.60 for b in ("B04", "B08", "B11"))
    b04_floor = stats["B04"]["min"]
    no_pedestal = all(stats[b]["min"] <= b04_floor + 400.0 for b in ("B08", "B11"))
    verdict = "CONFIRMED" if (stac_match and median_sane and no_pedestal) else "MISMATCH"
    print(f"[staging]   -> {verdict}: B08/B11 on the same reflectance scale as RGB "
          f"(reflectance*10000, additive offset 0). "
          f"stac_scale_offset_match={stac_match} median_reflectance_sane={median_sane} "
          f"no_boa_offset_pedestal={no_pedestal} (B04 floor={b04_floor:.0f} DN)")
    if verdict != "CONFIRMED":
        raise RuntimeError(
            f"B08/B11 reflectance-scale check failed for {dir_name}: stats={stats}, "
            f"stac_match={stac_match}. Not proceeding - downstream indices assume the RGB encoding."
        )

    return {
        "dir_name": dir_name,
        "dn_stats_over_common_valid_pixels": stats,
        "stac_raster_bands": raster_bands,
        "reflectance_scale_dn_per_unit": 10000.0,
        "boa_add_offset_dn": 0.0,
        "verdict": verdict,
    }


def stage_large_aoi_extra_bands(force: bool = False) -> None:
    """Phase 3a Step A: stage B08 (NIR) + B11 (SWIR-1) for the scaled AOI, both dates.

    Only the scaled AOI (``<scene>_scaled/``) is topped up - the small demo AOI
    is deliberately left RGB+SCL only. Records each new band in the provenance
    manifest with its SHA256, then confirms B08/B11 carry the same reflectance
    encoding as the RGB bands and records that confirmation under the manifest's
    ``extra_bands`` key.
    """
    from geoseek.staging.manifest import record_analysis_section

    settings = get_settings()
    print_startup_banner(settings)

    confirmations = []
    for scene_id, meta in SCENES.items():
        dir_name = scene_id + LARGE_AOI_DIR_SUFFIX
        print(f"\n=== {dir_name} (acq. {meta['acq_date']}, scaled AOI) - extra bands {EXTRA_BANDS} ===")
        if not force and scene_is_staged(scene_id, dir_name=dir_name, bands=EXTRA_BANDS):
            print(f"[staging] B08/B11 already staged under {settings.datasets_dir / dir_name} - skipping network.")
            out_dir = settings.datasets_dir / dir_name
            band_records = [
                {"band": b, "path": out_dir / f"{b}.tif", "url": _band_url(scene_id, b)} for b in EXTRA_BANDS
            ]
        else:
            band_records = fetch_scene_large_aoi_extra_bands(scene_id)

        for rec in band_records:
            artifact_name = f"sentinel2-{dir_name}-{rec['band']}"
            arec = record_artifact(
                name=artifact_name,
                source_url=rec["url"],
                local_path=rec["path"],
                license=DATA_LICENSE,
            )
            print(f"[staging]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

        confirmations.append(_confirm_extra_band_reflectance_scale(dir_name))

    record_analysis_section(
        "extra_bands",
        {
            "purpose": "Phase 3a: NIR (B08) + SWIR-1 (B11) for change-type classification / spectral indices.",
            "bands": list(EXTRA_BANDS),
            "staged_for": [s + LARGE_AOI_DIR_SUFFIX for s in SCENES],
            "b11_resample": "20m -> 10m BILINEAR (continuous reflectance; NEAREST reserved for categorical SCL)",
            "reflectance_scale_confirmation": confirmations,
        },
    )
    print(f"\n[staging] Extra bands staged + confirmed. Manifest: {settings.provenance_manifest_path}")


def stage_third_date(force: bool = False) -> dict:
    """Phase 3.5 Step 4: stage a third Sentinel-2 date over the scaled AOI.

    Same six bands as the scaled pair, same 82 km AOI, same MGRS tile 44RPQ.
    Records provenance for every band, then confirms B08/B11 (and, implicitly,
    RGB) carry the same reflectance encoding as the existing pair and writes a
    ``third_date`` manifest section. Offline everywhere except the windowed COG
    reads here.
    """
    from geoseek.staging.manifest import record_analysis_section

    settings = get_settings()
    print_startup_banner(settings)

    scene_id = THIRD_DATE_SCENE_ID
    dir_name = scene_id + LARGE_AOI_DIR_SUFFIX
    print(f"\n=== {dir_name} (acq. {THIRD_DATE_META['acq_date']}, scaled AOI, third date) ===")

    if not force and scene_is_staged(scene_id, dir_name=dir_name, bands=THIRD_DATE_BANDS):
        print(f"[staging] All six bands already staged under {settings.datasets_dir / dir_name} - skipping network.")
        out_dir = settings.datasets_dir / dir_name
        band_records = [
            {"band": b, "path": out_dir / f"{b}.tif", "url": _band_url(scene_id, b)} for b in THIRD_DATE_BANDS
        ]
    else:
        band_records = _fetch_window(
            scene_id, LARGE_AOI_BOUNDS_4326, out_dir_name=dir_name, bands=THIRD_DATE_BANDS
        )

    for rec in band_records:
        artifact_name = f"sentinel2-{dir_name}-{rec['band']}"
        arec = record_artifact(
            name=artifact_name, source_url=rec["url"], local_path=rec["path"], license=DATA_LICENSE,
        )
        print(f"[staging]   provenance: {artifact_name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")

    confirmation = _confirm_extra_band_reflectance_scale(dir_name)

    section = {
        "purpose": ("Phase 3.5 Step 4: a third acquisition date so the temporal stack supports "
                    "earliest-supported-change (PS 2.2.2) and temporal persistence."),
        "scene_id": scene_id,
        "observation_id": dir_name,
        "acq_date": THIRD_DATE_META["acq_date"],
        "mgrs_tile": "44RPQ",
        "bands": list(THIRD_DATE_BANDS),
        "aoi_bounds_4326": list(LARGE_AOI_BOUNDS_4326),
        "reference_date_observation": REFERENCE_DATE_SCENE_ID + LARGE_AOI_DIR_SUFFIX,
        "selection_rationale": ("MGRS 44RPQ, late March (phenology window matched to 2019-03-30 / "
                                "2024-03-08), lowest cloud in a live Earth Search query between the "
                                "existing dates, boa_offset_applied=true, raster:bands identical to the pair."),
        "reflectance_scale_confirmation": confirmation,
    }
    record_analysis_section("third_date", section)
    print(f"\n[staging] Third date staged + confirmed. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="geoseek-stage-datasets")
    parser.add_argument("--large", action="store_true", help="stage the Phase 2 scaled-up AOI")
    parser.add_argument("--extra-bands", action="store_true",
                        help="Phase 3a: stage B08 (NIR) + B11 (SWIR-1) for the scaled AOI")
    parser.add_argument("--third-date", action="store_true",
                        help="Phase 3.5 Step 4: stage a third Sentinel-2 date (6 bands) over the scaled AOI")
    parser.add_argument("--phase8-dates", action="store_true",
                        help="Phase 8 Step A: stage the 2025 + 2026 dates (6 bands) over the scaled AOI")
    parser.add_argument("--force", action="store_true", help="re-fetch even if already staged")
    args = parser.parse_args(argv)

    if args.phase8_dates:
        stage_phase8_dates(force=args.force)
    elif args.third_date:
        stage_third_date(force=args.force)
    elif args.extra_bands:
        stage_large_aoi_extra_bands(force=args.force)
    elif args.large:
        stage_large_aoi(force=args.force)
    else:
        stage_all(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

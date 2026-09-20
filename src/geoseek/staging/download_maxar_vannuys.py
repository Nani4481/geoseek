"""Stage a THIRD Maxar Open Data quadkey - Phase 8F-1's dense-vehicle scene -
into the existing ``maxar-opendata`` collection alongside the two
India-Floods-Oct-2023 quadkeys (lake + Chungthang, see ``download_maxar`` /
``download_maxar_chungthang``).

Why this quadkey
-----------------
Both existing maxar-opendata quadkeys have few or no confidently-resolvable
vehicles (the lake quadkey has none at all; Chungthang has one marginal
cluster of ~3-4 m blobs, "plausible, not confidently resolvable" - see
``geoseek-phase8-stepA-maxar-chungthang`` project notes). Phase 8F needs a
scene with UNAMBIGUOUS, densely-packed vehicle instances to demo
vehicle-scale object detection.

Surveyed the live Maxar Open Data STAC catalog (55 events) for candidates in
the "urban centres, ports, airports, large car parks, highway interchanges"
categories, then opened remote-COG quicklooks (not assumed from bbox alone)
of the top candidates:

  1. Van Nuys Airport, Los Angeles (THIS quadkey) - CHOSEN. Full 100% data
     coverage (tile:data_area 28.2 / 28.2 km^2), 0% cloud. Shows ~25
     individually-resolved business jets on the FlightSafety/Signature
     Aviation ramp AND, immediately adjacent (NE corner), a dense vehicle
     storage/dealership lot with several hundred individually-resolvable
     parked vehicles in rows (distinct roof colors - white/black/red -
     clearly separable at 0.3 m/px). Verified by cropping a native-resolution
     window over both areas, not just the downsampled quicklook.
  2. Sepulveda Blvd / I-405-I-10 interchange corridor, Westwood/Rancho Park,
     LA (same acquisition, quadkey 031311102321) - a real freeway
     interchange + dense mixed-use urban corridor, but only ~65% tile data
     coverage (rest is nodata).
  3. Santa Monica Airport + Marina del Rey, LA (same acquisition, quadkey
     031311102323) - a second airport apron plus a large boat marina
     (hundreds of boats, not land vehicles), ~53% tile data coverage.
  Also surveyed Kahramanmaras-turkey-earthquake-23 (downtown Adana, Turkey,
  quadkey 120022103201, 2022-10-17) as a non-US, genuinely dense city-centre
  candidate (river, park, bridges, dense blocks) - real and usable, but
  street-level traffic there is far less densely packed than the Van Nuys
  vehicle lot and its tile is only ~50% covered (rest nodata).
  Van Nuys was chosen for full data coverage, 0% cloud, and the sheer density
  and unambiguity of its vehicle instances.

Event / item details (confirmed live via the STAC catalog, not recalled)
--------------------------------------------------------------------------
Event: ``WildFires-LosAngeles-Jan-2025`` ("Los Angeles Wildfires 2025"),
one of 55 events live in the root catalog at authoring time. Acquisition
collection ``103001010C12B000`` (WV02, 2025-01-16, sensor gsd 0.61 m,
50 items). Item quadkey ``031311102120`` -> bbox (WSEN, EPSG:4326)
[-118.5219823608871, 34.1951844887466, -118.46348122939429, 34.24378135341177]
(covers Van Nuys Airport (KVNY) and the surrounding San Fernando Valley
built-up area). Note this quadkey path uses UTM zone folder ``11`` (11N,
California), NOT ``45`` (the India event's UTM zone) - the ARD path
convention embeds the local UTM zone, not a fixed constant.

This is a SINGLE-DATE observation (only one acquisition of this quadkey
exists in the live catalog covering the Jan 2025 fires) - no pre/post pair,
cannot support change detection. It exists purely as an object-detection
resolution/density demo, exactly like the Chungthang quadkey.

Same 4-class cloud-mask-raster gotcha as the other two quadkeys: this item
has NO ``cloud-mask-raster`` asset (only ``data-mask``/``ms_analytic``/
``pan_analytic``/``visual``). ``tile:clouds_percent=0`` and full data_area,
so an all-clear CLOUDMASK.tif is synthesized locally rather than downloaded -
identical logic to ``download_maxar_chungthang._stage_one`` (raises instead
of guessing if clouds_percent were ever non-zero with no asset present).

Everything else (band split, ARD-grid GSD note, attribution, CC-BY-NC-4.0
non-commercial licence, 1024x1024 tiling) is IDENTICAL to the other two
maxar-opendata quadkeys - the shared download/stage helpers from
``download_maxar`` are reused rather than duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.download_maxar import ATTRIBUTION, DATA_LICENSE, _download, _get_json, _split_bands
from geoseek.staging.manifest import record_analysis_section, record_artifact

EVENT_ID = "WildFires-LosAngeles-Jan-2025"
EVENT_BASE_URL = f"https://maxar-opendata.s3.amazonaws.com/events/{EVENT_ID}"
UTM_ZONE_FOLDER = "11"  # UTM zone 11N (California) - the India event used "45"

CATALOG_ID = "103001010C12B000"
ACQ_DATE = "2025-01-16"
PLATFORM = "WV02"
SENSOR_NATIVE_GSD_M = 0.61
QUADKEY = "031311102120"
ROLE = "single (dense-vehicle resolution/density demo; no matching second-date footprint, no change pair)"

LOCATION_NOTE = (
    "Van Nuys Airport (KVNY), San Fernando Valley, Los Angeles, California - a general-aviation "
    "airport with a business-jet ramp, immediately adjacent to a dense vehicle storage/dealership "
    "lot. Chosen out of 3 visually-inspected LA-area candidates (this quadkey, a 405/10 freeway-"
    "interchange corridor, and a Santa Monica Airport + Marina del Rey quadkey) plus a downtown "
    "Adana, Turkey candidate, for full tile data coverage, 0% cloud, and the density/clarity of "
    "its vehicle instances - see module docstring for the full survey."
)

# The two candidates NOT staged, kept for the record (see module docstring for why).
CANDIDATES_SURVEYED_NOT_STAGED = {
    "sepulveda_405_10_interchange": {
        "event_id": EVENT_ID, "catalog_id": CATALOG_ID, "quadkey": "031311102321",
        "date": ACQ_DATE, "data_area_km2": 18.5, "tile_full_km2": 28.2, "clouds_percent": 0,
        "description": "I-405/I-10 interchange, Westwood/Rancho Park, LA - freeway interchange "
                        "+ dense mixed-use urban corridor. Only ~65% tile data coverage.",
    },
    "santa_monica_airport_marina": {
        "event_id": EVENT_ID, "catalog_id": CATALOG_ID, "quadkey": "031311102323",
        "date": ACQ_DATE, "data_area_km2": 15.1, "tile_full_km2": 28.2, "clouds_percent": 0,
        "description": "Santa Monica Airport (KSMO) + Marina del Rey - a second airport apron and "
                        "a large boat marina (hundreds of boats, not land vehicles). ~53% coverage.",
    },
    "adana_turkey_downtown": {
        "event_id": "Kahramanmaras-turkey-earthquake-23", "catalog_id": "104001007DAE2D00",
        "quadkey": "120022103201", "date": "2022-10-17", "data_area_km2": 14.2,
        "tile_full_km2": 28.2, "clouds_percent": 0,
        "description": "Downtown Adana, Turkey (Seyhan River, central park, bridges, dense city "
                        "blocks) - genuinely dense non-US city centre, but street traffic is far "
                        "less densely packed than the Van Nuys vehicle lot. ~50% tile coverage.",
    },
}


def maxar_root() -> Path:
    return get_settings().datasets_dir / "maxar" / EVENT_ID.lower()


def _obs_dir(catalog_id: str, quadkey: str) -> Path:
    return maxar_root() / f"{catalog_id}_{quadkey}"


def _item_url(catalog_id: str, quadkey: str) -> str:
    return f"{EVENT_BASE_URL}/ard/{UTM_ZONE_FOLDER}/{quadkey}/{ACQ_DATE}/{catalog_id}.json"


def fetch_item() -> dict:
    return _get_json(_item_url(CATALOG_ID, QUADKEY))


def _stage_one(item: dict, force: bool = False) -> dict:
    out_dir = _obs_dir(CATALOG_ID, QUADKEY)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"{EVENT_BASE_URL}/ard/{UTM_ZONE_FOLDER}/{QUADKEY}/{ACQ_DATE}"
    visual_url = f"{base_url}/{CATALOG_ID}-visual.tif"

    visual_path = out_dir / "visual.tif"
    if force and visual_path.exists():
        visual_path.unlink()
    if not visual_path.is_file():
        _download(visual_url, visual_path)
    else:
        print(f"[staging] {visual_path} already staged - skipping network.")

    band_paths = _split_bands(visual_path, out_dir)

    cloud_percent = item["properties"].get("tile:clouds_percent")
    has_cloud_asset = "cloud-mask-raster" in item.get("assets", {})
    cloud_path = None
    if has_cloud_asset:
        cloud_url = f"{base_url}/{CATALOG_ID}-clouds.tif"
        cloud_path = out_dir / "cloud_mask_raw.tif"
        if force and cloud_path.exists():
            cloud_path.unlink()
        if not cloud_path.is_file():
            _download(cloud_url, cloud_path)
        else:
            print(f"[staging] {cloud_path} already staged - skipping network.")
        with rasterio.open(visual_path) as ref, rasterio.open(cloud_path) as cm:
            cm_path = out_dir / "CLOUDMASK.tif"
            profile = ref.profile.copy()
            profile.update(count=1, dtype="uint8", compress="deflate", photometric="minisblack")
            with WarpedVRT(cm, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                           resampling=Resampling.nearest) as vrt:
                data = vrt.read(1)
            with rasterio.open(cm_path, "w", **profile) as dst:
                dst.write(data, 1)
    else:
        if cloud_percent != 0:
            raise RuntimeError(
                f"No cloud-mask-raster asset for {CATALOG_ID}/{QUADKEY}, and "
                f"tile:clouds_percent={cloud_percent} (not 0) - cannot safely "
                "synthesize an all-clear mask."
            )
        print(f"[staging]   no cloud-mask-raster asset on this item; synthesizing an "
              f"all-clear mask (tile:clouds_percent={cloud_percent}, tile:data_area="
              f"{item['properties'].get('tile:data_area')} = full tile).")
        with rasterio.open(visual_path) as ref:
            cm_path = out_dir / "CLOUDMASK.tif"
            profile = ref.profile.copy()
            profile.update(count=1, dtype="uint8", compress="deflate", photometric="minisblack")
            clear = np.ones((ref.height, ref.width), dtype="uint8")
            with rasterio.open(cm_path, "w", **profile) as dst:
                dst.write(clear, 1)
    band_paths["CLOUDMASK"] = cm_path

    with rasterio.open(visual_path) as ds:
        native_gsd_m = round(float(ds.res[0]), 8)
        crs = str(ds.crs)
        width, height = ds.width, ds.height

    cloudmask_entry = (
        (f"maxar-{CATALOG_ID}-{QUADKEY}-cloudmask", cloud_path,
         f"{base_url}/{CATALOG_ID}-clouds.tif", "binary cloud mask (staged)")
        if has_cloud_asset else
        (f"maxar-{CATALOG_ID}-{QUADKEY}-cloudmask", cm_path, "synthesized (no cloud-mask-raster asset; "
         "tile:clouds_percent=0 confirmed via STAC properties)", "all-clear mask (synthesized, not downloaded)")
    )
    provenance = []
    for name, path, url, kind in (
        (f"maxar-{CATALOG_ID}-{QUADKEY}-visual", visual_path, visual_url,
         "visual RGB (staged, pansharpened natural color)"),
        cloudmask_entry,
    ):
        rec = record_artifact(name=name, source_url=url, local_path=path, license=DATA_LICENSE)
        provenance.append({"name": rec.name, "sha256": rec.sha256, "byte_size": rec.byte_size, "kind": kind})
        print(f"[staging]   provenance: {rec.name} sha256={rec.sha256[:16]}... size={rec.byte_size}B")

    p = item["properties"]
    return {
        "role": ROLE,
        "catalog_id": CATALOG_ID,
        "quadkey": QUADKEY,
        "observation_id": f"{CATALOG_ID}_{QUADKEY}",
        "dataset_dir": str(out_dir),
        "acquired_at": p["datetime"],
        "platform": p["platform"],
        "sensor_native_gsd_m": p["gsd"],
        "native_gsd_m": native_gsd_m,
        "crs": crs,
        "width": width,
        "height": height,
        "cloud_percent_stac": p.get("tile:clouds_percent"),
        "data_area_km2": p.get("tile:data_area"),
        "bbox_4326": item["bbox"],
        "band_files": {k: str(v) for k, v in band_paths.items()},
        "provenance": provenance,
    }


def stage_vannuys_quadkey(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== Maxar Open Data: {EVENT_ID} (quadkey {QUADKEY}, Van Nuys) -> {maxar_root()} ===")
    print(f"[staging] {LOCATION_NOTE}")

    print(f"\n[staging] Fetching item {CATALOG_ID}/{QUADKEY}/{ACQ_DATE} ...")
    item = fetch_item()
    print(f"[staging]   bbox_4326={item['bbox']}  "
          f"data_area={item['properties'].get('tile:data_area')}km^2  "
          f"clouds={item['properties'].get('tile:clouds_percent')}%")

    print(f"\n[staging] Staging visual + cloud-mask for {CATALOG_ID}/{QUADKEY} ...")
    staged = _stage_one(item, force=force)
    print(f"\n[staging] Delivered ARD raster grid: {staged['native_gsd_m']}m/px "
          f"({staged['width']}x{staged['height']} px)")

    section = {
        "purpose": ("Phase 8F-1: stage a dense-vehicle Maxar quadkey (Van Nuys Airport, LA) into "
                    "the existing maxar-opendata collection, as a resolution/density demo scene "
                    "for the future object-detection track. STAGING ONLY - no detector built here."),
        "location": LOCATION_NOTE,
        "candidates_surveyed_not_staged": CANDIDATES_SURVEYED_NOT_STAGED,
        "staged": staged,
        "license": DATA_LICENSE,
        "attribution": ATTRIBUTION,
    }
    record_analysis_section("maxar_opendata_vannuys", section)
    print(f"\n[staging] Van Nuys quadkey staged + recorded. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-maxar-vannuys")
    p.add_argument("--force", action="store_true", help="re-download even if already staged")
    args = p.parse_args(argv)
    stage_vannuys_quadkey(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

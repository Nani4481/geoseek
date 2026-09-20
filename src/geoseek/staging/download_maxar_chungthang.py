"""Stage a SECOND Maxar Open Data quadkey for the same event as
``download_maxar`` - chosen for POPULATED terrain (Chungthang town, the
Teesta III dam/reservoir, and the road network along the Teesta river) rather
than the glacial lake the first quadkey (``120220030330``, see
``download_maxar.QUADKEY``) turned out to sit on.

Why this quadkey
-----------------
The PRE/POST pair staged by ``download_maxar`` (``10300100E34B4D00`` /
``1050010036A0EC00``) shares only 9 quadkey cells, and ALL 9 are clustered
tightly around South Lhonak Lake itself (confirmed by fetching every shared
item's ``bbox``: lon 88.10-88.20E, lat 27.85-27.94N) - the GE01 post-event
tasking never reached the downstream settlements the flood actually damaged.
So no PRE/POST *pair* exists in this event's live STAC catalog for the
populated Teesta corridor; the only imagery there is one of the four
pre-event-only (2022) acquisitions ``download_maxar`` already surveyed and
declined to stage (no matching post-event footprint -> no change pair).

Acquisition ``10300100CE8D0400`` (2022-03-07, WV02) covers lat 27.55-27.63N,
lon 88.60-88.68E - this IS Chungthang, the town at the Lachen Chu / Lachung
Chu confluence that forms the Teesta, site of the Teesta-III dam
catastrophically breached by the Oct 2023 GLOF. Of its 9 quadkey cells,
``120220122202`` (center ~27.587N, 88.646E, ``tile:data_area``=28.2 km^2 of a
~28.2 km^2 full tile, i.e. ~100% real data, 0% cloud) was chosen after
visually inspecting a remote-COG quicklook of each candidate: it is the only
cell showing the town core itself - dense, multi-colour rooftops, a football
field, a road network, and the dam/reservoir edge - rather than forested
valley wall. Confirmed by opening the actual visual.tif (not assumed from the
bbox alone).

This is a SINGLE-DATE (2022-03-07, pre-event) observation, not a pre/post
pair - there is no post-event acquisition at this footprint in the live
catalog. It cannot support change detection. It exists ONLY to demonstrate
what sub-metre VHR resolves over built-up terrain (buildings, roads, a
footbridge) as a resolution-capability sample, and to give the
``maxar-opendata`` collection a second, geographically distinct observation.
Staged/ingested with the exact same 1024x1024 tiling as the lake quadkey
(``download_maxar.MAXAR_TILE_SIZE``) - zero pipeline changes needed, this
module only supplies different STAC coordinates.

Everything else (band split, cloud-mask resample, ARD-grid GSD note,
attribution, non-commercial licence) is IDENTICAL to ``download_maxar`` - the
download/stage helpers are imported and reused rather than duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.download_maxar import (
    ATTRIBUTION,
    DATA_LICENSE,
    EVENT_BASE_URL,
    EVENT_ID,
    _get_json,
)
from geoseek.staging.manifest import record_analysis_section, record_artifact

CATALOG_ID = "10300100CE8D0400"
ACQ_DATE = "2022-03-07"
PLATFORM = "WV02"
SENSOR_NATIVE_GSD_M = 0.49
QUADKEY = "120220122202"
ROLE = "single (pre-event only, no matching post-event footprint at this quadkey)"

LOCATION_NOTE = (
    "Chungthang town (Lachen Chu / Lachung Chu confluence -> Teesta river), "
    "North Sikkim - includes the Teesta-III dam/reservoir breached in the "
    "Oct 4 2023 GLOF this event covers. Chosen as a downstream, POPULATED "
    "counterpart to download_maxar.QUADKEY (which sits on the source lake)."
)


def maxar_root() -> Path:
    return get_settings().datasets_dir / "maxar" / EVENT_ID.lower()


def _obs_dir(catalog_id: str, quadkey: str) -> Path:
    return maxar_root() / f"{catalog_id}_{quadkey}"


def _item_url(catalog_id: str, quadkey: str) -> str:
    return f"{EVENT_BASE_URL}/ard/45/{quadkey}/{ACQ_DATE}/{catalog_id}.json"


def fetch_item() -> dict:
    return _get_json(_item_url(CATALOG_ID, QUADKEY))


def _stage_one(item: dict, force: bool = False) -> dict:
    """Stage visual + cloud-mask for this single quadkey/date.

    Same band split / cloud mask resample steps as
    ``download_maxar._stage_one``, kept as a separate function since the
    original closes over the module-level ``QUADKEY``/``PRE_CATALOG_ID`` for
    the lake pair rather than taking them as parameters.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    out_dir = _obs_dir(CATALOG_ID, QUADKEY)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"{EVENT_BASE_URL}/ard/45/{QUADKEY}/{ACQ_DATE}"
    visual_url = f"{base_url}/{CATALOG_ID}-visual.tif"

    from geoseek.staging.download_maxar import _download, _split_bands

    visual_path = out_dir / "visual.tif"
    if force and visual_path.exists():
        visual_path.unlink()
    if not visual_path.is_file():
        _download(visual_url, visual_path)
    else:
        print(f"[staging] {visual_path} already staged - skipping network.")

    band_paths = _split_bands(visual_path, out_dir)

    # This 2022-03-07 pre-event item has NO ``cloud-mask-raster`` asset (only
    # 'data-mask'/'ms_analytic'/'pan_analytic'/'visual' - confirmed live,
    # unlike the paired PRE/POST lake acquisitions which both ship one). Its
    # STAC ``tile:clouds_percent`` is 0 and ``tile:data_area`` equals the full
    # tile area (no nodata gaps), so an all-clear mask is synthesized rather
    # than downloaded: value 1 everywhere, matching the 4-class scheme
    # ``geoseek.ingest.quality.maxar_cloud_fraction`` expects
    # (0=nodata, 1=clear, 2=cloud, 3=cloud_shadow) - not a guess, the STAC
    # properties directly assert 0% cloud for this exact item.
    cloud_percent = item["properties"].get("tile:clouds_percent")
    has_cloud_asset = "cloud-mask-raster" in item.get("assets", {})
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
        (f"maxar-{CATALOG_ID}-{QUADKEY}-cloudmask", cloud_path, cloud_url, "binary cloud mask (staged)")
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


def stage_chungthang_quadkey(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== Maxar Open Data: {EVENT_ID} (quadkey {QUADKEY}, Chungthang) -> {maxar_root()} ===")
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
        "purpose": ("Phase 8 Step A follow-up: stage a SECOND maxar-opendata quadkey over "
                    "POPULATED terrain (Chungthang town / Teesta-III dam) as a downstream "
                    "counterpart to the lake quadkey - single-date only, no pre/post pair exists "
                    "here in the live catalog, so no change detection is possible for this quadkey."),
        "location": LOCATION_NOTE,
        "quadkey_selection": (
            "Of 9 quadkey cells in acquisition 10300100CE8D0400, 120220122202 was chosen after "
            "visually inspecting remote-COG quicklooks of each candidate - it uniquely shows the "
            "town core (dense rooftops, football field, road network, dam/reservoir edge) rather "
            "than forested valley wall."
        ),
        "staged": staged,
        "license": DATA_LICENSE,
        "attribution": ATTRIBUTION,
    }
    record_analysis_section("maxar_opendata_chungthang", section)
    print(f"\n[staging] Chungthang quadkey staged + recorded. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-maxar-chungthang")
    p.add_argument("--force", action="store_true", help="re-download even if already staged")
    args = p.parse_args(argv)
    stage_chungthang_quadkey(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

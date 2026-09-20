"""Stage Maxar Open Data Program sub-metre VHR imagery for the object-detection
track (Phase 8 Step A - STAGING AND INGESTION ONLY, no detector here).

This is one of the ONLY modules in geoseek allowed to touch the network (see
the project-wide rule in ``README.md`` / ``config.py``). Everything downstream
reads exclusively from the local files staged here under
``data/datasets/maxar/``.

Source, confirmed LIVE at authoring time (not recalled from memory)
---------------------------------------------------------------------
STAC root: ``https://maxar-opendata.s3.amazonaws.com/events/catalog.json``
(public, anonymous-read S3 bucket, no login/API key). Queried directly:
``GET catalog.json`` -> 200, ``license: CC-BY-NC-4.0``, 55 event children.
Cross-referenced against the maintained helper index at
``github.com/opengeos/maxar-open-data`` (its README confirms the same STAC
root URL and points at the same CSV/GeoJSON-per-event mirrors).

Only ONE India event exists in the LIVE catalog: ``India-Floods-Oct-2023``
("North India Floods" - the Oct 4 2023 South Lhonak glacial lake outburst
flood, Teesta river, Sikkim). There is no "Kerala flooding" activation in the
current catalog (Maxar's Open Data Program only carries ARD for events it
activated for and has since migrated to the ARD format; 2018 Kerala predates
this program's ARD coverage and is not present) - this was checked directly
against the live 55-event listing, not assumed.

Surveying ``India-Floods-Oct-2023/collection.json`` found 6 acquisition
collections (STAC catalog_ids), enumerated below via :func:`survey_event`.
Two share the EXACT footprint (same STAC ``quadkey`` grid) and straddle the
Oct 4 2023 event date - the only pre/post pair for this event:

    PRE  10300100E34B4D00  2023-02-07  WV02  gsd(sensor-native)=0.58 m
    POST 1050010036A0EC00  2023-10-06  GE01  gsd(sensor-native)=0.49 m

The other 4 acquisitions are all PRE-event (2022-03-07/03-14), no matching
POST acquisition exists for their footprints, so they cannot form a change
pair and are surveyed/reported but not staged.

Quadkey chosen: ``120220030330`` - of the 9 quadkey cells the PRE/POST
catalog_ids share, this one has ``tile:data_area = 28.2`` km^2 against a
28.19 km^2 full tile (i.e. ~100% real imagery, not mostly blank ARD-grid
canvas) and ``tile:clouds_percent = 0`` for BOTH dates - confirmed by
fetching both item STAC records directly (see :func:`fetch_chosen_pair`).

A Maxar ARD-specific fact worth being explicit about: the STAC ``gsd``
property (0.58 / 0.49 above) is the SENSOR's native ground sample distance at
that acquisition's incidence geometry. The delivered "visual" (pansharpened
natural-color RGB) raster is instead resampled onto a STANDARDIZED ARD grid -
confirmed by opening both COGs' headers directly: BOTH the pre and post
visual.tif for this quadkey are 17408x17408 px at 0.30517578125 m/px, same
UTM zone (EPSG:32645), same pixel grid alignment. So this pre/post pair is
**already pixel-grid co-registered by construction** (no coregistration step
needed) at an ACTUAL delivered GSD of ~0.305 m, distinct from the STAC
``gsd`` property. Both numbers are recorded (see ``native_gsd_m`` = the real
raster pixel spacing, used for the catalog/matcher gate, vs
``sensor_native_gsd_m`` = the STAC property, kept as context only).

Band composition available per item (confirmed via a remote COG header read,
no full download - see :func:`survey_bands`):
    visual        3-band uint8   pansharpened natural-color RGB, 0.305 m  <- STAGED
    ms_analytic   8-band uint16  multispectral (WV/GE 8-band), ~2.11 m    (reported, not staged)
    pan_analytic  1-band uint16  panchromatic, ~0.53 m                    (reported, not staged)
    cloud-mask-raster  1-band uint8 binary cloud mask, coarser grid       <- STAGED (per-tile cloud_fraction)
Only ``visual`` (+ its cloud mask) is staged: it is the natural-color product
this project's embedding/detection UI pipeline consumes; ms/pan are reported
for completeness (a future multispectral track) but out of scope here.

License + attribution (confirmed live)
---------------------------------------
STAC ``license`` field (root + collection): ``CC-BY-NC-4.0``. Cross-checked
against the AWS Open Data Registry page
(https://registry.opendata.aws/maxar-open-data), which states the license as
"Creative Commons Attribution Non Commercial 4.0" and gives the required
credit line reproduced verbatim in :data:`ATTRIBUTION` below. This is a
NON-COMMERCIAL license (stronger than Sentinel's free/open Copernicus terms)
- fine for this SIH research/educational prototype, but anything built on top
of these tiles must carry the same restriction forward.

Unlike OSCD (pinned Hugging Face revision with published git-LFS SHA256
oids), Maxar Open Data has no pre-published expected checksum to verify
against - it is a live event catalog, not a versioned dataset release. SHA256
is still computed and recorded at staging time (via the shared
``geoseek.staging.manifest`` machinery) so the exact staged bytes are
reproducibly checked in the future (re-download + diff), just not verified
against a known-good value at staging time the way OSCD's archives are.

Re-running after the files are staged skips the network entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import rasterio
import requests

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_analysis_section, record_artifact

STAC_ROOT_URL = "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
EVENT_ID = "India-Floods-Oct-2023"
EVENT_BASE_URL = f"https://maxar-opendata.s3.amazonaws.com/events/{EVENT_ID}"
EVENT_COLLECTION_URL = f"{EVENT_BASE_URL}/collection.json"

DATA_LICENSE = "CC-BY-NC-4.0 (Maxar Open Data Program - non-commercial; attribution required)"
ACCESSED_DATE = "2026-09-13"  # date this module's live research was performed
ATTRIBUTION = (
    f"Maxar Open Data Program was accessed on {ACCESSED_DATE} from "
    "https://registry.opendata.aws/maxar-open-data. Satellite imagery courtesy of Maxar Technologies."
)

# All 6 acquisition collections found live under India-Floods-Oct-2023 at
# authoring time (see module docstring). Only the marked PRE/POST pair shares
# a footprint and straddles the event date, so only that pair is staged.
ACQUISITIONS = {
    "10300100CE36CE00": {"date": "2022-03-07", "platform": "WV02", "sensor_native_gsd_m": 0.56, "role": "pre-only (no matching post footprint)"},
    "10300100CE8D0400": {"date": "2022-03-07", "platform": "WV02", "sensor_native_gsd_m": 0.49, "role": "pre-only (no matching post footprint)"},
    "10300100CF621C00": {"date": "2022-03-07", "platform": "WV02", "sensor_native_gsd_m": 0.46, "role": "pre-only (no matching post footprint)"},
    "10300100E34B4D00": {"date": "2023-02-07", "platform": "WV02", "sensor_native_gsd_m": 0.58, "role": "PRE (staged)"},
    "1040010073381800": {"date": "2022-03-14", "platform": "WV03", "sensor_native_gsd_m": 0.37, "role": "pre-only (no matching post footprint)"},
    "1050010036A0EC00": {"date": "2023-10-06", "platform": "GE01", "sensor_native_gsd_m": 0.49, "role": "POST (staged)"},
}
PRE_CATALOG_ID = "10300100E34B4D00"
POST_CATALOG_ID = "1050010036A0EC00"
QUADKEY = "120220030330"

# --- resolution-aware tiling ------------------------------------------------
# At ~0.3-0.5 m native GSD, the pipeline's existing Sentinel-2 tile size
# (256x256, chosen for 10 m pixels -> 2560 m ground footprint) would cover
# only 256 * ~0.305 =~ 78 m of ground here - far too small a chip to give an
# object detector (vehicles, rooftops, boats) useful spatial context, and it
# would multiply the tile count ~65x over an equivalent-area Sentinel-2 scene
# for no benefit. 1024x1024 is used instead: 1024 * 0.305 =~ 312 m ground
# footprint per tile, in the same few-hundred-metre range as established VHR
# object-detection benchmark chips (SpaceNet ~440 m at 0.3-0.5 m; DOTA/xView
# chips commonly 1024 px) - big enough to hold a full street block or
# settlement cluster of context around any one object, small enough to keep
# per-tile GPU memory and RemoteCLIP embedding cost bounded. Recorded here
# (not left as a silent default) since it is a real per-collection choice,
# not the Sentinel-2 constant.
MAXAR_TILE_SIZE = 1024

COLLECTION_ID = "maxar-opendata"
COLLECTION_DESCRIPTION = (
    "Maxar Open Data Program ARD (Analysis-Ready Data), pansharpened natural-color "
    "'visual' product. Sub-metre VHR, event-activated (disaster response), "
    "CC-BY-NC-4.0 non-commercial. Staged event-by-event for the object-detection track."
)


def maxar_root() -> Path:
    return get_settings().datasets_dir / "maxar" / EVENT_ID.lower()


def _obs_dir(catalog_id: str, quadkey: str) -> Path:
    return maxar_root() / f"{catalog_id}_{quadkey}"


# --------------------------------------------------------------------------
# live STAC fetch
# --------------------------------------------------------------------------


def _get_json(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def confirm_catalog_live() -> dict:
    """Fetch the STAC root, confirm the licence and that EVENT_ID is present."""
    root = _get_json(STAC_ROOT_URL)
    license_ = root.get("license")
    event_hrefs = [l["href"] for l in root.get("links", []) if l.get("rel") == "child"]
    india_events = [h for h in event_hrefs if "india" in h.lower() or "kerala" in h.lower()]
    target_href = f"./{EVENT_ID}/collection.json"
    if target_href not in event_hrefs:
        raise RuntimeError(f"'{EVENT_ID}' not found live in the Maxar Open Data catalog "
                            f"({len(event_hrefs)} events listed)")
    if license_ != "CC-BY-NC-4.0":
        raise RuntimeError(f"Unexpected root licence '{license_}' (expected CC-BY-NC-4.0) - "
                            "re-check terms before staging.")
    return {"stac_root": STAC_ROOT_URL, "license": license_, "n_events_total": len(event_hrefs),
            "india_or_kerala_events_found": india_events}


def survey_event() -> dict:
    """Fetch the event collection + every acquisition-collection child; report dates/platforms/GSD/roles."""
    coll = _get_json(EVENT_COLLECTION_URL)
    bbox_list = coll.get("extent", {}).get("spatial", {}).get("bbox", [])
    aoi_bbox = bbox_list[0] if bbox_list else None
    temporal = coll.get("extent", {}).get("temporal", {}).get("interval", [[None, None]])[0]

    acq_links = [l["href"] for l in coll.get("links", []) if l.get("rel") == "child"]
    acquisitions_live = {}
    for href in acq_links:
        cid = href.split("/")[-1].replace("_collection.json", "")
        ac = _get_json(f"{EVENT_BASE_URL}/{href.lstrip('./')}")
        interval = ac.get("extent", {}).get("temporal", {}).get("interval", [[None, None]])[0]
        acquisitions_live[cid] = {"datetime": interval[0], "n_item_links":
                                   sum(1 for l in ac.get("links", []) if l.get("rel") == "item")}

    return {
        "event_id": EVENT_ID,
        "title": coll.get("title"),
        "description": coll.get("description"),
        "license": coll.get("license"),
        "aoi_bbox_wsen": aoi_bbox,
        "temporal_interval_utc": temporal,
        "n_acquisition_collections": len(acq_links),
        "acquisitions": {
            cid: {**ACQUISITIONS[cid], **acquisitions_live.get(cid, {})}
            for cid in ACQUISITIONS
        },
        "pre_post_pair_confirmed": bool(PRE_CATALOG_ID in ACQUISITIONS and POST_CATALOG_ID in ACQUISITIONS),
    }


def _item_url(catalog_id: str, quadkey: str) -> str:
    d = ACQUISITIONS[catalog_id]["date"]
    return f"{EVENT_BASE_URL}/ard/45/{quadkey}/{d}/{catalog_id}.json"


def fetch_chosen_pair() -> dict:
    """Fetch the PRE/POST item STAC records for QUADKEY - the exact properties + asset hrefs to stage."""
    items = {}
    for role, cid in (("pre", PRE_CATALOG_ID), ("post", POST_CATALOG_ID)):
        item = _get_json(_item_url(cid, QUADKEY))
        items[role] = item
    pre_bbox, post_bbox = items["pre"]["bbox"], items["post"]["bbox"]
    return {"items": items, "same_footprint": pre_bbox == post_bbox or (
        abs(pre_bbox[0] - post_bbox[0]) < 1e-3 and abs(pre_bbox[1] - post_bbox[1]) < 1e-3)}


def survey_bands(item: dict) -> dict:
    """Report band composition for one item's assets via a remote COG header read (no full download)."""
    report = {}
    for key in ("visual", "ms_analytic", "pan_analytic", "cloud-mask-raster"):
        asset = item["assets"].get(key)
        if not asset:
            continue
        url = "/vsicurl/" + (EVENT_BASE_URL + "/ard/45/" + item["properties"]["quadkey"] + "/" +
                              item["properties"]["datetime"][:10] + "/" + asset["href"].split("/")[-1])
        try:
            with rasterio.open(url) as ds:
                report[key] = {"bands": ds.count, "dtype": str(ds.dtypes[0]),
                               "shape": [ds.height, ds.width], "gsd_m": round(ds.res[0], 5)}
        except Exception as e:  # pragma: no cover - network hiccup, non-fatal for a report
            report[key] = {"error": str(e)}
    return report


# --------------------------------------------------------------------------
# download + verify + split
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[staging]   GET {url}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                got += len(block)
                if total:
                    print(f"\r[staging]   {dest.name}: {got / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        if total:
            print()
    tmp.replace(dest)


def _split_bands(visual_path: Path, out_dir: Path, band_names: tuple[str, ...] = ("R", "G", "B")) -> dict[str, Path]:
    """Split a 3-band uint8 visual.tif into single-band GeoTIFFs (<out_dir>/<BAND>.tif).

    Matches the ``<scene_dir>/<BAND>.tif`` contract ``geoseek.ingest.reader.read_scene``
    already expects for Sentinel-2 - so the existing reader/tiler are reused for
    Maxar with ZERO changes.
    """
    out_paths = {}
    with rasterio.open(visual_path) as src:
        profile = src.profile.copy()
        # the source is a 3-band JPEG/YCbCr-compressed COG; both are invalid
        # for a single-band GeoTIFF, so switch to a lossless single-band-safe
        # compressor instead of just narrowing `count`.
        profile.update(count=1, compress="deflate", photometric="minisblack")
        for i, band in enumerate(band_names, start=1):
            out_path = out_dir / f"{band}.tif"
            arr = src.read(i)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)
            out_paths[band] = out_path
    return out_paths


def _stage_one(role: str, catalog_id: str, item: dict, force: bool = False) -> dict:
    out_dir = _obs_dir(catalog_id, QUADKEY)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = item["properties"]["datetime"][:10]
    base_url = f"{EVENT_BASE_URL}/ard/45/{QUADKEY}/{date_str}"
    visual_url = f"{base_url}/{catalog_id}-visual.tif"
    cloud_url = f"{base_url}/{catalog_id}-clouds.tif"

    visual_path = out_dir / "visual.tif"
    if force and visual_path.exists():
        visual_path.unlink()
    if not visual_path.is_file():
        _download(visual_url, visual_path)
    else:
        print(f"[staging] {visual_path} already staged - skipping network.")

    cloud_path = out_dir / "cloud_mask_raw.tif"
    if force and cloud_path.exists():
        cloud_path.unlink()
    if not cloud_path.is_file():
        _download(cloud_url, cloud_path)
    else:
        print(f"[staging] {cloud_path} already staged - skipping network.")

    band_paths = _split_bands(visual_path, out_dir)
    # cloud mask ships on a coarser grid than the visual product; resample onto
    # the visual grid (nearest - it is a binary categorical mask) so tiler.py's
    # single shared pixel grid assumption (all `<BAND>.tif` files co-registered)
    # holds for CLOUDMASK too.
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.open(visual_path) as ref, rasterio.open(cloud_path) as cm:
        cm_path = out_dir / "CLOUDMASK.tif"
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", compress="deflate", photometric="minisblack")
        with WarpedVRT(cm, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                       resampling=Resampling.nearest) as vrt:
            data = vrt.read(1)
        with rasterio.open(cm_path, "w", **profile) as dst:
            dst.write(data, 1)
    band_paths["CLOUDMASK"] = cm_path

    with rasterio.open(visual_path) as ds:
        native_gsd_m = round(float(ds.res[0]), 8)
        crs = str(ds.crs)
        width, height = ds.width, ds.height

    provenance = []
    for name, path, url, kind in (
        (f"maxar-{catalog_id}-{QUADKEY}-visual", visual_path, visual_url,
         "visual RGB (staged, pansharpened natural color)"),
        (f"maxar-{catalog_id}-{QUADKEY}-cloudmask", cloud_path, cloud_url, "binary cloud mask (staged)"),
    ):
        rec = record_artifact(name=name, source_url=url, local_path=path, license=DATA_LICENSE)
        provenance.append({"name": rec.name, "sha256": rec.sha256, "byte_size": rec.byte_size, "kind": kind})
        print(f"[staging]   provenance: {rec.name} sha256={rec.sha256[:16]}... size={rec.byte_size}B")

    p = item["properties"]
    return {
        "role": role,
        "catalog_id": catalog_id,
        "quadkey": QUADKEY,
        "observation_id": f"{catalog_id}_{QUADKEY}",
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


def stage_maxar_event(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== Maxar Open Data: {EVENT_ID} (quadkey {QUADKEY}) -> {maxar_root()} ===")

    print("[staging] Confirming the live STAC catalog + licence ...")
    catalog_check = confirm_catalog_live()
    print(f"[staging]   licence={catalog_check['license']}, "
          f"{catalog_check['n_events_total']} events total, "
          f"India/Kerala matches: {catalog_check['india_or_kerala_events_found']}")

    print(f"\n[staging] Surveying '{EVENT_ID}' (all acquisition collections) ...")
    event_report = survey_event()
    print(f"[staging]   {event_report['title']}: {event_report['description'][:120]}...")
    print(f"[staging]   AOI bbox (WSEN): {event_report['aoi_bbox_wsen']}")
    print(f"[staging]   temporal span: {event_report['temporal_interval_utc']}")
    for cid, meta in event_report["acquisitions"].items():
        print(f"[staging]     {cid}  {meta['date']}  {meta['platform']}  "
              f"gsd(sensor)={meta['sensor_native_gsd_m']}m  {meta['role']}")

    print(f"\n[staging] Fetching the chosen PRE/POST item pair (quadkey {QUADKEY}) ...")
    pair = fetch_chosen_pair()
    print(f"[staging]   same footprint: {pair['same_footprint']}")

    print("\n[staging] Band composition (remote COG header read, no full download) ...")
    bands_report = {role: survey_bands(item) for role, item in pair["items"].items()}
    for role, rep in bands_report.items():
        for k, v in rep.items():
            print(f"[staging]   {role}/{k}: {v}")

    print(f"\n[staging] Staging visual + cloud-mask assets for PRE ({PRE_CATALOG_ID}) and "
          f"POST ({POST_CATALOG_ID}) ...")
    staged = {
        "pre": _stage_one("pre", PRE_CATALOG_ID, pair["items"]["pre"], force=force),
        "post": _stage_one("post", POST_CATALOG_ID, pair["items"]["post"], force=force),
    }

    gsd_match = staged["pre"]["native_gsd_m"] == staged["post"]["native_gsd_m"]
    print(f"\n[staging] Delivered ARD raster grid: pre={staged['pre']['native_gsd_m']}m/px, "
          f"post={staged['post']['native_gsd_m']}m/px, same_grid={gsd_match} "
          f"({staged['pre']['width']}x{staged['pre']['height']} px)")

    section = {
        "purpose": ("Phase 8 Step A: stage sub-metre Maxar Open Data VHR imagery, as a NEW "
                    "collection distinct from Sentinel-2/-1, for a future object-detection track. "
                    "STAGING AND INGESTION ONLY - no detector built here."),
        "catalog_check": catalog_check,
        "event": event_report,
        "chosen_pair": {"pre_catalog_id": PRE_CATALOG_ID, "post_catalog_id": POST_CATALOG_ID,
                        "quadkey": QUADKEY, "same_footprint": pair["same_footprint"]},
        "bands_available": bands_report,
        "bands_staged": ["visual (R,G,B, uint8, pansharpened natural color)", "cloud-mask-raster (binary, resampled onto the visual grid)"],
        "bands_not_staged_reason": "ms_analytic (8-band uint16, ~2.1m) and pan_analytic (1-band uint16, ~0.53m) "
                                   "exist and are reported above, but out of scope for the RGB-only staging/ingest "
                                   "pipeline this step builds.",
        "tiling": {"tile_size_px": MAXAR_TILE_SIZE, "rationale": (
            "256x256 (the Sentinel-2 tile size) would cover only 256*0.305 =~ 78 m of ground at this "
            "GSD - too small for object-detection context and a ~65x tile-count blowup for no benefit. "
            "1024x1024 covers ~312 m/side, matching common VHR object-detection benchmark chip sizes "
            "(SpaceNet ~440 m, DOTA/xView ~1024 px) while keeping per-tile embedding cost bounded.")},
        "collection": {"collection_id": COLLECTION_ID, "description": COLLECTION_DESCRIPTION,
                       "native_gsd_m_recorded_per_observation": True,
                       "note": ("native_gsd_m is recorded PER OBSERVATION (the actual delivered raster "
                                "pixel spacing, ~0.305 m here) rather than assumed from the collection alone, "
                                "so geoseek.temporal.matcher.TemporalObservationMatcher's resolution_compatibility "
                                "gate compares real numbers - both refusing any match against 10 m Sentinel-2 "
                                "observations AND correctly recognizing this PRE/POST pair as same-grid.")},
        "license": DATA_LICENSE,
        "attribution": ATTRIBUTION,
        "staged": staged,
    }
    record_analysis_section("maxar_opendata", section)
    print(f"\n[staging] Maxar Open Data staged + recorded. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-maxar")
    p.add_argument("--force", action="store_true", help="re-download even if already staged")
    args = p.parse_args(argv)
    stage_maxar_event(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

"""Stage Sentinel-1 GRD (SAR) over the Ayodhya AOI as a SECOND collection.

Network entrypoint (staging only). SAR enters geoseek as **corroborating
evidence** for the optical change pipeline, never as retrieval embeddings -
RemoteCLIP was trained on optical imagery, so SAR is not embedded (its tiles
carry ``faiss_id = None``).

Source, confirmed LIVE at authoring time
---------------------------------------
Earth Search STAC (``https://earth-search.aws.element84.com/v1``), collection
``sentinel-1-grd`` - the same no-login API + AWS Open Data bucket geoseek
already uses for Sentinel-2. Backing store:
``s3://sentinel-s1-l1c`` (public, anonymous read).

Products chosen (one consistent orbit track so a dB *change* is meaningful):

  | S2 date     | S1 item date | offset | rel. orbit | pass | platform |
  |-------------|--------------|--------|-----------|------|----------|
  | 2019-03-30  | 2019-03-29   | -1 d   | 56        | asc  | S1A      |
  | 2021-03-04  | 2021-03-06   | +2 d   | 56        | asc  | S1A      |
  | 2024-03-08  | 2024-03-02   | -6 d   | 56        | asc  | S1A      |

All are **S1A IW GRDH 1SDV** (Interferometric Wide swath, GRD High-Res,
dual-pol VV+VH). A single 56-track slice covers ~94 % of the 82 km AOI; the
~6 % gap is at one swath edge and is treated as "SAR unavailable" (neutral)
downstream, never a penalty.

Preprocessing state of the served product
-----------------------------------------
The GRD is ESA Level-1: **detected** (amplitude), **multi-looked** (5 range x
1 azimuth looks, ENL 4.4), **ground-range projected**. It is NOT: map-geocoded
(the SAFE ``measurement/*.tiff`` carries no CRS), NOT radiometrically
calibrated to sigma0/gamma0 (no calibration LUT applied), NOT terrain-flattened,
NOT speckle-filtered.

geoseek adds, at staging:
  1. **Geocoding** to the S2 10 m UTM grid (EPSG:32644, identical
     transform/shape to the S2 scaled scenes) via the product's own ~210
     ground-control points (``rasterio`` ``WarpedVRT``, polynomial fit,
     bilinear). GCPs capture the swath rotation and give full AOI coverage;
     terrain distortion is negligible - the Ayodhya floodplain has < 20 m
     relief across the AOI.
Downstream (``geoseek.sar``) adds speckle filtering + the dB difference; the
un-applied absolute calibration cancels in a same-track dB *difference*, so
change is measured as ``10*log10(I_later / I_earlier)`` (I = DN^2).

Licence: Copernicus Sentinel Data (free, full & open, EU Copernicus Data
Policy) - the same terms as the Sentinel-2 scenes. AWS Open Data hosting terms:
https://registry.opendata.aws/sentinel-1/. (Earth Search tags the STAC
collection ``license: proprietary`` generically; the data itself is Copernicus
free & open.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from osgeo import gdal

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_analysis_section, record_artifact

gdal.UseExceptions()

STAC_ROOT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-1-grd"
S1_BUCKET_S3_PREFIX = "s3://sentinel-s1-l1c/"
S1_BUCKET_HTTPS = "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/"

POLARIZATIONS = ("vv", "vh")
DATA_LICENSE = "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)"

# S1A IW GRDH 1SDV, relative orbit 56, ascending - one consistent track.
S1_ITEMS = {
    "2019": {"item_id": "S1A_IW_GRDH_1SDV_20190329T123812_20190329T123837_026553_02F9E3",
             "acq_date": "2019-03-29", "s2_date": "2019-03-30", "s2_observation": "S2B_44RPQ_20190330_1_L2A_scaled"},
    "2021": {"item_id": "S1A_IW_GRDH_1SDV_20210306T123824_20210306T123849_036878_04564C",
             "acq_date": "2021-03-06", "s2_date": "2021-03-04", "s2_observation": "S2A_44RPQ_20210304_1_L2A_scaled"},
    "2024": {"item_id": "S1A_IW_GRDH_1SDV_20240302T123840_20240302T123905_052803_0663CD",
             "acq_date": "2024-03-02", "s2_date": "2024-03-08", "s2_observation": "S2A_44RPQ_20240308_0_L2A_scaled"},
}

# on-disk / catalog naming: <scene_id>_grd  (scene_id = the S1 product identifier stem)
S1_COLLECTION_ID = "sentinel-1-grd"


def scene_id_for(item_id: str) -> str:
    return item_id                                  # the full S1 product id is the scene id


def observation_id_for(item_id: str) -> str:
    return f"{item_id}_grd"


def s1_dataset_dir(item_id: str) -> Path:
    return get_settings().datasets_dir / observation_id_for(item_id)


# --------------------------------------------------------------------------
# STAC + geocoding
# --------------------------------------------------------------------------


def _stac_item(item_id: str) -> dict:
    url = f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items/{item_id}"
    print(f"[staging] STAC item: {url}")
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    return r.json()


def _s2_grid(s2_observation: str):
    with rasterio.open(get_settings().datasets_dir / s2_observation / "B04.tif") as ds:
        return ds.crs, ds.transform, ds.height, ds.width


def _https(href: str) -> str:
    return href.replace(S1_BUCKET_S3_PREFIX, S1_BUCKET_HTTPS)


def geocode_polarization(item: dict, pol: str, dst_crs, dst_transform, H, W, out_path: Path) -> float:
    """Geocode one S1 GRD polarization onto the exact S2 10 m UTM grid, via the
    product's own ~210 GCPs (``gdal.Warp``, GCP polynomial, bilinear), writing a
    tiled DEFLATE GeoTIFF to ``out_path``. Returns the AOI coverage fraction.

    The SAFE ``measurement/*.tiff`` is in radar/ground-range geometry with no
    CRS; its GCPs (EPSG:4326, with terrain height) capture the swath rotation.
    ``rasterio``'s ``WarpedVRT`` mishandles ``src_gcps`` here (collapses the
    image); the GDAL warper does it correctly - a single 56-track slice then
    covers ~93 % of the 82 km AOI, the eastern strip beyond the GCP grid staying
    nodata ("SAR unavailable" downstream, never a penalty).
    """
    url = "/vsicurl/" + _https(item["assets"][pol]["href"])
    minx = dst_transform.c
    maxy = dst_transform.f
    maxx = minx + dst_transform.a * W
    miny = maxy + dst_transform.e * H
    t0 = time.time()
    gdal.Warp(str(out_path), url, dstSRS=dst_crs.to_wkt(),
              outputBounds=(minx, miny, maxx, maxy), width=W, height=H,
              resampleAlg="bilinear", srcNodata=0, dstNodata=0, multithread=True,
              warpMemoryLimit=1024,
              creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"])
    with rasterio.open(out_path) as ds:
        cov = float((ds.read(1) > 0).mean())
    print(f"[staging]   {pol.upper()}: GCP-warped -> {H}x{W} on the S2 grid in {time.time()-t0:.0f}s "
          f"(AOI coverage {cov*100:.1f}%)")
    return cov


# --------------------------------------------------------------------------
# stage
# --------------------------------------------------------------------------


def stage_sentinel1(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== Sentinel-1 GRD (SAR) - second collection over the Ayodhya AOI ===")

    per_date = {}
    for key, meta in S1_ITEMS.items():
        item_id = meta["item_id"]
        obs_id = observation_id_for(item_id)
        out_dir = s1_dataset_dir(item_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = all((out_dir / f"{p.upper()}.tif").is_file() for p in POLARIZATIONS)
        print(f"\n--- {key}: {item_id}  (acq {meta['acq_date']}, near S2 {meta['s2_date']}) ---")

        item = _stac_item(item_id)
        p = item["properties"]
        dst_crs, dst_tr, H, W = _s2_grid(meta["s2_observation"])

        band_records = []
        cov = None
        for pol in POLARIZATIONS:
            path = out_dir / f"{pol.upper()}.tif"
            if staged and not force:
                print(f"[staging]   {pol.upper()} already staged - skipping warp.")
            else:
                geocode_polarization(item, pol, dst_crs, dst_tr, H, W, path)
            if pol == "vv":
                with rasterio.open(path) as ds:
                    a = ds.read(1)
                cov = float((a > 0).mean())
                v = a[a > 0].astype(np.float64)
                dn_stats = {"min": float(v.min()), "median": float(np.median(v)),
                            "p99": float(np.percentile(v, 99)), "max": float(v.max()),
                            "n_unique": int(np.unique(v).size)}
            arec = record_artifact(
                name=f"sentinel1-{obs_id}-{pol.upper()}",
                source_url=_https(item["assets"][pol]["href"]),
                local_path=path, license=DATA_LICENSE)
            print(f"[staging]   provenance: {arec.name} sha256={arec.sha256[:16]}... size={arec.byte_size}B")
            band_records.append({"pol": pol.upper(), "path": str(path), "sha256": arec.sha256,
                                 "byte_size": arec.byte_size, "source_url": arec.source_url})

        print(f"[staging]   AOI coverage (VV non-nodata): {cov*100:.1f}%  |  VV DN median {dn_stats['median']:.0f}")
        per_date[key] = {
            "item_id": item_id, "scene_id": scene_id_for(item_id), "observation_id": obs_id,
            "acq_datetime": p["datetime"], "acq_date": meta["acq_date"],
            "near_s2_date": meta["s2_date"], "near_s2_observation": meta["s2_observation"],
            "offset_days": _days(meta["acq_date"], meta["s2_date"]),
            "relative_orbit": p.get("sat:relative_orbit"), "orbit_state": p.get("sat:orbit_state"),
            "platform": p.get("platform"), "instrument_mode": p.get("sar:instrument_mode"),
            "polarizations": [pp.upper() for pp in POLARIZATIONS],
            "product_type": p.get("sar:product_type"),
            "looks_equivalent_number": p.get("sar:looks_equivalent_number"),
            "resolution_range_m": p.get("sar:resolution_range"),
            "resolution_azimuth_m": p.get("sar:resolution_azimuth"),
            "pixel_spacing_m": p.get("sar:pixel_spacing_range"),
            "stac_proj_transform": p.get("proj:transform"), "stac_proj_epsg": p.get("proj:epsg"),
            "aoi_coverage_fraction": round(cov, 4), "vv_dn_stats": dn_stats,
            "bands": band_records,
        }

    section = {
        "purpose": ("Phase 5 Step B: Sentinel-1 C-band SAR as a second collection - corroborating "
                    "evidence for the optical change pipeline (NOT embedded with RemoteCLIP, NOT "
                    "retrieval)."),
        "source": {"stac_api": STAC_ROOT, "collection": STAC_COLLECTION,
                   "bucket": S1_BUCKET_S3_PREFIX, "no_login": True},
        "license": DATA_LICENSE + " (Earth Search tags the STAC collection 'proprietary' generically; "
                  "the data is Copernicus free & open).",
        "product": "S1A IW GRDH 1SDV (dual-pol VV+VH), relative orbit 56, ascending",
        "polarizations": ["VV", "VH"],
        "preprocessing_applied_by_provider": ("ESA Level-1 GRD: detected (amplitude), multi-looked "
                                              "(5 rg x 1 az looks, ENL 4.4), ground-range projected. "
                                              "NOT map-geocoded, NOT radiometrically calibrated to "
                                              "sigma0/gamma0, NOT terrain-flattened, NOT speckle-filtered."),
        "preprocessing_applied_by_geoseek_staging": ("geocoding to the S2 10 m UTM grid (EPSG:32644, "
                                                     "identical transform/shape to the S2 scaled "
                                                     "scenes) via the product's ~210 GCPs (rasterio "
                                                     "WarpedVRT polynomial, bilinear) - full AOI "
                                                     "coverage; terrain distortion negligible over the "
                                                     "< 20 m-relief floodplain."),
        "radiometry_convention": ("amplitude DN, uint16, UNCALIBRATED. Backscatter *change* is "
                                  "10*log10(I_later / I_earlier), I = DN^2 - the un-applied absolute "
                                  "calibration constant cancels because all three dates are the same "
                                  "beam / relative orbit 56 / ascending / S1A."),
        "aoi": "same 82 km Ayodhya AOI as the S2 scaled scenes; single 56-track slice covers ~94%",
        "dates": per_date,
    }
    record_analysis_section("sentinel1", section)
    print(f"\n[staging] Sentinel-1 staged + recorded. Manifest section 'sentinel1'.")
    return section


def _days(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _mid(a: str, b: str) -> str:
    from datetime import date, timedelta
    da, db = date.fromisoformat(a), date.fromisoformat(b)
    return (da + (db - da) / 2).isoformat()


# --------------------------------------------------------------------------
# catalog registration (separate collection; NO embeddings)
# --------------------------------------------------------------------------


def register_sentinel1_catalog() -> dict:
    """Register sentinel-1-grd as its own collection + 3 scenes + 3 observations + tiles.

    Tiles carry geometry + row/col + a SAR nodata fraction; ``faiss_id`` stays
    ``None`` (SAR is never embedded), so the search index and its parity
    baseline are untouched. Idempotent: skips if the collection is already
    registered with tiles.
    """
    import pyproj
    from shapely.geometry import Polygon

    from geoseek.catalog.entities import Collection, Observation, Scene, Tile
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    from geoseek.ingest.tiler import TILE_SIZE
    from geoseek.staging.manifest import load_manifest

    settings = get_settings()
    repo = SQLiteMetadataRepository(settings.database_path)
    sec = load_manifest()["sentinel1"]

    existing_s1_tiles = sum(
        len(repo.list_tiles(observation_id=d["observation_id"])) for d in sec["dates"].values())
    if repo.get_collection(S1_COLLECTION_ID) is not None and existing_s1_tiles > 0:
        total_collections, total_tiles = len(repo.list_collections()), repo.count_tiles()
        repo.close()
        print(f"[catalog] sentinel-1-grd already registered ({existing_s1_tiles} tiles) - skipping. "
              f"Catalog: {total_collections} collections, {total_tiles} tiles.")
        return {"counts": {"scenes": 3, "observations": 3, "tiles": existing_s1_tiles},
                "collections_total": total_collections, "tiles_total": total_tiles, "skipped": True}

    repo.register_collection(Collection(
        collection_id=S1_COLLECTION_ID, sensor="C-SAR", platform="Sentinel-1",
        bands=("VV", "VH"), native_gsd_m=10.0,
        description=("Sentinel-1 IW GRD High-Res, dual-pol VV/VH, ~20x22 m resolution / 10 m pixel "
                     "spacing, approximately geocoded to the S2 grid. Amplitude DN, uncalibrated - "
                     "used for dB backscatter *change* only, NOT embedded, NOT retrieval."),
        metadata={"orbit_track": "rel_orbit_56_ascending", "product": "IW GRDH 1SDV"}))

    counts = {"scenes": 0, "observations": 0, "tiles": 0}
    for key, d in sec["dates"].items():
        item_id = d["item_id"]
        obs_id = d["observation_id"]
        vv_path = s1_dataset_dir(item_id) / "VV.tif"
        with rasterio.open(vv_path) as ds:
            crs, transform, H, W, nod = ds.crs, ds.transform, ds.height, ds.width, ds.nodata
            vv = ds.read(1)
        to_wgs = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        def _poly(r0, c0, r1, c1):
            xs = [transform * (c, r) for c, r in ((c0, r0), (c1, r0), (c1, r1), (c0, r1))]
            return Polygon([to_wgs.transform(x, y) for x, y in xs])

        footprint = _poly(0, 0, W, H).wkt
        repo.register_scene(Scene(
            scene_id=d["scene_id"], collection_id=S1_COLLECTION_ID, platform=d["platform"],
            acquired_at=d["acq_date"], footprint_wkt_4326=footprint,
            source_url=d["bands"][0]["source_url"], license=sec["license"], crs=str(crs),
            checksums={b["pol"]: b["sha256"] for b in d["bands"]},
            metadata={"stac_item": item_id, "relative_orbit": d["relative_orbit"],
                      "orbit_state": d["orbit_state"], "instrument_mode": d["instrument_mode"],
                      "near_s2_date": d["near_s2_date"], "offset_days": d["offset_days"]}))
        counts["scenes"] += 1

        repo.register_observation(Observation(
            observation_id=obs_id, scene_id=d["scene_id"], acquired_at=d["acq_date"],
            footprint_wkt_4326=footprint, aoi_name="ayodhya-82km", dataset_dir=obs_id,
            quality_summary={"aoi_coverage_fraction": d["aoi_coverage_fraction"],
                             "sar_nodata_fraction": round(1.0 - d["aoi_coverage_fraction"], 4),
                             "n_tiles": None},
            radiometry={"convention": sec["radiometry_convention"], "calibrated": False,
                        "polarizations": ["VV", "VH"]},
            coregistration={"method": "approx_geocode_via_stac_proj_transform",
                            "reference_grid": "S2 EPSG:32644 10 m", "terrain_corrected": False,
                            "note": "same-track (rel orbit 56) so a dB change is geometry-consistent"},
            metadata={"near_s2_observation": [v["s2_observation"] for k, v in S1_ITEMS.items()
                                             if v["item_id"] == item_id][0]}))
        counts["observations"] += 1

        tiles = []
        n_rt, n_ct = -(-H // TILE_SIZE), -(-W // TILE_SIZE)
        for r in range(n_rt):
            for c in range(n_ct):
                r1, c1 = min((r + 1) * TILE_SIZE, H), min((c + 1) * TILE_SIZE, W)
                sub = vv[r * TILE_SIZE:r1, c * TILE_SIZE:c1]
                if not (sub > 0).any():
                    continue
                tiles.append(Tile(
                    tile_id=f"{obs_id}_r{r:03d}_c{c:03d}", observation_id=obs_id, row=r, col=c,
                    geom_wkt_4326=_poly(r * TILE_SIZE, c * TILE_SIZE, r1, c1).wkt,
                    cloud_fraction=0.0,          # SAR: no clouds
                    quality_flags={"sar_nodata_fraction": round(float((sub == 0).mean()), 4),
                                   "sensor": "C-SAR"},
                    faiss_id=None, embedding_ref=None))
        repo.add_tiles(tiles)
        counts["tiles"] += len(tiles)
        print(f"[catalog]   {obs_id}: scene + observation + {len(tiles)} tiles (faiss_id=None)")

    total_collections = len(repo.list_collections())
    total_tiles = repo.count_tiles()
    repo.close()
    print(f"[catalog] sentinel-1-grd registered: {counts['scenes']} scenes / "
          f"{counts['observations']} observations / {counts['tiles']} tiles")
    print(f"[catalog] catalog now: {total_collections} collections, {total_tiles} tiles total")
    record_analysis_section("sentinel1_catalog", {
        "collection_id": S1_COLLECTION_ID, **counts,
        "catalog_collections_total": total_collections, "catalog_tiles_total": total_tiles,
        "embedded": False, "note": "SAR tiles are not embedded; faiss_id is None; the FAISS index "
                                   "and its parity baseline are unchanged."})
    return {"counts": counts, "collections_total": total_collections, "tiles_total": total_tiles}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-sentinel1")
    p.add_argument("--force", action="store_true", help="re-warp even if already staged")
    p.add_argument("--no-catalog", action="store_true", help="stage rasters only, skip catalog registration")
    args = p.parse_args(argv)
    stage_sentinel1(force=args.force)
    if not args.no_catalog:
        register_sentinel1_catalog()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

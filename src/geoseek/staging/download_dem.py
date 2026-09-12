"""Stage a free DEM (terrain context, Phase 8 Step B) for the Ayodhya AOI.

This is one of the ONLY modules in geoseek allowed to touch the network (see
``download_datasets.py`` / ``download_models.py`` / ``download_sentinel1.py``
for the sibling staging entrypoints). Everything downstream
(``geoseek.terrain.*``) reads exclusively from the local file staged here.

Source, confirmed LIVE at authoring time (2026-09-12), not recalled from memory
----------------------------------------------------------------------------
**Copernicus DEM GLO-30 Public**, hosted on AWS Open Data by Sinergise:

  * Registry entry: https://registry.opendata.aws/copernicus-dem/ (fetched
    live; ``"license"``: "GLO-30 Public and GLO-90 are available on a free
    basis for the general public under the terms and conditions of the
    Licence found on
    https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM"
    - i.e. the Copernicus data licence, same free/open family as the
    Sentinel-2 data this project already uses).
  * Bucket ``s3://copernicus-dem-30m`` (``arn:aws:s3:::copernicus-dem-30m``,
    region eu-central-1), served over plain HTTPS at
    ``https://copernicus-dem-30m.s3.amazonaws.com/`` - **no AWS account, no
    login, no API key** (the registry page's own CLI example uses
    ``--no-sign-request``); confirmed here with a plain unauthenticated
    ``HEAD`` on two tile COGs (200 OK, real Content-Length) and a GET of the
    bucket's own ``readme.html``.
  * Format: one Cloud-Optimized GeoTIFF per 1x1 degree tile, EPSG:4326,
    native 30 m (3600x3600 px after the bucket's own edge-trim - see
    ``readme.html``), float32 metres, ``Copernicus DEM 2021 release``.
  * **This is a Digital Surface Model (DSM)**, not bare-earth terrain: per the
    registry description it "represents the surface of the Earth including
    buildings, infrastructure and vegetation." Elevation values over the
    Ayodhya township therefore include roof/canopy height, not ground level -
    stated plainly wherever this DEM's numbers are surfaced (PS "do not
    overclaim").

AOI: the same 82 km scaled Ayodhya AOI as the Sentinel-2 stack
(``LARGE_AOI_BOUNDS_4326``), which straddles the 27 N tile boundary, so TWO
1x1 degree tiles are needed: ``N26_00_E082_00`` and ``N27_00_E082_00``
(confirmed via a live bounds check against both tiles' own COG headers).

Both tiles are merged (native EPSG:4326) and then resampled (bilinear -
elevation is a continuous quantity) onto the EXACT 10 m UTM 44N grid the
Sentinel-2 reference observation (2024-03-08) uses, so every downstream
terrain sample shares pixel coordinates with the change-pipeline rasters -
no separate reprojection needed at query time.

Re-running this after the DEM is already staged skips the network entirely
(checks the local file first), matching every other staging module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_artifact, record_analysis_section

DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_DATASET = "Copernicus DEM GLO-30 Public (2021 release)"
DEM_LICENSE = (
    "Copernicus DEM licence (free & open for the general public) - "
    "https://dataspace.copernicus.eu/explore-data/data-collections/"
    "copernicus-contributing-missions/collections-description/COP-DEM ; "
    "hosted by Sinergise on AWS Open Data, registry: "
    "https://registry.opendata.aws/copernicus-dem/"
)
DEM_NOTE = ("Digital Surface Model (DSM): includes buildings/infrastructure/vegetation height, "
            "NOT bare-earth ground elevation.")

# The 1x1 degree DEM tiles covering the scaled Ayodhya AOI (82.01237,26.36132 -
# 82.84594,27.10975): confirmed by comparing the AOI bounds against each
# candidate tile's own COG bounds (a live rasterio.open().bounds check).
DEM_TILES = ["N26_00_E082_00", "N27_00_E082_00"]

OUT_DIR_NAME = "dem_44RPQ_scaled"
OUT_FILE = "elevation_m.tif"


def _tile_url(tile: str) -> str:
    name = f"Copernicus_DSM_COG_10_{tile}_DEM"
    return f"{DEM_BUCKET}/{name}/{name}.tif"


def dem_is_staged() -> bool:
    return (get_settings().datasets_dir / OUT_DIR_NAME / OUT_FILE).is_file()


def _reference_grid() -> tuple:
    """(crs, transform, width, height) of the Sentinel-2 2024-03-08 reference band -
    every terrain raster is warped onto this exact grid so bbox/pixel coordinates
    are shared with the change-pipeline rasters."""
    from geoseek.staging.download_datasets import LARGE_AOI_DIR_SUFFIX, REFERENCE_DATE_SCENE_ID

    ref_dir = get_settings().datasets_dir / (REFERENCE_DATE_SCENE_ID + LARGE_AOI_DIR_SUFFIX)
    with rasterio.open(ref_dir / "B04.tif") as ds:
        return ds.crs, ds.transform, ds.width, ds.height


def stage_dem(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    out_dir = settings.datasets_dir / OUT_DIR_NAME
    out_path = out_dir / OUT_FILE

    if not force and out_path.is_file():
        print(f"[staging] DEM already staged at {out_path} - skipping network.")
        section = _record_provenance(out_path, refetched=False)
        return section

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_crs, ref_transform, ref_w, ref_h = _reference_grid()
    print(f"[staging] target grid: {ref_crs}  {ref_w}x{ref_h} px @ 10 m (matches the S2 2024-03-08 reference)")

    print(f"[staging] Fetching {len(DEM_TILES)} Copernicus DEM GLO-30 tile(s) from {DEM_BUCKET} "
          "(public, no-sign-request, windowed reads over HTTPS range requests) ...")
    srcs = []
    tile_meta = []
    try:
        for tile in DEM_TILES:
            url = _tile_url(tile)
            print(f"[staging]   opening {url}")
            ds = rasterio.open(url)
            srcs.append(ds)
            tile_meta.append({"tile": tile, "url": url, "crs": str(ds.crs), "shape": [ds.width, ds.height]})

        from geoseek.staging.download_datasets import LARGE_AOI_BOUNDS_4326

        print(f"[staging] merging tiles clipped to the AOI bounds {LARGE_AOI_BOUNDS_4326} (EPSG:4326) ...")
        mosaic, mosaic_transform = merge(srcs, bounds=LARGE_AOI_BOUNDS_4326)
        mosaic = mosaic[0].astype(np.float32)
        print(f"[staging]   mosaic {mosaic.shape[1]}x{mosaic.shape[0]} px, "
              f"elevation range {np.nanmin(mosaic):.1f}-{np.nanmax(mosaic):.1f} m")

        # reproject the native EPSG:4326 30m mosaic onto the exact 10m UTM 44N
        # reference grid (bilinear - elevation is continuous)
        with MemoryFile() as mem:
            with mem.open(
                driver="GTiff", height=mosaic.shape[0], width=mosaic.shape[1], count=1,
                dtype="float32", crs=srcs[0].crs, transform=mosaic_transform, nodata=None,
            ) as tmp_ds:
                tmp_ds.write(mosaic, 1)
            with mem.open() as tmp_ds:
                with WarpedVRT(
                    tmp_ds, crs=ref_crs, transform=ref_transform, width=ref_w, height=ref_h,
                    resampling=Resampling.bilinear,
                ) as vrt:
                    warped = vrt.read(1)
        print(f"[staging]   warped onto reference grid: {warped.shape[1]}x{warped.shape[0]} px, "
              f"elevation range {warped.min():.1f}-{warped.max():.1f} m")

        profile = {
            "driver": "GTiff", "height": ref_h, "width": ref_w, "count": 1, "dtype": "float32",
            "crs": ref_crs, "transform": ref_transform, "nodata": float("nan"),
            "compress": "deflate", "predictor": 3, "tiled": True, "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(warped, 1)
    finally:
        for ds in srcs:
            ds.close()

    print(f"[staging] wrote {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")
    return _record_provenance(out_path, refetched=True, tile_meta=tile_meta)


def _record_provenance(out_path: Path, *, refetched: bool, tile_meta: list | None = None) -> dict:
    arec = record_artifact(
        name="copernicus-dem-glo30-ayodhya-scaled", source_url=_tile_url(DEM_TILES[0]) + " (+1 more tile, merged)",
        local_path=out_path, license=DEM_LICENSE,
    )
    print(f"[staging] provenance: sha256={arec.sha256} size={arec.byte_size}B")
    section = {
        "dataset": DEM_DATASET,
        "license": DEM_LICENSE,
        "note": DEM_NOTE,
        "source_bucket": "s3://copernicus-dem-30m (eu-central-1, no-sign-request, public)",
        "tiles": DEM_TILES,
        "tile_urls": [_tile_url(t) for t in DEM_TILES],
        "tile_metadata": tile_meta,
        "native_resolution_m": 30.0,
        "resampled_to": "the Sentinel-2 2024-03-08 reference grid (10 m, EPSG:32644), bilinear",
        "output_path": str(out_path),
        "sha256": arec.sha256,
        "byte_size": arec.byte_size,
        "refetched": refetched,
    }
    record_analysis_section("terrain_dem", section)
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="geoseek-stage-dem")
    parser.add_argument("--force", action="store_true", help="re-fetch even if already staged")
    args = parser.parse_args(argv)
    stage_dem(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

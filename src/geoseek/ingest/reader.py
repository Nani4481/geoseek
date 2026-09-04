"""Open a staged COG/GeoTIFF scene and read its bands, preserving geospatial context.

A "scene" here is a directory of single-band GeoTIFFs (as staged by
``geoseek.staging.download_datasets``): ``<scene_dir>/<BAND>.tif``. Bands are
read onto a common pixel grid — the first requested band is the reference
grid; any band on a different grid (CRS/transform/shape) is resampled onto it
via a WarpedVRT (nearest-neighbor for the categorical SCL band, bilinear for
continuous reflectance bands) so downstream tiling never has to think about
band misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


@dataclass
class SceneData:
    scene_id: str
    band_order: list[str]
    bands: dict[str, np.ndarray]
    crs: CRS
    transform: Affine
    nodata: float | None
    width: int
    height: int
    source_paths: dict[str, str] = field(default_factory=dict)
    source_tags: dict[str, str] = field(default_factory=dict)


def read_scene(scene_dir: Path, bands: list[str]) -> SceneData:
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    paths = {b: scene_dir / f"{b}.tif" for b in bands}

    missing = [b for b, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Scene '{scene_id}' missing band file(s) {missing} under {scene_dir}")

    ref_band = bands[0]
    with rasterio.open(paths[ref_band]) as ref_ds:
        ref_crs = ref_ds.crs
        ref_transform = ref_ds.transform
        ref_width, ref_height = ref_ds.width, ref_ds.height
        ref_nodata = ref_ds.nodata
        ref_tags = dict(ref_ds.tags())

    band_arrays: dict[str, np.ndarray] = {}
    for band in bands:
        with rasterio.open(paths[band]) as ds:
            same_grid = (
                ds.crs == ref_crs
                and ds.transform.almost_equals(ref_transform)
                and ds.width == ref_width
                and ds.height == ref_height
            )
            if same_grid:
                data = ds.read(1)
            else:
                resampling = Resampling.nearest if band.upper() == "SCL" else Resampling.bilinear
                with WarpedVRT(
                    ds, crs=ref_crs, transform=ref_transform,
                    width=ref_width, height=ref_height, resampling=resampling,
                ) as vrt:
                    data = vrt.read(1)
        band_arrays[band] = data

    return SceneData(
        scene_id=scene_id,
        band_order=list(bands),
        bands=band_arrays,
        crs=ref_crs,
        transform=ref_transform,
        nodata=ref_nodata,
        width=ref_width,
        height=ref_height,
        source_paths={b: str(p) for b, p in paths.items()},
        source_tags=ref_tags,
    )

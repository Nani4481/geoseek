"""Phase 8 Step B - distance to the nearest water channel / built-up area.

Both are **derived** from the existing Sentinel-2 2024-03-08 reference
observation's own indices, NOT an independent hydrology or road-network
dataset:

  * water channel : SCL class 6 ("water") on the reference scene - the same
    convention already used throughout ``geoseek.change`` (``SCL_WATER``).
  * built-up area  : NDBI > :data:`NDBI_BUILTUP_THRESHOLD` on the reference
    scene's Phase 3a NDBI raster (Zha et al. 2003's standard built-up
    threshold). This is a coarse spectral proxy, not a vector road/building
    layer - it also fires on bright bare soil, which is stated explicitly
    wherever it is surfaced (PS "do not overclaim": state what is measured
    vs. derived).

Distance is a plain Euclidean distance transform (``scipy.ndimage.
distance_transform_edt``) on the reference's 10 m grid, so 1 pixel of EDT
distance = 10 m - no reprojection needed since it is already the shared
change-pipeline grid. Computed once, cached to disk as GeoTIFFs alongside the
DEM outputs (:mod:`geoseek.terrain.dem`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

from geoseek.config import get_settings
from geoseek.terrain.dem import terrain_dir

SCL_WATER = 6
NDBI_BUILTUP_THRESHOLD = 0.0   # Zha et al. 2003; also fires on bright bare soil - see module docstring

DIST_WATER_FILE = "distance_to_water_m.tif"
DIST_BUILTUP_FILE = "distance_to_built_up_m.tif"


def water_path() -> Path:
    return terrain_dir() / DIST_WATER_FILE


def builtup_path() -> Path:
    return terrain_dir() / DIST_BUILTUP_FILE


def _reference_obs_dir() -> Path:
    from geoseek.change.analyze import DATE_TO_OBS

    return get_settings().datasets_dir / DATE_TO_OBS["2024"]


def compute_distance_rasters(force: bool = False) -> tuple[Path, Path]:
    d = terrain_dir()
    d.mkdir(parents=True, exist_ok=True)
    water_p, built_p = d / DIST_WATER_FILE, d / DIST_BUILTUP_FILE
    if not force and water_p.is_file() and built_p.is_file():
        return water_p, built_p

    ref_dir = _reference_obs_dir()
    with rasterio.open(ref_dir / "SCL.tif") as ds:
        scl = ds.read(1)
        profile = ds.profile
        px = abs(ds.transform.a)
    with rasterio.open(ref_dir / "NDBI.tif") as ds:
        ndbi = ds.read(1)

    water_mask = scl == SCL_WATER
    built_mask = np.isfinite(ndbi) & (ndbi > NDBI_BUILTUP_THRESHOLD)

    dist_water = distance_transform_edt(~water_mask) * px
    dist_built = distance_transform_edt(~built_mask) * px

    prof = dict(profile)
    prof.update(dtype="float32", nodata=float("nan"), compress="deflate", predictor=3,
                tiled=True, blockxsize=256, blockysize=256, count=1)
    with rasterio.open(water_p, "w", **prof) as dst:
        dst.write(dist_water.astype(np.float32), 1)
    with rasterio.open(built_p, "w", **prof) as dst:
        dst.write(dist_built.astype(np.float32), 1)
    return water_p, built_p

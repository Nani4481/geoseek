"""Phase 8 Step B - terrain from the staged DEM: elevation, slope, aspect.

Reads only ``data/datasets/dem_44RPQ_scaled/elevation_m.tif`` (staged by
``geoseek.staging.download_dem``, already resampled onto the exact 10 m UTM
44N grid the Sentinel-2 change pipeline uses). Offline; no network, no torch.

Slope and aspect use **Horn's (1981) method** - the standard 3x3
finite-difference formulation (the same one GDAL's ``gdaldem``, QGIS and
ArcGIS use), computed once over the full raster and cached to disk.

Elevation here is a **Digital Surface Model** value (Copernicus DEM GLO-30):
it includes building/canopy height, not bare-earth ground level - see
``geoseek.staging.download_dem`` for the full provenance note. Slope/aspect
inherit that same caveat (a building edge reads as a very steep "slope").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import convolve

from geoseek.config import get_settings

TERRAIN_DIR_NAME = "dem_44RPQ_scaled"
ELEVATION_FILE = "elevation_m.tif"
SLOPE_FILE = "slope_deg.tif"
ASPECT_FILE = "aspect_deg.tif"
ASPECT_FLAT_SENTINEL = -1.0   # no well-defined downhill direction (slope ~ 0)
FLAT_SLOPE_DEG = 0.05         # below this, aspect is undefined ("flat")

# Horn (1981) 3x3 kernels, row-major with row 0 = north (matches a north-up
# raster: transform.e < 0, row index increases southward).
_KX = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_KY = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def terrain_dir() -> Path:
    return get_settings().datasets_dir / TERRAIN_DIR_NAME


def elevation_path() -> Path:
    return terrain_dir() / ELEVATION_FILE


def slope_path() -> Path:
    return terrain_dir() / SLOPE_FILE


def aspect_path() -> Path:
    return terrain_dir() / ASPECT_FILE


def terrain_staged() -> bool:
    return elevation_path().is_file()


def _write_like(path: Path, arr: np.ndarray, profile: dict) -> None:
    prof = dict(profile)
    prof.update(dtype="float32", nodata=float("nan"), compress="deflate", predictor=3,
                tiled=True, blockxsize=256, blockysize=256, count=1)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)


def compute_slope_aspect(force: bool = False) -> tuple[Path, Path]:
    """Slope (degrees from horizontal, >=0) and aspect (compass bearing the
    slope faces, 0=N/90=E/180=S/270=W; :data:`ASPECT_FLAT_SENTINEL` where the
    surface is too flat for a direction to be meaningful). Cached to disk."""
    d = terrain_dir()
    slope_p, aspect_p = d / SLOPE_FILE, d / ASPECT_FILE
    if not force and slope_p.is_file() and aspect_p.is_file():
        return slope_p, aspect_p

    with rasterio.open(elevation_path()) as ds:
        z = ds.read(1)
        profile = ds.profile
        px = abs(ds.transform.a)   # 10.0 m pixel size

    valid = np.isfinite(z)
    zf = np.where(valid, z, np.nanmean(z[valid]) if valid.any() else 0.0)
    dzdx = convolve(zf, _KX, mode="nearest") / (8.0 * px)
    dzdy = convolve(zf, _KY, mode="nearest") / (8.0 * px)

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    # Standard math-angle -> compass-bearing conversion (ESRI/GDAL convention):
    # atan2(dz/dy, -dz/dx) as a mathematical angle (CCW from +x/east), remapped
    # to a compass bearing (CW from north); + 180 deg because ``dy`` here is
    # d(z)/d(row) and row increases SOUTHWARD in this north-up raster (the
    # textbook formula assumes y increasing north) - verified against 4
    # synthetic tilted planes of known orientation (N/E/S/W) before shipping.
    ang = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(ang < 0.0, 90.0 - ang,
             np.where(ang > 90.0, 360.0 - ang + 90.0, 90.0 - ang))
    aspect = (aspect + 180.0) % 360.0
    aspect = np.where(slope < FLAT_SLOPE_DEG, ASPECT_FLAT_SENTINEL, aspect)

    slope[~valid] = np.nan
    aspect[~valid] = np.nan

    _write_like(slope_p, slope, profile)
    _write_like(aspect_p, aspect, profile)
    return slope_p, aspect_p


COMPASS_16 = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass_label(aspect_deg: float | None) -> str | None:
    """16-point compass label for an aspect in degrees, or None if flat/unknown."""
    if aspect_deg is None or aspect_deg < 0 or not np.isfinite(aspect_deg):
        return None
    idx = int(round(aspect_deg / 22.5)) % 16
    return COMPASS_16[idx]

"""Phase 8 Step B - per-candidate terrain evidence (elevation, slope, aspect,
distance to water / built-up), assembled from the cached rasters in
:mod:`geoseek.terrain.dem` / :mod:`geoseek.terrain.proximity`.

All five rasters share the EXACT pixel grid every Sentinel-2 date in this
project is staged onto (same MGRS tile 44RPQ, same AOI window, same UTM 44N
10 m transform - a Sentinel-2 tile's grid is fixed by ESA, so every date
already lines up bit-for-bit; this is the same assumption the change pipeline
relies on when it reuses a candidate's ``bbox``/``centroid_rc`` against any
date's raster). So a candidate's ``centroid_rc`` from
:mod:`geoseek.change.analyze` indexes these rasters directly - no
reprojection, no coordinate transform, at query time.

Offline; no network, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

from geoseek.terrain.dem import ASPECT_FLAT_SENTINEL, aspect_path, compass_label, elevation_path, slope_path
from geoseek.terrain.proximity import builtup_path, water_path

MEASURED = "measured (Copernicus DEM GLO-30, resampled to the S2 10 m grid)"
DERIVED_WATER = "derived (distance to the nearest SCL 'water' pixel on the 2024-03-08 reference scene)"
DERIVED_BUILTUP = ("derived (distance to the nearest NDBI > 0 pixel on the 2024-03-08 reference scene - "
                    "a spectral built-up proxy, not a road/building vector layer; also fires on bright "
                    "bare soil)")


@dataclass(frozen=True)
class TerrainFeatures:
    elevation_m: float | None
    slope_deg: float | None
    aspect_deg: float | None
    aspect_compass: str | None
    distance_to_water_m: float | None
    distance_to_built_up_m: float | None

    def as_dict(self) -> dict:
        return {
            "elevation_m": _r(self.elevation_m, 1),
            "slope_deg": _r(self.slope_deg, 1),
            "aspect_deg": _r(self.aspect_deg, 1),
            "aspect_compass": self.aspect_compass,
            "distance_to_water_m": _r(self.distance_to_water_m, 0),
            "distance_to_built_up_m": _r(self.distance_to_built_up_m, 0),
            "provenance": {
                "elevation_m": MEASURED, "slope_deg": MEASURED + "; slope/aspect derived from it",
                "aspect_deg": MEASURED + "; slope/aspect derived from it",
                "distance_to_water_m": DERIVED_WATER,
                "distance_to_built_up_m": DERIVED_BUILTUP,
            },
            "plain_language": describe_terrain(self),
        }


def _r(x: float | None, n: int) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def describe_terrain(f: "TerrainFeatures") -> str:
    """One plain-language sentence, e.g. '94 m elevation, 2 degree slope
    (facing SE), 340 m from the river channel, 120 m from the nearest
    built-up area.' Omits any field that could not be measured."""
    parts = []
    if f.elevation_m is not None and np.isfinite(f.elevation_m):
        parts.append(f"{f.elevation_m:.0f} m elevation")
    if f.slope_deg is not None and np.isfinite(f.slope_deg):
        facing = f" (facing {f.aspect_compass})" if f.aspect_compass else " (flat)"
        parts.append(f"{f.slope_deg:.0f} degree slope{facing}")
    if f.distance_to_water_m is not None and np.isfinite(f.distance_to_water_m):
        parts.append(f"{f.distance_to_water_m:.0f} m from the river channel")
    if f.distance_to_built_up_m is not None and np.isfinite(f.distance_to_built_up_m):
        parts.append(f"{f.distance_to_built_up_m:.0f} m from the nearest built-up area")
    return ", ".join(parts) if parts else "terrain context unavailable"


class TerrainSampler:
    """Opens the 5 cached terrain rasters once; samples every candidate cheaply
    (a handful of pixels each, like :class:`geoseek.change.analyze.ProbLookup`)."""

    def __init__(self):
        self._paths = {
            "elevation_m": elevation_path(), "slope_deg": slope_path(), "aspect_deg": aspect_path(),
            "distance_to_water_m": water_path(), "distance_to_built_up_m": builtup_path(),
        }
        missing = [k for k, p in self._paths.items() if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"terrain rasters missing: {missing} - run `python -m geoseek.terrain.build` first"
            )
        self._ds = {k: rasterio.open(p) for k, p in self._paths.items()}

    def close(self) -> None:
        for ds in self._ds.values():
            ds.close()

    def __enter__(self) -> "TerrainSampler":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def sample_rc(self, row: float, col: float) -> TerrainFeatures:
        """Sample at a (row, col) in the shared reference grid (a candidate's
        ``centroid_rc``). A tiny 3x3 window median around the point damps single-
        pixel DEM/index noise; falls back to None fields if out of raster bounds."""
        vals: dict[str, float | None] = {}
        for k, ds in self._ds.items():
            r, c = int(round(row)), int(round(col))
            if not (0 <= r < ds.height and 0 <= c < ds.width):
                vals[k] = None
                continue
            r0, c0 = max(r - 1, 0), max(c - 1, 0)
            win = rasterio.windows.Window(c0, r0, min(3, ds.width - c0), min(3, ds.height - r0))
            arr = ds.read(1, window=win)
            finite = arr[np.isfinite(arr)]
            vals[k] = float(np.median(finite)) if finite.size else None

        aspect = vals["aspect_deg"]
        if aspect is not None and aspect < 0:   # ASPECT_FLAT_SENTINEL median-contaminated -> treat as flat
            aspect = None
        return TerrainFeatures(
            elevation_m=vals["elevation_m"], slope_deg=vals["slope_deg"], aspect_deg=aspect,
            aspect_compass=compass_label(aspect) if aspect is not None else None,
            distance_to_water_m=vals["distance_to_water_m"],
            distance_to_built_up_m=vals["distance_to_built_up_m"],
        )

"""Raster-space <-> EPSG:4326 helpers for the analyst UI.

A change candidate carries its footprint as a pixel bbox ``[r0, c0, r1, c1]`` in
the span probability raster's grid. The map view and the GeoJSON export both
need that as a real lon/lat polygon, so this module reprojects the bbox corners
through the raster's own transform + CRS (read once from the GeoTIFF header -
no pixel data touched) and caches the transformer.
"""

from __future__ import annotations

import functools
from pathlib import Path


@functools.lru_cache(maxsize=8)
def _grid(prob_raster_path: str) -> tuple:
    """(affine transform, pyproj Transformer to 4326) for a probability GeoTIFF."""
    import pyproj
    import rasterio

    with rasterio.open(prob_raster_path) as ds:
        transform = ds.transform
        crs = ds.crs
    to_wgs = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transform, to_wgs


def bbox_rc_to_polygon_4326(prob_raster_path: str | Path, bbox_rc) -> list[list[float]]:
    """``[r0, c0, r1, c1]`` (row/col, r1/c1 exclusive) -> a closed GeoJSON ring
    ``[[lon,lat], ...]`` (5 points, first == last), corners reprojected to 4326."""
    transform, to_wgs = _grid(str(prob_raster_path))
    r0, c0, r1, c1 = bbox_rc
    corners_rc = [(r0, c0), (r0, c1), (r1, c1), (r1, c0), (r0, c0)]
    a, b, c_, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    ring: list[list[float]] = []
    for r, col in corners_rc:
        x = a * col + b * r + c_          # affine (col, row) -> projected x, y
        y = d * col + e * r + f
        lon, lat = to_wgs.transform(x, y)
        ring.append([round(float(lon), 6), round(float(lat), 6)])
    return ring


def polygon_geojson(prob_raster_path: str | Path, bbox_rc) -> dict:
    return {"type": "Polygon", "coordinates": [bbox_rc_to_polygon_4326(prob_raster_path, bbox_rc)]}


def ring_bounds(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))

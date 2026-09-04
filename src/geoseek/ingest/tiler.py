"""Cut a SceneData into fixed-size tiles with lon/lat footprints.

Tiles are TILE_SIZE x TILE_SIZE pixels except at the scene's right/bottom
edge, where a smaller partial tile is kept (never padded, never dropped).
A tile is skipped only if every pixel of its reference band is nodata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Polygon

from geoseek.ingest.reader import SceneData

TILE_SIZE = 256

_TILE_ID_ROW_COL_RE = re.compile(r"_r(\d+)_c(\d+)$")


@dataclass
class Tile:
    tile_id: str
    row: int
    col: int
    x_off: int
    y_off: int
    width: int
    height: int
    bands: dict[str, np.ndarray]
    footprint_wkt_4326: str
    nodata: float | None


def _is_fully_nodata(arr: np.ndarray, nodata: float | None) -> bool:
    if nodata is not None:
        return bool(np.all(arr == nodata))
    return bool(np.all(arr == 0))


def tile_scene(scene: SceneData, tile_size: int = TILE_SIZE) -> list[Tile]:
    transformer = pyproj.Transformer.from_crs(scene.crs, "EPSG:4326", always_xy=True)
    ref_band = scene.band_order[0]

    tiles: list[Tile] = []
    for row0 in range(0, scene.height, tile_size):
        for col0 in range(0, scene.width, tile_size):
            row1 = min(row0 + tile_size, scene.height)
            col1 = min(col0 + tile_size, scene.width)
            th, tw = row1 - row0, col1 - col0

            band_crops = {b: arr[row0:row1, col0:col1] for b, arr in scene.bands.items()}
            if _is_fully_nodata(band_crops[ref_band], scene.nodata):
                continue

            win = Window(col0, row0, tw, th)
            win_transform = window_transform(win, scene.transform)
            corners_native = [
                win_transform @ (0, 0),
                win_transform @ (tw, 0),
                win_transform @ (tw, th),
                win_transform @ (0, th),
            ]
            corners_4326 = [transformer.transform(x, y) for x, y in corners_native]
            footprint = Polygon(corners_4326)

            row_idx, col_idx = row0 // tile_size, col0 // tile_size
            tile_id = f"{scene.scene_id}_r{row_idx:03d}_c{col_idx:03d}"

            tiles.append(
                Tile(
                    tile_id=tile_id,
                    row=row_idx,
                    col=col_idx,
                    x_off=col0,
                    y_off=row0,
                    width=tw,
                    height=th,
                    bands=band_crops,
                    footprint_wkt_4326=footprint.wkt,
                    nodata=scene.nodata,
                )
            )
    return tiles


def parse_tile_row_col(tile_id: str) -> tuple[int, int]:
    """Recover (row, col) from a tile_id in the `<scene_id>_rNNN_cNNN` format tile_scene() produces."""
    m = _TILE_ID_ROW_COL_RE.search(tile_id)
    if not m:
        raise ValueError(f"tile_id '{tile_id}' doesn't match the expected '..._rNNN_cNNN' format")
    return int(m.group(1)), int(m.group(2))


def read_tile_window(
    scene_dir: Path, row: int, col: int, bands: list[str], tile_size: int = TILE_SIZE
) -> tuple[dict[str, np.ndarray], float | None]:
    """Read just one tile's pixel window directly off the per-band GeoTIFFs.

    Used to regenerate a single tile (e.g. for a thumbnail endpoint) without
    re-tiling the whole scene. Bands are assumed pre-aligned to the same grid
    (true for everything staged via geoseek.staging.download_datasets).
    """
    scene_dir = Path(scene_dir)
    col0, row0 = col * tile_size, row * tile_size

    band_arrays: dict[str, np.ndarray] = {}
    nodata: float | None = None
    for band in bands:
        path = scene_dir / f"{band}.tif"
        with rasterio.open(path) as ds:
            col1 = min(col0 + tile_size, ds.width)
            row1 = min(row0 + tile_size, ds.height)
            if col0 >= ds.width or row0 >= ds.height:
                raise ValueError(f"tile row={row} col={col} is out of bounds for scene '{scene_dir.name}'")
            win = Window(col0, row0, col1 - col0, row1 - row0)
            band_arrays[band] = ds.read(1, window=win)
            if nodata is None:
                nodata = ds.nodata
    return band_arrays, nodata

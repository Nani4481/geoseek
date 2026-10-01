"""Region-level water context: distance of each tile to "the river" and to any open water.

Same method as the Phase 7a features (``scripts/eval_retrieval_features.py``), generalized from one AOI to any
region: union the per-date NDWI>0 water masks of the region, drop specks (3x3 opening), take the LARGEST connected
component as "the river" and the whole union as "any open water", then Euclidean distance transforms at 40 m.

Masks come from ``geoseek.spectral.descriptor`` (written in the same pass as the tile descriptors) on a GLOBAL 40 m
grid, so dates whose crop windows start at different pixels still line up.

Limitation, stated: "the largest water component" is a river in Ayodhya; in a delta (Sundarbans), a backwater
(Kerala) or a desert it is whatever the biggest water body is. The river-relative judge criteria are therefore only as
meaningful as that component is river-like; the evaluation reports results with and without the river-relative queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from geoseek.spectral.descriptor import TILE_SIZE, WATER_DECIMATE

NO_WATER_M = 1.0e6          # distance reported when a region has no water component at all
PX_M = 10.0 * WATER_DECIMATE


@dataclass(frozen=True)
class WaterMask:
    mask: np.ndarray          # bool, decimated, rows/cols = global 40 m units
    unit_y0: int              # global unit index of mask row 0 (global 10 m row index / 4)
    unit_x0: int
    origin_gx: int            # scene origin (column 0, row 0) on the global 10 m grid
    origin_gy: int
    width: int
    height: int


def load_mask(path: Path) -> WaterMask:
    z = np.load(path)
    step = int(z["step"])
    if step != WATER_DECIMATE:
        raise ValueError(f"{path}: decimation step {step} != {WATER_DECIMATE}")
    gy, gx = int(z["gy_first"]), int(z["gx_first"])
    if gy % step or gx % step:
        raise ValueError(f"{path}: first sample ({gy},{gx}) is not on the global {step}-px grid")
    return WaterMask(z["mask"].astype(bool), gy // step, gx // step, int(z["origin_gx"]), int(z["origin_gy"]),
                     int(z["width"]), int(z["height"]))


def union_masks(masks: Sequence[WaterMask]) -> tuple[np.ndarray, int, int]:
    """OR the masks on the shared global grid. Returns (union, unit_y0, unit_x0)."""
    y0 = min(m.unit_y0 for m in masks)
    x0 = min(m.unit_x0 for m in masks)
    y1 = max(m.unit_y0 + m.mask.shape[0] for m in masks)
    x1 = max(m.unit_x0 + m.mask.shape[1] for m in masks)
    out = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for m in masks:
        out[m.unit_y0 - y0: m.unit_y0 - y0 + m.mask.shape[0], m.unit_x0 - x0: m.unit_x0 - x0 + m.mask.shape[1]] |= m.mask
    return out, y0, x0


def river_and_water_distances(union: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(dist_river_m, dist_water_m, meta) on the union grid. No water at all -> both are ``NO_WATER_M`` everywhere."""
    any_water = ndi.binary_opening(union, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndi.label(any_water, structure=np.ones((3, 3)))
    if n == 0:
        far = np.full(union.shape, NO_WATER_M, dtype=np.float32)
        return far, far.copy(), {"n_water_components": 0, "river_component_cells": 0, "union_water_cells": 0}
    sizes = ndi.sum(any_water, labels, index=np.arange(1, n + 1))
    river = labels == (int(np.argmax(sizes)) + 1)
    dist_river = ndi.distance_transform_edt(~river) * PX_M
    dist_water = ndi.distance_transform_edt(~any_water) * PX_M
    return (dist_river.astype(np.float32), dist_water.astype(np.float32),
            {"n_water_components": int(n), "river_component_cells": int(river.sum()),
             "union_water_cells": int(any_water.sum())})


def tile_distances(scene: WaterMask, row: int, col: int, dist_river: np.ndarray, dist_water: np.ndarray,
                   unit_y0: int, unit_x0: int) -> tuple[float, float]:
    """Min distance (m) over the 40 m samples inside tile (row, col) of ``scene``; nearest sample if it holds none."""
    y0, x0 = row * TILE_SIZE, col * TILE_SIZE
    y1, x1 = min(y0 + TILE_SIZE, scene.height), min(x0 + TILE_SIZE, scene.width)
    gy0, gy1 = scene.origin_gy + y0, scene.origin_gy + y1
    gx0, gx1 = scene.origin_gx + x0, scene.origin_gx + x1
    uy0, uy1 = -(-gy0 // WATER_DECIMATE), -(-gy1 // WATER_DECIMATE)            # ceil: samples at multiples of 4 inside [g0, g1)
    ux0, ux1 = -(-gx0 // WATER_DECIMATE), -(-gx1 // WATER_DECIMATE)
    if uy1 <= uy0:
        uy1 = uy0 + 1
    if ux1 <= ux0:
        ux1 = ux0 + 1
    H, W = dist_river.shape
    ys = slice(max(uy0 - unit_y0, 0), min(uy1 - unit_y0, H))
    xs = slice(max(ux0 - unit_x0, 0), min(ux1 - unit_x0, W))
    if ys.start >= ys.stop or xs.start >= xs.stop:                              # tile outside the union grid
        return NO_WATER_M, NO_WATER_M
    return float(dist_river[ys, xs].min()), float(dist_water[ys, xs].min())


def region_context(union_over: Sequence[Path], apply_to: dict[Path, Sequence[tuple[str, int, int]]]) -> tuple[list[tuple[str, float, float]], dict]:
    """River/water distances for every tile of the ``apply_to`` scenes, from the union of ``union_over`` masks.

    ``apply_to`` maps a scene's mask file -> its tiles ``(tile_id, row, col)``. Returns (rows, meta) with rows
    ``(tile_id, dist_river_m, dist_water_m)``.
    """
    masks = {p: load_mask(p) for p in {*union_over, *apply_to}}
    union, uy0, ux0 = union_masks([masks[p] for p in union_over])
    dr, dw, meta = river_and_water_distances(union)
    rows: list[tuple[str, float, float]] = []
    for path, tiles in apply_to.items():
        m = masks[path]
        for tile_id, r, c in tiles:
            a, b = tile_distances(m, r, c, dr, dw, uy0, ux0)
            rows.append((tile_id, round(a, 1), round(b, 1)))
    meta.update({"union_scenes": len(union_over), "union_shape": list(union.shape), "pixel_m": PX_M})
    return rows, meta

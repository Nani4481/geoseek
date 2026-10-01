"""Per-tile spectral descriptor: NDVI / NDWI / NDBI statistics, class fractions, texture and band-ratio moments.

ONE pass serves two consumers (so it is computed once): the retrieval JUDGE (the Phase 7a graders read exactly
these features) and the constrained-retrieval layer (range predicates) / a re-ranking term. The formulas are
those of ``geoseek.change.indices`` (single source of truth); validity, thresholds and the Sobel texture measure
are those of ``scripts/eval_retrieval_features.py`` so a descriptor computed here is interchangeable with a Phase 7a
feature for the same tile (checked against the stored Ayodhya features in the Block A evaluation).

Everything is computed from bands the retrieval models never see (NIR / SWIR): B03, B04 appear in the true-colour
input too, but B08 / B11 and every index built on them do not.

Scenes are processed in tile-row strips, never whole: one strip is 256 x W pixels x 5 bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi

from geoseek.change.indices import normalized_difference
from geoseek.spectral.fields import DESCRIPTOR_VERSION, INDEX_STATS, INDICES, RATIOS

TILE_SIZE = 256
MIN_VALID_FRAC = 0.05            # below this a tile is not "usable" (same rule as the Phase 7a features)
SCL_VALID = (2, 4, 5, 6, 7, 11)  # dark, vegetation, bare, water, unclassified, snow/ice
NDWI_WATER = 0.0                 # McFeeters water threshold
EDGE_NDVI_STEP = 0.10            # per-pixel NDVI gradient that counts as a boundary
RATIO_CLIP = 20.0                # band ratios are clipped to [0, 20] so their moments stay finite on dark pixels
WATER_DECIMATE = 4               # decimated water mask step (10 m -> 40 m), on a GLOBAL grid so dates align
BANDS = ("B03", "B04", "B08", "B11", "SCL")
WATER_MASK_NAME = "WATER_DEC4.npz"


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    row: int
    col: int


def _sobel_mag(a: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(a, axis=1, mode="nearest")
    gy = ndi.sobel(a, axis=0, mode="nearest")
    return np.hypot(gx, gy) / 8.0           # ndi.sobel sums a [1,2,1]*[-1,0,1] kernel -> /8 = gradient per pixel


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a.astype(np.float32) / np.maximum(b.astype(np.float32), 1.0), 0.0, RATIO_CLIP)


def describe_tile(b03: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray, scl: np.ndarray) -> dict:
    """Descriptor of one tile window (all inputs the same HxW). Unusable tiles carry only the header fields."""
    ndvi = normalized_difference(b08, b04)
    ndwi = normalized_difference(b03, b08)
    ndbi = normalized_difference(b11, b08)
    valid = np.isin(scl, SCL_VALID) & np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi)
    n, total = int(valid.sum()), ndvi.size
    head = {"descriptor_version": DESCRIPTOR_VERSION, "valid_frac": n / total, "n_valid": n}
    if n < MIN_VALID_FRAC * total:
        return {**head, "usable": 0}

    out: dict = {**head, "usable": 1}
    vals = {"ndvi": ndvi[valid], "ndwi": ndwi[valid], "ndbi": ndbi[valid]}
    for name in INDICES:
        v = vals[name]
        p10, p50, p90 = np.percentile(v, (10, 50, 90))
        out.update({f"{name}_mean": float(v.mean()), f"{name}_std": float(v.std()),
                    f"{name}_p10": float(p10), f"{name}_p50": float(p50), f"{name}_p90": float(p90)})
    dv, dw, db = vals["ndvi"], vals["ndwi"], vals["ndbi"]
    filled = np.where(np.isfinite(ndvi), ndvi, np.median(dv))
    out["edge_density"] = float(np.mean(_sobel_mag(filled)[valid] > EDGE_NDVI_STEP))
    out.update({
        "water_frac": float(np.mean(dw > NDWI_WATER)),
        "veg_frac": float(np.mean(dv > 0.35)),
        "dense_veg_frac": float(np.mean(dv > 0.55)),
        "bare_frac": float(np.mean((dv < 0.20) & (dw < 0.0) & (db > -0.15))),
        "built_frac": float(np.mean((db > -0.05) & (dv < 0.30) & (dw < 0.0))),
    })
    for name, (a, b) in {"sr_nir_red": (b08, b04), "sr_swir_nir": (b11, b08), "sr_green_red": (b03, b04)}.items():
        r = _ratio(a, b)[valid]
        out[f"{name}_mean"], out[f"{name}_std"] = float(r.mean()), float(r.std())
    return out


def _check_alignment(refs: dict[str, rasterio.DatasetReader]) -> None:
    ref = refs["B04"]
    for name, ds in refs.items():
        if (ds.width, ds.height) != (ref.width, ref.height) or not ds.transform.almost_equals(ref.transform):
            raise ValueError(f"{name} is not on the B04 grid (shape/transform differ) - cannot describe tiles")


def describe_scene(scene_dir: Path, tiles: Sequence[TileSpec], *, write_water_mask: bool = True,
                   log: Callable[[str], None] | None = None) -> list[dict]:
    """Describe the given catalog tiles of one scene directory, strip by strip. Returns one row dict per tile.

    ``tiles`` come from the catalog, so tile ids / geometry are never regenerated here - a (row, col) outside the
    raster is an error, not a silent skip. As a by-product of the same pass, a decimated NDWI>0 water mask is written
    next to the bands (``WATER_DEC4.npz``) for the region-level river-distance step.
    """
    scene_dir = Path(scene_dir)
    by_row: dict[int, list[TileSpec]] = {}
    for t in tiles:
        by_row.setdefault(t.row, []).append(t)
    rows_out: list[dict] = []
    datasets = {b: rasterio.open(scene_dir / f"{b}.tif") for b in BANDS}
    try:
        _check_alignment(datasets)
        ref = datasets["B04"]
        H, W = ref.height, ref.width
        res_x, res_y = abs(ref.transform.a), abs(ref.transform.e)
        if abs(res_x - 10) > 1e-6 or abs(res_y - 10) > 1e-6:
            raise ValueError(f"expected a 10 m grid, got {res_x} x {res_y}")
        gx0, gy0 = round(ref.transform.c / 10.0), round(-ref.transform.f / 10.0)     # absolute 10 m grid indices
        j0, i0 = (-gx0) % WATER_DECIMATE, (-gy0) % WATER_DECIMATE
        sample_cols = np.arange(j0, W, WATER_DECIMATE)
        water_rows: dict[int, np.ndarray] = {}

        for r in sorted(by_row):
            y0 = r * TILE_SIZE
            if y0 >= H:
                raise ValueError(f"tile row {r} is outside {scene_dir.name} (height {H})")
            y1 = min(y0 + TILE_SIZE, H)
            strip = {b: datasets[b].read(1, window=Window(0, y0, W, y1 - y0)) for b in BANDS}
            for t in by_row[r]:
                x0 = t.col * TILE_SIZE
                if x0 >= W:
                    raise ValueError(f"tile col {t.col} is outside {scene_dir.name} (width {W})")
                x1 = min(x0 + TILE_SIZE, W)
                sl = (slice(None), slice(x0, x1))
                row = describe_tile(*(strip[b][sl] for b in ("B03", "B04", "B08", "B11", "SCL")))
                row["tile_id"] = t.tile_id
                rows_out.append(row)
            if write_water_mask:
                ndwi = normalized_difference(strip["B03"], strip["B08"])
                for i in range(y0, y1):
                    if (i - i0) % WATER_DECIMATE == 0:
                        water_rows[i] = (ndwi[i - y0, sample_cols] > NDWI_WATER)
            if log and (r % 10 == 0):
                log(f"    {scene_dir.name}: strip {r + 1}/{max(by_row) + 1}")
        if write_water_mask:
            sample_rows = sorted(water_rows)
            mask = np.stack([water_rows[i] for i in sample_rows]) if sample_rows else np.zeros((0, len(sample_cols)), bool)
            tmp = scene_dir / (WATER_MASK_NAME + ".tmp.npz")
            np.savez_compressed(tmp, mask=mask, gy_first=gy0 + (sample_rows[0] if sample_rows else 0),
                                gx_first=gx0 + (int(sample_cols[0]) if len(sample_cols) else 0),
                                step=WATER_DECIMATE, crs=str(ref.crs),
                                origin_gx=gx0, origin_gy=gy0, width=W, height=H)    # scene origin on the global 10 m grid
            tmp.replace(scene_dir / WATER_MASK_NAME)
    finally:
        for ds in datasets.values():
            ds.close()
    return rows_out

"""Normalized-difference spectral indices for the two-date pair.

    NDVI = (B08 - B04) / (B08 + B04)   vegetation vigour   (high over cropland)
    NDWI = (B03 - B08) / (B03 + B08)   open water          (high over the river)
           (McFeeters 1996 - green/NIR formulation)
    NDBI = (B11 - B08) / (B11 + B08)   built-up / bare     (high over the town)
           (Zha et al. 2003)

All three are computed on the *normalized* reflectance (subject date mapped
into reference-date space by :mod:`geoseek.change.normalize`) so the two
dates' indices are directly comparable in Phase 3b. Each index is a ratio,
so a common multiplicative scale cancels - reflectance or DN both work as
input - but the additive inter-date offset does not fully cancel, which is
exactly why normalization runs first. No network.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

INDEX_BANDS = {
    "NDVI": ("B08", "B04"),
    "NDWI": ("B03", "B08"),
    "NDBI": ("B11", "B08"),
}


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), float32, 0 where the sum is non-positive, clipped to [-1, 1]."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = a + b
    out = np.zeros_like(denom, dtype=np.float32)
    np.divide(a - b, denom, out=out, where=denom > 0)
    return np.clip(out, -1.0, 1.0)


def ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    return normalized_difference(b08, b04)


def ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    return normalized_difference(b03, b08)


def ndbi(b11: np.ndarray, b08: np.ndarray) -> np.ndarray:
    return normalized_difference(b11, b08)


def compute_indices(bands: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """{'NDVI','NDWI','NDBI'} from a mapping providing B03, B04, B08, B11.

    ``bands`` values may be reflectance or DN; a nodata pixel (any input band
    exactly 0) yields 0 in every index and is reported via the returned
    ``valid`` mask below only if you compute it separately.
    """
    missing = {b for pair in INDEX_BANDS.values() for b in pair} - set(bands)
    if missing:
        raise ValueError(f"compute_indices missing band(s): {sorted(missing)}")
    return {
        "NDVI": ndvi(bands["B08"], bands["B04"]),
        "NDWI": ndwi(bands["B03"], bands["B08"]),
        "NDBI": ndbi(bands["B11"], bands["B08"]),
    }


def valid_mask(bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """True where every band used by any index is present (non-zero / finite)."""
    used = {b for pair in INDEX_BANDS.values() for b in pair}
    m = None
    for b in used:
        arr = bands[b]
        ok = np.isfinite(arr) & (arr != 0)
        m = ok if m is None else (m & ok)
    return m


def range_summary(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """min / p2 / mean / median / p98 / max over the valid pixels of ``arr``."""
    v = arr[mask] if mask is not None else arr.ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in ("min", "p2", "mean", "median", "p98", "max")}
    return {
        "min": float(v.min()),
        "p2": float(np.percentile(v, 2)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p98": float(np.percentile(v, 98)),
        "max": float(v.max()),
    }

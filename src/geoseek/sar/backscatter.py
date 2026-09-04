"""Speckle-filtered Sentinel-1 GRD backscatter change (VV / VH, dB).

The staged GRD is uncalibrated amplitude DN (see
``geoseek.staging.download_sentinel1``). Absolute sigma0/gamma0 is NOT
recovered; instead the **change** between two acquisitions of the same beam /
relative orbit is

    dB = 10 * log10( I_later / I_earlier ),   I = DN^2  (intensity)

in which the constant calibration term cancels. Speckle is suppressed with an
adaptive **Lee filter** (Lee, 1980) in the intensity domain before the ratio -
hand-rolled on numpy/scipy (scikit-image is deliberately not a geoseek
dependency, consistent with the rest of the codebase).

No network.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# IW GRDH equivalent number of looks (ESA product spec: ~5 range x 1 azimuth,
# ENL ~ 4.4). Used as the Lee filter's a-priori speckle level.
ENL_IW_GRDH = 4.4
LEE_WINDOW = 7          # ~70 m - enough looks to knock speckle down, small enough to keep edges


def lee_filter(intensity: np.ndarray, *, window: int = LEE_WINDOW, enl: float = ENL_IW_GRDH,
               valid: np.ndarray | None = None) -> np.ndarray:
    """Adaptive Lee speckle filter on a SAR **intensity** image.

    ``filtered = mean + W * (pixel - mean)`` with
    ``W = max(0, 1 - Cu^2 / Ci^2)``, ``Cu^2 = 1/ENL`` (a-priori speckle),
    ``Ci^2 = local_var / local_mean^2`` (observed). Over a homogeneous patch
    ``Ci^2 -> Cu^2`` so ``W -> 0`` and the output is the local mean (full
    smoothing); over an edge / point target ``Ci^2 >> Cu^2`` so ``W -> 1`` and
    the pixel is preserved.
    """
    x = intensity.astype(np.float64)
    if valid is not None:
        x = np.where(valid, x, np.nan)
    # nan-aware local mean / mean-of-squares via a validity-normalised boxcar
    m = np.isfinite(x)
    xf = np.where(m, x, 0.0)
    n = uniform_filter(m.astype(np.float64), window, mode="nearest")
    n = np.where(n > 0, n, np.nan)
    mean = uniform_filter(xf, window, mode="nearest") / n
    mean_sq = uniform_filter(xf * xf, window, mode="nearest") / n
    var = np.maximum(mean_sq - mean * mean, 0.0)

    cu2 = 1.0 / float(enl)
    ci2 = var / np.maximum(mean * mean, 1e-12)
    w = np.clip(1.0 - cu2 / np.maximum(ci2, 1e-12), 0.0, 1.0)
    out = mean + w * (np.where(np.isfinite(x), x, mean) - mean)
    return np.where(np.isfinite(intensity if valid is None else x), out, np.nan)


def db_change(amp_earlier: np.ndarray, amp_later: np.ndarray, *, speckle_filter: bool = True,
              clip_db: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """(dB change, valid mask) for one polarization.

    ``amp_*`` are amplitude DN (uint16, 0 = nodata). Returns ``dB`` (NaN where
    invalid) clipped to ``+/- clip_db`` and the boolean valid mask.
    """
    a1 = amp_earlier.astype(np.float64)
    a2 = amp_later.astype(np.float64)
    valid = (a1 > 0) & (a2 > 0)
    i1 = a1 * a1
    i2 = a2 * a2
    if speckle_filter:
        i1 = lee_filter(i1, valid=valid)
        i2 = lee_filter(i2, valid=valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(i2 / i1)
    db = np.where(valid & np.isfinite(db), np.clip(db, -clip_db, clip_db), np.nan)
    return db.astype(np.float32), valid


def summarize_db(db: np.ndarray, mask: np.ndarray) -> dict:
    """median / p16 / p84 / mean dB over ``mask`` (speckle-robust: report the median)."""
    v = db[mask & np.isfinite(db)]
    if v.size == 0:
        return {"n_px": 0, "median_db": None, "mean_db": None, "p16_db": None, "p84_db": None}
    return {"n_px": int(v.size), "median_db": float(np.median(v)), "mean_db": float(np.mean(v)),
            "p16_db": float(np.percentile(v, 16)), "p84_db": float(np.percentile(v, 84))}

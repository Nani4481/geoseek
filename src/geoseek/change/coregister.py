"""Sub-pixel co-registration check for a two-date scene pair.

Before comparing two acquisition dates we must confirm their pixels actually
line up on the ground. This module measures the residual translation between
the dates with **FFT phase correlation** on a spatially stable band, reports
the median ``(dy, dx)`` shift over several overlapping tiles, and - only if
that shift is larger than a sub-pixel threshold - resamples one date onto the
other to correct it.

Convention
----------
``phase_correlation_shift(reference, moving) -> (dy, dx)`` returns the shift
you pass to ``scipy.ndimage.shift(moving, (dy, dx))`` to bring ``moving`` into
alignment with ``reference`` (same sign convention as
``skimage.registration.phase_cross_correlation``). Positive ``dy`` moves
content downward (towards higher row index), positive ``dx`` rightward.

No network. Pure NumPy/SciPy - scikit-image is not a project dependency, so
the phase-correlation core (Hann-windowed cross-power spectrum + parabolic
peak interpolation for the sub-pixel term) is implemented here directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import shift as ndi_shift

# Residual shift at or below this (in pixels, on the 10m grid) is treated as
# "already aligned" - within the Sentinel-2 L1C multi-temporal registration
# spec - and left uncorrected.
SUBPIXEL_THRESHOLD_PX = 0.5


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def phase_correlation_shift(
    reference: np.ndarray, moving: np.ndarray, *, window: bool = True
) -> tuple[float, float]:
    """Estimate the sub-pixel translation ``(dy, dx)`` aligning ``moving`` to ``reference``.

    ``scipy.ndimage.shift(moving, (dy, dx))`` then overlays ``reference``.

    FFT phase correlation: the normalized cross-power spectrum of the two
    images has unit magnitude and a linear phase ramp whose slope is the
    shift; its inverse FFT is a sharp peak at the (wrapped) integer shift.
    A parabolic fit through the peak and its two neighbours along each axis
    provides the sub-pixel term. An optional Hann window suppresses the
    spectral leakage from the images' non-periodic edges.
    """
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape:
        raise ValueError(
            f"reference {reference.shape} and moving {moving.shape} must have the same shape"
        )
    if reference.ndim != 2:
        raise ValueError("phase_correlation_shift expects 2-D arrays")

    ref = reference - reference.mean()
    mov = moving - moving.mean()
    if window:
        w = _hann2d(ref.shape)
        ref = ref * w
        mov = mov * w

    fref = np.fft.fft2(ref)
    fmov = np.fft.fft2(mov)
    cross = fref * np.conj(fmov)
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    corr = np.fft.ifft2(cross / mag).real

    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)

    shifts: list[float] = []
    for axis, p in enumerate(peak):
        n = corr.shape[axis]
        idx_m = list(peak)
        idx_p = list(peak)
        idx_m[axis] = (p - 1) % n
        idx_p[axis] = (p + 1) % n
        y_m = float(corr[tuple(idx_m)])
        y_0 = float(corr[peak])
        y_p = float(corr[tuple(idx_p)])
        denom = y_m - 2.0 * y_0 + y_p
        delta = (0.5 * (y_m - y_p) / denom) if denom != 0.0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        s = p + delta
        if s > n // 2:  # wrap to a signed shift
            s -= n
        shifts.append(float(s))

    return shifts[0], shifts[1]


def apply_shift_to_array(
    arr: np.ndarray, dy: float, dx: float, *, order: int = 1, nodata: float | None = 0.0
) -> np.ndarray:
    """Resample one band by ``(dy, dx)`` sub-pixel, preserving dtype and nodata.

    Uses spline interpolation of the given ``order`` (1 = bilinear, the safe
    default for reflectance). Areas shifted in from outside are filled with
    ``nodata`` (0 by convention here). Integer inputs are rounded and clipped
    back into range.
    """
    src_dtype = arr.dtype
    cval = 0.0 if nodata is None else float(nodata)
    out = ndi_shift(arr.astype(np.float64), shift=(dy, dx), order=order, mode="constant", cval=cval)
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(src_dtype)


@dataclass
class CoregResult:
    band: str
    reference_scene: str
    moving_scene: str
    tile_px: int
    per_tile: list[dict] = field(default_factory=list)
    median_dy: float = 0.0
    median_dx: float = 0.0
    mad_dy: float = 0.0
    mad_dx: float = 0.0
    threshold_px: float = SUBPIXEL_THRESHOLD_PX
    correction_applied: bool = False

    @property
    def median_magnitude_px(self) -> float:
        return float(np.hypot(self.median_dy, self.median_dx))

    @property
    def correction_needed(self) -> bool:
        return self.median_magnitude_px > self.threshold_px

    def to_manifest_dict(self) -> dict:
        return {
            "method": "fft_phase_correlation_hann_parabolic_subpixel",
            "stable_band": self.band,
            "reference_scene": self.reference_scene,
            "moving_scene": self.moving_scene,
            "tile_px": self.tile_px,
            "n_tiles": len(self.per_tile),
            "per_tile_shifts_px": self.per_tile,
            "median_shift_px": {"dy": round(self.median_dy, 4), "dx": round(self.median_dx, 4)},
            "shift_mad_px": {"dy": round(self.mad_dy, 4), "dx": round(self.mad_dx, 4)},
            "median_magnitude_px": round(self.median_magnitude_px, 4),
            "subpixel_threshold_px": self.threshold_px,
            "correction_needed": self.correction_needed,
            "correction_applied": self.correction_applied,
            "note": (
                "shift = scipy.ndimage.shift(moving, (dy,dx)) to align moving onto reference; "
                "moving is the newer date."
            ),
        }


def _read_band(scene_dir: Path, band: str) -> tuple[np.ndarray, float | None]:
    with rasterio.open(Path(scene_dir) / f"{band}.tif") as ds:
        return ds.read(1), ds.nodata


def estimate_pair_shift(
    reference_scene_dir: Path,
    moving_scene_dir: Path,
    *,
    band: str = "B08",
    tile_px: int = 1024,
    margin_px: int = 512,
    grid: int = 3,
    max_nodata_frac: float = 0.02,
) -> CoregResult:
    """Measure the residual translation between two staged scenes on ``band``.

    Runs :func:`phase_correlation_shift` on a ``grid`` x ``grid`` set of
    ``tile_px`` windows spread across the scene interior (``margin_px`` kept
    clear of every edge), skipping any window with more than
    ``max_nodata_frac`` nodata in either date. Reports the per-tile shifts and
    their component-wise median + MAD.
    """
    ref_dir = Path(reference_scene_dir)
    mov_dir = Path(moving_scene_dir)
    ref, ref_nodata = _read_band(ref_dir, band)
    mov, _ = _read_band(mov_dir, band)
    if ref.shape != mov.shape:
        raise ValueError(
            f"{band}: reference {ref.shape} vs moving {mov.shape} - scenes not on the same grid"
        )
    h, w = ref.shape
    nd = 0.0 if ref_nodata is None else ref_nodata

    usable_h = h - 2 * margin_px - tile_px
    usable_w = w - 2 * margin_px - tile_px
    if usable_h < 0 or usable_w < 0:
        raise ValueError("scene too small for the requested tile_px / margin_px")
    row_starts = [int(margin_px + usable_h * i / max(grid - 1, 1)) for i in range(grid)]
    col_starts = [int(margin_px + usable_w * i / max(grid - 1, 1)) for i in range(grid)]

    per_tile: list[dict] = []
    for r0 in row_starts:
        for c0 in col_starts:
            a = ref[r0:r0 + tile_px, c0:c0 + tile_px]
            b = mov[r0:r0 + tile_px, c0:c0 + tile_px]
            if (a == nd).mean() > max_nodata_frac or (b == nd).mean() > max_nodata_frac:
                continue
            dy, dx = phase_correlation_shift(a, b)
            per_tile.append({"row": r0, "col": c0, "dy": round(dy, 4), "dx": round(dx, 4)})

    if not per_tile:
        raise RuntimeError("no usable (mostly nodata-free) tiles found for co-registration")

    dys = np.array([t["dy"] for t in per_tile])
    dxs = np.array([t["dx"] for t in per_tile])
    med_dy, med_dx = float(np.median(dys)), float(np.median(dxs))

    return CoregResult(
        band=band,
        reference_scene=ref_dir.name,
        moving_scene=mov_dir.name,
        tile_px=tile_px,
        per_tile=per_tile,
        median_dy=med_dy,
        median_dx=med_dx,
        mad_dy=float(np.median(np.abs(dys - med_dy))),
        mad_dx=float(np.median(np.abs(dxs - med_dx))),
    )

"""Relative radiometric normalization of a two-date scene pair.

Method: **pseudo-invariant feature (PIF) additive normalization**, per band,
with a **spatially-varying offset** over a large (82 x 82 km) scene.

1. Select pixels that are very likely unchanged between the two dates
   (:func:`select_pif_mask`): valid in every band at both dates, an SCL
   "good" class at both dates, not water, and low ``|NDVI|`` at both dates -
   i.e. bare soil / built-up, whose reflectance is stable between two March
   acquisitions, so any residual inter-date difference over them is
   atmospheric rather than a real surface change.
2. Fit an additive per-band correction ``reference_DN = subject_DN +
   offset``. Two flavours:

   * ``fit_linear_per_band(method="additive")`` - a single scene-wide
     ``offset`` = the robust (MAD-clipped) median of ``reference - subject``
     over the PIF pixels. Over genuine no-change targets the two dates sit
     within ~70-180 DN in green/red/NIR/SWIR, while blue carries ~700 DN of
     nearly-uncorrelated additive haze. (A free 2-parameter or RMA fit
     over-reacts to the PIF scatter and *amplifies* the difference on
     unchanged tiles; a constrained additive offset cannot. ``"rma"``,
     ``"ols"`` and ``"affine_clamped"`` remain available.)
   * :func:`fit_local_offset_surface` - the recommended choice here. The
     atmosphere is not uniform across 82 km, so a single global offset still
     leaves 100-200 DN on individual no-change tiles far from the scene mean.
     This fits ``offset_b(x, y)`` as a smooth surface: robust median of
     ``reference - subject`` over the PIF pixels of each ~5 km block, empty
     blocks filled from their nearest neighbour, lightly Gaussian-smoothed,
     bilinearly upsampled to full resolution.
3. Apply the per-band correction to the subject date
   (:meth:`RadiometricNormalization.apply`).

"Subject" is the date being adjusted; "reference" is the date left as-is.
Here the subject is 2019 (the hazier date - normalizing it onto 2024
removes haze rather than adding it) and the reference is 2024.

The per-band offset statistics + the local-offset block grid are recorded
in the provenance manifest so the normalization is reproducible from the
manifest alone. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from geoseek.ingest.quality import BAD_SCL_CLASSES

REFLECTANCE_SCALE = 10000.0  # S2 L2A: reflectance = DN / 10000 (additive offset 0 for both staged dates)
SCL_WATER = 6

# Bands the normalization is fit + applied on (all the change-detection bands).
NORM_BANDS = ("B04", "B03", "B02", "B08", "B11")

# Physical bound on the multiplicative term for haze/transmittance differences
# between two clear-sky acquisitions - used by method="affine_clamped".
GAIN_CLAMP = (0.8, 1.25)


def _safe_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    out = np.zeros_like(denom)
    np.divide(nir - red, denom, out=out, where=denom > 0)
    return out


def select_pif_mask(
    ref_bands: Mapping[str, np.ndarray],
    subj_bands: Mapping[str, np.ndarray],
    *,
    ref_scl: np.ndarray | None = None,
    subj_scl: np.ndarray | None = None,
    ndvi_abs_max: float = 0.25,
    min_nir_dn: float = 400.0,
    stable_fraction: float = 1.0,
) -> np.ndarray:
    """Boolean mask of pseudo-invariant (very-likely-unchanged) pixels.

    ``ref_bands`` / ``subj_bands`` must each provide at least B04, B03, B02,
    B08 as raw DN arrays of identical shape.

    * valid    : every provided band > 0 at both dates
    * SCL-good : if ``ref_scl`` / ``subj_scl`` are given, both dates' SCL must
      be outside :data:`geoseek.ingest.quality.BAD_SCL_CLASSES` and not water
    * bare/built, not vegetation : ``|NDVI| < ndvi_abs_max`` at both dates and
      NIR above ``min_nir_dn`` at both dates (cropland greens/senesces between
      dates, so it is excluded - only surfaces that don't change seasonally
      are kept)
    * stable   : if ``stable_fraction < 1.0``, keep only that fraction of
      what's left with the smallest brightness-matched multi-band residual
      between dates (off by default - the SCL + NDVI gate is usually enough
      and the fit's own MAD-clipping handles the rest)
    """
    bands = [b for b in ("B04", "B03", "B02", "B08") if b in ref_bands and b in subj_bands]
    if "B08" not in bands or "B04" not in bands:
        raise ValueError("select_pif_mask needs at least B04 and B08 at both dates")

    shape = ref_bands[bands[0]].shape
    valid = np.ones(shape, dtype=bool)
    for b in bands:
        valid &= (ref_bands[b] > 0) & (subj_bands[b] > 0)

    if ref_scl is not None and subj_scl is not None:
        bad = np.array(sorted(BAD_SCL_CLASSES))
        valid &= ~np.isin(ref_scl, bad) & ~np.isin(subj_scl, bad)
        valid &= (ref_scl != SCL_WATER) & (subj_scl != SCL_WATER)

    ndvi_ref = _safe_ndvi(ref_bands["B08"], ref_bands["B04"])
    ndvi_subj = _safe_ndvi(subj_bands["B08"], subj_bands["B04"])
    non_veg = (
        (np.abs(ndvi_ref) < ndvi_abs_max)
        & (np.abs(ndvi_subj) < ndvi_abs_max)
        & (ref_bands["B08"] > min_nir_dn)
        & (subj_bands["B08"] > min_nir_dn)
    )
    cand = valid & non_veg
    if not cand.any() or stable_fraction >= 1.0:
        return cand

    # Brightness-matched multi-band residual over the candidate pixels.
    resid = np.zeros(shape, dtype=np.float32)
    for b in bands:
        r = ref_bands[b].astype(np.float32)
        s = subj_bands[b].astype(np.float32)
        ratio = np.median(r[cand]) / max(np.median(s[cand]), 1e-6)
        resid[cand] += np.abs(r[cand] - ratio * s[cand]) / (r[cand] + 1.0)

    thresh = np.quantile(resid[cand], stable_fraction)
    return cand & (resid <= thresh)


def _mad(v: np.ndarray) -> float:
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def _round_or_none(x, ndigits: int = 4):
    """round(x, n), or None when x is missing / NaN (keeps the manifest JSON-clean)."""
    if x is None or x != x:
        return None
    return round(float(x), ndigits)


def _fit_line(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """gain, offset for ``y ~ gain*x + offset``. ``method`` in {additive, ols, rma, affine_clamped}.

    * ``additive``       - gain fixed at 1, offset = median(y - x). The right
      model when the inter-date difference is dominated by additive path
      radiance (haze) and the multiplicative term is ~1.
    * ``ols``            - ordinary least squares. Biased toward gain 0 by
      noise in x.
    * ``rma``            - reduced major axis (geometric-mean) regression;
      symmetric in x and y. Over-reacts when the PIF scatter is broad.
    * ``affine_clamped`` - OLS gain clamped to :data:`GAIN_CLAMP`, then offset
      re-solved as median(y - gain*x).
    """
    if x.size < 2:
        return 1.0, 0.0
    if method == "additive":
        return 1.0, float(np.median(y - x))
    sx = x.std()
    if method == "ols" or sx == 0:
        gain, offset = np.polyfit(x, y, 1)
        return float(gain), float(offset)
    if method == "rma":
        sign = 1.0 if np.cov(x, y)[0, 1] >= 0 else -1.0
        gain = sign * y.std() / sx
        return float(gain), float(y.mean() - gain * x.mean())
    if method == "affine_clamped":
        gain, _ = np.polyfit(x, y, 1)
        gain = float(np.clip(gain, *GAIN_CLAMP))
        return gain, float(np.median(y - gain * x))
    raise ValueError(f"unknown fit method {method!r}")


def fit_linear_per_band(
    ref_bands: Mapping[str, np.ndarray],
    subj_bands: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    bands: tuple[str, ...] = NORM_BANDS,
    method: str = "additive",
    n_sigma: float = 2.5,
    n_iter: int = 3,
) -> dict[str, dict[str, float]]:
    """Fit ``reference_DN = gain * subject_DN + offset`` per band on the masked pixels.

    See :func:`_fit_line` for the models. ``n_iter`` rounds of ``n_sigma``
    residual clipping shrug off any change pixels that slipped through PIF
    selection. Returns ``{band: {gain, offset, r2, corr, n, rmse_dn, method}}``
    with gain/offset in DN space (``corr`` is Pearson r between the dates over
    the kept pixels - low for a band where the inter-date difference is
    mostly uncorrelated haze, e.g. blue).
    """
    out: dict[str, dict[str, float]] = {}
    base = np.asarray(mask, dtype=bool)
    for b in bands:
        if b not in ref_bands or b not in subj_bands:
            continue
        x_all = subj_bands[b].astype(np.float64)[base]
        y_all = ref_bands[b].astype(np.float64)[base]
        keep = np.ones(x_all.shape, dtype=bool)
        gain, offset = 1.0, 0.0
        for _ in range(n_iter + 1):
            x, y = x_all[keep], y_all[keep]
            if x.size < 100:
                break
            gain, offset = _fit_line(x, y, method)
            resid = y_all - (gain * x_all + offset)
            sigma = resid[keep].std()
            if sigma <= 1e-9:
                break
            new_keep = np.abs(resid - np.median(resid[keep])) <= n_sigma * sigma
            if new_keep.sum() < max(100, 0.5 * keep.sum()):
                break
            keep = new_keep
        x, y = x_all[keep], y_all[keep]
        if x.size == 0:
            x, y = x_all, y_all
        pred = gain * x + offset
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        out[b] = {
            "gain": float(gain),
            "offset": float(offset),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "corr": corr,
            "n": int(x.size),
            "rmse_dn": float(np.sqrt(ss_res / x.size)) if x.size else float("nan"),
            "method": method,
        }
    return out


def iterate_pif_normalization(
    ref_bands: Mapping[str, np.ndarray],
    subj_bands: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    bands: tuple[str, ...] = NORM_BANDS,
    method: str = "additive",
    n_iter: int = 2,
    n_mad: float = 4.0,
) -> tuple[dict[str, dict[str, float]], np.ndarray]:
    """Fit per band, then drop only the *gross* cross-band outliers and refit.

    After a fit, a PIF pixel is dropped if its residual exceeds ``n_mad`` times
    the per-band residual MAD in *any* band - a loose gate (default 4 MAD) that
    removes pixels which clearly changed without letting the no-change set
    collapse onto a self-consistent sub-population (which is what a tight,
    many-round refinement does, especially with a free-gain fit). Returns the
    final per-band coefficients and the refined mask.
    """
    m = np.asarray(mask, dtype=bool).copy()
    per_band = fit_linear_per_band(ref_bands, subj_bands, m, bands=bands, method=method)
    for _ in range(n_iter):
        idx = m
        within = np.ones(int(m.sum()), dtype=bool)
        for b in bands:
            if b not in per_band:
                continue
            c = per_band[b]
            x = subj_bands[b].astype(np.float64)[idx]
            y = ref_bands[b].astype(np.float64)[idx]
            r = y - (c["gain"] * x + c["offset"])
            scale = max(_mad(r), max(c["rmse_dn"], 1.0))
            within &= np.abs(r - np.median(r)) <= n_mad * scale
        new_m = np.zeros_like(m)
        new_m[m] = within
        if new_m.sum() == m.sum() or new_m.sum() < max(5000, 0.5 * m.sum()):
            break
        m = new_m
        per_band = fit_linear_per_band(ref_bands, subj_bands, m, bands=bands, method=method)
    return per_band, m


# --------------------------------------------------------------------------
# Spatially-varying (local) additive offset
# --------------------------------------------------------------------------


def fit_local_offset_surface(
    ref_band: np.ndarray,
    subj_band: np.ndarray,
    pif_mask: np.ndarray,
    *,
    block_px: int = 512,
    min_pif_per_block: int = 120,
    smooth_sigma_blocks: float = 0.8,
    clamp_block_support: bool = False,
    block_deadband_mad_frac: float = 0.0,
    unreliable_block_shrink_k: float = 0.0,
) -> tuple[np.ndarray, float, dict]:
    """Robust ``reference - subject`` offset (DN) as a coarse, smooth block grid.

    The scene is diced into ``block_px`` squares (~5 km at 10 m). Each block's
    offset is the median of ``reference - subject`` over the PIF pixels inside
    it; blocks with fewer than ``min_pif_per_block`` PIFs are filled from their
    nearest populated neighbour; the grid is then lightly Gaussian-smoothed
    (``smooth_sigma_blocks``). Returns ``(grid, global_offset, stats)`` - the
    small ``grid`` (n_by x n_bx float32, DN), the scene-wide robust offset (the
    fallback / summary), and a stats dict.

    Safe / non-overshooting options (all off by default; ``fit_local_normalization``
    turns them on):

    * ``unreliable_block_shrink_k`` - when > 0, a block whose *internal* spatial
      scatter (within-block MAD of ``reference - subject`` over its PIFs) is
      large relative to the offset it proposes is under-trusted: its offset is
      multiplied by ``|d| / (|d| + k * within_block_MAD)`` -> when the block
      cannot resolve the correction reliably, APPLY LESS, not more.
    * ``block_deadband_mad_frac`` - when > 0, a block whose own ``|median diff|``
      is below ``frac * within-block MAD`` (dates already agree within local
      noise) gets offset 0 - no correction where none is supported.
    * ``clamp_block_support`` - the final per-block offset is clamped to the
      closed interval between 0 and that block's *own* robust ``reference -
      subject`` median. It can never flip sign or exceed the local evidence, so
      the corrected subject value stays between the two real observations: no
      opposite-sign error, no phantom near-zero/negative reflectance. Blocks
      with no PIF support are filled with 0 (leave them alone) rather than
      borrowing a neighbouring block's (possibly large) offset.
    """
    ref_band = np.asarray(ref_band)
    subj_band = np.asarray(subj_band)
    pif_mask = np.asarray(pif_mask, dtype=bool)
    h, w = ref_band.shape
    n_by = -(-h // block_px)
    n_bx = -(-w // block_px)

    diff = ref_band.astype(np.float64) - subj_band.astype(np.float64)
    if pif_mask.any():
        dpif = diff[pif_mask]
        global_med = float(np.median(dpif))
        pixel_mad = float(np.median(np.abs(dpif - global_med)) * 1.4826)
    else:
        global_med = 0.0
        pixel_mad = 0.0

    grid = np.full((n_by, n_bx), np.nan, dtype=np.float64)
    block_mad = np.full((n_by, n_bx), np.nan, dtype=np.float64)
    counts = np.zeros((n_by, n_bx), dtype=np.int64)
    for by in range(n_by):
        ys = slice(by * block_px, min((by + 1) * block_px, h))
        for bx in range(n_bx):
            xs = slice(bx * block_px, min((bx + 1) * block_px, w))
            m = pif_mask[ys, xs]
            k = int(m.sum())
            counts[by, bx] = k
            if k >= min_pif_per_block:
                bvals = diff[ys, xs][m]
                bmed = float(np.median(bvals))
                grid[by, bx] = bmed
                block_mad[by, bx] = float(np.median(np.abs(bvals - bmed)) * 1.4826)

    populated = np.isfinite(grid)
    n_filled = int((~populated).sum())
    raw_grid = grid.copy()  # each block's own robust reference-subject median (NaN where unsupported)
    mad_f = np.where(np.isfinite(block_mad), block_mad, 0.0)

    # Under-correct by the block's own uncertainty: keep only the part of the
    # offset that the block resolves *confidently*. safe = sign(d) * max(0,
    # |d| - k * within-block MAD). Where the block cannot resolve its correction
    # (scatter >= |d|/k, i.e. sub-km blue haze the ~2.5 km block can't see) this
    # collapses to 0 - APPLY LESS, never more. This is a one-sided confidence
    # bound on the true offset, so it can never overshoot into an opposite-sign
    # residual.
    n_shrunk = 0
    if unreliable_block_shrink_k > 0 and populated.any():
        safe_mag = np.maximum(0.0, np.abs(grid) - unreliable_block_shrink_k * mad_f)
        shrunk = np.sign(grid) * safe_mag
        n_shrunk = int((populated & (np.abs(shrunk) < np.abs(grid) - 1.0)).sum())
        grid = np.where(populated, shrunk, grid)

    # deadband: no correction at all where the two dates already agree within local noise
    n_deadbanded = 0
    if block_deadband_mad_frac > 0 and populated.any():
        dead = populated & (np.abs(raw_grid) < block_deadband_mad_frac * mad_f)
        n_deadbanded = int(dead.sum())
        grid[dead] = 0.0

    if not populated.any():
        grid[:] = 0.0 if clamp_block_support else global_med
    elif n_filled:
        if clamp_block_support:
            grid = np.where(np.isfinite(grid), grid, 0.0)  # under-correct, don't borrow a neighbour's offset
        else:
            idx = distance_transform_edt(~populated, return_distances=False, return_indices=True)
            grid = grid[tuple(idx)]

    support = grid.copy()  # the confidently-resolved, under-corrected offset per block (pre-smoothing)

    if smooth_sigma_blocks > 0 and min(grid.shape) > 1:
        grid = gaussian_filter(grid, smooth_sigma_blocks, mode="nearest")

    # smoothing may not re-inflate a block past its own under-corrected value, nor
    # flip its sign: clamp each block into [0, support] (or [support, 0]).
    if clamp_block_support:
        grid = np.clip(grid, np.minimum(support, 0.0), np.maximum(support, 0.0))

    grid = grid.astype(np.float32)
    stats = {
        "block_px": block_px,
        "grid_shape": [n_by, n_bx],
        "blocks_populated": int(populated.sum()),
        "blocks_filled_from_neighbour": n_filled,
        "min_pif_per_block": min_pif_per_block,
        "smooth_sigma_blocks": smooth_sigma_blocks,
        "global_offset_dn": round(global_med, 3),
        "pif_pixel_mad_dn": round(pixel_mad, 3),
        "clamped_to_block_support": bool(clamp_block_support),
        "block_deadband_mad_frac": block_deadband_mad_frac,
        "unreliable_block_shrink_k": unreliable_block_shrink_k,
        "blocks_deadbanded": n_deadbanded,
        "blocks_shrunk": n_shrunk,
        "fill_mode": "zero_undercorrect" if clamp_block_support else "nearest_neighbour",
        "offset_dn_mean": round(float(grid.mean()), 3),
        "offset_dn_std": round(float(grid.std()), 3),
        "offset_dn_min": round(float(grid.min()), 3),
        "offset_dn_max": round(float(grid.max()), 3),
        "zeroed_insignificant": False,
    }
    return grid, global_med, stats


def expand_offset_grid(
    grid: np.ndarray, block_px: int, y0: int, x0: int, h: int, w: int, *, row_chunk: int = 2048
) -> np.ndarray:
    """Bilinearly upsample a block ``grid`` to the pixel region ``[y0:y0+h, x0:x0+w]`` (DN, float32).

    Done in horizontal strips of ``row_chunk`` rows so a full-scene expansion
    never materialises a giant coordinate array.
    """
    grid = np.asarray(grid, dtype=np.float32)
    # block (i, j) is centred at pixel ((i + 0.5) * block_px, (j + 0.5) * block_px)
    gx = ((np.arange(x0, x0 + w, dtype=np.float32) + 0.5) / block_px) - 0.5
    out = np.empty((h, w), dtype=np.float32)
    for r0 in range(0, h, row_chunk):
        r1 = min(r0 + row_chunk, h)
        gy = ((np.arange(y0 + r0, y0 + r1, dtype=np.float32) + 0.5) / block_px) - 0.5
        cy, cx = np.meshgrid(gy, gx, indexing="ij")
        out[r0:r1] = map_coordinates(grid, [cy, cx], order=1, mode="nearest").astype(np.float32)
    return out


def fit_local_normalization(
    ref_bands: Mapping[str, np.ndarray],
    subj_bands: Mapping[str, np.ndarray],
    pif_mask: np.ndarray,
    *,
    bands: tuple[str, ...] = NORM_BANDS,
    block_px: int = 512,
    min_pif_per_block: int = 120,
    smooth_sigma_blocks: float = 0.8,
    zero_if_insignificant: bool = True,
    zero_mad_frac: float = 0.5,
    zero_min_corr: float = 0.6,
    clamp_block_support: bool = True,
    block_deadband_mad_frac: float = 0.5,
    unreliable_block_shrink_k: float = 2.0,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Per-band spatially-varying additive normalization.

    Returns ``(per_band, offset_grids)``: ``per_band[b]`` is the scalar
    summary (``gain`` 1.0, ``offset`` = scene-wide robust offset, plus the
    surface stats and inter-date correlation); ``offset_grids[b]`` is the
    small block grid used by :meth:`RadiometricNormalization.apply`.

    ``zero_if_insignificant``: for a well-correlated band
    (``corr >= zero_min_corr``) whose scene-wide offset is small relative to
    the per-pixel PIF scatter (``|offset| < zero_mad_frac * pixel_MAD``), the
    two dates already agree within the noise - the offset surface is forced to
    zero so the normalization does not *add* a spurious correction that would
    degrade an already-matched band.
    """
    per_band: dict[str, dict[str, float]] = {}
    grids: dict[str, np.ndarray] = {}
    m = np.asarray(pif_mask, dtype=bool)
    for b in bands:
        if b not in ref_bands or b not in subj_bands:
            continue
        grid, global_off, stats = fit_local_offset_surface(
            ref_bands[b], subj_bands[b], m,
            block_px=block_px, min_pif_per_block=min_pif_per_block,
            smooth_sigma_blocks=smooth_sigma_blocks,
            clamp_block_support=clamp_block_support,
            block_deadband_mad_frac=block_deadband_mad_frac,
            unreliable_block_shrink_k=unreliable_block_shrink_k,
        )
        x = subj_bands[b].astype(np.float64)[m]
        y = ref_bands[b].astype(np.float64)[m]
        corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        if (
            zero_if_insignificant
            and corr == corr and corr >= zero_min_corr
            and stats["pif_pixel_mad_dn"] > 0
            and abs(global_off) < zero_mad_frac * stats["pif_pixel_mad_dn"]
        ):
            grid = np.zeros_like(grid)
            global_off = 0.0
            stats["zeroed_insignificant"] = True
            for k in ("offset_dn_mean", "offset_dn_std", "offset_dn_min", "offset_dn_max"):
                stats[k] = 0.0
        grids[b] = grid
        # residual RMSE after applying the local surface, over the PIF pixels
        local = expand_offset_grid(grid, stats["block_px"], 0, 0, *ref_bands[b].shape)[m]
        resid = y - (x + local)
        per_band[b] = {
            "gain": 1.0,
            "offset": float(global_off),
            "corr": corr,
            "n": int(m.sum()),
            "rmse_dn": float(np.sqrt(np.mean(resid ** 2))),
            "method": "additive_local",
            "surface": stats,
        }
    return per_band, grids


@dataclass
class RadiometricNormalization:
    """Per-band additive DN->DN normalization of the subject date onto the reference date.

    ``offset_grids`` (optional) holds a small per-band block grid for a
    spatially-varying offset; when present, :meth:`apply` interpolates it over
    the region being normalized (pass ``y0``/``x0`` for a tile that is not at
    the scene origin). Without it, the scalar ``per_band[b]["offset"]`` is used.
    """

    reference_scene: str
    subject_scene: str
    reference_date: str
    subject_date: str
    per_band: dict[str, dict[str, float]]
    method: str = "pif_additive_offset_mad_clipped"
    n_pif: int = 0
    pif_selection: dict | None = None
    offset_grids: dict[str, np.ndarray] | None = None
    grid_block_px: int = 512

    def _offset(self, band: str, shape: tuple[int, int], y0: int, x0: int):
        if self.offset_grids and band in self.offset_grids:
            return expand_offset_grid(self.offset_grids[band], self.grid_block_px, y0, x0, *shape)
        return self.per_band[band]["offset"]

    def apply(self, dn: np.ndarray, band: str, *, y0: int = 0, x0: int = 0) -> np.ndarray:
        """Map subject-date DN for ``band`` into reference-date DN space.

        ``y0``/``x0`` are this array's pixel offset within the full scene (only
        relevant when a local ``offset_grids`` is set). nodata (0) stays 0;
        results are clipped at 0 and returned in the input dtype (rounded for
        integer input).
        """
        if band not in self.per_band:
            raise KeyError(f"no normalization coefficients for band {band!r}")
        c = self.per_band[band]
        src_dtype = dn.dtype
        x = dn.astype(np.float64)
        out = c["gain"] * x + self._offset(band, dn.shape, y0, x0)
        out = np.clip(out, 0.0, None)
        out[dn == 0] = 0.0
        if np.issubdtype(src_dtype, np.integer):
            info = np.iinfo(src_dtype)
            out = np.clip(np.round(out), info.min, info.max)
        return out.astype(src_dtype)

    def apply_reflectance(self, dn: np.ndarray, band: str, *, y0: int = 0, x0: int = 0) -> np.ndarray:
        """Convenience: normalized subject DN -> surface reflectance (float32, NaN at nodata)."""
        norm = self.apply(dn, band, y0=y0, x0=x0).astype(np.float32)
        refl = norm / REFLECTANCE_SCALE
        refl[dn == 0] = np.nan
        return refl

    def to_manifest_dict(self) -> dict:
        return {
            "method": self.method,
            "direction": f"{self.subject_date} (subject) -> {self.reference_date} (reference)",
            "reference_scene": self.reference_scene,
            "subject_scene": self.subject_scene,
            "reference_date": self.reference_date,
            "subject_date": self.subject_date,
            "reflectance_scale_dn_per_unit": REFLECTANCE_SCALE,
            "boa_add_offset_dn": 0.0,
            "n_pseudo_invariant_pixels": self.n_pif,
            "pif_selection": self.pif_selection or {},
            "spatially_varying": bool(self.offset_grids),
            "grid_block_px": self.grid_block_px,
            "per_band_dn": {
                b: {
                    "gain": round(c["gain"], 6),
                    "offset_dn": round(c["offset"], 3),
                    "fit_method": c.get("method", "additive"),
                    "inter_date_corr": _round_or_none(c.get("corr")),
                    "r2": _round_or_none(c.get("r2"), 5),
                    "n": c["n"],
                    "rmse_dn": round(c["rmse_dn"], 2),
                    "offset_surface": c.get("surface"),
                    "offset_grid_dn": (
                        [[round(float(v), 2) for v in row] for row in self.offset_grids[b]]
                        if self.offset_grids and b in self.offset_grids else None
                    ),
                }
                for b, c in self.per_band.items()
            },
        }

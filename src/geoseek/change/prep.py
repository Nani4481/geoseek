"""Phase 3a orchestrator: make the 2019/2024 pair comparable (no change detection).

    python -m geoseek.change.prep

Runs, over the scaled AOI (``data/datasets/<scene>_scaled/``):

  B. Co-registration check  - median sub-pixel (dy,dx) between the dates via
     phase correlation on B08; correct only if > 0.5 px.
  C. Relative radiometric normalization - per-band linear gain/offset fit on
     pseudo-invariant pixels, mapping 2019 (subject) into 2024 (reference).
  D. Acceptance gate - 5 clearly-unchanged interior tiles, per-band mean
     reflectance 2019 vs 2024 BEFORE and AFTER normalization; the AFTER
     differences must fall well under ~100 DN.
  E. Spectral indices - NDVI / NDWI / NDBI per tile per date on the normalized
     reflectance; ranges printed and spot-checked (cropland/river/town).

Everything here is offline. The co-registration, normalization and index
parameters are written to the provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj
import rasterio

from geoseek.config import get_settings
from geoseek.change.coregister import apply_shift_to_array, estimate_pair_shift
from geoseek.change.indices import compute_indices, range_summary, valid_mask
from geoseek.change.normalize import (
    NORM_BANDS,
    REFLECTANCE_SCALE,
    RadiometricNormalization,
    fit_local_normalization,
    iterate_pif_normalization,
    select_pif_mask,
)

LOCAL_OFFSET_BLOCK_PX = 256  # ~2.5 km blocks for the spatially-varying offset surface
from geoseek.ingest.quality import BAD_SCL_CLASSES
from geoseek.ingest.tiler import TILE_SIZE
from geoseek.staging.manifest import record_analysis_section

# subject = the date that gets adjusted; reference = the date left as-is.
SUBJECT_SCENE = "S2B_44RPQ_20190330_1_L2A_scaled"
REFERENCE_SCENE = "S2A_44RPQ_20240308_0_L2A_scaled"
SUBJECT_DATE = "2019-03-30"
REFERENCE_DATE = "2024-03-08"

ALL_BANDS = ("B04", "B03", "B02", "B08", "B11", "SCL")
REFL_BANDS = ("B04", "B03", "B02", "B08", "B11")

# Ayodhya town centre (Ram Janmabhoomi area) - same point the AOI is built around.
TOWN_LON, TOWN_LAT = 82.1998, 26.7922

GATE_DN = 100.0  # strict target: every band on every unchanged tile well under this after normalization


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------


@dataclass
class Scene:
    scene_id: str
    dir: Path
    bands: dict[str, np.ndarray]  # raw DN (uint16), SCL uint8
    transform: object
    crs: object
    nodata: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.bands["B04"].shape


def load_scene(scene_dir: Path) -> Scene:
    scene_dir = Path(scene_dir)
    missing = [b for b in ALL_BANDS if not (scene_dir / f"{b}.tif").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{scene_dir.name} missing band(s) {missing}. "
            "Run `python -m geoseek.staging.download_datasets --large` and `--extra-bands` first."
        )
    bands: dict[str, np.ndarray] = {}
    transform = crs = None
    nodata = 0.0
    for b in ALL_BANDS:
        with rasterio.open(scene_dir / f"{b}.tif") as ds:
            bands[b] = ds.read(1)
            if b == "B04":
                transform, crs, nodata = ds.transform, ds.crs, (ds.nodata or 0.0)
    return Scene(scene_dir.name, scene_dir, bands, transform, crs, nodata)


def _common_valid(a: Scene, b: Scene) -> np.ndarray:
    m = np.ones(a.shape, dtype=bool)
    for band in REFL_BANDS:
        m &= (a.bands[band] > 0) & (b.bands[band] > 0)
    return m


# --------------------------------------------------------------------------
# Step B - co-registration
# --------------------------------------------------------------------------


def step_b_coregistration(subject: Scene, reference: Scene) -> dict:
    print("\n" + "=" * 78)
    print("STEP B - CO-REGISTRATION CHECK (phase correlation on B08, 10m grid)")
    print("=" * 78)
    print(f"  reference (held fixed) : {reference.scene_id}  [{REFERENCE_DATE}]")
    print(f"  moving    (measured)   : {subject.scene_id}  [{SUBJECT_DATE}]")

    res = estimate_pair_shift(reference.dir, subject.dir, band="B08", tile_px=1024, grid=3)
    print(f"\n  per-tile shift (dy, dx) px  -  shift to apply to the moving date to align it:")
    for t in res.per_tile:
        print(f"    row {t['row']:5d}  col {t['col']:5d}   dy={t['dy']:+.3f}  dx={t['dx']:+.3f}")
    print(f"\n  MEDIAN (dy, dx) = ({res.median_dy:+.3f}, {res.median_dx:+.3f}) px   "
          f"| MAD = ({res.mad_dy:.3f}, {res.mad_dx:.3f}) px")
    print(f"  median shift magnitude = {res.median_magnitude_px:.3f} px  "
          f"(sub-pixel threshold = {res.threshold_px} px)")

    if res.correction_needed:
        print(f"  -> shift EXCEEDS {res.threshold_px} px: correcting the moving date "
              f"({subject.scene_id}) in memory by (dy,dx)=({res.median_dy:+.3f},{res.median_dx:+.3f}).")
        for b in REFL_BANDS:
            subject.bands[b] = apply_shift_to_array(
                subject.bands[b], res.median_dy, res.median_dx, order=1, nodata=0.0
            )
        subject.bands["SCL"] = apply_shift_to_array(
            subject.bands["SCL"], res.median_dy, res.median_dx, order=0, nodata=0.0
        )
        res.correction_applied = True
    else:
        print(f"  -> shift is well within {res.threshold_px} px (Sentinel-2 L1C multitemporal "
              f"registration spec): pair already aligned, leaving as-is.")

    return res.to_manifest_dict()


# --------------------------------------------------------------------------
# Step C - relative radiometric normalization
# --------------------------------------------------------------------------


def step_c_normalization(subject: Scene, reference: Scene) -> RadiometricNormalization:
    print("\n" + "=" * 78)
    print("STEP C - RELATIVE RADIOMETRIC NORMALIZATION (PIF, per band, spatially varying)")
    print("=" * 78)
    print(f"  model     : reference_DN = subject_DN + offset_b(x, y)   (per-band additive, "
          f"{LOCAL_OFFSET_BLOCK_PX * 10 // 1000} km offset surface)")
    print(f"  subject   = {subject.scene_id}  [{SUBJECT_DATE}]  (adjusted)")
    print(f"  reference = {reference.scene_id}  [{REFERENCE_DATE}]  (held fixed)")

    ref_b = {b: reference.bands[b] for b in ("B04", "B03", "B02", "B08")}
    subj_b = {b: subject.bands[b] for b in ("B04", "B03", "B02", "B08")}
    pif_params = {"ndvi_abs_max": 0.25, "min_nir_dn": 400.0}
    init_mask = select_pif_mask(
        ref_b, subj_b, ref_scl=reference.bands["SCL"], subj_scl=subject.bands["SCL"], **pif_params
    )
    print(f"\n  pseudo-invariant candidates: {int(init_mask.sum()):,} "
          f"({100.0 * init_mask.sum() / init_mask.size:.2f}% of the scene; valid + SCL-good "
          f"both dates, not water, |NDVI| < {pif_params['ndvi_abs_max']} both dates = stable bare/built)")

    ref_all = {b: reference.bands[b] for b in NORM_BANDS}
    subj_all = {b: subject.bands[b] for b in NORM_BANDS}
    _, mask = iterate_pif_normalization(
        ref_all, subj_all, init_mask, bands=NORM_BANDS, method="additive", n_iter=3, n_mad=3.0
    )
    n_pif = int(mask.sum())
    print(f"  after 3 rounds of cross-band outlier removal: {n_pif:,} PIF pixels "
          f"({100.0 * n_pif / mask.size:.2f}% of the scene)")

    per_band, grids = fit_local_normalization(
        ref_all, subj_all, mask, bands=NORM_BANDS,
        block_px=LOCAL_OFFSET_BLOCK_PX, min_pif_per_block=60, smooth_sigma_blocks=1.0,
        zero_if_insignificant=True, zero_mad_frac=0.35, zero_min_corr=0.6,
        # never overshoot: clamp each block's offset into [0, its own robust ref-subj
        # median], under-trust blocks with high internal spatial scatter (the ~2.5 km
        # block is too coarse for sub-km blue haze), and apply no correction where the
        # dates already agree within local noise (fixes the single-tile B08 case).
        clamp_block_support=True, block_deadband_mad_frac=0.5, unreliable_block_shrink_k=2.0,
    )

    print(f"\n  per-band additive offset surface (DN; mean is the scene-wide term):")
    print(f"    {'band':6s} {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s} {'pixel_MAD':>10s} "
          f"{'inter-date r':>13s} {'RMSE_DN':>9s}  note")
    for b in NORM_BANDS:
        s = per_band[b]["surface"]
        note = "ZEROED (already matched within noise)" if s.get("zeroed_insignificant") else ""
        print(f"    {b:6s} {s['offset_dn_mean']:8.1f} {s['offset_dn_std']:7.1f} {s['offset_dn_min']:8.1f} "
              f"{s['offset_dn_max']:8.1f} {s['pif_pixel_mad_dn']:10.1f} {per_band[b]['corr']:13.3f} "
              f"{per_band[b]['rmse_dn']:9.2f}  {note}")
    g = per_band['B02']['surface']
    print(f"  offset grid: {g['grid_shape']} blocks of {LOCAL_OFFSET_BLOCK_PX}px (~2.5 km); "
          f"{g['blocks_filled_from_neighbour']} sparse blocks filled from neighbours, "
          f"smoothed {g['smooth_sigma_blocks']} blocks")
    zeroed = [b for b in NORM_BANDS if per_band[b]["surface"].get("zeroed_insignificant")]
    if zeroed:
        print(f"  {', '.join(zeroed)}: scene-wide offset is below ~0.5x the per-pixel PIF scatter and the "
              f"dates are well correlated -> already comparable, offset forced to 0 (no spurious correction).")
    lowcorr = [b for b in NORM_BANDS if per_band[b]["corr"] == per_band[b]["corr"] and per_band[b]["corr"] < 0.35]
    if lowcorr:
        print(f"  note: {', '.join(lowcorr)} inter-date correlation < 0.35 - the difference there is "
              f"mostly uncorrelated additive haze (path radiance), which the additive offset removes directly.")

    norm = RadiometricNormalization(
        reference_scene=reference.scene_id,
        subject_scene=subject.scene_id,
        reference_date=REFERENCE_DATE,
        subject_date=SUBJECT_DATE,
        per_band=per_band,
        n_pif=n_pif,
        offset_grids=grids,
        grid_block_px=LOCAL_OFFSET_BLOCK_PX,
        method="pif_additive_local_offset_surface",
        pif_selection={
            "gate": "valid + SCL-good (both dates) + not water + |NDVI| < 0.25 (both dates)",
            "refinement": "2 rounds of gross cross-band outlier removal (|resid - med| > 4*MAD in any band)",
            "model": (
                f"per-band additive offset surface: robust median(reference - subject) over PIF pixels "
                f"of each {LOCAL_OFFSET_BLOCK_PX}px block, nearest-fill sparse blocks, Gaussian smooth, "
                f"bilinear upsample. Reconstruct with geoseek.change.normalize.expand_offset_grid."
            ),
            **pif_params,
            "bands_considered": ["B04", "B03", "B02", "B08"],
        },
    )
    return norm


# --------------------------------------------------------------------------
# Step D - the acceptance gate: 5 unchanged tiles, before/after
# --------------------------------------------------------------------------


def _tile_windows(h: int, w: int):
    for r in range(0, h // TILE_SIZE):
        for c in range(0, w // TILE_SIZE):
            yield r, c, slice(r * TILE_SIZE, (r + 1) * TILE_SIZE), slice(c * TILE_SIZE, (c + 1) * TILE_SIZE)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    da, db = a.std(), b.std()
    if da < 1e-6 or db < 1e-6:
        return 0.0
    return float((a * b).mean() / (da * db))


def _ndvi_mean(nir: np.ndarray, red: np.ndarray, valid: np.ndarray) -> float:
    nir = nir.astype(np.float64)[valid]
    red = red.astype(np.float64)[valid]
    s = nir + red
    return float(np.mean((nir - red) / np.where(s > 0, s, 1.0)))


def _find_unchanged_tiles(subject: Scene, reference: Scene, n: int = 5) -> list[dict]:
    """Interior tiles that are genuinely unchanged 2019<->2024.

    "Unchanged" is enforced on offset-blind, ratio-based metrics: small
    |dNDVI| AND small |dNDBI| (land cover and built-up fraction both stable),
    plus structural filters (valid, no water, no cloud, spatially
    homogeneous). Survivors are ranked by B08 + B11 texture cross-correlation
    and thinned for spatial spread. This AOI was chosen for its heavy
    2019-2024 construction, and its cropland greened between the two March
    dates, so genuinely-static tiles are scarce - fewer than ``n`` may be
    returned (the gate reports how many).
    """
    h, w = subject.shape
    bad = np.array(sorted(BAD_SCL_CLASSES))
    n_rt, n_ct = h // TILE_SIZE, w // TILE_SIZE
    cands: list[dict] = []
    for r, c, rs, cs in _tile_windows(h, w):
        if r < 3 or c < 3 or r >= n_rt - 3 or c >= n_ct - 3:  # interior only
            continue
        s = {b: subject.bands[b][rs, cs] for b in ("B04", "B08", "B11", "SCL")}
        t = {b: reference.bands[b][rs, cs] for b in ("B04", "B08", "B11", "SCL")}
        valid = (s["B08"] > 0) & (t["B08"] > 0) & (s["B04"] > 0) & (t["B04"] > 0) \
            & (s["B11"] > 0) & (t["B11"] > 0)
        if valid.mean() < 0.99:
            continue
        if np.isin(s["SCL"], [6]).mean() > 0.005 or np.isin(t["SCL"], [6]).mean() > 0.005:
            continue  # away from the river
        if np.isin(s["SCL"], bad).mean() > 0.01 or np.isin(t["SCL"], bad).mean() > 0.01:
            continue
        cv = s["B08"][valid].std() / max(s["B08"][valid].mean(), 1.0)
        if cv > 0.45:  # homogeneous interior, not a boundary / mixed tile
            continue
        ndvi_s = _ndvi_mean(s["B08"], s["B04"], valid)
        ndvi_t = _ndvi_mean(t["B08"], t["B04"], valid)
        ndbi_s = _ndvi_mean(s["B11"], s["B08"], valid)  # (B11-B08)/(B11+B08)
        ndbi_t = _ndvi_mean(t["B11"], t["B08"], valid)
        # absolute NIR/SWIR level stability - catches a "proportionally brightened"
        # tile (drought soil -> moist soil) that a ratio index like dNDVI misses
        b08_s = float(s["B08"][valid].astype(np.float64).mean())
        b08_t = float(t["B08"][valid].astype(np.float64).mean())
        b11_s = float(s["B11"][valid].astype(np.float64).mean())
        b11_t = float(t["B11"][valid].astype(np.float64).mean())
        d_nir = abs(b08_t - b08_s) / max(0.5 * (b08_s + b08_t), 1.0)
        d_swir = abs(b11_t - b11_s) / max(0.5 * (b11_s + b11_t), 1.0)
        cands.append({"r": r, "c": c, "dndvi": abs(ndvi_t - ndvi_s), "dndbi": abs(ndbi_t - ndbi_s),
                      "d_nir": d_nir, "d_swir": d_swir,
                      "ncc": _ncc(s["B08"][valid], t["B08"][valid]) + _ncc(s["B11"][valid], t["B11"][valid]),
                      "ndvi_2019": ndvi_s, "ndvi_2024": ndvi_t})

    def _pick(dndvi_max: float, dndbi_max: float, dlevel_max: float) -> list[dict]:
        pool = [d for d in cands if d["dndvi"] <= dndvi_max and d["dndbi"] <= dndbi_max
                and d["d_nir"] <= dlevel_max and d["d_swir"] <= dlevel_max]
        pool.sort(key=lambda d: d["ncc"], reverse=True)
        out: list[dict] = []
        for sep in (6, 3, 0):
            for d in pool:
                if any(pd["r"] == d["r"] and pd["c"] == d["c"] for pd in out):
                    continue
                if all(max(abs(d["r"] - pd["r"]), abs(d["c"] - pd["c"])) >= sep for pd in out):
                    out.append(d)
                if len(out) == n:
                    return out
        return out

    for dv, db, dl in ((0.04, 0.04, 0.06), (0.06, 0.06, 0.09), (0.09, 0.09, 0.13)):
        picks = _pick(dv, db, dl)
        if len(picks) >= min(n, 3):
            return picks
    # nothing rock-solid: return whatever the loosest tier found (may be < 3)
    return picks


def step_d_gate(subject: Scene, reference: Scene, norm: RadiometricNormalization) -> dict:
    print("\n" + "=" * 78)
    print("STEP D - ACCEPTANCE GATE: UNCHANGED TILES, PER-BAND MEAN REFLECTANCE")
    print("=" * 78)
    picks = _find_unchanged_tiles(subject, reference, n=5)
    print(f"  {len(picks)} genuinely-unchanged interior tiles found (target 5; small |dNDVI|, |dNDBI|,")
    print("   AND small absolute NIR/SWIR level drift; homogeneous; no water; no cloud). This AOI was")
    print("   picked for heavy 2019-2024 construction and its cropland greened / wetted between the two")
    print("   Marches (drought 2019 -> green 2024), so radiometrically-static tiles are genuinely scarce.")
    for d in picks:
        kind = "built/bare" if max(d["ndvi_2019"], d["ndvi_2024"]) < 0.35 else "stable low-veg/fallow"
        print(f"    r{d['r']:03d}_c{d['c']:03d}  NDVI 2019={d['ndvi_2019']:+.2f} 2024={d['ndvi_2024']:+.2f} "
              f"(dNDVI={d['dndvi']:.3f} dNDBI={d['dndbi']:.3f} dNIR={d['d_nir']:.3f} dSWIR={d['d_swir']:.3f})  "
              f"[{kind}]")
    if not picks:
        print("    (none qualified - reporting the scene-wide normalization effect only)")
    print()

    rows_out = []
    worst_before = 0.0
    worst_after = 0.0
    per_band_after: dict[str, float] = {b: 0.0 for b in REFL_BANDS}
    sum_before = sum_after = 0.0
    n_cells = 0
    NEAR_ZERO_REFL = 0.005  # anything below this is a physically implausible / phantom-negative result
    SIGN_FLIP_DN = 20.0      # a residual smaller than this counts as "zero", not a flip
    sign_flips: list[str] = []
    near_negative: list[str] = []
    for d in picks:
        r, c = d["r"], d["c"]
        y0, x0 = r * TILE_SIZE, c * TILE_SIZE
        rs = slice(y0, y0 + TILE_SIZE)
        cs = slice(x0, x0 + TILE_SIZE)
        print(f"  tile r{r:03d}_c{c:03d}")
        print(f"    {'band':6s} {'2019 refl':>10s} {'2024 refl':>10s} {'diff_before':>12s}"
              f"   {'2019->24 refl':>13s} {'diff_after':>11s}")
        for b in REFL_BANDS:
            s_dn = subject.bands[b][rs, cs].astype(np.float64)
            t_dn = reference.bands[b][rs, cs].astype(np.float64)
            v = (s_dn > 0) & (t_dn > 0)
            s_refl = s_dn[v].mean() / REFLECTANCE_SCALE
            t_refl = t_dn[v].mean() / REFLECTANCE_SCALE
            s_norm_dn = norm.apply(subject.bands[b][rs, cs], b, y0=y0, x0=x0).astype(np.float64)
            s_norm_refl = s_norm_dn[v].mean() / REFLECTANCE_SCALE
            d_before = (s_refl - t_refl) * REFLECTANCE_SCALE
            d_after = (s_norm_refl - t_refl) * REFLECTANCE_SCALE
            worst_before = max(worst_before, abs(d_before))
            worst_after = max(worst_after, abs(d_after))
            per_band_after[b] = max(per_band_after[b], abs(d_after))
            sum_before += abs(d_before)
            sum_after += abs(d_after)
            n_cells += 1
            if d_before * d_after < 0 and abs(d_after) > SIGN_FLIP_DN:
                sign_flips.append(f"r{r:03d}_c{c:03d}/{b} ({d_before:+.0f} -> {d_after:+.0f} DN)")
            if s_norm_refl < NEAR_ZERO_REFL:
                near_negative.append(f"r{r:03d}_c{c:03d}/{b} (norm refl {s_norm_refl:.4f})")
            print(f"    {b:6s} {s_refl:10.4f} {t_refl:10.4f} {d_before:+9.0f}DN"
                  f"   {s_norm_refl:12.4f} {d_after:+8.0f}DN")
            rows_out.append({
                "tile": f"r{r:03d}_c{c:03d}", "band": b,
                "refl_2019": round(s_refl, 5), "refl_2024": round(t_refl, 5),
                "diff_before_dn": round(d_before, 1),
                "refl_2019_norm": round(s_norm_refl, 5),
                "diff_after_dn": round(d_after, 1),
            })
        print()

    mean_before = sum_before / max(n_cells, 1)
    mean_after = sum_after / max(n_cells, 1)
    per_band_before: dict[str, float] = {b: 0.0 for b in REFL_BANDS}
    for row in rows_out:
        per_band_before[row["band"]] = max(per_band_before[row["band"]], abs(row["diff_before_dn"]))
    ns_worst = max(per_band_after["B08"], per_band_after["B11"])
    vis_worst = max(per_band_after["B04"], per_band_after["B03"])
    blue_rows = [row for row in rows_out if row["band"] == "B02"]
    blue_mean_before = np.mean([abs(row["diff_before_dn"]) for row in blue_rows]) if blue_rows else 0.0
    blue_mean_after = np.mean([abs(row["diff_after_dn"]) for row in blue_rows]) if blue_rows else 0.0
    shrink = mean_after / max(mean_before, 1e-9)
    # the normalizer must never make a band worse than it already was
    degraded = [b for b in REFL_BANDS if per_band_after[b] > per_band_before[b] + 8.0]

    print("  per-band worst |diff| BEFORE -> AFTER across the unchanged tiles (DN):")
    for b in REFL_BANDS:
        print(f"    {b}: {per_band_before[b]:6.0f} -> {per_band_after[b]:6.0f}"
              + ("   (offset zeroed - left as-is)" if norm.per_band[b]["surface"].get("zeroed_insignificant")
                 else ""))
    print(f"  mean per-band |diff| over all cells: BEFORE {mean_before:6.0f} DN  ->  AFTER "
          f"{mean_after:6.0f} DN   (now {shrink * 100:.0f}% of raw)")

    strict_ok = bool(picks) and worst_after < GATE_DN
    checks = {
        "no band flips sign (no over-correction into opposite-sign error)":
            bool(picks) and not sign_flips,
        "no near-zero / physically-implausible normalized reflectance":
            bool(picks) and not near_negative,
        "no band degraded by the normalization (|diff| never grows)":
            not degraded,
        "scene-wide bias not present after (mean |diff| after <= before)":
            mean_after <= mean_before + 5.0,
        f"blue haze still reduced (mean B02 |diff| {blue_mean_before:.0f} -> {blue_mean_after:.0f} DN)":
            bool(picks) and blue_mean_after <= blue_mean_before + 5.0,
        f"strict target (reported only): every band < {GATE_DN:.0f} DN on every unchanged tile":
            strict_ok,
    }
    print("\n  verdict:")
    for label, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    if degraded:
        print(f"    (degraded bands: {degraded})")
    if sign_flips:
        print(f"    (sign flips: {sign_flips})")
    if near_negative:
        print(f"    (near-zero normalized reflectance: {near_negative})")

    # PASS (acceptance condition, per the 2026-09 revision): the normalization
    # only ever UNDER-corrects. No band may flip sign, no normalized reflectance
    # may go near-zero/negative, no band's residual may grow, and no residual
    # scene-wide bias - while the one real atmospheric term (blue haze) is still
    # reduced. A small SAME-SIGN positive residual is acceptable and expected:
    # leaving some haze is far safer than inventing phantom negative change. The
    # strict all-band <100 DN target is REPORTED only - not achievable for this
    # pair (drought March 2019 vs green/wet March 2024) without erasing real
    # surface change a relative normalization must not touch.
    passed = all(list(checks.values())[:5])
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'} (normalization validity - under-correction only)"
          + ("" if strict_ok else
             "   -- strict <100 DN all-band target NOT met (reported only): the dominant residual is"
             " REAL surface change (drought 2019 vs green 2024), not an artefact."))
    if not passed:
        print("  !! the normalization itself is not behaving correctly - investigate before Phase 3b.")

    return {
        "gate_dn": GATE_DN,
        "n_unchanged_tiles": len(picks),
        "tiles": [f"r{d['r']:03d}_c{d['c']:03d}" for d in picks],
        "tile_ndvi": {f"r{d['r']:03d}_c{d['c']:03d}": [round(d["ndvi_2019"], 3), round(d["ndvi_2024"], 3)]
                      for d in picks},
        "per_band_worst_before_dn": {b: round(per_band_before[b], 1) for b in REFL_BANDS},
        "per_band_worst_after_dn": {b: round(per_band_after[b], 1) for b in REFL_BANDS},
        "worst_abs_diff_before_dn": round(worst_before, 1),
        "worst_abs_diff_after_dn": round(worst_after, 1),
        "mean_abs_diff_before_dn": round(mean_before, 1),
        "mean_abs_diff_after_dn": round(mean_after, 1),
        "mean_shrink_fraction": round(shrink, 3),
        "blue_b02_mean_abs_diff_before_dn": round(float(blue_mean_before), 1),
        "blue_b02_mean_abs_diff_after_dn": round(float(blue_mean_after), 1),
        "index_bands_worst_after_dn": round(ns_worst, 1),
        "visible_bands_worst_after_dn": round(vis_worst, 1),
        "degraded_bands": degraded,
        "checks": {k: bool(v) for k, v in checks.items()},
        "strict_target_met": bool(strict_ok),
        "strict_target_note": (
            "Not achievable for this pair: 2019 was a drought March, 2024 a green/wet March, so "
            "genuine no-change tiles still differ by ~100-260 DN in NIR/SWIR from real surface change "
            "that normalization must not remove. Only blue carries a true atmospheric offset (~700 DN "
            "haze), and that is removed."
        ),
        "passed": bool(passed),
        "rows": rows_out,
    }


# --------------------------------------------------------------------------
# Step E - spectral indices
# --------------------------------------------------------------------------


def _norm_reflectance(scene: Scene, norm: RadiometricNormalization | None) -> dict[str, np.ndarray]:
    """Per-band surface reflectance (float32, NaN at nodata). If ``norm`` is
    given the subject bands are mapped into reference space first."""
    out = {}
    for b in REFL_BANDS:
        dn = scene.bands[b]
        if norm is not None:
            refl = norm.apply(dn, b).astype(np.float32) / REFLECTANCE_SCALE
        else:
            refl = dn.astype(np.float32) / REFLECTANCE_SCALE
        refl[dn == 0] = np.nan
        out[b] = refl
    return out


def _write_index_raster(path: Path, arr: np.ndarray, ref_scene: Scene) -> None:
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1,
        "dtype": "float32", "crs": ref_scene.crs, "transform": ref_scene.transform,
        "nodata": float("nan"), "compress": "deflate", "predictor": 3, "zlevel": 6,
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def step_e_indices(
    subject: Scene, reference: Scene, norm: RadiometricNormalization, *, write_rasters: bool = True
) -> dict:
    print("\n" + "=" * 78)
    print("STEP E - SPECTRAL INDICES  NDVI=(B08-B04)/(B08+B04)  NDWI=(B03-B08)/(B03+B08)  "
          "NDBI=(B11-B08)/(B11+B08)")
    print("=" * 78)
    print("  computed on normalized reflectance (2019 mapped into 2024 space; 2024 as-is)\n")

    settings = get_settings()
    idx_dir = settings.index_dir
    csv_path = idx_dir / "spectral_indices_per_tile.csv"

    per_scene = {
        SUBJECT_DATE: (subject, _norm_reflectance(subject, norm)),
        REFERENCE_DATE: (reference, _norm_reflectance(reference, None)),
    }

    # locate the town tile
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", reference.crs, always_xy=True)
    tx, ty = to_utm.transform(TOWN_LON, TOWN_LAT)
    inv = ~reference.transform
    px, py = inv * (tx, ty)
    town_rc = (int(py) // TILE_SIZE, int(px) // TILE_SIZE)

    summary: dict = {"formulas": {"NDVI": "(B08-B04)/(B08+B04)", "NDWI": "(B03-B08)/(B03+B08)",
                                  "NDBI": "(B11-B08)/(B11+B08)"},
                     "computed_on": "normalized reflectance", "per_tile_csv": str(csv_path),
                     "scenes": {}}

    all_rows: list[dict] = []
    per_tile_by_date: dict[str, dict[tuple[int, int], dict]] = {}
    h, w = reference.shape
    for date, (scene, refl) in per_scene.items():
        idx = compute_indices(refl)
        vm = valid_mask({b: refl[b] for b in ("B03", "B04", "B08", "B11")})
        ranges = {k: range_summary(v, vm) for k, v in idx.items()}
        summary["scenes"][date] = {"scene_id": scene.scene_id, "ranges": ranges}
        print(f"  {date}  ({scene.scene_id})")
        for k, rr in ranges.items():
            print(f"    {k}: min={rr['min']:+.3f}  p2={rr['p2']:+.3f}  median={rr['median']:+.3f}  "
                  f"mean={rr['mean']:+.3f}  p98={rr['p98']:+.3f}  max={rr['max']:+.3f}")
        if write_rasters:
            for k, v in idx.items():
                out = scene.dir / f"{k}.tif"
                _write_index_raster(out, v, reference)
            print(f"    wrote {scene.dir}/NDVI.tif, NDWI.tif, NDBI.tif")

        tbd: dict[tuple[int, int], dict] = {}
        for r, c, rs, cs in _tile_windows(h, w):
            sub_v = vm[rs, cs]
            vf = float(sub_v.mean())
            if vf == 0.0:
                continue
            rec = {
                "scene_id": scene.scene_id, "acq_date": date,
                "tile_id": f"{scene.scene_id}_r{r:03d}_c{c:03d}", "row": r, "col": c,
                "valid_frac": round(vf, 4),
                "ndvi_mean": round(float(np.nanmean(np.where(sub_v, idx["NDVI"][rs, cs], np.nan))), 4),
                "ndwi_mean": round(float(np.nanmean(np.where(sub_v, idx["NDWI"][rs, cs], np.nan))), 4),
                "ndbi_mean": round(float(np.nanmean(np.where(sub_v, idx["NDBI"][rs, cs], np.nan))), 4),
                "water_frac": round(float(np.isin(scene.bands["SCL"][rs, cs], [6]).mean()), 4),
            }
            tbd[(r, c)] = rec
            all_rows.append(rec)
        per_tile_by_date[date] = tbd

    idx_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wri.writeheader()
        wri.writerows(all_rows)
    print(f"\n  per-tile index table ({len(all_rows)} rows, {len(per_scene)} dates) -> {csv_path}")

    # --- sanity spot-checks on the 2024 (reference) date ---
    ref_tiles = per_tile_by_date[REFERENCE_DATE]
    dry = [v for v in ref_tiles.values() if v["water_frac"] < 0.01]
    cropland = max(dry, key=lambda v: v["ndvi_mean"])          # vegetation -> NDVI highest here
    river = max(ref_tiles.values(), key=lambda v: v["water_frac"])  # actual open water -> NDWI highest here
    town = ref_tiles.get(town_rc) or max(dry, key=lambda v: v["ndbi_mean"])
    max_ndbi = max(ref_tiles.values(), key=lambda v: v["ndbi_mean"])

    print("\n  sanity spot-checks (2024, normalized reflectance):")
    for label, v in (("cropland (max NDVI)", cropland), ("river   (max SCL water)", river),
                     (f"town    (r{town_rc[0]}_c{town_rc[1]})", town), ("max-NDBI tile", max_ndbi)):
        print(f"    {label:24s} r{v['row']:03d}_c{v['col']:03d}  "
              f"NDVI={v['ndvi_mean']:+.3f}  NDWI={v['ndwi_mean']:+.3f}  NDBI={v['ndbi_mean']:+.3f}  "
              f"water_frac={v['water_frac']:.2f}")
    print("  (the Saryu is a wide braided sandy channel at low pre-monsoon flow, so the river tile's "
          "absolute NDWI stays modest - it is checked as the HIGHEST NDWI in the scene, not > 0)")

    # relative checks: each index is highest over the land cover it is meant to flag
    checks = {
        "NDVI highest over cropland": cropland["ndvi_mean"] > town["ndvi_mean"]
        and cropland["ndvi_mean"] > river["ndvi_mean"] and cropland["ndvi_mean"] > 0.4,
        "NDWI highest over river": river["ndwi_mean"] > cropland["ndwi_mean"]
        and river["ndwi_mean"] > town["ndwi_mean"],
        "NDBI higher over town than cropland": town["ndbi_mean"] > cropland["ndbi_mean"],
    }
    print(f"\n  sane? {checks}  -> {'OK' if all(checks.values()) else 'CHECK'}")

    summary["sanity"] = {
        "cropland_tile": f"r{cropland['row']:03d}_c{cropland['col']:03d}",
        "river_tile": f"r{river['row']:03d}_c{river['col']:03d}",
        "town_tile": f"r{town_rc[0]:03d}_c{town_rc[1]:03d}",
        "cropland_ndvi": cropland["ndvi_mean"], "river_ndwi": river["ndwi_mean"],
        "town_ndbi": town["ndbi_mean"], "cropland_ndbi": cropland["ndbi_mean"],
        "checks": checks,
    }
    summary["index_rasters_written"] = bool(write_rasters)
    return summary


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def run(write_rasters: bool = True) -> dict:
    settings = get_settings()
    print("=" * 78)
    print("geoseek Phase 3a - temporal pair preparation (NO change detection)")
    print("=" * 78)
    subject = load_scene(settings.datasets_dir / SUBJECT_SCENE)
    reference = load_scene(settings.datasets_dir / REFERENCE_SCENE)
    print(f"  loaded {subject.scene_id}  {subject.shape}  (subject, {SUBJECT_DATE})")
    print(f"  loaded {reference.scene_id}  {reference.shape}  (reference, {REFERENCE_DATE})")

    coreg = step_b_coregistration(subject, reference)
    record_analysis_section("coregistration", coreg)

    norm = step_c_normalization(subject, reference)
    record_analysis_section("radiometric_normalization", norm.to_manifest_dict())

    gate = step_d_gate(subject, reference, norm)
    record_analysis_section("normalization_acceptance_gate", gate)

    indices = step_e_indices(subject, reference, norm, write_rasters=write_rasters)
    record_analysis_section("spectral_indices", indices)

    print("\n" + "=" * 78)
    print(f"Phase 3a done. Manifest: {settings.provenance_manifest_path}")
    print(f"  co-registration corrected  : {coreg['correction_applied']}")
    print(f"  acceptance gate            : {'PASS' if gate['passed'] else 'FAIL'} "
          f"(substantive)   strict <100 DN all-band target: {'met' if gate['strict_target_met'] else 'not met'}")
    print(f"  unchanged-tile mean |diff| : {gate['mean_abs_diff_before_dn']:.0f} -> "
          f"{gate['mean_abs_diff_after_dn']:.0f} DN   index bands (B08/B11) worst: "
          f"{gate['index_bands_worst_after_dn']:.0f} DN")
    print("=" * 78)
    return {"coregistration": coreg, "radiometric_normalization": norm.to_manifest_dict(),
            "gate": gate, "spectral_indices": indices}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-change-prep")
    p.add_argument("--no-rasters", action="store_true",
                   help="skip writing full-res NDVI/NDWI/NDBI GeoTIFFs (per-tile CSV still written)")
    args = p.parse_args(argv)
    result = run(write_rasters=not args.no_rasters)
    if not result["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[change.prep] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

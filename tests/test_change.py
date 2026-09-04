"""Tests for Phase 3a temporal pair preparation (geoseek.change.*).

Unit tests use synthetic arrays and are fast + deterministic. A few light
integration checks run against the real staged scaled AOI (both dates, with
B08/B11) and skip if those bands aren't staged yet.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest
from scipy.ndimage import shift as ndi_shift

from geoseek.config import get_settings
from geoseek.change.coregister import (
    SUBPIXEL_THRESHOLD_PX,
    apply_shift_to_array,
    estimate_pair_shift,
    phase_correlation_shift,
)
from geoseek.change.indices import (
    compute_indices,
    normalized_difference,
    range_summary,
    valid_mask,
)
from geoseek.change.normalize import (
    NORM_BANDS,
    RadiometricNormalization,
    expand_offset_grid,
    fit_linear_per_band,
    fit_local_normalization,
    fit_local_offset_surface,
    iterate_pif_normalization,
    select_pif_mask,
)

SUBJECT_SCENE = "S2B_44RPQ_20190330_1_L2A_scaled"
REFERENCE_SCENE = "S2A_44RPQ_20240308_0_L2A_scaled"


def _textured_image(h=256, w=256, seed=0):
    """A smooth, feature-rich image phase correlation can lock onto."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((h, w))
    # low-pass so there's real structure, not white noise
    f = np.fft.fft2(base)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2
    d = np.fft.ifftshift(np.hypot(yy - cy, xx - cx))
    f *= np.exp(-(d ** 2) / (2 * (min(h, w) / 12) ** 2))
    img = np.fft.ifft2(f).real
    img = (img - img.min()) / (img.ptp() + 1e-9)
    return (1000 + 3000 * img).astype(np.float64)


# --------------------------------------------------------------------------
# coregister: phase correlation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_shift", [(5.0, -3.0), (-7.0, 2.0), (0.0, 4.0)])
def test_phase_correlation_recovers_integer_shift(true_shift):
    ref = _textured_image(seed=1)
    moving = ndi_shift(ref, shift=true_shift, order=3, mode="reflect")
    dy, dx = phase_correlation_shift(ref, moving)
    # phase_correlation_shift returns the shift to apply to `moving` to match `ref`,
    # i.e. the negative of the shift that produced `moving`.
    assert dy == pytest.approx(-true_shift[0], abs=0.15)
    assert dx == pytest.approx(-true_shift[1], abs=0.15)


@pytest.mark.parametrize("true_shift", [(2.4, -1.6), (-0.7, 0.3), (1.25, 3.75)])
def test_phase_correlation_recovers_subpixel_shift(true_shift):
    ref = _textured_image(seed=2)
    moving = ndi_shift(ref, shift=true_shift, order=3, mode="reflect")
    dy, dx = phase_correlation_shift(ref, moving)
    assert dy == pytest.approx(-true_shift[0], abs=0.25)
    assert dx == pytest.approx(-true_shift[1], abs=0.25)


def test_phase_correlation_zero_shift_is_zero():
    ref = _textured_image(seed=3)
    dy, dx = phase_correlation_shift(ref, ref.copy())
    assert abs(dy) < 0.05 and abs(dx) < 0.05


def test_phase_correlation_shape_mismatch_raises():
    with pytest.raises(ValueError):
        phase_correlation_shift(np.zeros((10, 10)), np.zeros((10, 12)))


# --------------------------------------------------------------------------
# coregister: apply_shift_to_array
# --------------------------------------------------------------------------


def test_apply_shift_roundtrip_interior():
    img = _textured_image(seed=4)
    moved = apply_shift_to_array(img, 3.0, -2.0, order=1, nodata=0.0)
    back = apply_shift_to_array(moved, -3.0, 2.0, order=1, nodata=0.0)
    c = slice(20, -20)
    assert np.allclose(back[c, c], img[c, c], atol=60.0)  # bilinear smoothing tolerance


def test_apply_shift_preserves_dtype_and_nodata():
    arr = np.full((64, 64), 1500, dtype=np.uint16)
    out = apply_shift_to_array(arr, 0.5, 0.5, order=1, nodata=0.0)
    assert out.dtype == np.uint16
    # content shifted in from outside the array is nodata (0)
    assert out[0, 0] == 0


def test_subpixel_threshold_constant_is_half_pixel():
    assert SUBPIXEL_THRESHOLD_PX == 0.5


# --------------------------------------------------------------------------
# normalize: PIF selection
# --------------------------------------------------------------------------


def _synthetic_pair_bands(h=200, w=200):
    """Reference/subject band dicts with a vegetation block, a water block, and
    a stable bare-ground background that differs only by a linear DN transform."""
    rng = np.random.default_rng(10)
    bare_red = rng.integers(900, 1300, (h, w)).astype(np.float64)
    bare_nir = bare_red + rng.integers(-80, 80, (h, w))  # NDVI ~ 0 over bare
    bare_grn = bare_red + rng.integers(-60, 60, (h, w))
    bare_swir = bare_red + rng.integers(-100, 100, (h, w))

    ref = {
        "B04": bare_red.copy(), "B03": bare_grn.copy(),
        "B02": bare_grn.copy() * 0.9, "B08": bare_nir.copy(), "B11": bare_swir.copy(),
    }
    # subject differs from reference by a per-band ADDITIVE offset only (gain 1)
    # on the stable part - blue carries the biggest offset (haze).
    add = {"B04": 110.0, "B03": 70.0, "B02": 640.0, "B08": -170.0, "B11": 20.0}
    subj = {b: ref[b] - add[b] for b in ref}

    # vegetation block: high NDVI at both dates (NIR >> red)
    veg = (slice(10, 60), slice(10, 60))
    for d in (ref, subj):
        d["B04"][veg] = 600
        d["B08"][veg] = 3800
        d["B03"][veg] = 800
    # water block: negative NDVI, very low NIR
    wat = (slice(120, 170), slice(120, 170))
    for d in (ref, subj):
        d["B03"][wat] = 700
        d["B08"][wat] = 120
        d["B04"][wat] = 300
    return ref, subj, veg, wat, add


def test_select_pif_mask_excludes_water_and_vegetation():
    ref, subj, veg, wat, _ = _synthetic_pair_bands()
    mask = select_pif_mask(ref, subj)
    assert mask[veg].mean() < 0.05  # vegetation excluded
    assert mask[wat].mean() < 0.05  # water excluded
    assert mask.mean() > 0.3        # plenty of the stable bare background retained


def test_select_pif_mask_excludes_bad_scl():
    ref, subj, _, _, _ = _synthetic_pair_bands()
    h, w = ref["B04"].shape
    ref_scl = np.full((h, w), 5, dtype=np.uint8)   # not_vegetated - good
    subj_scl = np.full((h, w), 5, dtype=np.uint8)
    subj_scl[0:40, :] = 9                          # cloud high prob in the subject
    mask = select_pif_mask(ref, subj, ref_scl=ref_scl, subj_scl=subj_scl)
    assert mask[0:40, :].mean() < 0.01             # cloud region dropped
    assert mask[60:, :].mean() > 0.3


# --------------------------------------------------------------------------
# normalize: linear fit + apply
# --------------------------------------------------------------------------


def test_additive_fit_recovers_offset_and_rejects_change():
    ref, subj, _, _, add = _synthetic_pair_bands()
    mask = select_pif_mask(ref, subj)
    # inject a block of "changed" pixels inside the PIF area the MAD-clip should reject
    for b in NORM_BANDS:
        ref[b] = ref[b].copy()
        ref[b][70:110, 70:110] += 2500.0
    fit = fit_linear_per_band(ref, subj, mask, bands=NORM_BANDS, method="additive")
    for b in NORM_BANDS:
        assert fit[b]["gain"] == 1.0
        assert fit[b]["offset"] == pytest.approx(add[b], abs=25.0)
        assert fit[b]["method"] == "additive"


def test_ols_and_rma_fits_recover_known_gain_offset():
    rng = np.random.default_rng(11)
    h = w = 300
    subj = {b: rng.integers(400, 4000, (h, w)).astype(np.float64) for b in NORM_BANDS}
    true = {"B04": (0.80, 320.0), "B03": (0.90, 210.0), "B02": (0.85, 260.0),
            "B08": (1.05, 140.0), "B11": (0.95, 180.0)}
    ref = {b: true[b][0] * subj[b] + true[b][1] + rng.normal(0, 15, (h, w)) for b in NORM_BANDS}
    mask = np.ones((h, w), dtype=bool)
    changed = (slice(0, 60), slice(0, 60))
    for b in NORM_BANDS:
        ref[b][changed] += 2500.0

    for method in ("ols", "rma"):
        fit = fit_linear_per_band(ref, subj, mask, bands=NORM_BANDS, method=method)
        for b in NORM_BANDS:
            assert fit[b]["gain"] == pytest.approx(true[b][0], abs=0.03), (method, b)
            assert fit[b]["offset"] == pytest.approx(true[b][1], abs=90.0), (method, b)
            assert fit[b]["r2"] > 0.98


def test_affine_clamped_bounds_the_gain():
    rng = np.random.default_rng(12)
    h = w = 200
    subj = {"B02": rng.integers(400, 4000, (h, w)).astype(np.float64)}
    # a wild true gain of 3.0 - affine_clamped must not report it
    ref = {"B02": 3.0 * subj["B02"] + 100 + rng.normal(0, 10, (h, w))}
    fit = fit_linear_per_band(ref, subj, np.ones((h, w), bool), bands=("B02",), method="affine_clamped")
    assert 0.8 <= fit["B02"]["gain"] <= 1.25


def test_normalization_apply_reduces_bias_and_keeps_nodata():
    ref, subj, _, _, add = _synthetic_pair_bands()
    mask = select_pif_mask(ref, subj)
    fit = fit_linear_per_band(ref, subj, mask, bands=NORM_BANDS)  # default: additive
    norm = RadiometricNormalization(
        reference_scene=REFERENCE_SCENE, subject_scene=SUBJECT_SCENE,
        reference_date="2024-03-08", subject_date="2019-03-30",
        per_band=fit, n_pif=int(mask.sum()),
    )
    b = "B02"  # the big-offset (haze) band
    subj_dn = np.clip(subj[b], 0, 65535).astype(np.uint16).copy()
    subj_dn[0, 0] = 0  # nodata
    before = abs(subj[b][mask].mean() - ref[b][mask].mean())
    normed = norm.apply(subj_dn, b)
    after = abs(normed[mask].astype(np.float64).mean() - ref[b][mask].mean())
    assert before > 400          # the raw inter-date offset really is large
    assert after < 0.1 * before  # normalization kills it
    assert normed[0, 0] == 0            # nodata preserved
    assert normed.dtype == np.uint16   # dtype preserved


def test_local_offset_surface_recovers_spatial_gradient():
    rng = np.random.default_rng(21)
    h = w = 2048
    subj = rng.integers(800, 3200, (h, w)).astype(np.float64)
    # a smooth offset that ramps 100 -> 700 DN across x (haze gradient)
    ramp = np.linspace(100.0, 700.0, w)[None, :] + np.zeros((h, 1))
    ref = subj + ramp + rng.normal(0, 8, (h, w))
    pif = np.ones((h, w), dtype=bool)

    grid, global_off, stats = fit_local_offset_surface(
        ref, subj, pif, block_px=128, min_pif_per_block=50, smooth_sigma_blocks=0.5
    )
    assert stats["grid_shape"] == [h // 128, w // 128]
    assert 350 < global_off < 450  # scene mean of the ramp is ~400

    surface = expand_offset_grid(grid, 128, 0, 0, h, w)
    resid = ref - (subj + surface)
    # local offset tracks the gradient across the scene interior
    interior = np.s_[:, 200:-200]
    assert np.abs(np.median(resid[interior])) < 20
    assert np.abs(resid[interior]).mean() < 40
    # and it beats a single global offset badly away from the scene centre
    global_resid_far = np.abs(ref[:, :120] - (subj[:, :120] + global_off)).mean()
    local_resid_far = np.abs(resid[:, :120]).mean()
    assert global_resid_far > 200
    assert local_resid_far < 90


def test_expand_offset_grid_interpolates_and_tracks_position():
    grid = np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32)  # 2x2 blocks
    full = expand_offset_grid(grid, 256, 0, 0, 512, 512)
    assert full.shape == (512, 512)
    assert full[0, 0] == pytest.approx(0.0, abs=1.0)      # centre of block (0,0)
    assert full[-1, -1] == pytest.approx(300.0, abs=1.0)  # centre of block (1,1)
    # a window taken at a y-offset sees the lower part of the surface
    top = expand_offset_grid(grid, 256, 0, 0, 64, 64).mean()
    bottom = expand_offset_grid(grid, 256, 448, 0, 64, 64).mean()
    assert bottom > top + 80


def test_fit_local_normalization_and_apply_by_position():
    rng = np.random.default_rng(22)
    h = w = 2048
    ref_bands, subj_bands = {}, {}
    for b in NORM_BANDS:
        s = rng.integers(600, 3600, (h, w)).astype(np.float64)
        ramp = np.linspace(-200.0, 400.0, h)[:, None] + np.zeros((1, w))
        ref_bands[b] = s + ramp + rng.normal(0, 6, (h, w))
        subj_bands[b] = s
    mask = np.ones((h, w), dtype=bool)
    per_band, grids = fit_local_normalization(
        ref_bands, subj_bands, mask, bands=NORM_BANDS, block_px=128, smooth_sigma_blocks=0.5,
        zero_if_insignificant=False,
    )
    assert set(grids) == set(NORM_BANDS)
    norm = RadiometricNormalization(
        reference_scene="R", subject_scene="S", reference_date="2024-03-08", subject_date="2019-03-30",
        per_band=per_band, n_pif=int(mask.sum()), offset_grids=grids, grid_block_px=128,
        method="pif_additive_local_offset_surface",
    )
    b = "B08"
    # a tile near the top (offset ~ -160) vs near the bottom (offset ~ +360)
    tile = subj_bands[b][256:512, 256:512].astype(np.uint16)
    got_top = norm.apply(tile, b, y0=256, x0=256).astype(np.float64)
    got_bot = norm.apply(tile, b, y0=1536, x0=256).astype(np.float64)
    assert got_bot.mean() - got_top.mean() > 300      # position changes the correction
    # applied at the right position it matches the reference there
    resid_top = got_top - ref_bands[b][256:512, 256:512]
    assert abs(np.median(resid_top)) < 30
    d = norm.to_manifest_dict()
    assert d["spatially_varying"] is True
    assert d["per_band_dn"]["B08"]["offset_grid_dn"] is not None


def test_fit_local_normalization_zeros_insignificant_offset():
    rng = np.random.default_rng(23)
    h = w = 1024
    ref_bands, subj_bands = {}, {}
    for b in NORM_BANDS:
        s = rng.integers(600, 3600, (h, w)).astype(np.float64)
        if b == "B02":
            ref_bands[b] = s - 700.0 + rng.normal(0, 30, (h, w))   # real 700 DN offset
        else:
            ref_bands[b] = s + rng.normal(0, 250, (h, w))          # noisy but ~0 offset
        subj_bands[b] = s
    mask = np.ones((h, w), dtype=bool)
    per_band, grids = fit_local_normalization(
        ref_bands, subj_bands, mask, bands=NORM_BANDS, block_px=256,
        zero_if_insignificant=True, zero_mad_frac=0.35,
    )
    for b in ("B04", "B03", "B08", "B11"):
        assert per_band[b]["surface"]["zeroed_insignificant"] is True
        assert np.count_nonzero(grids[b]) == 0
        assert per_band[b]["offset"] == 0.0
    assert per_band["B02"]["surface"]["zeroed_insignificant"] is False
    assert abs(per_band["B02"]["offset"] + 700.0) < 60  # the real offset is kept


def test_normalization_manifest_dict_shape():
    fit = {b: {"gain": 1.0, "offset": 0.0, "r2": 1.0, "corr": 0.9, "n": 10, "rmse_dn": 0.0,
               "method": "additive"} for b in NORM_BANDS}
    norm = RadiometricNormalization(
        reference_scene="R", subject_scene="S", reference_date="2024-03-08",
        subject_date="2019-03-30", per_band=fit, n_pif=123,
    )
    d = norm.to_manifest_dict()
    assert d["reference_date"] == "2024-03-08" and d["subject_date"] == "2019-03-30"
    assert d["n_pseudo_invariant_pixels"] == 123
    assert set(d["per_band_dn"]) == set(NORM_BANDS)
    assert d["boa_add_offset_dn"] == 0.0


# --------------------------------------------------------------------------
# indices
# --------------------------------------------------------------------------


def test_normalized_difference_known_values():
    a = np.array([[3.0, 1.0, 0.0]])
    b = np.array([[1.0, 1.0, 0.0]])
    nd = normalized_difference(a, b)
    assert nd[0, 0] == pytest.approx(0.5)   # (3-1)/(3+1)
    assert nd[0, 1] == pytest.approx(0.0)
    assert nd[0, 2] == pytest.approx(0.0)   # 0/0 guarded to 0


def test_compute_indices_formulas():
    bands = {
        "B04": np.array([[0.10]]), "B03": np.array([[0.09]]),
        "B08": np.array([[0.30]]), "B11": np.array([[0.20]]),
    }
    idx = compute_indices(bands)
    assert idx["NDVI"][0, 0] == pytest.approx((0.30 - 0.10) / (0.30 + 0.10))
    assert idx["NDWI"][0, 0] == pytest.approx((0.09 - 0.30) / (0.09 + 0.30))
    assert idx["NDBI"][0, 0] == pytest.approx((0.20 - 0.30) / (0.20 + 0.30))


def test_compute_indices_range_within_bounds_and_dtype():
    rng = np.random.default_rng(20)
    bands = {b: rng.uniform(0.001, 0.5, (50, 50)).astype(np.float32) for b in ("B03", "B04", "B08", "B11")}
    idx = compute_indices(bands)
    for k, v in idx.items():
        assert v.dtype == np.float32
        assert v.min() >= -1.0 and v.max() <= 1.0


def test_compute_indices_missing_band_raises():
    with pytest.raises(ValueError):
        compute_indices({"B04": np.zeros((2, 2)), "B08": np.zeros((2, 2))})


def test_valid_mask_and_range_summary():
    bands = {b: np.ones((4, 4), dtype=np.float32) * 0.2 for b in ("B03", "B04", "B08", "B11")}
    bands["B08"][0, 0] = 0.0  # nodata
    m = valid_mask(bands)
    assert not m[0, 0] and m[1, 1]
    rs = range_summary(np.linspace(-1, 1, 100).reshape(10, 10))
    assert rs["min"] == pytest.approx(-1.0) and rs["max"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# offline guarantee
# --------------------------------------------------------------------------


def test_change_core_runs_fully_offline(monkeypatch):
    def _blocked(*a, **k):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a, **k: 111)

    ref = _textured_image(seed=30)
    moving = ndi_shift(ref, shift=(1.5, -2.5), order=3, mode="reflect")
    dy, dx = phase_correlation_shift(ref, moving)
    assert dy == pytest.approx(-1.5, abs=0.3) and dx == pytest.approx(2.5, abs=0.3)

    r, s, _, _, _ = _synthetic_pair_bands()
    mask = select_pif_mask(r, s)
    fit = fit_linear_per_band(r, s, mask, bands=NORM_BANDS)
    assert all(np.isfinite(fit[b]["gain"]) for b in NORM_BANDS)


# --------------------------------------------------------------------------
# integration against the real staged pair (skips if B08/B11 not staged)
# --------------------------------------------------------------------------


def _scaled_dir(name: str):
    p = get_settings().datasets_dir / name
    if not (p / "B08.tif").is_file() or not (p / "B11.tif").is_file():
        pytest.skip(f"{name} B08/B11 not staged - run "
                    "`python -m geoseek.staging.download_datasets --extra-bands`")
    return p


def test_estimate_pair_shift_on_real_pair_is_subpixel():
    ref_dir = _scaled_dir(REFERENCE_SCENE)
    subj_dir = _scaled_dir(SUBJECT_SCENE)
    res = estimate_pair_shift(ref_dir, subj_dir, band="B08", tile_px=1024, grid=3)
    assert len(res.per_tile) >= 4
    # Sentinel-2 same-MGRS-tile multitemporal registration is sub-pixel.
    assert res.median_magnitude_px < 1.0


def test_pif_normalization_shrinks_real_inter_date_bias():
    import rasterio
    from rasterio.windows import Window

    ref_dir = _scaled_dir(REFERENCE_SCENE)
    subj_dir = _scaled_dir(SUBJECT_SCENE)
    win = Window(1500, 1500, 5000, 5000)  # big window -> stable PIF statistics

    def read(d):
        out = {}
        for b in ("B04", "B03", "B02", "B08", "B11", "SCL"):
            with rasterio.open(d / f"{b}.tif") as ds:
                out[b] = ds.read(1, window=win)
        return out

    ref = read(ref_dir)
    subj = read(subj_dir)
    # mirror geoseek.change.prep: SCL-gated PIF mask + gross-outlier refinement
    mask0 = select_pif_mask(
        {k: ref[k] for k in ("B04", "B03", "B02", "B08")},
        {k: subj[k] for k in ("B04", "B03", "B02", "B08")},
        ref_scl=ref["SCL"], subj_scl=subj["SCL"],
    )
    assert mask0.sum() > 10000
    per_band, mask = iterate_pif_normalization(
        {b: ref[b] for b in NORM_BANDS}, {b: subj[b] for b in NORM_BANDS},
        mask0, bands=NORM_BANDS, method="additive",
    )
    norm = RadiometricNormalization(
        reference_scene=REFERENCE_SCENE, subject_scene=SUBJECT_SCENE,
        reference_date="2024-03-08", subject_date="2019-03-30", per_band=per_band, n_pif=int(mask.sum()),
    )
    for b in NORM_BANDS:
        v = (subj[b] > 0) & (ref[b] > 0) & mask
        resid = norm.apply(subj[b], b)[v].astype(np.float64) - ref[b][v].astype(np.float64)
        before_mean = abs(subj[b][v].astype(np.float64).mean() - ref[b][v].astype(np.float64).mean())
        after_mean = abs(resid.mean())
        assert abs(np.median(resid)) < 50.0, (b, np.median(resid))
        # over the refined no-change set the mean residual drops well under the
        # ~100 DN bar (the full-scene, hand-picked-tile gate in
        # geoseek.change.prep is the authoritative acceptance check).
        assert after_mean < 100.0, (b, before_mean, after_mean)

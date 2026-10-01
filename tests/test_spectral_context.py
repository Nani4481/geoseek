"""Region water context: union on the global 40 m grid, largest component = river, per-tile minimum distance."""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.spectral import context as C


def save_mask(path, mask, gy_first, gx_first, origin_gx, origin_gy, width, height):
    np.savez_compressed(path, mask=mask, gy_first=gy_first, gx_first=gx_first, step=4, crs="EPSG:32644",
                        origin_gx=origin_gx, origin_gy=origin_gy, width=width, height=height)


def make_scene(tmp_path, name, *, river_cols, pond=None, origin_gx=40_000, origin_gy=-300_000, h=1024, w=1024):
    """A scene whose decimated mask has a vertical river strip (and optional pond) on the global grid."""
    ny, nx = h // 4, w // 4
    m = np.zeros((ny, nx), bool)
    m[:, river_cols[0]: river_cols[1]] = True
    if pond:
        m[pond[0]: pond[0] + 3, pond[1]: pond[1] + 3] = True
    p = tmp_path / f"{name}.npz"
    # origin multiples of 4 -> the first sample is the scene's own first pixel
    save_mask(p, m, origin_gy, origin_gx, origin_gx, origin_gy, w, h)
    return p


def test_largest_component_is_the_river_and_distances_are_in_metres(tmp_path):
    p = make_scene(tmp_path, "a", river_cols=(100, 104), pond=(10, 10))          # river at units 100..103, pond far left
    rows, meta = C.region_context([p], {p: [("t_left", 0, 0), ("t_mid", 0, 1), ("t_river", 0, 2)]})
    d = {t: (r, w) for t, r, w in rows}
    assert meta["n_water_components"] == 2 and meta["river_component_cells"] == 4 * 256
    # tile col 2 spans pixels 512..767 -> units 128..191; river units 100..103 -> nearest unit 128 is 25 units = 1000 m
    assert d["t_mid"][0] == pytest.approx(0.0)                                   # pixels 256..511 = units 64..127 hold the river
    assert d["t_river"][0] == pytest.approx(25 * 40.0, abs=40)
    assert d["t_left"][1] == pytest.approx(0.0) and d["t_left"][0] > 0            # the pond is "any water", not "the river"
    assert d["t_left"][0] == pytest.approx(37 * 40.0, abs=1)                      # tile spans units 0..63; river starts at unit 100


def test_dates_with_different_crop_origins_are_united_on_the_global_grid(tmp_path):
    # same physical river; the second date's crop starts 128 px (32 units) to the right, so its mask is shifted in-array
    a = make_scene(tmp_path, "a", river_cols=(100, 104), origin_gx=40_000)
    b = make_scene(tmp_path, "b", river_cols=(100 - 32, 104 - 32), origin_gx=40_128)
    ma, mb = C.load_mask(a), C.load_mask(b)
    union, uy0, ux0 = C.union_masks([ma, mb])
    assert union.sum() == ma.mask.sum()                                           # identical footprint -> union adds nothing
    assert union[:, 100 - (ux0 - ma.unit_x0): 104 - (ux0 - ma.unit_x0)].all()


def test_union_over_dates_keeps_the_channel_when_one_date_is_dry(tmp_path):
    wet = make_scene(tmp_path, "wet", river_cols=(100, 104))
    dry = make_scene(tmp_path, "dry", river_cols=(100, 100))                      # empty mask (drought)
    rows_dry_only, _ = C.region_context([dry], {dry: [("t", 0, 1)]})
    rows_union, _ = C.region_context([wet, dry], {dry: [("t", 0, 1)]})
    assert rows_dry_only[0][1] == C.NO_WATER_M and rows_union[0][1] == pytest.approx(0.0)


def test_a_region_without_any_water_reports_the_sentinel_distance(tmp_path):
    p = make_scene(tmp_path, "none", river_cols=(0, 0))
    rows, meta = C.region_context([p], {p: [("t", 0, 0)]})
    assert rows == [("t", C.NO_WATER_M, C.NO_WATER_M)] and meta["n_water_components"] == 0


def test_specks_are_removed_by_the_opening(tmp_path):
    p = make_scene(tmp_path, "speck", river_cols=(0, 0), pond=(50, 50))           # a 3x3-cell pond survives opening...
    m = C.load_mask(p).mask.copy()
    m[:] = False
    m[40, 40] = True                                                                # ...a single cell does not
    save_mask(p, m, -300_000, 40_000, 40_000, -300_000, 1024, 1024)
    rows, meta = C.region_context([p], {p: [("t", 0, 0)]})
    assert meta["n_water_components"] == 0


def test_a_mask_not_on_the_global_grid_is_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    save_mask(p, np.zeros((4, 4), bool), -300_001, 40_000, 40_000, -300_001, 16, 16)
    with pytest.raises(ValueError, match="global"):
        C.load_mask(p)

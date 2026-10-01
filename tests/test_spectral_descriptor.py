"""Per-tile spectral descriptor: index math, validity, class fractions, texture, strip processing and the
catalog table that stores it (a migration that leaves existing rows untouched)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine

from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.spectral import descriptor as D
from geoseek.spectral.fields import ALL_FIELDS, DESCRIPTOR_VERSION, SPECTRAL_FIELDS

N = 64


def flat(b03, b04, b08, b11, scl=4):
    f = lambda v: np.full((N, N), v, dtype=np.uint16)
    return f(b03), f(b04), f(b08), f(b11), np.full((N, N), scl, dtype=np.uint8)


def test_vegetation_tile_hand_computed_values():
    d = D.describe_tile(*flat(800, 500, 3500, 1800))
    assert d["usable"] == 1 and d["n_valid"] == N * N and d["valid_frac"] == 1.0
    assert d["ndvi_mean"] == pytest.approx((3500 - 500) / (3500 + 500))                # 0.75
    assert d["ndwi_mean"] == pytest.approx((800 - 3500) / (800 + 3500), abs=1e-6)       # -0.628
    assert d["ndbi_mean"] == pytest.approx((1800 - 3500) / (1800 + 3500), abs=1e-6)     # -0.321
    assert d["ndvi_std"] == pytest.approx(0, abs=1e-7) and d["ndvi_p10"] == pytest.approx(d["ndvi_p90"])
    assert (d["veg_frac"], d["dense_veg_frac"], d["water_frac"], d["bare_frac"], d["built_frac"]) == (1, 1, 0, 0, 0)
    assert d["edge_density"] == 0.0                                                      # uniform tile: no texture
    assert d["sr_nir_red_mean"] == pytest.approx(7.0) and d["sr_swir_nir_mean"] == pytest.approx(1800 / 3500, abs=1e-6)


def test_water_bare_and_built_signatures():
    water = D.describe_tile(*flat(900, 700, 300, 150, scl=6))
    assert water["water_frac"] == 1 and water["ndwi_mean"] > 0.4 and water["veg_frac"] == 0
    bare = D.describe_tile(*flat(2000, 2500, 2800, 3500, scl=5))
    assert bare["bare_frac"] == 1 and bare["veg_frac"] == 0 and bare["water_frac"] == 0
    built = D.describe_tile(*flat(1700, 1800, 2000, 2800, scl=5))
    assert built["built_frac"] == 1 and built["ndbi_mean"] > -0.05


def test_cloud_pixels_are_excluded_and_a_mostly_cloudy_tile_is_not_usable():
    b03, b04, b08, b11, scl = flat(800, 500, 3500, 1800)
    scl[:, : N // 2] = 9                                      # left half cloud (high probability)
    half = D.describe_tile(b03, b04, b08, b11, scl)
    assert half["usable"] == 1 and half["valid_frac"] == pytest.approx(0.5)
    scl[:, :] = 9
    assert D.describe_tile(b03, b04, b08, b11, scl) == {"descriptor_version": DESCRIPTOR_VERSION, "valid_frac": 0.0,
                                                         "n_valid": 0, "usable": 0}
    scl[:, :] = 4
    scl[: int(N * N * 0.96 / N), :] = 9                       # 4% valid < the 5% floor
    assert D.describe_tile(b03, b04, b08, b11, scl)["usable"] == 0


def test_texture_and_spread_respond_to_field_boundaries():
    b03, b04, b08, b11, scl = flat(800, 500, 3500, 1800)
    b08[:, N // 2:] = 1200                                   # a sharp NIR step: a field boundary
    d = D.describe_tile(b03, b04, b08, b11, scl)
    # one vertical step is seen by the 3x3 Sobel kernel in exactly the two columns either side of it: 2 / 64
    assert d["edge_density"] == pytest.approx(2 / N) and d["ndvi_std"] > 0.1 and d["ndvi_p10"] < d["ndvi_p90"]


def test_ratio_moments_are_clipped_on_dark_pixels():
    d = D.describe_tile(*flat(800, 0, 3500, 1800))           # red = 0 would give an infinite ratio
    assert d["sr_nir_red_mean"] == D.RATIO_CLIP


def test_every_field_the_schema_stores_is_produced_for_a_usable_tile():
    d = D.describe_tile(*flat(800, 500, 3500, 1800))
    missing = [f for f in ALL_FIELDS if f not in d and f not in ("dist_river_m", "dist_water_m")]
    assert not missing and set(SPECTRAL_FIELDS) <= set(d)


# ---------------------------------------------------------------- scene level (real GeoTIFFs)


def _write_scene(tmp_path, h=300, w=520, x_ul=399_970.0, y_ul=3_000_020.0):   # 10 m-aligned, offset NOT a multiple of 40 m
    sd = tmp_path / "scene"
    sd.mkdir()
    transform = Affine(10, 0, x_ul, 0, -10, y_ul)
    rng = np.random.default_rng(0)
    veg = lambda: np.clip(rng.normal(1, 0.02, (h, w)), 0.9, 1.1)
    data = {"B03": 800 * veg(), "B04": 500 * veg(), "B08": 3500 * veg(), "B11": 1800 * veg()}
    data["B08"][:, 260:] = 300                                # right half of the scene is water
    data["B03"][:, 260:] = 900
    prof = dict(driver="GTiff", height=h, width=w, count=1, crs="EPSG:32644", transform=transform, nodata=0)
    for b, a in data.items():
        with rasterio.open(sd / f"{b}.tif", "w", dtype="uint16", **prof) as dst:
            dst.write(a.astype(np.uint16), 1)
    with rasterio.open(sd / "SCL.tif", "w", dtype="uint8", **prof) as dst:
        dst.write(np.full((h, w), 4, np.uint8), 1)
    return sd


def test_scene_strips_produce_one_row_per_catalog_tile_and_a_global_grid_water_mask(tmp_path):
    sd = _write_scene(tmp_path)                                # 300 x 520 px -> 2 x 3 tiles (partial edges kept)
    specs = [D.TileSpec(f"s_r{r:03d}_c{c:03d}", r, c) for r in range(2) for c in range(3)]
    rows = D.describe_scene(sd, specs)
    assert [r["tile_id"] for r in rows] == [s.tile_id for s in specs]
    assert rows[0]["ndvi_mean"] > 0.6 and rows[0]["water_frac"] == 0           # vegetated left tile
    assert rows[2]["water_frac"] == 1 and rows[2]["ndvi_mean"] < 0               # right tile: all water
    assert rows[1]["water_frac"] == pytest.approx(252 / 256, abs=0.01)           # columns 256..259 vegetation, 260..511 water
    z = np.load(sd / D.WATER_MASK_NAME)
    step, gx, gy = int(z["step"]), int(z["gx_first"]), int(z["gy_first"])
    assert step == 4 and gx % 4 == 0 and gy % 4 == 0, "decimated samples must sit on the GLOBAL 40 m grid so dates align"
    mask = z["mask"]
    cols = np.arange(mask.shape[1]) * 4 + gx - round(399_970.0 / 10)           # back to scene pixel columns
    assert mask[:, cols < 260].sum() == 0 and mask[:, cols >= 260].all()
    assert cols[0] == 3 and (gy - round(-3_000_020.0 / 10)) == 2, "first samples are offset by the grid phase, not at 0"


def test_a_catalog_tile_outside_the_raster_is_an_error_not_a_silent_skip(tmp_path):
    sd = _write_scene(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        D.describe_scene(sd, [D.TileSpec("s_r009_c000", 9, 0)])
    with pytest.raises(ValueError, match="outside"):
        D.describe_scene(sd, [D.TileSpec("s_r000_c009", 0, 9)])


def test_misaligned_band_grids_are_rejected(tmp_path):
    sd = _write_scene(tmp_path)
    with rasterio.open(sd / "B11.tif", "r+") as ds:
        ds.transform = Affine(10, 0, 399_980.0, 0, -10, 3_000_020.0)            # shifted by one pixel
    with pytest.raises(ValueError, match="not on the B04 grid"):
        D.describe_scene(sd, [D.TileSpec("s_r000_c000", 0, 0)])


# ---------------------------------------------------------------- catalog migration


def _repo(tmp_path):
    repo = SQLiteMetadataRepository(tmp_path / "c.sqlite")
    poly = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    repo.register_collection(Collection("sentinel-2-l2a", "MSI", "Sentinel-2", ("B04",), 10.0))
    repo.register_scene(Scene("s", "sentinel-2-l2a", "P", "2024-01-01", poly))
    repo.register_observation(Observation("o", "s", "2024-01-01", poly))
    repo.add_tiles([Tile(f"t{i}", "o", 0, i, poly, 0.1 * i, faiss_id=i) for i in range(3)])
    return repo


def test_the_migration_leaves_every_existing_table_untouched(tmp_path):
    repo = _repo(tmp_path)
    before = repo.table_fingerprints()
    repo.upsert_tile_spectral([{"tile_id": "t0", "usable": 1, "valid_frac": 1.0, "n_valid": 10, "ndvi_mean": 0.4}])
    repo.set_tile_spectral_context([("t0", 120.0, 30.0)])
    assert repo.table_fingerprints() == before                    # tiles / scenes / observations / ... byte-for-byte
    assert repo.spectral_tile_ids(version=DESCRIPTOR_VERSION) == {"t0"}
    row = repo.get_tile_spectral("t0")
    assert row["ndvi_mean"] == 0.4 and row["dist_river_m"] == 120.0 and row["descriptor_version"] == DESCRIPTOR_VERSION


def test_redescribing_a_tile_keeps_its_region_context_and_a_reopen_is_a_no_op(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_tile_spectral([{"tile_id": "t1", "usable": 1, "valid_frac": 1.0, "n_valid": 10, "ndvi_mean": 0.4}])
    repo.set_tile_spectral_context([("t1", 77.0, 5.0)])
    repo.upsert_tile_spectral([{"tile_id": "t1", "usable": 1, "valid_frac": 1.0, "n_valid": 10, "ndvi_mean": 0.9}])
    assert repo.get_tile_spectral("t1")["dist_river_m"] == 77.0 and repo.get_tile_spectral("t1")["ndvi_mean"] == 0.9
    repo.close()
    again = SQLiteMetadataRepository(tmp_path / "c.sqlite")       # migration re-runs on open: must be idempotent
    assert again.get_tile_spectral("t1")["ndvi_mean"] == 0.9 and again.count_tiles() == 3


def test_list_and_chunked_lookup(tmp_path):
    repo = _repo(tmp_path)
    repo.upsert_tile_spectral([{"tile_id": f"t{i}", "usable": 1, "valid_frac": 1.0, "n_valid": 1, "ndvi_mean": i} for i in range(3)])
    assert set(repo.list_tile_spectral()) == {"t0", "t1", "t2"}
    assert set(repo.list_tile_spectral(["t2", "missing"])) == {"t2"}
    assert repo.get_tile_spectral("missing") is None


def test_a_row_for_a_nonexistent_tile_violates_the_foreign_key(tmp_path):
    repo = _repo(tmp_path)
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        repo.upsert_tile_spectral([{"tile_id": "ghost", "usable": 1, "valid_frac": 1.0, "n_valid": 1}])

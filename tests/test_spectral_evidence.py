"""Spectral evidence overlays: per-pixel NDVI/NDWI/NDBI that must agree, number for number, with the catalog's own descriptor."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from geoseek.change.indices import normalized_difference
from geoseek.spectral import evidence as ev
from geoseek.spectral.descriptor import describe_tile

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------- query relevance ------------------------


def test_water_queries_point_at_ndwi_built_queries_at_ndbi_and_vegetation_at_ndvi():
    def idx(q):
        return {m["index"] for m in ev.relevant_indices(q)["matches"]}

    assert idx("an open water reservoir or pond") == {"ndwi"}
    assert idx("a river with wide sandbars") >= {"ndwi"}
    assert idx("dense urban buildings and rooftops") == {"ndbi"}
    assert idx("trees and dense vegetation") == {"ndvi"}
    assert idx("bare dry open ground") == {"ndvi", "ndbi"}                 # bare ground = low NDVI and high NDBI
    r = ev.relevant_indices("something unrelated")
    assert r["matches"] == [] and "all three are shown" in r["note"]
    m = ev.relevant_indices("flooded paddy fields")["matches"]
    assert {x["index"] for x in m} == {"ndwi", "ndvi"} and all(x["terms"] and x["why"] for x in m)
    assert ev.relevant_indices(None)["matches"] == []


# ------------------------------------------------------------------------- colour mapping -------------------------


def test_colorize_hits_the_legend_stops_exactly_and_makes_invalid_pixels_transparent():
    for name, spec in ev.INDEX_SPECS.items():
        vals = np.array([[s[0] for s in spec["stops"]]], dtype=np.float32)
        out = ev.colorize(vals, np.ones(vals.shape, bool), name)
        for k, (_, hexc) in enumerate(spec["stops"]):
            assert tuple(int(x) for x in out[0, k, :3]) == ev._hex_rgb(hexc), (name, k)
        assert (out[..., 3] == 255).all()
        assert (ev.colorize(vals, np.zeros(vals.shape, bool), name)[..., 3] == 0).all()
    # outside the fixed domain the colour saturates at the end stop rather than extrapolating
    o = ev.colorize(np.array([[-1.0, 1.0]], dtype=np.float32), np.ones((1, 2), bool), "ndvi")
    assert tuple(o[0, 0, :3]) == ev._hex_rgb(ev.INDEX_SPECS["ndvi"]["stops"][0][1]) and tuple(o[0, 1, :3]) == ev._hex_rgb(ev.INDEX_SPECS["ndvi"]["stops"][-1][1])


# ------------------------------------------------------------------------- a synthetic scene ----------------------

N = 256
# a 2.56 km x 2.56 km footprint at 26.6 degrees north (256 px at 10 m): the patch size must come out of THIS geometry
TILE_WKT = "POLYGON((82.0 26.6, 82.02575 26.6, 82.02575 26.62315, 82.0 26.62315, 82.0 26.6))"


def _scene(tmp_path, *, cloudy=False, drop_band=None):
    d = tmp_path / "datasets" / "S2X_TEST_20240101_0_L2A"
    d.mkdir(parents=True)
    col = np.tile(np.arange(N), (N, 1))
    water = col < 100                                          # left strip: water-like (green > NIR)
    b03 = np.where(water, 1500, 900).astype("uint16")
    b04 = np.where(water, 800, 500).astype("uint16")
    b08 = np.where(water, 400, 3500).astype("uint16")
    b11 = np.where(water, 300, 1800).astype("uint16")
    scl = np.where(water, 6, 4).astype("uint8")                # 6 = water, 4 = vegetation: both valid
    if cloudy:
        scl[:] = 9                                             # cloud high probability: nothing valid
    else:
        scl[10:30, 200:220] = 9                                # a cloud patch inside the vegetated part
    for name, arr in {"B03": b03, "B04": b04, "B08": b08, "B11": b11, "SCL": scl}.items():
        if name == drop_band:
            continue
        with rasterio.open(d / f"{name}.tif", "w", driver="GTiff", width=N, height=N, count=1, dtype=arr.dtype, crs="EPSG:32644",
                           transform=from_origin(600000, 2950000, 10, 10)) as ds:
            ds.write(arr, 1)
    repo = SimpleNamespace(
        get_tile=lambda t: SimpleNamespace(observation_id="obs1", row=0, col=0, geom_wkt_4326=TILE_WKT) if t == "tile1" else None,
        get_observation=lambda o: SimpleNamespace(dataset_dir=d.name, observation_id=o, scene_id="S2X_TEST_20240101_0_L2A", acquired_at="2024-01-01"))
    return repo, tmp_path / "datasets", (b03, b04, b08, b11, scl)


def test_statistics_equal_the_catalog_descriptor_and_classes_use_its_rules(tmp_path):
    ev._window.cache_clear()
    repo, ds, (b03, b04, b08, b11, scl) = _scene(tmp_path)
    out = ev.tile_spectral(repo, ds, "tile1", "an open water reservoir")
    want = describe_tile(b03, b04, b08, b11, scl)
    assert out["usable"] and out["gsd_m"] == 10.0 and (out["width_px"], out["height_px"]) == (N, N)
    for name in ("ndvi", "ndwi", "ndbi"):
        for k in ("mean", "std", "p10", "p50", "p90"):
            assert out["layers"][name]["stats"][k] == pytest.approx(want[f"{name}_{k}"], abs=1e-9)
    assert out["classes"]["water_frac"] == pytest.approx(want["water_frac"]) and out["classes"]["veg_frac"] == pytest.approx(want["veg_frac"])
    # ground truth by construction: water strip 100 px wide of 256, minus nothing for water; clouds only remove vegetated pixels
    valid = N * N - 20 * 20
    assert out["classes"]["water_frac"] == pytest.approx(100 * N / valid, abs=1e-6)
    assert out["valid_fraction"] == pytest.approx(valid / (N * N))
    assert [m["index"] for m in out["relevance"]["matches"]] == ["ndwi"]
    assert out["layers"]["ndwi"]["image_url"] == "/ui/tiles/tile1/spectral/ndwi.png" and out["embedding_patch_m"] == pytest.approx(2560 / 7, abs=6) and f"~{out['embedding_patch_m']} m patches" in out["caveat"]


def test_png_pixels_are_the_index_colours_at_ten_metres_with_clouds_transparent(tmp_path):
    ev._window.cache_clear(); ev._png.cache_clear()
    repo, ds, (b03, b04, b08, b11, scl) = _scene(tmp_path)
    for name, (a, b) in {"ndwi": (b03, b08), "ndvi": (b08, b04), "ndbi": (b11, b08)}.items():
        png = ev.tile_index_png(repo, ds, "tile1", name)
        im = np.array(Image.open(io.BytesIO(png)))
        assert im.shape == (N, N, 4)                                       # one pixel per 10 m pixel, never resampled
        v = normalized_difference(a, b)
        want = ev.colorize(v, np.isin(scl, ev.SCL_VALID), name)
        assert (im == want).all(), name
        assert (im[10:30, 200:220, 3] == 0).all() and (im[100:110, 150:160, 3] == 255).all()
    ndwi = np.array(Image.open(io.BytesIO(ev.tile_index_png(repo, ds, "tile1", "ndwi"))))
    assert ndwi[50, 50, 2] > ndwi[50, 50, 0] and ndwi[50, 50, 0] < 100         # the water strip is dark blue, not brown
    with pytest.raises(ValueError):
        ev.tile_index_png(repo, ds, "tile1", "evi")


def test_a_fully_clouded_tile_reports_no_statistics_instead_of_inventing_them(tmp_path):
    ev._window.cache_clear()
    repo, ds, _ = _scene(tmp_path, cloudy=True)
    out = ev.tile_spectral(repo, ds, "tile1")
    assert out["usable"] is False and out["classes"] is None and out["unusable_reason"]
    assert all(l["stats"] is None for l in out["layers"].values())


def test_tiles_without_nir_swir_are_a_clear_404_not_a_fabrication(tmp_path):
    ev._window.cache_clear()
    repo, ds, _ = _scene(tmp_path, drop_band="B11")
    with pytest.raises(ev.SpectralUnavailable, match="B11"):
        ev.tile_spectral(repo, ds, "tile1")
    with pytest.raises(KeyError):
        ev.tile_spectral(repo, ds, "unknown-tile")


def test_http_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from geoseek.search import api

    ev._window.cache_clear(); ev._png.cache_clear()
    repo, ds, _ = _scene(tmp_path)
    monkeypatch.setattr(api, "_engine", SimpleNamespace(repo=repo, settings=SimpleNamespace(datasets_dir=ds)))
    c = TestClient(api.app)
    j = c.get("/ui/tiles/tile1/spectral", params={"q": "trees and dense vegetation"}).json()
    assert j["relevance"]["matches"][0]["index"] == "ndvi" and set(j["layers"]) == {"ndvi", "ndwi", "ndbi"}
    r = c.get("/ui/tiles/tile1/spectral/ndbi.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and Image.open(io.BytesIO(r.content)).size == (N, N)
    assert c.get("/ui/tiles/tile1/spectral/evi.png").status_code == 400
    assert c.get("/ui/tiles/nope/spectral").status_code == 404


# ------------------------------------------------------------------------- against the production catalog ----------


def test_overlay_statistics_match_the_stored_tile_spectral_rows_of_real_tiles():
    db = ROOT / "data" / "index" / "tiles.sqlite"
    ds = ROOT / "data" / "datasets"
    if not db.is_file() or not ds.is_dir():
        pytest.skip("production catalog / staged scenes not on this machine")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT t.tile_id, t.observation_id, t.row, t.col, o.dataset_dir, o.scene_id, o.acquired_at, s.ndvi_mean, s.ndwi_mean, s.ndbi_mean, "
        "s.ndvi_p90, s.water_frac, s.veg_frac, s.built_frac FROM tile_spectral s JOIN tiles t ON t.tile_id = s.tile_id "
        "JOIN observations o ON o.observation_id = t.observation_id WHERE s.usable = 1 "
        "AND o.scene_id LIKE 'S2%' ORDER BY (t.row * 7919 + t.col * 104729) % 1009 LIMIT 40").fetchall()
    con.close()
    byid = {r[0]: r for r in rows}
    repo = SimpleNamespace(
        get_tile=lambda t: SimpleNamespace(observation_id=byid[t][1], row=byid[t][2], col=byid[t][3]) if t in byid else None,
        get_observation=lambda o: next(SimpleNamespace(dataset_dir=r[4], observation_id=o, scene_id=r[5], acquired_at=r[6]) for r in rows if r[1] == o))
    checked = 0
    for tid, r in byid.items():
        if not all((ds / (r[4] or r[1]) / f"{b}.tif").is_file() for b in ev.BANDS):
            continue
        out = ev.tile_spectral(repo, ds, tid)
        assert out["usable"], tid
        assert out["layers"]["ndvi"]["stats"]["mean"] == pytest.approx(r[7], abs=1e-6), tid
        assert out["layers"]["ndwi"]["stats"]["mean"] == pytest.approx(r[8], abs=1e-6), tid
        assert out["layers"]["ndbi"]["stats"]["mean"] == pytest.approx(r[9], abs=1e-6), tid
        assert out["layers"]["ndvi"]["stats"]["p90"] == pytest.approx(r[10], abs=1e-6), tid
        assert out["classes"]["water_frac"] == pytest.approx(r[11], abs=1e-6) and out["classes"]["veg_frac"] == pytest.approx(r[12], abs=1e-6)
        assert out["classes"]["built_frac"] == pytest.approx(r[13], abs=1e-6)
        checked += 1
        if checked >= 12:
            break
    assert checked >= 5, "too few catalogued tiles with staged NIR/SWIR to check against"

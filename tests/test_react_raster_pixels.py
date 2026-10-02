"""Browser-side raster maths (frontend-react/src/lib/rasterPixels.ts + rasterMath.ts) verified against numpy / rasterio.

The TypeScript runs under Node on REAL GeoTIFFs cut from the archive's staged Sentinel-2 scenes (tests/raster_fixtures.py). Every number
it prints - band statistics, NDVI / NDWI statistics, stretched composite samples, the Web-Mercator resample, the indicative visual
difference - is compared with an independent computation here, the same way the spectral-evidence panel is checked against
tile_spectral. Percentiles and the population standard deviation follow numpy's defaults, so the agreement is expected to be exact
to floating-point precision, not "close".
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from raster_fixtures import available, build

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "frontend-react"
NODE = shutil.which("node")

pytestmark = [pytest.mark.skipif(NODE is None, reason="node is not installed (the console build tooling is optional)"),
              pytest.mark.skipif(not available(), reason="staged Sentinel-2 scenes are not on this machine")]


def _node(*args: str) -> dict:
    r = subprocess.run([NODE, "tools/selftest-pixels.mjs", *args], capture_output=True, text=True, cwd=FRONT, timeout=300)
    assert r.returncode == 0, r.stderr[-2500:]
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("rasterfx")
    ref = build(d)
    return ref, _node(str(d / "reference.json"))


def test_band_statistics_equal_numpy(run):
    ref, got = run
    for date in ("2019-03-30", "2024-03-08"):
        for i, name in enumerate(("B02", "B03", "B04", "B08")):
            want, have = ref["files"][date]["bands"][name], got["files"][date]["bandStats"][i]
            assert have["n"] == want["n"], (date, name)
            for k in ("mean", "std", "min", "max", "p2", "p10", "p50", "p90", "p98"):
                assert have[k] == pytest.approx(want[k], rel=1e-9, abs=1e-9), (date, name, k)


def test_spectral_index_statistics_equal_numpy(run):
    ref, got = run
    for date in ("2019-03-30", "2024-03-08"):
        for idx in ("ndvi", "ndwi"):
            want, have = ref["files"][date]["indices"][idx], got["files"][date]["indices"][idx]
            assert have["n"] == want["n"], (date, idx)
            for k in ("mean", "std", "min", "max", "p2", "p10", "p50", "p90", "p98"):
                assert have[k] == pytest.approx(want[k], abs=2e-6), (date, idx, k)         # the browser holds an index as float32


def test_percentile_stretched_true_colour_samples_equal_numpy(run):
    ref, got = run
    for date in ("2019-03-30", "2024-03-08"):
        wants = ref["files"][date]["composite_samples"]
        assert wants, "fixture has no valid sample pixels"
        for w, h in zip(wants, got["files"][date]["compositeSamples"]):
            assert (w["row"], w["col"]) == (h["row"], h["col"])
            assert h["rgb"] == w["rgb"], (date, w)


def test_band_roles_and_defaults(run):
    _, got = run
    four = got["files"]["2024-03-08"]
    assert four["rgb"] == [2, 1, 0] and four["roles"] == {"blue": 0, "green": 1, "red": 2, "nir": 3} and not four["ndbiAvailable"]   # B04/B03/B02; no SWIR
    three = got["files"]["3band"]
    assert three["bandNames"] == ["B04", "B03", "B02"] and three["roles"] == {"red": 0, "green": 1, "blue": 2} and three["rgb"] == [0, 1, 2]
    assert "band descriptions" in three["basis"]
    one = got["files"]["1band"]
    assert one["rgb"] is None and one["roles"] == {}
    assert got["files"]["2019-03-30"]["acquisition"].startswith("2019-03-30")


def _reproject(path, band, grid):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    dst = np.full((grid["h"], grid["w"]), np.nan, dtype="float32")
    with rasterio.open(path) as ds:
        reproject(ds.read(band).astype("float32"), dst, src_transform=ds.transform, src_crs=ds.crs, src_nodata=0,
                  dst_transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]), dst_crs="EPSG:3857", dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    return dst


def _pyproj_nearest(path, band, grid):
    """Exact reference: every destination pixel centre -> EPSG:32644 with pyproj -> the source pixel that contains it."""
    import pyproj
    import rasterio

    jj, ii = np.mgrid[0:grid["h"], 0:grid["w"]]
    mx = grid["x0"] + (ii + 0.5) * grid["res"]
    my = grid["y0"] - (jj + 0.5) * grid["res"]
    with rasterio.open(path) as ds:
        x, y = pyproj.Transformer.from_crs("EPSG:3857", ds.crs, always_xy=True).transform(mx, my)
        src = ds.read(band).astype("float32")
        a, b, c, d, e, f = ds.transform.a, ds.transform.b, ds.transform.c, ds.transform.d, ds.transform.e, ds.transform.f
    det = a * e - b * d
    col = np.floor((e * (x - c) - b * (y - f)) / det).astype(int)
    row = np.floor((-d * (x - c) + a * (y - f)) / det).astype(int)
    ok = (col >= 0) & (row >= 0) & (col < src.shape[1]) & (row < src.shape[0])
    out = np.full(mx.shape, np.nan, dtype="float32")
    out[ok] = src[row[ok], col[ok]]
    out[out == 0] = np.nan                                                      # nodata
    return out


def test_mercator_resample_equals_an_exact_pyproj_nearest_reference(run):
    ref, got = run
    w = got["warp"]
    for key, band in (("b04_2019", "2019-03-30"), ("b04_2024", "2024-03-08")):
        mine = np.frombuffer(base64.b64decode(w[key]), dtype="float32").reshape(w["grid"]["h"], w["grid"]["w"])
        want = _pyproj_nearest(ref["files"][band]["path"], 3, w["grid"])
        assert np.isfinite(want).sum() > 50_000
        assert (np.isfinite(mine) == np.isfinite(want)).mean() > 0.9999         # same footprint
        both = np.isfinite(mine) & np.isfinite(want)
        assert (mine[both] == want[both]).mean() > 0.9999, band                 # same source pixel (the browser's UTM + Mercator maths = pyproj's)


def test_mercator_resample_agrees_with_gdal_to_its_warp_tolerance(run):
    """GDAL's warper uses an approximate transformer (error threshold 0.125 source px), so it can pick the neighbouring source pixel
    for pixels that fall near a boundary; the exact pyproj reference above is the strict check, this is the cross-tool sanity check."""
    ref, got = run
    w = got["warp"]
    mine = np.frombuffer(base64.b64decode(w["b04_2019"]), dtype="float32").reshape(w["grid"]["h"], w["grid"]["w"])
    gdal = _reproject(ref["files"]["2019-03-30"]["path"], 3, w["grid"])
    both = np.isfinite(mine) & np.isfinite(gdal)
    assert (np.isfinite(mine) == np.isfinite(gdal)).mean() > 0.999
    assert (mine[both] == gdal[both]).mean() > 0.92


def test_visual_difference_agrees_with_an_independent_numpy_computation(run):
    ref, got = run
    w = got["warp"]
    lum = {}
    for date in ("2019-03-30", "2024-03-08"):
        p = ref["files"][date]["path"]
        r, g, b = (_pyproj_nearest(p, k, w["grid"]) for k in (3, 2, 1))        # B04, B03, B02
        lum[date] = (0.299 * r + 0.587 * g + 0.114 * b).astype("float32")
    a, b = lum["2019-03-30"], lum["2024-03-08"]
    both = np.isfinite(a) & np.isfinite(b)
    ra = np.percentile(a[both].astype("float64"), [2, 98]); rb = np.percentile(b[both].astype("float64"), [2, 98])
    x = np.clip((a[both] - ra[0]) / (ra[1] - ra[0]), 0, 1); y = np.clip((b[both] - rb[0]) / (rb[1] - rb[0]), 0, 1)
    d = y - x
    want = {"n": int(both.sum()), "meanAbs": float(np.abs(d).mean()), "shareAbove": float((np.abs(d) > 0.25).mean()), "correlation": float(np.corrcoef(x, y)[0, 1])}
    have = w["diff"]
    assert abs(have["n"] - want["n"]) <= 0.0005 * want["n"]
    assert have["meanAbs"] == pytest.approx(want["meanAbs"], abs=2e-3)
    assert have["shareAbove"] == pytest.approx(want["shareAbove"], abs=2e-3)
    assert have["correlation"] == pytest.approx(want["correlation"], abs=2e-3)
    assert have["rangeA"] == pytest.approx(list(ra), rel=2e-3) and have["rangeB"] == pytest.approx(list(rb), rel=2e-3)


def test_preview_modes_native_overview_and_decimated(tmp_path):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin

    def write(path, w, h, overviews):
        yy, xx = np.mgrid[0:h, 0:w]
        with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint16", crs="EPSG:32644",
                           transform=from_origin(600000, 2950000, 10, 10), tiled=True, blockxsize=256, blockysize=256) as ds:
            ds.write((xx + 3 * yy).astype("uint16"), 1)
            if overviews:
                ds.build_overviews([2, 4, 8], Resampling.nearest)
    big_ov, big_plain, small = tmp_path / "ov.tif", tmp_path / "plain.tif", tmp_path / "small.tif"
    write(big_ov, 3200, 3200, True)
    write(big_plain, 3200, 2000, False)
    write(small, 700, 500, False)

    s = _node("--decode", str(small))
    assert (s["mode"], s["width"], s["height"]) == ("native", 700, 500) and s["scale"] == [1, 1]
    with rasterio.open(small) as ds:
        assert s["row0"] == ds.read(1)[0].tolist()                                  # exactly the file's pixels: nothing resampled

    o = _node("--decode", str(big_ov))
    assert o["mode"] == "overview" and (o["width"], o["height"]) == (800, 800) and o["scale"] == [0.25, 0.25]   # the 1:4 overview: the largest that fits 1536 px
    with rasterio.open(big_ov) as ds:
        assert ds.overviews(1) == [2, 4, 8]
        ov = ds.read(1, out_shape=(800, 800), resampling=Resampling.nearest)       # GDAL serves this from the same overview level
        assert o["rowMid"] == ov[400].tolist()

    d = _node("--decode", str(big_plain))
    assert d["mode"] == "decimated" and max(d["width"], d["height"]) == 1536 and (d["fullWidth"], d["fullHeight"]) == (3200, 2000)
    assert "decimation" in d["note"]
    k = 3200 / d["width"]
    # nearest decimation of value = col + 3 * row: row 0 holds the source column picked for each preview column, within one source pixel of the centre
    assert all(abs(v - (i + 0.5) * k) <= k for i, v in enumerate(d["row0"]))


def test_spectral_profile_over_the_shared_ground_equals_an_independent_numpy_pyproj_computation(run):
    """The upload-mode profile: footprint overlap, grid choice, resampling, masking and every per-band / NDVI / NDWI statistic."""
    ref, got = run
    want, have = ref["profile"], got["profile"]
    assert have["members"] == ["2019-03-30", "2024-03-08"]
    assert have["z"] == want["z"] and have["grid"]["w"] == want["grid"]["w"] and have["grid"]["h"] == want["grid"]["h"]
    assert have["grid"]["x0"] == pytest.approx(want["grid"]["x0"], abs=1e-6) and have["grid"]["res"] == pytest.approx(want["grid"]["res"], rel=1e-12)
    assert have["box"] == pytest.approx(want["box"], abs=2e-6)
    assert have["cells"] == want["cells"]
    assert {"blue", "green", "red", "nir", "ndvi", "ndwi"} <= set(have["keys"]) and "ndbi" not in have["keys"]      # no SWIR band -> no NDBI offered
    for date in ("2019-03-30", "2024-03-08"):
        for metric, w in want["files"][date].items():
            h = have["files"][date][metric]
            assert h["n"] == w["n"], (date, metric)                                   # the same valid pixels, to the pixel
            for k in ("mean", "std", "p10", "p50", "p90"):
                if not metric.startswith("nd"):
                    assert h[k] == pytest.approx(w[k], rel=1e-9, abs=1e-9), (date, metric, k)       # integer DNs: exact to floating point
                else:
                    assert h[k] == pytest.approx(w[k], abs=2e-6), (date, metric, k)                 # an index is held as float32 in the browser

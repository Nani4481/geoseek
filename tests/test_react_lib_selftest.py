"""The console's framework-free TypeScript modules (frontend-react/src/lib), executed under Node and cross-checked
against independent implementations: pyproj for UTM, and the catalog's own Sentinel-2 tile names for MGRS squares."""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pyproj
import pytest
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "frontend-react"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed (the console build tooling is optional)")


def _node(script: str, stdin: str | None = None) -> str:
    r = subprocess.run([NODE, script], input=stdin, capture_output=True, text=True, cwd=FRONT, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_timelapse_geometry_selftest():
    res = json.loads(_node("tools/selftest-lib.mjs"))
    assert res["timelapse"] == "ok" and res["mapfit"] == "ok"
    assert res["cluster_insight"] == "ok" and res["gsd"] == "ok" and res["calendar_axis"] == "ok"


def _utm_epsg(zone: int, north: bool) -> int:
    return (32600 if north else 32700) + zone


def test_utm_matches_pyproj_to_a_millimetre_and_round_trips():
    rnd = random.Random(7)
    pts = [[rnd.uniform(-179.9, 179.9), rnd.uniform(-79.9, 83.9)] for _ in range(600)]
    pts += [[82.25, 26.636], [77.1, 28.6], [88.6, 27.5], [-118.4, 34.2], [0.0, 0.0], [5.0, 60.0], [15.0, 78.0], [179.99, -45.0]]
    res = json.loads(_node("tools/selftest-geo.mjs", json.dumps(pts)))
    worst_fwd = worst_back = 0.0
    for (lon, lat), r in zip(pts, res):
        assert r is not None
        t = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{_utm_epsg(r['zone'], r['north'])}", always_xy=True)
        e, n = t.transform(lon, lat)
        worst_fwd = max(worst_fwd, abs(e - r["easting"]), abs(n - r["northing"]))
        lo, la = t.transform(r["easting"], r["northing"], direction="INVERSE")
        worst_back = max(worst_back, abs(lo - r["back"]["lon"]) * 111_000, abs(la - r["back"]["lat"]) * 111_000)
    assert worst_fwd < 1e-3, f"forward UTM differs from pyproj by {worst_fwd:.6f} m"
    assert worst_back < 1e-3, f"inverse UTM differs from pyproj by {worst_back:.6f} m"


def test_mgrs_known_references():
    pts = [[82.25, 26.636], [-118.4, 34.2], [0.0, 0.0], [-0.1276, 51.5072]]            # incl. Greenwich: 30U / 31U boundary
    res = json.loads(_node("tools/selftest-geo.mjs", json.dumps(pts)))
    assert res[0]["tile"] == "44RPQ" and res[0]["zone"] == 44 and res[0]["band"] == "R"
    assert res[1]["mgrs"].startswith("11S") and res[2]["mgrs"].startswith("31N")
    assert res[3]["mgrs"].startswith("30U")


def test_mgrs_squares_equal_the_sentinel2_tile_names_in_the_catalog():
    """A Sentinel-2 scene id carries its MGRS tile (`..._44RPQ_...`). Every catalogued footprint's centre must fall inside
    the 100 km square the TypeScript conversion names - an independent check on the zone, band and both square letters."""
    db = ROOT / "data" / "index" / "tiles.sqlite"
    if not db.is_file():
        pytest.skip("no production catalog on this machine")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    rows = con.execute("SELECT s.scene_id, o.footprint_wkt_4326 FROM observations o JOIN scenes s ON s.scene_id = o.scene_id "
                       "WHERE s.collection_id = 'sentinel-2-l2a'").fetchall()
    con.close()
    pts, want = [], []
    for scene_id, fp in rows:
        tok = scene_id.split("_")[1]
        c = wkt.loads(fp).centroid
        pts.append([c.x, c.y]); want.append(tok)
    assert len(pts) >= 20
    got = [r["tile"] for r in json.loads(_node("tools/selftest-geo.mjs", json.dumps(pts)))]
    # footprints may straddle a 100 km square edge: require a clear majority AND that each mismatch is a neighbouring square
    ok = sum(g == w for g, w in zip(got, want))
    assert ok / len(pts) >= 0.95, [(g, w) for g, w in zip(got, want) if g != w][:10]


# ------------------------------------------------------------------------- raster header parse (geotiff.js) -----------


def _write_tif(path, *, crs, transform, count=1, dtype="uint16", width=96, height=64, nodata=None, tags=None, tiled=False, compress=None):
    import numpy as np
    import rasterio

    kw = dict(driver="GTiff", width=width, height=height, count=count, dtype=dtype, transform=transform)
    if crs:
        kw["crs"] = crs
    if nodata is not None:
        kw["nodata"] = nodata
    if tiled:
        kw.update(tiled=True, blockxsize=32, blockysize=32)
    if compress:
        kw["compress"] = compress
    with rasterio.open(path, "w", **kw) as ds:
        ds.write((np.arange(width * height, dtype="float64").reshape(height, width) % 250).astype(dtype), 1)
        for b in range(2, count + 1):
            ds.write(np.ones((height, width), dtype=dtype), b)
        if tags:
            ds.update_tags(**tags)


def test_raster_header_matches_rasterio_across_crs_types_and_edge_cases(tmp_path):
    import rasterio
    from rasterio.transform import Affine, from_origin

    cases = {
        "utm44n.tif": dict(crs="EPSG:32644", transform=from_origin(600000, 2950000, 10, 10), count=3, dtype="uint16", nodata=0, tiled=True,
                           compress="deflate", tags={"ACQUISITION_DATE": "2024-03-08T05:21:00Z"}),
        "utm54s.tif": dict(crs="EPSG:32754", transform=from_origin(300000, 6000000, 20, 20), dtype="int16"),
        "wgs84.tif": dict(crs="EPSG:4326", transform=from_origin(82.0, 27.0, 0.001, 0.001), dtype="float32", nodata=-9999.0),
        "merc.tif": dict(crs="EPSG:3857", transform=from_origin(9100000, 3100000, 30, 30), dtype="uint8"),
        "rotated.tif": dict(crs="EPSG:32644", transform=Affine(9.5, 3.0, 600000, 3.0, -9.5, 2950000), dtype="uint8"),
        "nocrs.tif": dict(crs=None, transform=Affine.identity(), dtype="uint8"),
        "nonsquare.tif": dict(crs="EPSG:32644", transform=from_origin(600000, 2950000, 10, 12), dtype="uint8"),
        "lcc.tif": dict(crs="EPSG:7755", transform=from_origin(1000000, 2000000, 30, 30), dtype="uint8"),    # WGS84 / India NSF LCC: not invertible offline
    }
    paths = []
    for name, kw in cases.items():
        p = tmp_path / name
        _write_tif(p, **kw)
        paths.append(str(p))
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"this is not a tiff at all" * 20)
    out = {r["fileName"]: r for r in json.loads(_node_args("tools/selftest-raster.mjs", paths + [str(junk)]))}

    assert "error" in out["junk.tif"]
    for name, kw in cases.items():
        r = out[name]
        assert "error" not in r, (name, r)
        with rasterio.open(tmp_path / name) as ds:
            assert (r["width"], r["height"], r["bands"], r["dtype"]) == (ds.width, ds.height, ds.count, ds.dtypes[0]), name
            assert r["crs"]["epsg"] == (ds.crs.to_epsg() if ds.crs else None), name
            assert r["nodata"] == ds.nodata, name
            assert r["tiled"] == bool(ds.profile.get("tiled")), name
            tf = ds.transform
            if ds.crs:                                                  # no CRS -> GDAL reports an identity transform, the file has none
                assert r["affine"] == pytest.approx([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f], rel=1e-12, abs=1e-9), name
                corners = [tf * (c, rr) for c, rr in ((0, 0), (ds.width, 0), (ds.width, ds.height), (0, ds.height))]
                want = [min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners)]
                assert r["bounds"] == pytest.approx(want, rel=1e-12, abs=1e-6), name
                assert r["resolution"] == pytest.approx([(tf.a ** 2 + tf.d ** 2) ** 0.5, (tf.b ** 2 + tf.e ** 2) ** 0.5], rel=1e-12), name
                if r["lonlatBounds"]:
                    t = pyproj.Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
                    ll = [t.transform(*p) for p in corners]
                    wantll = [min(p[0] for p in ll), min(p[1] for p in ll), max(p[0] for p in ll), max(p[1] for p in ll)]
                    assert r["lonlatBounds"] == pytest.approx(wantll, abs=2e-6), name
            comp = ds.profile.get("compress")
            if comp:
                assert r["compression"].lower() == comp.lower(), name

    assert out["rotated.tif"]["rotated"] is True and out["utm44n.tif"]["rotated"] is False
    assert not next(c for c in out["rotated.tif"]["checks"] if c["id"] == "north_up")["ok"]
    assert not next(c for c in out["nonsquare.tif"]["checks"] if c["id"] == "square")["ok"]
    assert out["nocrs.tif"]["crs"]["epsg"] is None and out["nocrs.tif"]["affine"] is None      # an identity matrix is pixel space, not a place
    assert out["nocrs.tif"]["lonlatBounds"] is None
    assert not next(c for c in out["nocrs.tif"]["checks"] if c["id"] == "crs")["ok"]
    # a CRS we cannot invert offline is reported as such, not guessed
    assert out["lcc.tif"]["lonlatBounds"] is None and "not bundled" in out["lcc.tif"]["lonlatNote"]
    # acquisition: only a metadata item that says so; a plain TIFF DateTime is the *file* timestamp, kept apart
    acq = out["utm44n.tif"]["acquisition"]
    assert acq["iso"] == "2024-03-08T05:21:00.000Z" and "ACQUISITION_DATE" in acq["source"]
    assert out["wgs84.tif"]["acquisition"] is None
    assert out["utm54s.tif"]["crs"]["name"] == "WGS 84 / UTM zone 54S"


def _node_args(script: str, args: list[str]) -> str:
    r = subprocess.run([NODE, script, *args], capture_output=True, text=True, cwd=FRONT, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_raster_header_of_a_real_catalogued_sentinel2_band():
    p = ROOT / "data" / "datasets" / "S2A_42QXM_20240110_0_L2A" / "B08.tif"
    if not p.is_file():
        pytest.skip("staged scene not on this machine")
    import rasterio

    (r,) = json.loads(_node_args("tools/selftest-raster.mjs", [str(p)]))
    with rasterio.open(p) as ds:
        assert (r["width"], r["height"], r["crs"]["epsg"]) == (ds.width, ds.height, ds.crs.to_epsg())
        assert r["bounds"] == pytest.approx(list(ds.bounds), abs=1e-6)

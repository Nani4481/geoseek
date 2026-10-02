"""The local raster basemap: geometry checked against an independent projection, honest gaps, capped caches, no disk writes.

Everything runs on small synthetic rasters and a fake catalog (no production data needed). What is asserted is that tiles are
assembled from the imagery the catalog references, land where an independent pyproj/shapely computation says they must,
stay transparent where nothing is staged, and that the in-memory caches never exceed their hard caps.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyproj
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from shapely import contains_xy
from shapely.geometry import Polygon

from geoseek.analyst import basemap as bm
from geoseek.catalog.entities import Collection, Observation, Scene, TileRecord

TO_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
UTM44 = "EPSG:32644"


# --------------------------------------------------------------------------- fakes


class FakeRepo:
    def __init__(self, tiles=(), observations=(), scenes=(), collections=()):
        self.tiles = list(tiles)
        self.obs = {o.observation_id: o for o in observations}
        self.scenes = {s.scene_id: s for s in scenes}
        self.colls = {c.collection_id: c for c in collections}

    def iter_tile_records(self):
        return iter(self.tiles)

    def query_tiles(self, *, bbox=None, collection=None, **_):
        w, s, e, n = bbox
        out = []
        for t in self.tiles:
            if collection and t.collection_id != collection:
                continue
            if Polygon(_ring(t.geom_wkt_4326)).intersects(Polygon([(w, s), (e, s), (e, n), (w, n)])):
                out.append(t)
        return out

    def get_observation(self, oid):
        return self.obs.get(oid)

    def get_scene(self, sid):
        return self.scenes.get(sid)

    def get_collection(self, cid):
        return self.colls.get(cid)


def _ring(wkt: str):
    nums = [float(v) for v in bm._NUM_RE.findall(wkt)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _utm_square_wkt(easting, northing, size_m, crs=UTM44):
    tr = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    pts = [tr.transform(easting + dx, northing + dy) for dx, dy in ((0, size_m), (size_m, size_m), (size_m, 0), (0, 0), (0, size_m))]
    return "POLYGON ((" + ", ".join(f"{lo:.7f} {la:.7f}" for lo, la in pts) + "))", pts


def _tile_xyz_containing(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def _jpeg(rgb, size=256):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _decode(body):
    im = Image.open(io.BytesIO(body)).convert("RGBA")
    return np.asarray(im)


def _record(tid, obs, wkt, cloud, date="2024-01-10", coll=bm.S2_COLLECTION):
    return TileRecord(tile_id=tid, observation_id=obs, scene_id=obs, collection_id=coll, sensor="MSI", platform="Sentinel-2A",
                      acq_date=date, geom_wkt_4326=wkt, cloud_fraction=cloud, faiss_id=None)


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    monkeypatch.setattr(bm, "_tile_cache", bm.ByteLRU(bm.TILE_CACHE_CAP_BYTES))
    monkeypatch.setattr(bm, "_overview_cache", bm.ByteLRU(bm.OVERVIEW_CAP_BYTES))
    bm._archive_cache.update(repo=None, value=None)
    yield
    bm._archive_cache.update(repo=None, value=None)


# --------------------------------------------------------------------------- maths, caches


def test_web_mercator_tile_bounds_match_pyproj():
    for z, x, y in ((0, 0, 0), (5, 22, 13), (14, 11935, 6933), (19, 1, 2)):
        b = bm.tile_bounds_3857(z, x, y)
        w, s, e, n = bm.tile_bbox_lonlat(z, x, y)
        ex0, ey0 = TO_3857.transform(w, s)
        ex1, ey1 = TO_3857.transform(e, n)
        assert (ex0, ey0, ex1, ey1) == pytest.approx(b, abs=1e-3)
        assert b[2] - b[0] == pytest.approx(2 * bm.ORIGIN / 2 ** z)


def test_tiles_outside_the_world_grid_are_rejected():
    for bad in ((3, 8, 0), (3, 0, 8), (-1, 0, 0), (23, 0, 0)):
        with pytest.raises(bm.BasemapError):
            bm.validate_zxy(*bad)


def test_byte_lru_enforces_a_hard_cap_and_evicts_least_recently_used():
    c = bm.ByteLRU(1000)
    for i in range(10):
        c.put(i, f"v{i}", 300)
        assert c.bytes <= 1000
    st = c.stats()
    assert st["entries"] == 3 and st["bytes"] == 900 and st["peak_bytes"] <= 1000 and st["evictions"] == 7
    assert c.get(9) == "v9" and c.get(0) is None
    c.get(7)                                      # touch 7 -> 8 is now the oldest
    c.put("new", "n", 300)
    assert c.get(8) is None and c.get(7) == "v7"
    c.put("huge", "x", 5000)                       # larger than the whole cap: never admitted, never evicts anything
    assert c.get("huge") is None and c.stats()["bytes"] <= 1000


# --------------------------------------------------------------------------- Sentinel-2 thumbnail composite


def _one_tile_repo(cloud=0.0, obs="S2A_44RPQ_20240110_0_L2A", e0=500_000.0, n0=2_950_000.0):
    wkt, pts = _utm_square_wkt(e0, n0, 2560.0)
    return FakeRepo(tiles=[_record("t1", obs, wkt, cloud)]), pts


def test_thumbnail_lands_where_an_independent_projection_says_and_nothing_else_is_painted():
    repo, pts = _one_tile_repo()
    lon, lat = np.mean([p[0] for p in pts[:4]]), np.mean([p[1] for p in pts[:4]])
    z = 13
    x, y = _tile_xyz_containing(lon, lat, z)
    body, mime, meta = bm.render_sentinel2(repo, lambda tid: _jpeg((200, 40, 40)), z, x, y)
    assert meta["status"] == "ok" and meta["composited"] == 1
    px = _decode(body)
    painted = px[..., 3] > 0
    mx0, my0, mx1, my1 = bm.tile_bounds_3857(z, x, y)
    poly = Polygon([((a - mx0) / (mx1 - mx0) * 256, (my1 - b) / (my1 - my0) * 256) for a, b in (TO_3857.transform(*p) for p in pts[:4])])
    xs, ys = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)
    truth = contains_xy(poly, xs, ys)
    iou = (painted & truth).sum() / max((painted | truth).sum(), 1)
    assert truth.sum() > 2000, "test geometry should cover a meaningful part of the map tile"
    assert iou > 0.97, f"painted area differs from the footprint: IoU {iou:.3f}"
    r, g, b_ = (px[..., i][painted & truth].mean() for i in range(3))
    assert r > 150 and g < 90 and b_ < 90, "the thumbnail's own colour must come through, not a made-up fill"


def test_unstaged_areas_are_transparent_never_filled():
    repo, _ = _one_tile_repo()
    body, mime, meta = bm.render_sentinel2(repo, lambda tid: _jpeg((10, 200, 10)), 13, 0, 0)       # nowhere near India
    assert meta["status"] == "no-coverage" and mime == "image/png"
    assert _decode(body).shape == (1, 1, 4) and _decode(body)[0, 0, 3] == 0


def test_a_catalog_tile_whose_bands_are_not_on_this_machine_paints_nothing():
    repo, pts = _one_tile_repo()
    x, y = _tile_xyz_containing(pts[0][0], pts[0][1], 13)

    def missing(_tid):
        raise KeyError("not staged")

    body, mime, meta = bm.render_sentinel2(repo, missing, 13, x, y)
    assert meta["status"] == "no-coverage" and _decode(body)[0, 0, 3] == 0


def test_the_clearest_acquisition_is_drawn_on_top_where_two_granules_overlap():
    wkt, pts = _utm_square_wkt(500_000.0, 2_950_000.0, 2560.0)
    repo = FakeRepo(tiles=[_record("cloudy", "S2A_44RPQ_20231001_0_L2A", wkt, 0.9, "2023-10-01"),
                           _record("clear", "S2B_44RPQ_20240110_0_L2A", wkt, 0.0, "2024-01-10")])
    x, y = _tile_xyz_containing(*np.mean(pts[:4], axis=0), 13)
    thumbs = {"cloudy": _jpeg((250, 250, 250)), "clear": _jpeg((30, 60, 200))}
    # both observations are the SAME granule (44RPQ): the selection rule keeps only the clearest one
    body, _, meta = bm.render_sentinel2(repo, thumbs.__getitem__, 13, x, y)
    px = _decode(body)
    c = px[px[..., 3] > 0]
    assert meta["composited"] == 1 and c[:, 2].mean() > 150 and c[:, 0].mean() < 80, "the cloudy acquisition must not be shown"


def test_selection_rule_is_lowest_mean_cloud_then_newest_per_granule():
    wkt, _ = _utm_square_wkt(500_000.0, 2_950_000.0, 2560.0)
    repo = FakeRepo(tiles=[_record("a", "S2A_44RPQ_20231001_0_L2A", wkt, 0.30, "2023-10-01"),
                           _record("b", "S2B_44RPQ_20240110_0_L2A", wkt, 0.05, "2024-01-10"),
                           _record("c", "S2A_44RPQ_20240209_0_L2A", wkt, 0.05, "2024-02-09"),
                           _record("d", "S2A_43RGM_20231205_0_L2A", wkt, 0.5, "2023-12-05")])
    pref = bm.archive(repo).preferred(None)
    assert pref["44RPQ"] == "S2A_44RPQ_20240209_0_L2A", "tie on cloud -> the newest"
    assert pref["43RGM"] == "S2A_43RGM_20231205_0_L2A"
    assert bm.archive(repo).preferred("2023")["44RPQ"] == "S2A_44RPQ_20231001_0_L2A"


def test_a_request_touching_too_many_catalog_tiles_is_refused_not_served_slowly(monkeypatch):
    repo, pts = _one_tile_repo()
    monkeypatch.setattr(bm, "MAX_COMPOSE_TILES", 0)
    x, y = _tile_xyz_containing(pts[0][0], pts[0][1], 13)
    _, _, meta = bm.render_sentinel2(repo, lambda t: _jpeg((1, 2, 3)), 13, x, y)
    assert meta["status"] == "too-coarse"


# --------------------------------------------------------------------------- Sentinel-2 overviews


def _write_granule(root: Path, obs: str, e0=500_000.0, n0=2_950_000.0, px=1600, dn=(1800, 1500, 900)):
    d = root / obs
    d.mkdir(parents=True)
    t = from_origin(e0, n0, 10.0, 10.0)
    for band, v in zip(("B04", "B03", "B02"), dn):
        with rasterio.open(d / f"{band}.tif", "w", driver="GTiff", height=px, width=px, count=1, dtype="uint16", crs=UTM44, transform=t) as ds:
            ds.write(np.full((px, px), v, np.uint16), 1)
    return d


def _granule_repo(root: Path, obs="S2A_44RPQ_20240110_0_L2A", px=1600):
    e0, n0 = 500_000.0, 2_950_000.0
    wkt, pts = _utm_square_wkt(e0, n0 - px * 10.0, px * 10.0)       # footprint of the raster (origin is its NW corner)
    repo = FakeRepo(tiles=[_record("g1", obs, wkt, 0.02)],
                    observations=[Observation(observation_id=obs, scene_id=obs, acquired_at="2024-01-10", footprint_wkt_4326=wkt, dataset_dir=obs)],
                    scenes=[Scene(scene_id=obs, collection_id=bm.S2_COLLECTION, platform="Sentinel-2A", acquired_at="2024-01-10", footprint_wkt_4326=wkt)])
    return repo, pts


def test_low_zoom_overview_is_warped_from_the_same_bands_and_stays_inside_the_footprint(tmp_path):
    from geoseek.ingest.embed import make_true_color_uint8

    root = tmp_path / "datasets"
    _write_granule(root, "S2A_44RPQ_20240110_0_L2A")
    before = sorted(p.name for p in root.rglob("*"))
    repo, pts = _granule_repo(root)
    z = 8
    x, y = _tile_xyz_containing(*np.mean(pts[:4], axis=0), z)
    body, mime, meta = bm.render_sentinel2_overview(repo, root, z, x, y)
    assert meta["status"] == "ok" and meta["granules"] == 1
    px = _decode(body)
    painted = px[..., 3] > 0
    mx0, my0, mx1, my1 = bm.tile_bounds_3857(z, x, y)
    poly = Polygon([((a - mx0) / (mx1 - mx0) * 256, (my1 - b) / (my1 - my0) * 256) for a, b in (TO_3857.transform(*p) for p in pts[:4])])
    xs, ys = np.meshgrid(np.arange(256) + 0.5, np.arange(256) + 0.5)
    truth = contains_xy(poly, xs, ys)
    assert truth.sum() > 300
    near = contains_xy(poly.buffer(1.0), xs, ys)
    assert not (painted & ~near).any(), "overview painted more than one pixel outside the granule footprint"
    assert (painted & truth).sum() >= 0.9 * truth.sum()
    bands = {"B04": np.full((4, 4), 1800, np.uint16), "B03": np.full((4, 4), 1500, np.uint16), "B02": np.full((4, 4), 900, np.uint16)}
    want = make_true_color_uint8(bands, nodata=None)[0, 0]
    got = px[painted & truth][:, :3].mean(axis=0)
    assert np.abs(got - want).max() <= 3, f"colour {got} vs the same stretch applied to the same DN {want}"
    assert sorted(p.name for p in root.rglob("*")) == before, "the basemap must not write anything next to the data"


def test_zoom_picks_the_overview_level_and_both_levels_are_real_pixels_in_the_right_place(tmp_path):
    from geoseek.ingest.embed import make_true_color_uint8

    root = tmp_path / "datasets"
    _write_granule(root, "S2A_44RPQ_20240110_0_L2A", px=2048)
    repo, pts = _granule_repo(root, px=2048)
    bands = {"B04": np.full((4, 4), 1800, np.uint16), "B03": np.full((4, 4), 1500, np.uint16), "B02": np.full((4, 4), 900, np.uint16)}
    want = make_true_color_uint8(bands, nodata=None)[0, 0]
    seen = {}
    for z in (5, 7, 8, 10, 11):
        x, y = _tile_xyz_containing(*np.mean(pts[:4], axis=0), z)
        body, _, meta = bm.render_tile(repo, lambda t: _jpeg((0, 0, 0)), z, x, y, datasets_dir=root)
        assert meta["status"] == "ok", (z, meta)
        seen[z] = meta["level"]
        px = _decode(body)
        got = px[px[..., 3] > 0][:, :3].mean(axis=0)
        assert np.abs(got - want).max() <= 3, (z, got, want)
    assert seen == {5: "coarse", 7: "coarse", 8: "fine", 10: "fine", 11: "mid"}
    by = bm.cache_stats()["overviews"]["bytes_by_level"]
    assert set(by) == {"coarse", "fine", "mid"} and by["coarse"] < by["fine"] / 8 and by["fine"] < by["mid"], "levels must get coarser as they get smaller"
    assert bm.cache_stats()["overviews"]["bytes"] <= bm.OVERVIEW_CAP_BYTES


def test_overviews_are_lazy_capped_and_shared(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    for i in range(5):
        _write_granule(root, f"S2A_44R{'ABCDE'[i]}Q_20240110_0_L2A", px=800)
    wkt, _ = _utm_square_wkt(500_000.0, 2_942_000.0, 8000.0)
    ids = [f"S2A_44R{'ABCDE'[i]}Q_20240110_0_L2A" for i in range(5)]
    repo = FakeRepo(tiles=[_record(f"t{i}", o, wkt, 0.0) for i, o in enumerate(ids)],
                    observations=[Observation(observation_id=o, scene_id=o, acquired_at="2024-01-10", footprint_wkt_4326=wkt, dataset_dir=o) for o in ids],
                    scenes=[Scene(scene_id=o, collection_id=bm.S2_COLLECTION, platform="Sentinel-2A", acquired_at="2024-01-10", footprint_wkt_4326=wkt) for o in ids])
    assert bm._overview_cache.stats()["entries"] == 0, "nothing is decoded before a tile asks for it"
    one = bm.overview(repo, root, ids[0])
    assert bm._overview_cache.stats()["entries"] == 1 and bm.overview(repo, root, ids[0]) is one, "warm read returns the cached array"
    per = one["bytes"]
    monkeypatch.setattr(bm, "_overview_cache", bm.ByteLRU(int(per * 2.5)))                       # room for two
    for o in ids:
        assert bm.overview(repo, root, o) is not None
        assert bm._overview_cache.bytes <= bm._overview_cache.cap
    st = bm._overview_cache.stats()
    assert st["entries"] == 2 and st["evictions"] == 3 and st["peak_bytes"] <= st["cap_bytes"]


def test_missing_bands_give_no_overview_rather_than_a_guess(tmp_path):
    repo, _ = _granule_repo(tmp_path)               # footprint known to the catalog, but no raster files on disk
    assert bm.overview(repo, tmp_path, "S2A_44RPQ_20240110_0_L2A") is None


# --------------------------------------------------------------------------- Maxar scene


def _maxar_repo(root: Path, size=2048):
    obs = "10300100TEST_031311102120"
    d = root / obs
    d.mkdir(parents=True)
    t = from_origin(360_000.0, 3_787_000.0, 0.3, 0.3)
    for band, v in zip("RGB", (230, 120, 20)):
        with rasterio.open(d / f"{band}.tif", "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8", crs="EPSG:32611", transform=t) as ds:
            ds.write(np.full((size, size), v, np.uint8), 1)
    repo = FakeRepo(observations=[Observation(observation_id=obs, scene_id=obs, acquired_at="2025-01-16", footprint_wkt_4326="", dataset_dir=str(d))],
                    scenes=[Scene(scene_id=obs, collection_id=bm.MAXAR_COLLECTION, platform="WV02", acquired_at="2025-01-16", footprint_wkt_4326="")],
                    collections=[Collection(collection_id=bm.MAXAR_COLLECTION, sensor="VHR Optical", platform="Maxar", bands=("R", "G", "B"), native_gsd_m=0.3)])
    return repo, obs


def test_maxar_scene_tiles_come_from_the_scene_and_are_transparent_outside_it(tmp_path):
    repo, obs = _maxar_repo(tmp_path)
    info = bm._scene_info(repo, obs)
    w, s, e, n = info["bbox"]
    lon, lat = (w + e) / 2, (s + n) / 2
    for z in (19, 15):                                     # full-resolution window, and the cached 1/8 overview
        x, y = _tile_xyz_containing(lon, lat, z)
        body, mime, meta = bm.render_scene(repo, z, x, y, obs)
        assert meta["status"] == "ok", (z, meta)
        px = _decode(body)
        inside = px[px[..., 3] > 0]
        assert len(inside) > 100 and abs(inside[:, 0].mean() - 230) < 4 and abs(inside[:, 2].mean() - 20) < 4
    far = bm.render_scene(repo, 19, *_tile_xyz_containing(lon + 0.5, lat, 19), obs)
    assert far[2]["status"] == "no-coverage" and _decode(far[0])[0, 0, 3] == 0


def test_scene_ids_are_looked_up_in_the_catalog_never_used_as_paths(tmp_path):
    repo, obs = _maxar_repo(tmp_path)
    for bad in ("../../etc/passwd", "no-such-observation", str(tmp_path)):
        with pytest.raises(bm.BasemapError):
            bm._scene_info(repo, bad)
    s2_obs = Observation(observation_id="S2X", scene_id="S2X", acquired_at="2024-01-01", footprint_wkt_4326="", dataset_dir=str(tmp_path))
    repo.obs["S2X"] = s2_obs
    repo.scenes["S2X"] = Scene(scene_id="S2X", collection_id=bm.S2_COLLECTION, platform="Sentinel-2A", acquired_at="2024-01-01", footprint_wkt_4326="")
    repo.colls[bm.S2_COLLECTION] = Collection(collection_id=bm.S2_COLLECTION, sensor="MSI", platform="Sentinel-2", bands=("B04",), native_gsd_m=10.0)
    with pytest.raises(bm.BasemapError):
        bm._scene_info(repo, "S2X")                        # a Sentinel-2 observation is not a Maxar scene


# --------------------------------------------------------------------------- coverage + HTTP


def test_coverage_states_dates_fraction_and_the_selection_rule(tmp_path):
    root = tmp_path / "datasets"
    _write_granule(root, "S2A_44RPQ_20240110_0_L2A")
    repo, pts = _granule_repo(root)
    lon, lat = np.mean(pts[:4], axis=0)
    inside = bm.coverage(repo, (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02))
    assert inside["available"] and inside["fraction"] == pytest.approx(1.0, abs=0.02)
    assert inside["dates"] == ["2024-01-10"] and "lowest mean tile cloud" in inside["selection_rule"]
    half = bm.coverage(repo, (lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5))
    assert 0 < half["fraction"] < 0.5
    nowhere = bm.coverage(repo, (0.0, 0.0, 1.0, 1.0))
    assert nowhere["available"] is False and nowhere["fraction"] == 0.0 and nowhere["scenes"] == []


def _client(monkeypatch, repo, datasets_dir, thumbs=None):
    from fastapi.testclient import TestClient

    import geoseek.search.api as api

    eng = SimpleNamespace(repo=repo, get_tile_thumbnail_png=thumbs or (lambda t: _jpeg((9, 9, 9))), settings=SimpleNamespace(datasets_dir=datasets_dir))
    monkeypatch.setattr(api, "_engine", eng)
    return TestClient(api.app)


def test_http_tile_endpoint_headers_errors_and_caching(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    _write_granule(root, "S2A_44RPQ_20240110_0_L2A")
    repo, pts = _granule_repo(root)
    c = _client(monkeypatch, repo, root)
    x, y = _tile_xyz_containing(*np.mean(pts[:4], axis=0), 8)
    r1 = c.get(f"/ui/basemap/8/{x}/{y}")
    assert r1.status_code == 200 and r1.headers["x-basemap-status"] == "ok" and r1.headers["x-basemap-cached"] == "0"
    assert r1.headers["content-type"] in ("image/png", "image/jpeg") and _decode(r1.content)[..., 3].max() == 255
    r2 = c.get(f"/ui/basemap/8/{x}/{y}")
    assert r2.content == r1.content and r2.headers["x-basemap-cached"] == "1"
    empty = c.get("/ui/basemap/8/0/0")
    assert empty.status_code == 200 and empty.headers["x-basemap-status"] == "no-coverage" and _decode(empty.content)[0, 0, 3] == 0
    assert c.get("/ui/basemap/3/9/0").status_code == 400
    assert c.get(f"/ui/basemap/8/{x}/{y}?year=20x4").status_code == 400
    assert c.get(f"/ui/basemap/8/{x}/{y}?scene=nope").status_code == 404
    st = c.get("/ui/basemap/stats").json()
    assert st["overviews"]["bytes"] <= st["overviews"]["cap_bytes"] and st["tiles"]["bytes"] <= st["tiles"]["cap_bytes"]


def test_http_coverage_validates_the_bbox(tmp_path, monkeypatch):
    repo, _ = _granule_repo(tmp_path)
    c = _client(monkeypatch, repo, tmp_path)
    assert c.get("/ui/basemap/coverage?bbox=1,2,3").status_code == 400
    assert c.get("/ui/basemap/coverage?bbox=5,2,3,4").status_code == 400
    ok = c.get("/ui/basemap/coverage?bbox=70,20,80,30")
    assert ok.status_code == 200 and {"available", "fraction", "scenes", "selection_rule", "native_max_zoom"} <= set(ok.json())


def test_the_catalog_summary_is_built_lazily_and_exactly_once_under_concurrent_first_requests():
    import threading

    wkt, _ = _utm_square_wkt(500_000.0, 2_950_000.0, 2560.0)
    reads = []

    class CountingRepo(FakeRepo):
        def iter_tile_records(self):
            reads.append(1)
            return super().iter_tile_records()

    repo = CountingRepo(tiles=[_record("a", "S2A_44RPQ_20240110_0_L2A", wkt, 0.1)])
    assert reads == [], "constructing the repo / importing the module must not read the catalog (nothing is warmed at boot)"
    got = []
    ts = [threading.Thread(target=lambda: got.append(bm.archive(repo))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(reads) == 1 and len({id(g) for g in got}) == 1, f"{len(reads)} catalog reads for 8 simultaneous first requests"

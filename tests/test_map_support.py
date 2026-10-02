"""Read-only data the console's maps are built from: oriented-box sizes on detection points, and per-cluster geography."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pyproj
import pytest

from geoseek.analyst import threat_rings as tr
from geoseek.analyst import ui_support
from geoseek.catalog.entities import TileRecord

GEOD = pyproj.Geod(ellps="WGS84")


def _oriented_box(lon, lat, length_m, width_m, heading_deg):
    """Corner ring of a length x width box whose long axis points along ``heading_deg`` - built with geodesic forward steps."""
    def step(p, az, d):
        lo, la, _ = GEOD.fwd(p[0], p[1], az, d)
        return (lo, la)

    a = step(step((lon, lat), heading_deg, -length_m / 2), heading_deg + 90, -width_m / 2)
    b = step(a, heading_deg, length_m)
    c = step(b, heading_deg + 90, width_m)
    d = step(c, heading_deg, -length_m)
    return [list(a), list(b), list(c), list(d), list(a)]


def test_detection_points_carry_box_dimensions_measured_from_the_stored_footprint(tmp_path):
    d = tmp_path / "detections" / "obsX"
    d.mkdir(parents=True)
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [_oriented_box(-118.5, 34.2, ln, w, hd)]},
              "properties": {"tile_id": "obsX_r000_c000", "class": "small-vehicle", "score": 0.7, "long_side_px": ln / 0.3, "heading_deg": hd}}
             for ln, w, hd in ((6.6, 3.4, 20.0), (4.2, 1.9, 275.0), (40.0, 12.0, 130.0))]
    (d / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    pts = tr.detection_points(tmp_path, "obsX")
    for p, (ln, w, hd) in zip(pts, ((6.6, 3.4, 20.0), (4.2, 1.9, 275.0), (40.0, 12.0, 130.0))):
        assert p["length_m"] == pytest.approx(ln, abs=0.05) and p["width_m"] == pytest.approx(w, abs=0.05)
        assert p["heading_deg"] == hd and p["long_side_px"] == pytest.approx(ln / 0.3)
        assert p["class"] == "small-vehicle" and p["score"] == 0.7


def _rec(tid, lon, lat, half=0.0115):
    wkt = f"POLYGON (({lon - half} {lat + half}, {lon + half} {lat + half}, {lon + half} {lat - half}, {lon - half} {lat - half}, {lon - half} {lat + half}))"
    return TileRecord(tile_id=tid, observation_id=tid.rsplit("_r", 1)[0], scene_id="s", collection_id="sentinel-2-l2a", sensor="MSI",
                      platform="S2", acq_date="2024-01-01", geom_wkt_4326=wkt, cloud_fraction=0.0, faiss_id=None)


def test_cluster_geo_counts_equal_the_stored_run_and_cells_sit_where_the_tiles_are(tmp_path):
    ui_support._geo_cache.update(key=None, value=None)
    tiles = [_rec("o1_r000_c000", 82.20, 26.60), _rec("o1_r000_c001", 82.201, 26.601),     # same 0.03-degree cell
             _rec("o1_r001_c000", 82.40, 26.60), _rec("o2_r000_c000", 77.10, 28.60), _rec("o2_r000_c001", 77.50, 28.70),
             _rec("unclustered_r000_c000", 70.0, 20.0)]
    run = {"tile_cluster": {"o1_r000_c000": 0, "o1_r000_c001": 0, "o1_r001_c000": 1, "o2_r000_c000": 1, "o2_r000_c001": 1}, "sizes": {"0": 2, "1": 3}}
    (tmp_path / "tile_clusters.json").write_text(json.dumps(run), encoding="utf-8")
    svc = SimpleNamespace(settings=SimpleNamespace(index_dir=tmp_path), repo=SimpleNamespace(iter_tile_records=lambda: iter(tiles)))
    g = ui_support.cluster_geo(svc)
    assert g["available"] and g["n_clustered_tiles"] == 5 and g["n_unplaced"] == 0
    assert {k: v["n_tiles"] for k, v in g["clusters"].items()} == {k: v for k, v in run["sizes"].items()}
    c0 = g["clusters"]["0"]
    assert c0["n_cells"] == 1 and c0["cells"][0][2] == 2, "two tiles in one 0.03-degree cell are one cell with n=2"
    assert abs(c0["cells"][0][0] - 82.2) < 0.03 and abs(c0["cells"][0][1] - 26.6) < 0.03
    w, s, e, n = g["clusters"]["1"]["bbox"]
    assert (w, s) == pytest.approx((77.1 - 0.0115, 26.6 - 0.0115)) and (e, n) == pytest.approx((82.4 + 0.0115, 28.7 + 0.0115))
    assert sum(c[2] for v in g["clusters"].values() for c in v["cells"]) == 5
    assert ui_support.cluster_geo(svc) is g, "second call is served from the cache"


def test_cluster_geo_without_a_clustering_run_says_so(tmp_path):
    ui_support._geo_cache.update(key=None, value=None)
    svc = SimpleNamespace(settings=SimpleNamespace(index_dir=tmp_path), repo=None)
    assert ui_support.cluster_geo(svc)["available"] is False


def test_cluster_geo_is_built_once_when_two_first_requests_arrive_together(tmp_path):
    import threading

    ui_support._geo_cache.update(key=None, value=None)
    (tmp_path / "tile_clusters.json").write_text(json.dumps({"tile_cluster": {"o1_r000_c000": 0}}), encoding="utf-8")
    reads = []

    def records():
        reads.append(1)
        return iter([_rec("o1_r000_c000", 82.2, 26.6)])

    svc = SimpleNamespace(settings=SimpleNamespace(index_dir=tmp_path), repo=SimpleNamespace(iter_tile_records=records))
    out = []
    ts = [threading.Thread(target=lambda: out.append(ui_support.cluster_geo(svc))) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(reads) == 1 and all(o is out[0] for o in out)


def _repo_with_regions(tiles, obs_regions):
    obs = [SimpleNamespace(observation_id=o, aoi_name=r) for o, r in obs_regions.items()]
    return SimpleNamespace(iter_tile_records=lambda: iter(tiles), list_observations=lambda: obs)


def test_cluster_geo_examples_are_spread_over_the_regions_that_matter_and_are_repeatable(tmp_path):
    ui_support._geo_cache.update(key=None, value=None)
    # cluster 0: 14 tiles in region A, 10 in region B, 1 stray in region C (< 5 %: contributes no example); cluster 1 lives in A only
    tiles = ([_rec(f"a_r{i:03d}_c000", 80.0 + i * 0.1, 26.0) for i in range(14)] + [_rec(f"b_r{i:03d}_c000", 70.0 + i * 0.1, 24.0) for i in range(10)]
             + [_rec("c_r000_c000", 77.0, 28.0)] + [_rec(f"a_r{i:03d}_c001", 81.0 + i * 0.1, 26.5) for i in range(10)])
    run = {"tile_cluster": {**{t.tile_id: 0 for t in tiles[:25]}, **{t.tile_id: 1 for t in tiles[25:]}}}
    (tmp_path / "tile_clusters.json").write_text(json.dumps(run), encoding="utf-8")
    svc = SimpleNamespace(settings=SimpleNamespace(index_dir=tmp_path),
                          repo=_repo_with_regions(tiles, {"a": "ayodhya_44RPQ_diverse", "b": "kutch_42QZG_diverse", "c": "deccan_43PGQ_diverse"}))
    g = ui_support.cluster_geo(svc)
    ex0 = g["clusters"]["0"]["examples"]
    assert len(ex0) == ui_support.CLUSTER_EXAMPLES
    assert {e["region"] for e in ex0} == {"ayodhya", "kutch"}, "the 1-in-25 stray region supplies no example"
    assert [e["region"] for e in ex0][:2] == ["ayodhya", "kutch"], "regions take turns, largest first"
    assert all(set(e) == {"tile_id", "region", "acq_date", "cloud_fraction", "centroid_lonlat"} for e in ex0), "no private sort key leaks"
    ex1 = g["clusters"]["1"]["examples"]
    assert len(ex1) == ui_support.CLUSTER_EXAMPLES and {e["region"] for e in ex1} == {"ayodhya"} and len({e["tile_id"] for e in ex1}) == 4
    assert all(t["tile_id"] in run["tile_cluster"] and run["tile_cluster"][t["tile_id"]] == 1 for t in ex1)
    ui_support._geo_cache.update(key=None, value=None)
    assert ui_support.cluster_geo(svc)["clusters"]["0"]["examples"] == ex0, "the same rule gives the same examples on a rebuild"


def test_cluster_geo_examples_prefer_the_clearest_tiles():
    pools = {"r": [{"tile_id": "x", "_k": (0.0, 1)}, {"tile_id": "y", "_k": (0.4, 0)}]}
    assert [e["tile_id"] for e in ui_support._pick_examples(pools, {"r": 2}, n=1)] == ["x"]
    assert ui_support._pick_examples({}, {}) == []


def test_discovery_clusters_exposes_the_stored_region_counts(tmp_path):
    from geoseek.analyst.service import AnalystService

    d = {"n_clusters": 1, "noise_count": 0, "n_tiles": 3, "sizes": {"0": 3}, "tile_cluster": {"o_r000_c000": 0},
         "region_purity": {"0": {"size": 3, "dominant_region": "kutch", "purity": 0.667, "regions": {"kutch": 2, "kanha": 1}}}}
    (tmp_path / "tile_clusters.json").write_text(json.dumps(d), encoding="utf-8")
    out = AnalystService.discovery_clusters(SimpleNamespace(settings=SimpleNamespace(index_dir=tmp_path)))
    assert out["region_purity"]["0"]["regions"] == {"kutch": 2, "kanha": 1}
    assert sum(out["region_purity"]["0"]["regions"].values()) == out["sizes"]["0"]

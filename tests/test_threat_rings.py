"""Threat buffer rings: exact geometry against independently computed geodesic distances, on small fakes."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pyproj
import pytest

from geoseek.analyst import threat_rings as tr

GEOD = pyproj.Geod(ellps="WGS84")
LON0, LAT0 = 82.2500, 26.6360          # inside the "Forward Operating Base Alpha" demo zone


def _square(lon, lat, half_m):
    dlat = half_m / 110_574.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return {"type": "Polygon", "coordinates": [[[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat],
                                                 [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]]}


def _offset(lon, lat, east_m, north_m):
    """A point east_m / north_m from (lon, lat), by geodesic forward computation (independent of the code under test)."""
    az = math.degrees(math.atan2(east_m, north_m))
    d = math.hypot(east_m, north_m)
    lo, la, _ = GEOD.fwd(lon, lat, az, d)
    return lo, la


def _svc(tmp_path, candidates=(), watch=()):
    return SimpleNamespace(
        details=list(candidates), settings=SimpleNamespace(data_dir=tmp_path), list_watch_areas=lambda: list(watch))


def _cand(cid, lon, lat, half_m=20, **kw):
    return {"candidate_id": cid, "_geometry": _square(lon, lat, half_m), "change_type": "construction", "confidence": 0.9,
            "area_m2": 1600.0, "persistence": "persistent", **kw}


# ------------------------------------------------------------------------- radii -----------------------------------


def test_parse_radii_sorts_dedups_and_validates():
    assert tr.parse_radii(None) == list(tr.DEFAULT_RADII_M)
    assert tr.parse_radii("2500, 500,500,1000") == [500.0, 1000.0, 2500.0]
    for bad in ("", "abc", "5", "300000", ",".join(str(100 * i) for i in range(1, 12))):
        if bad == "":
            assert tr.parse_radii(bad) == list(tr.DEFAULT_RADII_M)
            continue
        with pytest.raises(ValueError):
            tr.parse_radii(bad)


# ------------------------------------------------------------------------- distances -------------------------------


def test_candidate_distance_matches_the_independent_geodesic_and_lands_in_the_right_ring(tmp_path):
    lon, lat = _offset(LON0, LAT0, 1200.0, 0.0)             # 1.2 km east; footprint half-width 20 m -> nearest edge ~1180 m
    out = tr.threat_rings(_svc(tmp_path, [_cand("c1", lon, lat)]), LON0, LAT0, [500, 1000, 2500, 5000], kinds=("change_candidate",))
    (f,) = out["features"]
    assert f["kind"] == "change_candidate" and f["id"] == "candidate:c1"
    centre_to_centroid = GEOD.inv(LON0, LAT0, lon, lat)[2]
    assert centre_to_centroid == pytest.approx(1200.0, abs=1.0)
    assert f["distance_m"] == pytest.approx(centre_to_centroid - 20.0, abs=2.5)       # centroid distance minus the half-width
    assert f["ring_index"] == 2 and f["ring_m"] == 2500 and not f["contains_centre"]
    assert [r["total"] for r in out["rings"]] == [0, 0, 1, 1]                           # cumulative: in every ring from #3 outward
    assert out["sources"]["change_candidate"] == {"features_searched": 1, "within_largest_ring": 1}


def test_restricted_zone_distance_is_to_the_box_edge_and_zero_when_inside(tmp_path):
    inside = tr.threat_rings(_svc(tmp_path), LON0, LAT0, [500, 1000], kinds=("restricted_zone",))
    z = next(f for f in inside["features"] if f["name"] == "Forward Operating Base Alpha")
    assert z["distance_m"] == 0.0 and z["contains_centre"] and z["ring_index"] == 0 and z["detail"]["level"] == "critical"

    west_edge = 82.235                                       # zone's min_lon; stand 450 m west of it at the same latitude
    lon, lat = _offset(west_edge, 26.636, -450.0, 0.0)
    out = tr.threat_rings(_svc(tmp_path), lon, lat, [300, 600, 2000], kinds=("restricted_zone",))
    z = next(f for f in out["features"] if f["name"] == "Forward Operating Base Alpha")
    want = GEOD.inv(lon, lat, west_edge, 26.636)[2]
    assert want == pytest.approx(450.0, abs=1.0)
    assert z["distance_m"] == pytest.approx(want, abs=1.5)
    assert z["ring_index"] == 1 and not z["contains_centre"]
    assert out["sources"]["restricted_zone"]["features_searched"] == 4


def test_features_beyond_the_largest_ring_are_not_reported(tmp_path):
    lon, lat = _offset(LON0, LAT0, 9000.0, 0.0)
    out = tr.threat_rings(_svc(tmp_path, [_cand("far", lon, lat)]), LON0, LAT0, [500, 5000], kinds=("change_candidate",))
    assert out["features"] == [] and out["sources"]["change_candidate"]["within_largest_ring"] == 0
    assert [r["total"] for r in out["rings"]] == [0, 0]


def test_exclude_removes_the_ring_s_own_centre_feature(tmp_path):
    svc = _svc(tmp_path, [_cand("self", LON0, LAT0), _cand("near", *_offset(LON0, LAT0, 150, 0))])
    both = tr.threat_rings(svc, LON0, LAT0, [500], kinds=("change_candidate",))
    assert {f["name"] for f in both["features"]} == {"self", "near"}
    assert next(f for f in both["features"] if f["name"] == "self")["contains_centre"]
    one = tr.threat_rings(svc, LON0, LAT0, [500], kinds=("change_candidate",), exclude="candidate:self")
    assert [f["name"] for f in one["features"]] == ["near"] and one["excluded"] == "candidate:self"


def test_results_are_sorted_by_distance_and_truncation_is_reported_not_hidden(tmp_path):
    cands = [_cand(f"c{i}", *_offset(LON0, LAT0, 100 + 40 * i, 0)) for i in range(10)]
    out = tr.threat_rings(_svc(tmp_path, cands), LON0, LAT0, [1000], kinds=("change_candidate",), limit=4)
    d = [f["distance_m"] for f in out["features"]]
    assert d == sorted(d) and len(d) == 4 and out["truncated"] is True and out["n_features"] == 10
    assert out["rings"][0]["total"] == 10                       # counts are over everything, not just the returned page


# ------------------------------------------------------------------------- detections / watch areas ----------------


def _write_detections(tmp_path, obs, items):
    d = tmp_path / "detections" / obs
    d.mkdir(parents=True)
    feats = [{"type": "Feature", "geometry": _square(lo, la, 3), "properties": {"class": cls, "score": sc, "tile_id": f"{obs}_r000_c000"}}
             for lo, la, cls, sc in items]
    (d / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def test_stored_detections_are_intersected_and_exposed_as_points(tmp_path):
    near, far = _offset(LON0, LAT0, 0, 220.0), _offset(LON0, LAT0, 0, 3000.0)
    _write_detections(tmp_path, "obsA", [(*near, "plane", 0.91), (*far, "small-vehicle", 0.55)])
    out = tr.threat_rings(_svc(tmp_path), LON0, LAT0, [300, 1000], kinds=("detection",))
    (f,) = out["features"]
    assert f["id"] == "detection:obsA:0" and f["detail"]["class"] == "plane" and f["ring_index"] == 0
    assert f["distance_m"] == pytest.approx(220.0, abs=4.0)
    # the whole layer counts as searched even though the far detection was cheaply pre-filtered out before projection
    assert out["sources"]["detection"] == {"features_searched": 2, "within_largest_ring": 1}
    pts = tr.detection_points(tmp_path, "obsA")
    assert [p["id"] for p in pts] == ["detection:obsA:0", "detection:obsA:1"]
    assert pts[0]["lon"] == pytest.approx(near[0], abs=1e-5) and pts[0]["class"] == "plane" and pts[1]["score"] == 0.55
    with pytest.raises(FileNotFoundError):
        tr.detection_points(tmp_path, "nope")


def test_watch_areas_with_a_bbox_are_intersected(tmp_path):
    lo, la = _offset(LON0, LAT0, 800, 0)
    w = {"watch_id": "w1", "name": "River bend", "bbox": [lo - 0.001, la - 0.001, lo + 0.001, la + 0.001], "active": True}
    out = tr.threat_rings(_svc(tmp_path, watch=[w]), LON0, LAT0, [1000], kinds=("watch_area",))
    (f,) = out["features"]
    assert f["name"] == "River bend" and 600 < f["distance_m"] < 800


def test_notes_state_what_was_not_searched(tmp_path):
    out = tr.threat_rings(_svc(tmp_path), LON0, LAT0, [500])
    assert any("no road, building or infrastructure layer" in n for n in out["notes"])
    assert set(out["sources"]) == set(tr.KINDS)


# ------------------------------------------------------------------------- HTTP ----------------------------------


def test_http_endpoints_validate_and_serve(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from geoseek.search import api

    _write_detections(tmp_path, "obsB", [(LON0, LAT0 + 0.001, "ship", 0.7)])
    monkeypatch.setattr(api, "_analyst", _svc(tmp_path, [_cand("c1", *_offset(LON0, LAT0, 300, 0))]))
    c = TestClient(api.app)
    r = c.get("/ui/threat-rings", params={"lon": LON0, "lat": LAT0, "radii_m": "500,2000", "exclude": "candidate:zzz"})
    assert r.status_code == 200
    body = r.json()
    assert {f["kind"] for f in body["features"]} >= {"restricted_zone", "change_candidate", "detection"}
    assert body["radii_m"] == [500.0, 2000.0]
    assert c.get("/ui/threat-rings", params={"lon": LON0, "lat": LAT0, "radii_m": "5"}).status_code == 400
    assert c.get("/ui/threat-rings", params={"lon": 500, "lat": 0}).status_code == 422
    p = c.get("/ui/detections/obsB/points")
    assert p.status_code == 200 and p.json()["points"][0]["class"] == "ship"
    assert c.get("/ui/detections/missing/points").status_code == 404

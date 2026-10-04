"""Backend support for the React console: the /react/ mount, and the read-only /ui/* projections.

Everything runs against small fakes (no production catalog needed): what is asserted is that figures are DERIVED from
the sources they claim (checkpoint model card, detector eval JSON, catalog repository, trajectory) and never invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geoseek.analyst import ui_support


# ------------------------------------------------------------------------- mount ----------------------------------


def _client():
    from fastapi.testclient import TestClient

    from geoseek.search.api import app

    return TestClient(app)          # no `with`: the lifespan (model + index load) is deliberately not started


def test_react_console_is_served_at_react_and_the_existing_ui_is_unchanged():
    from geoseek.search.api import _REACT_DIR, _WEB_DIR

    assert _REACT_DIR.is_dir() and (_REACT_DIR / "index.html").is_file(), "run `npm run build` in frontend-react/"
    c = _client()
    r = c.get("/react/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "GeoSeek Console" in r.text
    asset = next(p for p in (_REACT_DIR / "assets").glob("index-*.js"))
    assert c.get(f"/react/assets/{asset.name}").status_code == 200
    # the pre-existing frontend still serves, from the same mount as before, and its directory is not the React one
    assert _WEB_DIR != _REACT_DIR
    old = c.get("/app/")
    assert old.status_code == 200 and "text/html" in old.headers["content-type"]
    body = c.get("/").json()
    assert body["ui"] == "/app/" and body["react_ui"] == "/react/"


def test_react_assets_never_leave_the_react_directory():
    c = _client()
    assert c.get("/react/../analyst/service.py").status_code in (400, 404)
    assert c.get("/react/%2e%2e/%2e%2e/config.py").status_code in (400, 404)


# ------------------------------------------------------------------------- timeline ----------------------------------

DATES = ["2019-03-30", "2021-03-04", "2024-03-08", "2025-03-08", "2026-03-08"]


def _cand(persistence: str, changed: list[bool], window: list[str] | None):
    pairs = [{"window": [DATES[i], DATES[i + 1]], "changed": ch, "probability": 0.9 if ch else 0.05, "comparable": True}
             for i, ch in enumerate(changed)]
    return {
        "candidate_id": "c1", "persistence": persistence,
        "trajectory": {
            "persistence": persistence, "persistence_confidence": 0.5, "consecutive_pairs": pairs,
            "earliest_supported_change": {"window": window, "caveat": "cav"} if window else {},
            "span_pair": {"changed": True}, "notes": [],
        },
    }


def _svc(cand):
    return SimpleNamespace(_by_id={"c1": cand}, _observation_dates=lambda: list(DATES))


def test_timeline_persistent_counts_every_date_from_first_detection_on():
    tl = ui_support.candidate_timeline(_svc(_cand("persistent", [True, False, False, False], ["2019-03-30", "2021-03-04"])), "c1")
    assert tl["dates"] == DATES and [p["date"] for p in tl["points"]] == DATES
    assert [p["role"] for p in tl["points"]] == ["baseline", "first_detected", "supported", "supported", "supported"]
    assert tl["first_detected"]["date"] == "2021-03-04" and tl["first_detected"]["bracket"] == ["2019-03-30", "2021-03-04"]
    assert (tl["n_supporting"], tl["n_total"]) == (4, 5)
    assert tl["supporting_dates"] == DATES[1:]


def test_timeline_transient_does_not_claim_persistence():
    tl = ui_support.candidate_timeline(_svc(_cand("transient", [True, False, False, False], ["2019-03-30", "2021-03-04"])), "c1")
    assert [p["role"] for p in tl["points"]][1:] == ["first_detected", "not_supported", "not_supported", "not_supported"]
    assert (tl["n_supporting"], tl["n_total"]) == (1, 5)


def test_timeline_recent_change_starts_late():
    tl = ui_support.candidate_timeline(_svc(_cand("recent", [False, False, False, True], ["2025-03-08", "2026-03-08"])), "c1")
    assert [p["role"] for p in tl["points"]] == ["baseline", "before_detection", "before_detection", "before_detection", "first_detected"]
    assert (tl["n_supporting"], tl["n_total"]) == (1, 5)


def test_timeline_with_no_change_has_no_first_detection_and_never_invents_dates():
    tl = ui_support.candidate_timeline(_svc(_cand("none", [False] * 4, None)), "c1")
    assert tl["first_detected"] is None and tl["n_supporting"] == 0 and tl["n_total"] == 5
    assert {p["role"] for p in tl["points"][1:]} == {"no_change_seen"}
    assert set(p["date"] for p in tl["points"]) <= set(DATES)       # only catalog acquisition dates, ever


def test_timeline_unknown_candidate_is_none():
    assert ui_support.candidate_timeline(_svc(_cand("none", [False] * 4, None)), "nope") is None


# ------------------------------------------------------------------------- footprints ---------------------------------


class _FakeRepo:
    def __init__(self, tiles):
        self.tiles = tiles

    def get_tile(self, tid):
        return self.tiles.get(tid)


def test_tile_footprints_come_from_catalog_geometry():
    wkt = "POLYGON((82.0 26.0, 82.0259 26.0, 82.0259 26.0233, 82.0 26.0233, 82.0 26.0))"
    repo = _FakeRepo({"t1": SimpleNamespace(geom_wkt_4326=wkt, cloud_fraction=0.25, observation_id="o1")})
    out = ui_support.tile_footprints(repo, ["t1", "missing"])
    assert [t["tile_id"] for t in out] == ["t1"]                      # unknown ids are skipped, not invented
    t = out[0]
    assert t["bbox"] == [82.0, 26.0, 82.0259, 26.0233] and t["cloud_fraction"] == 0.25
    assert 2.4 < t["width_km"] < 2.7 and 2.4 < t["height_km"] < 2.7
    assert t["area_km2"] == pytest.approx(t["width_km"] * t["height_km"], rel=0.01)


# ------------------------------------------------------------------------- latency probe -----------------------------


class _FakeEngine:
    def __init__(self, ms):
        self.ms, self.calls = list(ms), 0

    def search_text(self, q, k=10):
        self.calls += 1
        return [], self.ms[(self.calls - 1) % len(self.ms)]

    def count(self):
        return 123


def test_latency_probe_reports_engine_measured_times_and_is_cached():
    ui_support._probe_cache.update(at=0.0, value=None)
    eng = _FakeEngine([99.0, 10.0, 20.0, 30.0, 40.0, 50.0])      # first call is the discarded warm-up
    out = ui_support.latency_probe(eng)
    assert eng.calls == 1 + len(ui_support.PROBE_QUERIES)
    assert out["median_ms"] == 30.0 and out["min_ms"] == 10.0 and out["max_ms"] == 50.0 and out["p95_ms"] == 50.0
    assert out["n_queries"] == 5 and out["vectors"] == 123
    again = ui_support.latency_probe(eng)
    assert again is out and eng.calls == 1 + len(ui_support.PROBE_QUERIES)    # served from the short cache
    ui_support._probe_cache.update(at=0.0, value=None)


# ------------------------------------------------------------------------- metrics -----------------------------------


def _metrics_svc(tmp_path: Path, *, with_eval: bool, with_card: bool, monkeypatch):
    if with_eval:
        d = tmp_path / "data" / "detect_eval"
        d.mkdir(parents=True)
        (d / "eval_results.json").write_text(json.dumps({
            "val_full_image_v15": {
                "per_class": {"small-vehicle": {"AP50": 0.8453}},
                "groups": {"ground_vehicles": {"macro_AP50": 0.8536}, "all_kept_classes": {"macro_AP50": 0.8174}},
                "counts": {"n_images": 458}},
            "val_bootstrap_v15": {"per_class": {"small-vehicle": {"AP50_ci": [0.7, 0.9]}},
                                  "groups": {"ground_vehicles": {"macro_AP50_ci": [0.78, 0.89]}, "all_kept_classes": {"macro_AP50_ci": [0.76, 0.84]}}},
        }), encoding="utf-8")
    ckpt = tmp_path / "fc.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(ui_support, "_model_card_eval", lambda path: {
        "test_at_precision_favouring": {"threshold": 0.8, "f1": 0.5528, "precision": 0.6034, "recall": 0.5101, "iou": 0.382},
        "test_at_0.50": {"f1": 0.5596}, "validation_at_chosen_threshold": {"f1": 0.5107}} if with_card else None)
    coll = [SimpleNamespace(collection_id="s2", sensor="MSI", platform="Sentinel-2", native_gsd_m=10.0),
            SimpleNamespace(collection_id="s1", sensor="C-SAR", platform="Sentinel-1", native_gsd_m=10.0),
            SimpleNamespace(collection_id="s2b", sensor="MSI", platform="Sentinel-2", native_gsd_m=10.0)]
    scenes = [SimpleNamespace(collection_id="s2")] * 3 + [SimpleNamespace(collection_id="s1")]
    repo = SimpleNamespace(count_tiles=lambda: 1000, list_collections=lambda: coll, list_scenes=lambda: scenes,
                           list_analyst_decisions=lambda: [1, 2])
    regions = [{"name": "a", "bbox": [0, 0, 1, 1], "n_observations": 2}, {"name": "b", "bbox": [5, 5, 6, 6], "n_observations": 1}]
    details = [{"centroid_lonlat": [0.5, 0.5], "change_type": "road"}, {"centroid_lonlat": [0.2, 0.9], "change_type": "road"},
               {"centroid_lonlat": [50, 50], "change_type": "other"}]
    return SimpleNamespace(
        repo=repo, details=details, list_regions=lambda: regions, _engine=SimpleNamespace(count=lambda: 900),
        report={"aoi": "AOI"}, model_info={"name": "M", "checkpoint": str(ckpt)}, out_dir=tmp_path,
        settings=SimpleNamespace(project_root=tmp_path), _observation_dates=lambda: ["2020-01-01"],
        detector_model_info=lambda: {"operating_points": {"small-vehicle": {"AP50": 0.531, "source": "x.json"},
                                                          "large-vehicle": {"AP50": 0.315}}})


def test_console_metrics_are_derived_from_their_sources(tmp_path, monkeypatch):
    m = ui_support.console_metrics(_metrics_svc(tmp_path, with_eval=True, with_card=True, monkeypatch=monkeypatch))
    c = m["counters"]
    assert (c["tiles_indexed"], c["vectors_searchable"], c["scenes"], c["regions"], c["collections"]) == (1000, 900, 4, 2, 3)
    assert c["sensors"] == 2                                              # MSI counted once: distinct sensors, not collections
    assert c["change_candidates"] == 3 and c["analyst_decisions"] == 2
    assert [(r["name"], r["candidates"]) for r in m["findings_by_region"]] == [("a", 2), ("b", 0)]   # point-in-bbox, 1 outside both
    assert m["findings_by_type"] == {"road": 2, "other": 1}
    cm = m["change_model"]
    assert cm["available"] and cm["f1"] == 0.5528 and cm["precision"] == 0.6034 and cm["operating_threshold"] == 0.8
    d = m["detector"]["dota_val"]
    assert (d["small_vehicle_ap50"], d["ground_vehicles_ap50"], d["all_classes_ap50"], d["n_images"]) == (0.8453, 0.8536, 0.8174, 458)
    assert d["ground_vehicles_ap50_ci"] == [0.78, 0.89]
    assert m["detector"]["xview_test"]["small_vehicle_ap50"] == 0.531     # the harder transfer figure travels with it
    assert "do not transfer" in m["detector"]["caveat"]


def test_console_metrics_report_missing_sources_instead_of_inventing_numbers(tmp_path, monkeypatch):
    m = ui_support.console_metrics(_metrics_svc(tmp_path, with_eval=False, with_card=False, monkeypatch=monkeypatch))
    assert m["change_model"] == {"available": False}
    assert "dota_val" not in m["detector"]                                 # no eval file -> no DOTA numbers at all


# ------------------------------------------------------------------ region catalog figures (Findings by region) --------


def _catalog_svc(*, report_obs):
    def obs(oid, scene, when, aoi, coreg):
        return SimpleNamespace(observation_id=oid, scene_id=scene, acquired_at=when, aoi_name=aoi, coregistration=coreg)

    observations = [
        obs("o1", "sc1", "2019-03-30T05:00:00Z", "alpha_44RPQ_scaled", {"method": "fft"}),
        obs("o2", "sc2", "2021-03-04", "alpha_44RPQ_scaled", {"role": "reference"}),
        obs("o3", "sc3", "2021-03-06", "alpha-82km", {"method": "geocode"}),        # a SAR look in the same region
        obs("o4", "sc4", "2023-10-11", "beta_43RGP_diverse", {}),
        obs("o5", "sc5", "2023-10-11", "beta_43RGP_diverse", {}),                   # same date twice: one acquisition, two observations
        obs("o6", "sc6", "2024-02-01", "beta_43RGP_diverse", {}),
    ]
    scenes = [SimpleNamespace(scene_id=f"sc{i}", collection_id=c) for i, c in
              [(1, "s2"), (2, "s2"), (3, "s1"), (4, "s2"), (5, "s2"), (6, "s2")]]
    colls = [SimpleNamespace(collection_id="s2", sensor="MSI", platform="Sentinel-2"), SimpleNamespace(collection_id="s1", sensor="C-SAR", platform="Sentinel-1")]
    tiles = {"o1": 10, "o2": 10, "o3": 7, "o4": 5, "o5": 5, "o6": 5}
    repo = SimpleNamespace(list_scenes=lambda: scenes, list_collections=lambda: colls, list_observations=lambda: observations,
                           count_tiles_by_observation=lambda: tiles, count_tiles=lambda: sum(tiles.values()))
    return SimpleNamespace(repo=repo, report={"observations": report_obs}), tiles


def test_region_catalog_figures_are_counted_from_the_catalog_not_typed_in():
    svc, tiles = _catalog_svc(report_obs=["o1", "o2"])
    cat = ui_support._region_catalog(svc)
    assert set(cat) == {"alpha", "beta"}
    a, b = cat["alpha"], cat["beta"]
    assert a["analysed"] is True and b["analysed"] is False                      # from the pipeline's own observation list
    assert (a["n_tiles"], b["n_tiles"]) == (27, 15) and a["n_tiles"] + b["n_tiles"] == sum(tiles.values())
    assert (a["first_date"], a["last_date"]) == ("2019-03-30", "2021-03-06")      # across every sensor
    s2, sar = a["sensors"]                                                       # deepest history first
    assert s2["collection_id"] == "s2" and (s2["n_observations"], s2["n_acquisitions"], s2["n_coregistered"]) == (2, 2, 2)
    assert (s2["first_date"], s2["last_date"], s2["n_tiles"]) == ("2019-03-30", "2021-03-04", 20)
    assert sar["collection_id"] == "s1" and sar["sensor"] == "C-SAR" and sar["n_coregistered"] == 1
    (only,) = b["sensors"]
    assert (only["n_observations"], only["n_acquisitions"], only["n_coregistered"]) == (3, 2, 0)   # two granules on one date = one acquisition


def test_region_catalog_marks_nothing_analysed_when_the_pipeline_report_is_empty_and_degrades_without_a_catalog():
    svc, _ = _catalog_svc(report_obs=[])
    assert not any(r["analysed"] for r in ui_support._region_catalog(svc).values())
    broken = SimpleNamespace(repo=SimpleNamespace(), report={})                  # a repo that cannot answer: no figures, never a guess
    assert ui_support._region_catalog(broken) == {}


def test_region_counts_carry_the_breakdown_by_type_and_the_catalog_block():
    svc, _ = _catalog_svc(report_obs=["o1"])
    svc.list_regions = lambda: [{"name": "alpha", "bbox": [0, 0, 1, 1], "n_observations": 3}, {"name": "beta", "bbox": [5, 5, 6, 6], "n_observations": 3}]
    svc.details = [{"centroid_lonlat": [0.5, 0.5], "change_type": "road"}, {"centroid_lonlat": [0.2, 0.9], "change_type": "road"},
                   {"centroid_lonlat": [0.3, 0.3], "change_type": "other"}]
    a, b = ui_support._region_counts(svc)
    assert a["candidates"] == 3 and a["by_type"] == {"road": 2, "other": 1} and a["catalog"]["analysed"] is True
    assert b["candidates"] == 0 and b["by_type"] == {} and b["catalog"]["analysed"] is False

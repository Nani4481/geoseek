"""Phase 8 - archive refresh (Step A), terrain context (Step B), standing
watch areas (Step C), sector summary brief (Step D).

Step A: the date/pair-discovery helpers in geoseek.change.analyze are pure
functions of a {year: observation_id} mapping - tested here without a live
catalog. The full 5-date pipeline re-run itself is an integration script
(python -m geoseek.change.analyze), not exercised by pytest (like Phase 4/5).

Step B: the Horn's-method slope/aspect formula is tested against synthetic
tilted planes of known orientation (the same self-check done before shipping
it - see geoseek.terrain.dem).

Step C: the watch_areas / watch_notifications tables through the
MetadataRepository seam (never raw sqlite3), and the pure-function matching
logic in geoseek.watch.evaluator.

Step D: AnalystService.sector_brief's aggregation logic, exercised on a
fabricated in-memory report (does not require a live change report).
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.catalog.entities import AnalystDecision, WatchArea, WatchNotification
from geoseek.catalog.repository import MetadataRepository
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository

# --------------------------------------------------------------------------
# Step A - date / pair discovery (pure functions)
# --------------------------------------------------------------------------


def test_build_pair_keys_reproduces_the_original_3_date_set():
    from geoseek.change.analyze import build_pair_keys

    three = {"2019": "a", "2021": "b", "2024": "c"}
    assert build_pair_keys(three) == {
        "2019-2021": ("2019", "2021"), "2021-2024": ("2021", "2024"), "2019-2024": ("2019", "2024"),
    }


def test_build_pair_keys_extends_cleanly_to_5_dates():
    from geoseek.change.analyze import build_pair_keys

    five = {"2019": "a", "2021": "b", "2024": "c", "2025": "d", "2026": "e"}
    pairs = build_pair_keys(five)
    assert pairs == {
        "2019-2021": ("2019", "2021"), "2021-2024": ("2021", "2024"),
        "2024-2025": ("2024", "2025"), "2025-2026": ("2025", "2026"),
        "2019-2026": ("2019", "2026"),
    }


def test_discover_date_to_obs_falls_back_without_a_catalog():
    from geoseek.change.analyze import _STATIC_DATE_TO_OBS, discover_date_to_obs

    class _BrokenRepo:
        def list_observations(self, **kw):
            raise RuntimeError("no catalog")

    assert discover_date_to_obs(repo=_BrokenRepo()) == _STATIC_DATE_TO_OBS


def test_discover_date_to_obs_falls_back_under_3_observations():
    from geoseek.change.analyze import _STATIC_DATE_TO_OBS, discover_date_to_obs

    class _Obs:
        def __init__(self, oid, date):
            self.observation_id, self.acquired_at = oid, date

    class _TinyRepo:
        def list_observations(self, **kw):
            return [_Obs("only_one", "2024-03-08")]

    assert discover_date_to_obs(repo=_TinyRepo()) == _STATIC_DATE_TO_OBS


def test_discover_date_to_obs_sorts_by_acquisition_and_keys_by_year():
    from geoseek.change.analyze import discover_date_to_obs

    class _Obs:
        def __init__(self, oid, date):
            self.observation_id, self.acquired_at = oid, date

    class _Repo:
        def list_observations(self, **kw):
            # deliberately out of order
            return [_Obs("c", "2024-03-08"), _Obs("a", "2019-03-30"),
                    _Obs("e", "2026-03-08"), _Obs("b", "2021-03-04"), _Obs("d", "2025-03-08")]

    out = discover_date_to_obs(repo=_Repo())
    assert list(out.items()) == [("2019", "a"), ("2021", "b"), ("2024", "c"), ("2025", "d"), ("2026", "e")]


# --------------------------------------------------------------------------
# Step B - terrain (Horn's method self-check + plain-language formatting)
# --------------------------------------------------------------------------


def test_slope_aspect_orientation_matches_known_synthetic_planes():
    """Same 4-orientation self-check performed before shipping the formula -
    guards the +180 deg correction (row increases southward vs the textbook
    y-increases-north convention) from silently regressing."""
    from geoseek.terrain.dem import _KX, _KY
    from scipy.ndimage import convolve

    def slope_aspect(z, px=10.0):
        dzdx = convolve(z, _KX, mode="nearest") / (8.0 * px)
        dzdy = convolve(z, _KY, mode="nearest") / (8.0 * px)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        ang = np.degrees(np.arctan2(dzdy, -dzdx))
        aspect = np.where(ang < 0.0, 90.0 - ang, np.where(ang > 90.0, 360.0 - ang + 90.0, 90.0 - ang))
        aspect = (aspect + 180.0) % 360.0
        return slope, aspect

    n = 11
    rows, cols = np.arange(n).reshape(-1, 1), np.arange(n).reshape(1, -1)
    cases = [
        ((n - 1 - rows) * np.ones((n, n)) * 10.0, 180.0),   # high north / low south -> faces south
        (cols * np.ones((n, n)) * 10.0, 270.0),             # high east / low west -> faces west
        (rows * np.ones((n, n)) * 10.0, 0.0),               # high south / low north -> faces north
        ((n - 1 - cols) * np.ones((n, n)) * 10.0, 90.0),    # high west / low east -> faces east
    ]
    for z, expect in cases:
        slope, aspect = slope_aspect(z)
        c = n // 2
        assert slope[c, c] == pytest.approx(45.0, abs=0.5)
        assert aspect[c, c] == pytest.approx(expect, abs=0.5)


def test_compass_label():
    from geoseek.terrain.dem import compass_label

    assert compass_label(0.0) == "N"
    assert compass_label(90.0) == "E"
    assert compass_label(180.0) == "S"
    assert compass_label(270.0) == "W"
    assert compass_label(-1.0) is None   # flat sentinel


def test_describe_terrain_plain_language():
    from geoseek.terrain.features import TerrainFeatures, describe_terrain

    f = TerrainFeatures(elevation_m=94.0, slope_deg=2.0, aspect_deg=135.0, aspect_compass="SE",
                        distance_to_water_m=340.0, distance_to_built_up_m=120.0)
    s = describe_terrain(f)
    assert "94 m elevation" in s
    assert "2 degree slope" in s and "SE" in s
    assert "340 m from the river channel" in s
    assert "120 m from the nearest built-up area" in s


def test_describe_terrain_handles_missing_fields():
    from geoseek.terrain.features import TerrainFeatures, describe_terrain

    empty = TerrainFeatures(None, None, None, None, None, None)
    assert describe_terrain(empty) == "terrain context unavailable"


def test_terrain_features_as_dict_states_measured_vs_derived():
    from geoseek.terrain.features import TerrainFeatures

    f = TerrainFeatures(94.0, 2.0, 135.0, "SE", 340.0, 120.0)
    d = f.as_dict()
    assert "measured" in d["provenance"]["elevation_m"]
    assert "derived" in d["provenance"]["distance_to_water_m"]
    assert "derived" in d["provenance"]["distance_to_built_up_m"]


# --------------------------------------------------------------------------
# Step C - watch areas through the MetadataRepository seam
# --------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    r = SQLiteMetadataRepository(tmp_path / "watch.sqlite")
    yield r
    r.close()


def test_seam_still_complete_after_watch_methods_added():
    assert issubclass(SQLiteMetadataRepository, MetadataRepository)
    assert not getattr(SQLiteMetadataRepository, "__abstractmethods__", set())


def test_watch_area_create_get_update_delete(repo):
    w = repo.create_watch_area(WatchArea(
        watch_id="", name="Test watch", bbox=(82.0, 26.3, 82.9, 27.2),
        change_types=("construction",), min_confidence=0.6, created_by="tester",
    ))
    assert w.watch_id.startswith("watch_")
    assert w.created_at and w.updated_at == w.created_at

    got = repo.get_watch_area(w.watch_id)
    assert got == w

    import dataclasses
    updated = repo.update_watch_area(dataclasses.replace(got, name="Renamed", active=False))
    assert updated.name == "Renamed" and updated.active is False
    assert updated.updated_at >= w.updated_at

    assert repo.list_watch_areas(active_only=True) == []
    assert len(repo.list_watch_areas()) == 1

    repo.delete_watch_area(w.watch_id)
    assert repo.get_watch_area(w.watch_id) is None
    assert repo.list_watch_areas() == []


def test_update_watch_area_missing_raises(repo):
    from geoseek.catalog.sqlite_repository import CatalogError

    with pytest.raises(CatalogError):
        repo.update_watch_area(WatchArea(watch_id="nope", name="x"))


def test_watch_areas_are_editable_unlike_the_audit_trail(repo):
    """Contrast with analyst_decisions (Phase 6): watch areas are a normal,
    mutable/deletable table by design - update/delete must NOT raise."""
    w = repo.create_watch_area(WatchArea(watch_id="", name="Editable"))
    import dataclasses
    repo.update_watch_area(dataclasses.replace(w, name="Edited"))   # must not raise
    repo.delete_watch_area(w.watch_id)                              # must not raise
    # sanity: the audit trail's append-only guard is untouched by this change
    repo.record_analyst_decision(AnalystDecision(decision_id="", candidate_id="c1", decision="confirm"))
    d = repo.list_analyst_decisions(candidate_id="c1")[0]
    with pytest.raises(Exception):
        repo.connection.execute("UPDATE analyst_decisions SET decision='reject' WHERE decision_id=?",
                                (d.decision_id,))


def test_notifications_round_trip_and_seen_toggle(repo):
    w = repo.create_watch_area(WatchArea(watch_id="", name="W"))
    n = repo.record_notification(WatchNotification(
        notification_id="", watch_id=w.watch_id, observation_id="obs_2026",
        candidate_ids=("c1", "c2"),
    ))
    assert n.notification_id.startswith("notif_")
    assert repo.list_notifications() == [n]
    assert repo.list_notifications(watch_id=w.watch_id) == [n]
    assert repo.list_notifications(watch_id="other") == []
    assert repo.list_notifications(unseen_only=True) == [n]

    repo.mark_notification_seen(n.notification_id)
    assert repo.list_notifications(unseen_only=True) == []
    seen = repo.list_notifications()[0]
    assert seen.seen is True


# --------------------------------------------------------------------------
# Step C - evaluator matching logic (pure functions over plain dicts)
# --------------------------------------------------------------------------


def _cand(cid, *, lon=82.2, lat=26.8, change_type="construction", confidence=0.7, detail="new built-up"):
    return {"candidate_id": cid, "centroid_lonlat": [lon, lat], "change_type": change_type,
            "confidence": confidence, "classification": {"change_type": change_type, "rule": "x", "detail": detail}}


def test_candidate_matches_bbox_type_confidence_and_text():
    from geoseek.watch.evaluator import candidate_matches

    watch = WatchArea(watch_id="w1", name="w", bbox=(82.0, 26.5, 82.5, 27.0),
                      change_types=("construction",), min_confidence=0.6, text_query="river")
    inside_matching = _cand("a", lon=82.2, lat=26.8, detail="new construction near the river")
    assert candidate_matches(watch, inside_matching) is True

    outside_bbox = _cand("b", lon=90.0, lat=26.8, detail="new construction near the river")
    assert candidate_matches(watch, outside_bbox) is False

    wrong_type = _cand("c", lon=82.2, lat=26.8, change_type="water_gain", detail="river flooding")
    assert candidate_matches(watch, wrong_type) is False

    low_conf = _cand("d", lon=82.2, lat=26.8, confidence=0.3, detail="new construction near the river")
    assert candidate_matches(watch, low_conf) is False

    no_keyword = _cand("e", lon=82.2, lat=26.8, detail="new construction, dry area")
    assert candidate_matches(watch, no_keyword) is False


def test_candidate_matches_unrestricted_watch_matches_everything_meeting_type():
    from geoseek.watch.evaluator import candidate_matches

    watch = WatchArea(watch_id="w1", name="anywhere", change_types=("road",))
    assert candidate_matches(watch, _cand("a", change_type="road", lon=0.0, lat=89.0)) is True
    assert candidate_matches(watch, _cand("b", change_type="other", lon=0.0, lat=89.0)) is False


def test_candidate_matches_polygon_takes_precedence_over_bbox():
    from geoseek.watch.evaluator import candidate_matches

    # a tiny polygon around (82.2, 26.8), with a bbox that would ALSO match a
    # point far outside the polygon - the polygon must be the one that governs.
    poly = "POLYGON((82.19 26.79, 82.21 26.79, 82.21 26.81, 82.19 26.81, 82.19 26.79))"
    watch = WatchArea(watch_id="w1", name="poly", bbox=(0.0, 0.0, 179.0, 89.0), polygon_wkt_4326=poly)
    assert candidate_matches(watch, _cand("a", lon=82.2, lat=26.8)) is True
    assert candidate_matches(watch, _cand("b", lon=100.0, lat=50.0)) is False


def test_evaluate_and_notify_only_fires_for_new_matches(repo):
    from geoseek.watch.evaluator import evaluate_and_notify

    w = repo.create_watch_area(WatchArea(watch_id="", name="W", change_types=("construction",)))
    candidates = [_cand("c1"), _cand("c2"), _cand("c3", change_type="water_gain")]

    fired = evaluate_and_notify(repo, candidates, observation_id="obs_a")
    assert len(fired) == 1
    assert set(fired[0].candidate_ids) == {"c1", "c2"}

    # re-running with the SAME candidates must not re-fire (nothing new)
    fired_again = evaluate_and_notify(repo, candidates, observation_id="obs_b")
    assert fired_again == []

    # a genuinely new matching candidate DOES fire again, with only the new id
    candidates.append(_cand("c4"))
    fired_third = evaluate_and_notify(repo, candidates, observation_id="obs_c")
    assert len(fired_third) == 1
    assert fired_third[0].candidate_ids == ("c4",)


def test_evaluate_and_notify_skips_inactive_watch_areas(repo):
    from geoseek.watch.evaluator import evaluate_and_notify

    repo.create_watch_area(WatchArea(watch_id="", name="W", active=False, change_types=("construction",)))
    fired = evaluate_and_notify(repo, [_cand("c1")], observation_id="obs_a")
    assert fired == []


# --------------------------------------------------------------------------
# ingest-time watch trigger (this round's addition): geoseek.ingest.pipeline
# also calls evaluate_and_notify at the end of every ingest, against whatever
# candidates already exist for that observation - not only at the end of a
# full `python -m geoseek.change.analyze` run. Isolated via monkeypatched
# OUT_DIR (a tmp change-report directory), never touches the real report.
# --------------------------------------------------------------------------


def test_ingest_trigger_fires_from_an_already_existing_report(repo, tmp_path, monkeypatch):
    import json

    import geoseek.change.analyze as analyze_mod
    from geoseek.ingest.pipeline import _evaluate_watch_areas_on_ingest

    monkeypatch.setattr(analyze_mod, "OUT_DIR", tmp_path)
    detail_path = tmp_path / "ayodhya_change_ranked_detail.json"
    detail_path.write_text(json.dumps([
        {**_cand("2019_2026_000001"), "pair": "S2B_44RPQ_20190330->S2C_44RPQ_20260308"},
        {**_cand("2019_2026_000002", change_type="water_gain"), "pair": "S2B_44RPQ_20190330->S2C_44RPQ_20260308"},
    ]), encoding="utf-8")

    repo.create_watch_area(WatchArea(watch_id="", name="ingest-trigger watch", change_types=("construction",)))

    # brand-new observation with no candidates yet in the report: real no-op
    assert _evaluate_watch_areas_on_ingest(repo, "S2X_44RPQ_20270101_0_L2A") == 0
    assert repo.list_notifications() == []

    # the newly-ingested observation IS one of the report's pair endpoints -
    # fires immediately, without any change/analyze re-run in between
    n_fired = _evaluate_watch_areas_on_ingest(repo, "S2C_44RPQ_20260308")
    assert n_fired == 1
    notifs = repo.list_notifications()
    assert len(notifs) == 1
    assert notifs[0].candidate_ids == ("2019_2026_000001",)

    # re-ingesting the same observation again must not re-fire (already notified)
    assert _evaluate_watch_areas_on_ingest(repo, "S2C_44RPQ_20260308") == 0
    assert len(repo.list_notifications()) == 1


def test_ingest_trigger_is_a_cheap_noop_with_no_report_or_no_watch_areas(repo, tmp_path, monkeypatch):
    import geoseek.change.analyze as analyze_mod
    from geoseek.ingest.pipeline import _evaluate_watch_areas_on_ingest

    monkeypatch.setattr(analyze_mod, "OUT_DIR", tmp_path)
    # no watch areas at all -> short-circuits before even looking for a report
    assert _evaluate_watch_areas_on_ingest(repo, "any_obs") == 0

    repo.create_watch_area(WatchArea(watch_id="", name="W", change_types=("construction",)))
    # watch areas exist, but no report file on disk yet
    assert _evaluate_watch_areas_on_ingest(repo, "any_obs") == 0


# --------------------------------------------------------------------------
# Step D - sector brief aggregation (fabricated in-memory service state)
# --------------------------------------------------------------------------


def test_sector_brief_aggregates_counts_area_and_top_candidates(tmp_path):
    from geoseek.analyst.service import AnalystService

    svc = AnalystService.__new__(AnalystService)   # bypass __init__/reload (no live report needed)
    svc.repo = SQLiteMetadataRepository(tmp_path / "brief.sqlite")
    svc.details = [
        {"candidate_id": "a", "change_type": "construction", "confidence": 0.9, "area_m2": 500.0,
         "significance": 0.8, "queue_score": 0.85, "persistence": "persistent", "centroid_lonlat": [82.2, 26.8],
         "earliest_supported": ["2024-03-08", "2025-03-08"], "terrain": {"elevation_m": 90},
         "_geometry": {"type": "Point", "coordinates": [82.2, 26.8]}},
        {"candidate_id": "b", "change_type": "construction", "confidence": 0.7, "area_m2": 300.0,
         "significance": 0.5, "queue_score": 0.6, "persistence": "recent", "centroid_lonlat": [82.3, 26.9],
         "earliest_supported": ["2025-03-08", "2026-03-08"], "terrain": {},
         "_geometry": {"type": "Point", "coordinates": [82.3, 26.9]}},
        {"candidate_id": "c", "change_type": "water_gain", "confidence": 0.95, "area_m2": 1000.0,
         "significance": 0.9, "queue_score": 0.92, "persistence": "persistent", "centroid_lonlat": [82.1, 26.7],
         "earliest_supported": ["2019-03-30", "2024-03-08"], "terrain": {"elevation_m": 88},
         "_geometry": {"type": "Point", "coordinates": [82.1, 26.7]}},
    ]
    svc._by_id = {c["candidate_id"]: c for c in svc.details}
    svc.report = {"observations": ["S2B_44RPQ_20190330_1_L2A_scaled", "S2C_44RPQ_20260308_0_L2A_scaled"],
                 "pairs": {"2019-2026": {"context": {"radiometric_reliability": 0.5}}},
                 "domain_gap_statement": "test caveat", "sar_corroboration": {"available": False},
                 "aoi": "Ayodhya test AOI"}
    svc.span_pair_name = "2019-2026"
    svc.model_info = {"name": "FCSiamDiff", "threshold": 0.8, "weights_sha256": "abc123"}
    svc.git_commit = "deadbee"
    svc.pipeline_version = "0.1.0"
    svc.report_mtime = "2026-09-12T00:00:00+00:00"

    brief = svc.sector_brief()
    assert brief["total_candidates"] == 3
    by_type = {d["change_type"]: d for d in brief["by_change_type"]}
    assert by_type["construction"]["count"] == 2
    assert by_type["construction"]["area_m2"] == pytest.approx(800.0)
    assert by_type["water_gain"]["count"] == 1
    assert brief["total_area_m2"] == pytest.approx(1800.0)
    assert [m["candidate_id"] for m in brief["most_significant"]] == ["c", "a", "b"]
    assert "test caveat" in brief["imagery_and_quality_caveats"]
    assert any("radiometric reliability" in c for c in brief["imagery_and_quality_caveats"])
    assert any("terrain evidence" in c for c in brief["imagery_and_quality_caveats"])
    assert brief["provenance"]["weights_sha256"] == "abc123"
    assert "SECTOR SUMMARY BRIEF" in brief["text"]
    assert "construction" in brief["text"]
    svc.repo.close()


def test_sector_brief_bbox_filters_candidates(tmp_path):
    from geoseek.analyst.service import AnalystService

    svc = AnalystService.__new__(AnalystService)
    svc.repo = SQLiteMetadataRepository(tmp_path / "brief2.sqlite")
    svc.details = [
        {"candidate_id": "a", "change_type": "construction", "confidence": 0.9, "area_m2": 500.0,
         "significance": 0.8, "queue_score": 0.85, "persistence": "persistent", "centroid_lonlat": [82.2, 26.8],
         "earliest_supported": ["2024-03-08", "2025-03-08"], "_bounds": (82.19, 26.79, 82.21, 26.81),
         "terrain": {}, "_geometry": {"type": "Point", "coordinates": [82.2, 26.8]}},
        {"candidate_id": "far", "change_type": "construction", "confidence": 0.9, "area_m2": 500.0,
         "significance": 0.8, "queue_score": 0.85, "persistence": "persistent", "centroid_lonlat": [10.0, 10.0],
         "earliest_supported": ["2024-03-08", "2025-03-08"], "_bounds": (9.9, 9.9, 10.1, 10.1), "terrain": {},
         "_geometry": {"type": "Point", "coordinates": [10.0, 10.0]}},
    ]
    svc._by_id = {c["candidate_id"]: c for c in svc.details}
    svc.report = {"observations": [], "pairs": {}, "domain_gap_statement": "", "sar_corroboration": {},
                 "aoi": "test"}
    svc.span_pair_name = "x"
    svc.model_info, svc.git_commit, svc.pipeline_version, svc.report_mtime = {}, "", "", ""

    brief = svc.sector_brief(bbox=(82.0, 26.5, 82.5, 27.0))
    assert brief["total_candidates"] == 1
    assert brief["most_significant"][0]["candidate_id"] == "a"
    svc.repo.close()


# --------------------------------------------------------------------------
# API endpoints (integration, gated on the production catalog + change report
# existing - same client fixture pattern as test_phase6.py; writes are
# isolated to a throwaway COPY of the production catalog)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import shutil

    from geoseek.change.analyze import OUT_DIR
    from geoseek.config import get_settings

    idx = get_settings().index_dir
    if not (idx / "tiles.faiss").is_file():
        pytest.skip("no production index - run the ingest pipeline")
    if not (OUT_DIR / "ayodhya_change_report.json").is_file():
        pytest.skip("no change report - run `python -m geoseek.change.analyze`")

    from fastapi.testclient import TestClient

    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    from geoseek.search.api import app

    with TestClient(app) as c:
        if c.get("/health").json().get("vectors", 0) == 0:
            pytest.skip("production index empty")
        from geoseek.search import api as api_mod

        tmp_db = tmp_path_factory.mktemp("phase8_watch_audit") / "tiles.sqlite"
        shutil.copy2(idx / "tiles.sqlite", tmp_db)
        api_mod._analyst.repo = SQLiteMetadataRepository(tmp_db)
        yield c


def test_watch_area_crud_via_api(client):
    created = client.post("/watch-areas", json={
        "name": "pytest watch", "bbox": [82.0, 26.3, 82.9, 27.2],
        "change_types": ["construction"], "min_confidence": 0.5,
    }).json()
    wid = created["watch_id"]
    assert created["name"] == "pytest watch" and created["active"] is True

    assert client.get(f"/watch-areas/{wid}").json() == created
    assert any(w["watch_id"] == wid for w in client.get("/watch-areas").json()["watch_areas"])

    updated = client.put(f"/watch-areas/{wid}", json={"name": "renamed", "active": False}).json()
    assert updated["name"] == "renamed" and updated["active"] is False
    assert updated["bbox"] == created["bbox"]              # untouched field survives a partial update

    assert client.get("/watch-areas", params={"active_only": True}).json()["watch_areas"] == \
        [w for w in client.get("/watch-areas").json()["watch_areas"] if w["active"]]

    client.delete(f"/watch-areas/{wid}")
    assert client.get(f"/watch-areas/{wid}").status_code == 404


def test_watch_area_update_missing_is_404(client):
    assert client.put("/watch-areas/does_not_exist", json={"name": "x"}).status_code == 404


def test_notifications_endpoint_and_seen_toggle(client):
    created = client.post("/watch-areas", json={"name": "pytest notif watch", "change_types": []}).json()
    listing = client.get("/notifications", params={"watch_id": created["watch_id"]}).json()
    assert listing["notifications"] == []   # brand new watch area has no history yet
    client.delete(f"/watch-areas/{created['watch_id']}")

    # the demo watch area created before the Phase 8 pipeline run DID fire -
    # exercise the real notification shape + candidate links if it's there.
    all_notifs = client.get("/notifications").json()["notifications"]
    if all_notifs:
        n = all_notifs[0]
        assert {"notification_id", "watch_id", "observation_id", "candidates"} <= set(n)
        if n["candidates"]:
            assert "link" in n["candidates"][0]
        if not n["seen"]:
            client.post(f"/notifications/{n['notification_id']}/seen")
            assert client.get("/notifications", params={"watch_id": n["watch_id"]}).json()["notifications"][0]["seen"]


def test_sector_brief_endpoint(client):
    d = client.get("/sector-brief").json()
    assert d["total_candidates"] > 0
    assert "SECTOR SUMMARY BRIEF" in d["text"]
    assert d["provenance"]["weights_sha256"]
    assert "text_path" not in d   # write=False by default - nothing written to disk

    written = client.get("/sector-brief", params={"write": True}).json()
    assert written["text_path"].endswith(".txt") and written["json_path"].endswith(".json")
    from pathlib import Path
    assert Path(written["text_path"]).is_file() and Path(written["json_path"]).is_file()


def test_sector_brief_bbox_narrows_results(client):
    full = client.get("/sector-brief").json()
    narrow = client.get("/sector-brief", params={"bbox": "82.0,26.3,82.1,26.4"}).json()
    assert narrow["total_candidates"] <= full["total_candidates"]

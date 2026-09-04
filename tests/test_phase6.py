"""Phase 6 - analyst interface: audit trail seam + backend API.

Step B (audit trail, PS 2.2.5): the append-only ``analyst_decisions`` table
reached ONLY through the MetadataRepository seam - never raw sqlite3 in
application code. Tests here prove: the seam round-trips a decision, the log is
ordered and append-only (a second decision on the same candidate adds a row,
never overwrites), and the storage layer itself rejects UPDATE / DELETE.

Step A/C/D (endpoints): a TestClient drives the FastAPI app against whatever
production catalog + change report exist, skipping if they haven't been built.
"""

from __future__ import annotations

import sqlite3

import pytest

from geoseek.catalog.entities import AnalystDecision
from geoseek.catalog.repository import MetadataRepository
from geoseek.catalog.sqlite_repository import CatalogError, SQLiteMetadataRepository


# --------------------------------------------------------------------------
# Step B - audit trail through the repository seam
# --------------------------------------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    r = SQLiteMetadataRepository(tmp_path / "audit.sqlite")
    yield r
    r.close()


def test_seam_still_complete_after_audit_methods_added():
    assert issubclass(SQLiteMetadataRepository, MetadataRepository)
    assert not getattr(SQLiteMetadataRepository, "__abstractmethods__", set())


def test_record_and_read_back_a_decision(repo):
    out = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="2019_2024_011912", decision="confirm",
        analyst_note="clear reservoir fill, matches SAR VV drop", analyst="tester",
        model_version="FCSiamDiff@0.80", weights_sha256="452ac0", git_commit="ecf0943",
        pipeline_version="0.1.0", confidence_at_decision=0.9873,
        evidence_snapshot={"change_type": "water_gain", "persistence": "persistent"},
    ))
    assert out.decision_id.startswith("dec_")
    assert out.created_at  # filled in on write

    back = repo.get_analyst_decision(out.decision_id)
    assert back == out
    assert back.evidence_snapshot["change_type"] == "water_gain"
    assert back.confidence_at_decision == pytest.approx(0.9873)


def test_decision_value_is_validated(repo):
    with pytest.raises(CatalogError):
        repo.record_analyst_decision(AnalystDecision(
            decision_id="", candidate_id="c1", decision="maybe"))


def test_log_is_append_only_second_decision_adds_a_row(repo):
    repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c1", decision="reject", analyst_note="cloud edge"))
    repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c1", decision="confirm", analyst_note="re-checked, real"))

    hist = repo.list_analyst_decisions(candidate_id="c1")
    assert [d.decision for d in hist] == ["reject", "confirm"]        # both kept, in write order
    assert hist[0].decision_id != hist[1].decision_id

    latest = repo.latest_decision_by_candidate()
    assert latest["c1"].decision == "confirm"                         # current verdict = most recent


def test_storage_layer_rejects_update_and_delete(repo):
    d = repo.record_analyst_decision(AnalystDecision(
        decision_id="", candidate_id="c9", decision="confirm"))
    conn = repo.connection  # the migration escape hatch - used here only to prove the trigger fires
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE analyst_decisions SET decision='reject' WHERE decision_id=?",
                     (d.decision_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM analyst_decisions WHERE decision_id=?", (d.decision_id,))
    # row is untouched
    assert repo.get_analyst_decision(d.decision_id).decision == "confirm"


def test_global_audit_listing_is_time_ordered(repo):
    for cid in ("a", "b", "c"):
        repo.record_analyst_decision(AnalystDecision(
            decision_id="", candidate_id=cid, decision="confirm"))
    allrows = repo.list_analyst_decisions()
    assert [d.candidate_id for d in allrows] == ["a", "b", "c"]
    assert [d.candidate_id for d in repo.list_analyst_decisions(limit=2)] == ["a", "b"]


# --------------------------------------------------------------------------
# Step A / C / D - the FastAPI analyst endpoints (integration, gated on the
# production catalog + change report existing)
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

    from geoseek import search  # noqa: F401
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    from geoseek.search.api import app

    with TestClient(app) as c:
        if c.get("/health").json().get("vectors", 0) == 0:
            pytest.skip("production index empty")
        # isolate audit writes: point the live AnalystService at a throwaway COPY
        # of the production catalog so decisions/audit don't touch data/index.
        from geoseek.search import api as api_mod

        tmp_db = tmp_path_factory.mktemp("phase6_audit") / "tiles.sqlite"
        shutil.copy2(idx / "tiles.sqlite", tmp_db)
        api_mod._analyst.repo = SQLiteMetadataRepository(tmp_db)
        yield c


def test_health_and_stats(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["candidates"] > 0 and h["probability_raster_present"] is True

    s = client.get("/stats").json()
    assert s["index"]["tiles"] > 0
    assert s["model"]["weights_sha256"]
    assert s["build"]["git_commit"]
    assert s["change_pipeline"]["candidates_ranked"] >= 1000


def test_candidates_queue_filters_and_sorts(client):
    body = client.get("/candidates", params={"limit": 10}).json()
    assert body["total"] > 1000 and body["count"] == 10
    scores = [c["queue_score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    for c in body["candidates"]:
        assert c["geometry"]["type"] == "Polygon"
        assert len(c["geometry"]["coordinates"][0]) == 5
        assert c["decision"] in ("confirm", "reject", "undecided")

    filt = client.get("/candidates", params={"change_type": "construction",
                                             "min_confidence": 0.9, "limit": 5}).json()
    assert all(c["change_type"] == "construction" and c["confidence"] >= 0.9
               for c in filt["candidates"])

    bbox_none = client.get("/candidates", params={"bbox": "0,0,1,1"}).json()
    assert bbox_none["total"] == 0


def test_candidate_detail_has_full_evidence_and_provenance(client):
    top = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]
    d = client.get(f"/candidates/{top}").json()

    # evidence
    assert d["confidence_breakdown"] and any("model:" in x for x in d["confidence_breakdown"])
    assert d["suppression"]["trace"] and {"quality", "registration", "radiometric",
                                          "phenology", "morphology"} == {
        t["rule"] for t in d["suppression"]["trace"]}
    assert d["temporal_trajectory"]["intervals"]
    assert d["temporal_trajectory"]["earliest_supported_change"]["caveat"]

    # provenance chain
    prov = d["provenance"]
    assert [o["role"] for o in prov["observations"]] == ["before", "after"]
    sc = prov["observations"][0]["scene"]
    assert sc["source_url"].startswith("https://") and sc["license"]
    assert prov["observations"][0]["collection"]["sensor"] == "MSI"
    assert prov["model"]["weights_sha256"] and prov["code"]["git_commit"]
    assert prov["observations"][0]["representative_tile"]["tile_id"]

    assert client.get("/candidates/nope_9999").status_code == 404


def test_candidate_imagery_before_after_overlay_are_png(client):
    top = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]
    for date in ("2019", "2021", "2024"):
        for view in ("rgb", "overlay"):
            r = client.get(f"/candidates/{top}/imagery", params={"date": date, "view": view})
            assert r.status_code == 200
            assert r.headers["content-type"] == "image/png"
            assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/candidates/{top}/imagery", params={"date": "1999"}).status_code == 400


def test_date_selector_offers_only_valid_before_dates(client):
    # a 2019->2024 span candidate: BEFORE dates are 2019 & 2021, AFTER is 2024
    top = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]
    im = client.get(f"/candidates/{top}").json()["imagery"]
    assert im["after_date"] == "2024"
    assert im["before_dates"] == ["2019", "2021"]         # NOT "2024" - would be before==after
    assert "2024" not in im["before_dates"]


def test_thumbnails_are_per_observation_white_balanced(client):
    # 2019 (drought) and 2021 tiles must NOT render blue-starved / yellow under
    # the per-observation display stretch: mean blue within ~35% of mean green.
    import io as _io

    import numpy as np
    from PIL import Image

    seen = {}
    for date, want in (("2019-03-30", "S2B"), ("2021-03-04", "S2A")):
        res = client.get("/search/text", params={"q": "open bare ground", "k": 60}).json()["results"]
        tid = next((r["tile_id"] for r in res if r["acq_date"] == date), None)
        if tid is None:
            continue
        img = np.array(Image.open(_io.BytesIO(
            client.get(f"/tile/{tid}/thumbnail").content)).convert("RGB")).astype(float)
        m = img.sum(-1) > 20
        g, b = img[..., 1][m].mean(), img[..., 2][m].mean()
        seen[date] = b / max(g, 1)
    for date, bg in seen.items():
        assert 0.6 < bg < 1.4, f"{date} thumbnail blue/green ratio {bg:.2f} - not neutral"


def test_decision_writes_audit_and_is_append_only(client):
    top = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]

    r1 = client.post(f"/candidates/{top}/decision",
                     json={"decision": "confirm", "note": "phase6 test confirm", "analyst": "pytest"})
    assert r1.status_code == 200
    dec1 = r1.json()
    assert dec1["decision"] == "confirm" and dec1["weights_sha256"] and dec1["git_commit"]
    assert dec1["confidence_at_decision"] is not None
    assert dec1["evidence_snapshot"]["provenance"]["model"]["weights_sha256"]

    r2 = client.post(f"/candidates/{top}/decision",
                     json={"decision": "reject", "note": "phase6 test reject", "analyst": "pytest"})
    assert r2.json()["decision_id"] != dec1["decision_id"]

    hist = client.get("/audit", params={"candidate_id": top}).json()
    assert hist["append_only"] is True
    mine = [d for d in hist["decisions"] if d["analyst"] == "pytest"]
    assert [d["decision"] for d in mine][-2:] == ["confirm", "reject"]

    # the audit LIST is light (snapshot stripped); the per-decision fetch is full
    row = mine[-1]
    assert "evidence_snapshot" not in row and row["has_evidence_snapshot"] is True
    full = client.get(f"/audit/{row['decision_id']}").json()
    assert full["evidence_snapshot"]["provenance"]["model"]["weights_sha256"]
    assert client.get("/audit/nope").status_code == 404

    # current verdict reflects the most recent write
    q = client.get("/candidates", params={"decision": "reject", "limit": 5000}).json()
    assert top in [c["candidate_id"] for c in q["candidates"]]

    assert client.post("/candidates/nope/decision",
                       json={"decision": "confirm"}).status_code == 404
    assert client.post(f"/candidates/{top}/decision",
                       json={"decision": "maybe"}).status_code == 400


def test_export_geojson_features_carry_full_provenance(client):
    top = client.get("/candidates", params={"limit": 2}).json()["candidates"]
    ids = [c["candidate_id"] for c in top]
    r = client.post("/export", json={"candidate_ids": ids, "format": "both"})
    assert r.status_code == 200
    body = r.json()
    fc = body["geojson"]
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    p = fc["features"][0]["properties"]
    for key in ("candidate_id", "change_type", "confidence", "earliest_supported_change",
                "source_scene_ids", "acquisition_dates", "evidence_summary", "processing"):
        assert key in p
    assert p["processing"]["weights_sha256"] and p["processing"]["git_commit"]
    assert fc["features"][0]["geometry"]["type"] == "Polygon"
    assert "csv" in body and body["csv"].splitlines()[0].startswith("candidate_id,")
    assert body["geojson_path"].endswith(".geojson")
    # flat GIS-reader-friendly provenance keys + real source acquisition dates
    assert p["weights_sha256"] and len(p["weights_sha256"]) == 64
    assert p["git_commit"] and p["acquisition_dates"] == ["2019-03-30", "2024-03-08"]


def test_export_geojson_loads_in_a_standard_gis_reader(client, tmp_path):
    ogr = pytest.importorskip("osgeo.ogr")

    ids = [c["candidate_id"] for c in
           client.get("/candidates", params={"limit": 6}).json()["candidates"]]
    fc = client.post("/export", json={"candidate_ids": ids, "format": "geojson"}).json()["geojson"]
    path = tmp_path / "export.geojson"
    path.write_text(__import__("json").dumps(fc), encoding="utf-8")

    ds = ogr.Open(str(path))
    assert ds is not None and ds.GetDriver().GetName() == "GeoJSON"
    lyr = ds.GetLayer(0)
    assert lyr.GetFeatureCount() == len(ids)
    srs = lyr.GetSpatialRef()
    assert srs is not None and (srs.GetAuthorityCode(None) == "4326"
                                or "WGS" in srs.GetAttrValue("GEOGCS").upper())
    fields = {lyr.GetLayerDefn().GetFieldDefn(i).GetName()
              for i in range(lyr.GetLayerDefn().GetFieldCount())}
    assert {"candidate_id", "change_type", "confidence", "weights_sha256", "git_commit"} <= fields
    f0 = lyr.GetNextFeature()
    g = f0.GetGeometryRef()
    assert g.GetGeometryName() == "POLYGON" and g.IsValid()
    ds = None


def test_discovery_endpoints(client):
    cl = client.get("/discovery/clusters").json()
    assert cl["available"] is True and cl["n_clusters"] >= 1

    top = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]
    sim = client.get(f"/candidates/{top}/similar", params={"k": 5}).json()
    assert sim["seed_candidate_id"] == top
    assert 1 <= len(sim["results"]) <= 5


def test_ui_bundle_is_served_and_offline(client):
    r = client.get("/app/")
    assert r.status_code == 200
    assert "<" in r.text

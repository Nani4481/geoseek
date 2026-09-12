"""Presentation layer - the demo-facing overview + guided-demo support that sits
on top of the analyst UI.

Everything here is a read-only projection of data the analyst service already
holds; these tests prove the projection is well-formed, that its ids resolve
through the real analyst endpoints (the guided demo uses the live API - no mock
data), and that the existing analyst endpoints are byte-for-byte unaffected.

Gated on the production catalog + change report existing, exactly like the rest
of test_phase6.
"""

from __future__ import annotations

import pytest

from geoseek.analyst.service import _confidence_band, _region_of


# --------------------------------------------------------------------------
# pure helpers - no fixtures needed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aoi,region", [
    ("ayodhya-82km", "ayodhya"),
    ("ayodhya_44RPQ_scaled_82km", "ayodhya"),
    ("kerala_backwaters_43PFL_diverse", "kerala_backwaters"),
    ("delhi_ncr_43RGM_diverse", "delhi_ncr"),
    ("jaisalmer_42RYR_diverse", "jaisalmer"),
    (None, "unknown"),
])
def test_region_of_strips_mgrs_and_staging_suffixes(aoi, region):
    assert _region_of(aoi) == region


def test_confidence_band_thresholds():
    assert _confidence_band(0.97) == "High"
    assert _confidence_band(0.85) == "High"
    assert _confidence_band(0.84) == "Medium"
    assert _confidence_band(0.60) == "Medium"
    assert _confidence_band(0.59) == "Low"
    assert _confidence_band(None) == "Low"


# --------------------------------------------------------------------------
# the endpoint (integration)
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

        tmp_db = tmp_path_factory.mktemp("phase6_pres_audit") / "tiles.sqlite"
        shutil.copy2(idx / "tiles.sqlite", tmp_db)
        api_mod._analyst.repo = SQLiteMetadataRepository(tmp_db)
        yield c


def test_summary_counters_are_present_and_sane(client):
    s = client.get("/presentation/summary").json()
    assert s["offline"] is True
    c = s["counters"]
    for k in ("tiles_indexed", "regions", "scenes", "change_candidates", "high_confidence"):
        assert isinstance(c[k], int) and c[k] >= 0
    # was >=1000 at 1104 candidates (3-date, 2019-2024 span); Phase 8 extended the
    # stack to 5 dates and the span pair to 2019-2026, giving 841.
    assert c["change_candidates"] >= 500
    assert c["tiles_indexed"] > c["scenes"] > 0
    assert c["regions"] >= 1
    # counters agree with the analyst /stats + /candidates views they summarise
    st = client.get("/stats").json()
    assert c["tiles_indexed"] == st["index"]["tiles"]
    assert c["scenes"] == st["index"]["scenes"]
    assert c["change_candidates"] == client.get("/candidates", params={"limit": 1}).json()["total"]
    assert c["change_candidates"] == sum(s["change_type_distribution"].values())


def test_summary_featured_are_real_resolvable_candidates(client):
    s = client.get("/presentation/summary").json()
    feat = s["featured"]
    assert 1 <= len(feat) <= 4
    ids = [f["candidate_id"] for f in feat]
    types = [f["change_type"] for f in feat]
    assert len(ids) == len(set(ids))                                    # no repeats
    # diversified: as many distinct change types as cards, up to what exists
    assert len(set(types)) == len(feat) or len(feat) < 4

    for f in feat:
        for k in ("candidate_id", "change_type_human", "confidence_band",
                  "persistence_human", "caption", "before_date", "after_date"):
            assert f[k], f"featured card missing {k}"
        assert f["confidence_band"] in ("High", "Medium", "Low")
        assert f["before_date"] != f["after_date"]
        # the id resolves through the real analyst detail endpoint
        d = client.get(f"/candidates/{f['candidate_id']}")
        assert d.status_code == 200
        assert d.json()["change_type"] == f["change_type"]
        # and both before/after thumbnails actually render as PNG
        for url in (f["imagery"]["before"], f["imagery"]["after"]):
            r = client.get(url)
            assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_summary_demo_block_targets_a_water_gain_candidate(client):
    s = client.get("/presentation/summary").json()
    demo = s["demo"]
    assert demo["search_query"]
    assert len(demo["steps"]) == 4
    cid = demo["water_gain_candidate_id"]
    assert demo["discovery_seed"] == cid
    d = client.get(f"/candidates/{cid}").json()
    assert d["change_type"] == "water_gain"
    # the guided demo's discovery step calls this exact endpoint
    sim = client.get(f"/candidates/{cid}/similar", params={"k": 6}).json()
    assert len(sim["results"]) >= 1


def test_persistence_phrasing_matches_the_trajectory(client):
    """'Confirmed across N later observations' must equal the number of
    acquisitions at/after the earliest supported change window."""
    s = client.get("/presentation/summary").json()
    obs_dates = s["observation_dates"]
    assert len(obs_dates) == 5   # Phase 8 Step A: 2019/2021/2024/2025/2026
    for f in s["featured"]:
        d = client.get(f"/candidates/{f['candidate_id']}").json()
        win = d["temporal_trajectory"]["earliest_supported_change"]["window"]
        expect = sum(1 for x in obs_dates if x >= win[1]) or 1
        if f["persistence"] in ("persistent", "progressive"):
            assert f"{expect} later observation" in f["persistence_human"]
        assert f["later_observations"] == expect


def test_imagery_scale_param_upsamples_and_default_is_unchanged(client):
    import io

    from PIL import Image

    cid = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]
    base = client.get(f"/candidates/{cid}/imagery", params={"date": "2019", "view": "rgb"})
    x3 = client.get(f"/candidates/{cid}/imagery", params={"date": "2019", "view": "rgb", "scale": 3})
    assert base.status_code == x3.status_code == 200
    w0, h0 = Image.open(io.BytesIO(base.content)).size
    w3, h3 = Image.open(io.BytesIO(x3.content)).size
    assert (w3, h3) == (w0 * 3, h0 * 3)
    # scale=1 is the native render, i.e. the pre-existing behaviour
    one = client.get(f"/candidates/{cid}/imagery", params={"date": "2019", "view": "rgb", "scale": 1})
    assert one.content == base.content
    assert client.get(f"/candidates/{cid}/imagery",
                      params={"date": "2019", "scale": 9}).status_code == 422


def test_existing_analyst_endpoints_unaffected(client):
    """The presentation layer must not have perturbed the analyst API surface."""
    q = client.get("/candidates", params={"limit": 5}).json()
    assert q["total"] > 500 and len(q["candidates"]) == 5
    top = q["candidates"][0]["candidate_id"]
    d = client.get(f"/candidates/{top}").json()
    assert d["confidence_breakdown"] and len(d["suppression"]["trace"]) == 5
    assert d["provenance"]["observations"][0]["scene"]["source_url"].startswith("https://")
    assert d["imagery"]["after_date"] not in d["imagery"]["before_dates"]
    assert len(d["imagery"]["before_dates"]) >= 2
    assert client.get("/app/").status_code == 200
    # the SPA bundle is still self-contained: no external origins / hosts referenced
    for asset in ("/app/app.js", "/app/style.css", "/app/index.html"):
        low = client.get(asset).text.lower()
        assert "http://" not in low and "https://" not in low
        assert "fonts.googleapis" not in low and "cdn." not in low
        assert 'src="//' not in low and "url(http" not in low

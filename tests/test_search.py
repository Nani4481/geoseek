"""Tests for the geoseek Phase 2 semantic search module.

Uses the small Phase 1 demo AOI (already staged, 9+9 tiles) ingested into a
throwaway temp index, so these tests are fast and don't depend on the
Phase 2 scale-up having run. A couple of light integration checks against
the real production index (data/index) are skipped if it's empty.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from geoseek.config import get_settings
from geoseek.ingest.pipeline import ingest_scene
from geoseek.search.engine import SearchEngine, SearchFilters

SCENE_A = "S2B_44RPQ_20190330_1_L2A"  # 2019-03-30, Sentinel-2B
SCENE_B = "S2A_44RPQ_20240308_0_L2A"  # 2024-03-08, Sentinel-2A


def _scene_dir(scene_id: str):
    path = get_settings().datasets_dir / scene_id
    if not path.is_dir():
        pytest.skip(f"staged scene not found at {path} - run `python -m geoseek.staging.download_datasets`")
    return path


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("search_index")
    ingest_scene(_scene_dir(SCENE_A), save_sample=False, index_dir=index_dir, record_manifest=False)
    ingest_scene(_scene_dir(SCENE_B), save_sample=False, index_dir=index_dir, record_manifest=False)

    eng = SearchEngine(index_dir=index_dir)
    yield eng
    eng.close()


# --------------------------------------------------------------------------
# SearchFilters (pure unit tests, no model/index needed)
# --------------------------------------------------------------------------


def test_filters_bbox_matches():
    f = SearchFilters(bbox=(82.0, 26.5, 82.5, 27.0))
    assert f.matches(lon=82.2, lat=26.8, acq_date="2020-01-01", sensor="Sentinel-2A", cloud_fraction=0.0)
    assert not f.matches(lon=90.0, lat=26.8, acq_date="2020-01-01", sensor="Sentinel-2A", cloud_fraction=0.0)


def test_filters_date_range_matches():
    f = SearchFilters(date_start="2020-01-01", date_end="2020-12-31")
    assert f.matches(lon=0, lat=0, acq_date="2020-06-15", sensor="x", cloud_fraction=0.0)
    assert not f.matches(lon=0, lat=0, acq_date="2019-06-15", sensor="x", cloud_fraction=0.0)
    assert not f.matches(lon=0, lat=0, acq_date="2021-06-15", sensor="x", cloud_fraction=0.0)


def test_filters_sensor_matches():
    f = SearchFilters(sensor="Sentinel-2A")
    assert f.matches(lon=0, lat=0, acq_date="x", sensor="Sentinel-2A", cloud_fraction=0.0)
    assert not f.matches(lon=0, lat=0, acq_date="x", sensor="Sentinel-2B", cloud_fraction=0.0)


def test_filters_max_cloud_fraction_matches():
    f = SearchFilters(max_cloud_fraction=0.2)
    assert f.matches(lon=0, lat=0, acq_date="x", sensor="s", cloud_fraction=0.1)
    assert not f.matches(lon=0, lat=0, acq_date="x", sensor="s", cloud_fraction=0.5)


def test_filters_none_means_no_constraint():
    f = SearchFilters()
    assert f.matches(lon=999, lat=999, acq_date="whenever", sensor="anything", cloud_fraction=1.0)


# --------------------------------------------------------------------------
# SearchEngine: text-to-image
# --------------------------------------------------------------------------


def test_search_text_returns_ranked_results(engine):
    results, latency_ms = engine.search_text("a river with sandbars", k=5)
    assert 1 <= len(results) <= 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)  # descending by score
    for r in results:
        assert -1.0001 <= r.score <= 1.0001  # cosine similarity range (unit-norm vectors)
        assert r.tile_id
        assert r.scene_id in (SCENE_A, SCENE_B)


def test_search_text_latency_under_1s(engine):
    _, latency_ms = engine.search_text("dense urban buildings", k=5)
    assert latency_ms < 1000.0


def test_search_text_provenance_fields_present(engine):
    results, _ = engine.search_text("open bare ground", k=3)
    for r in results:
        d = r.to_dict()
        for key in ("tile_id", "score", "lon", "lat", "acq_date", "sensor", "cloud_fraction", "scene_id"):
            assert key in d


# --------------------------------------------------------------------------
# SearchEngine: image-to-image
# --------------------------------------------------------------------------


def test_search_image_by_tile_id_self_match_is_top1(engine):
    any_tile_id = next(iter(engine._rows.values()))["tile_id"]
    results, _ = engine.search_image(tile_id=any_tile_id, k=5)
    assert results[0].tile_id == any_tile_id
    assert results[0].score == pytest.approx(1.0, abs=1e-4)  # cosine sim of a vector with itself


def test_search_image_unknown_tile_id_raises(engine):
    with pytest.raises(KeyError):
        engine.search_image(tile_id="does-not-exist", k=5)


def test_search_image_requires_exactly_one_arg(engine):
    with pytest.raises(ValueError):
        engine.search_image(k=5)  # neither tile_id nor image given
    with pytest.raises(ValueError):
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        any_tile_id = next(iter(engine._rows.values()))["tile_id"]
        engine.search_image(tile_id=any_tile_id, image_rgb_uint8=dummy, k=5)  # both given


def test_search_image_from_raw_image(engine):
    rgb = np.random.default_rng(0).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    results, latency_ms = engine.search_image(image_rgb_uint8=rgb, k=5)
    assert 1 <= len(results) <= 5
    assert latency_ms < 1000.0


# --------------------------------------------------------------------------
# filters applied through the real engine
# --------------------------------------------------------------------------


def test_search_text_filter_by_sensor(engine):
    filters = SearchFilters(sensor="Sentinel-2B")
    results, _ = engine.search_text("a river", k=20, filters=filters)
    assert results  # scene A (S2B) tiles exist
    assert all(r.sensor == "Sentinel-2B" for r in results)


def test_search_text_filter_by_date_range(engine):
    filters = SearchFilters(date_start="2024-01-01", date_end="2024-12-31")
    results, _ = engine.search_text("a river", k=20, filters=filters)
    assert results
    assert all(r.acq_date.startswith("2024") for r in results)


def test_search_text_filter_excludes_everything_outside_bbox(engine):
    filters = SearchFilters(bbox=(0.0, 0.0, 1.0, 1.0))  # nowhere near Ayodhya
    results, _ = engine.search_text("a river", k=20, filters=filters)
    assert results == []


# --------------------------------------------------------------------------
# thumbnail regeneration
# --------------------------------------------------------------------------


def test_thumbnail_is_valid_and_natural_looking(engine):
    any_tile_id = next(iter(engine._rows.values()))["tile_id"]
    img_bytes = engine.get_tile_thumbnail_png(any_tile_id)
    img = Image.open(__import__("io").BytesIO(img_bytes))
    assert img.format == "JPEG"  # thumbnails are JPEG q85 (small on the wire + in the LRU cache)
    img = img.convert("RGB")
    arr = np.array(img)
    assert arr.dtype == np.uint8
    # A properly scaled true-color tile spans a natural range: highlights near
    # white, real midtones - NOT the near-black cluster raw reflectance cast
    # straight to uint8 gives (DN~1200 median would render as u8~5, max~16).
    # Bounds are FIXED S2 true-color (0-0.3 reflectance, same for every tile and
    # date), plus the Phase 6 per-band cross-date harmonization; this tiny ~6 km
    # demo AOI has no deep water, so its darkest soil/built pixels land ~u8 50.
    assert arr.max() > 200
    assert arr.min() < 90
    assert 25 < float(arr.mean()) < 210


def test_thumbnail_unknown_tile_id_raises(engine):
    with pytest.raises(KeyError):
        engine.get_tile_thumbnail_png("does-not-exist")


# --------------------------------------------------------------------------
# offline search
# --------------------------------------------------------------------------


def test_search_text_works_fully_offline(engine, monkeypatch):
    def _blocked_connect(*args, **kwargs):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a, **k: 111)

    results, latency_ms = engine.search_text("agricultural fields", k=5)
    assert 1 <= len(results) <= 5
    assert latency_ms < 1000.0


# --------------------------------------------------------------------------
# FastAPI app (light integration check against whatever real index exists)
# --------------------------------------------------------------------------


def test_api_search_text_and_thumbnail_endpoints():
    settings = get_settings()
    index_path = settings.index_dir / "tiles.faiss"
    if not index_path.is_file():
        pytest.skip("no production index built yet - run the ingest pipeline first")

    from geoseek.search.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        if resp.json()["vectors"] == 0:
            pytest.skip("production index is empty")

        resp = client.get("/search/text", params={"q": "a river with sandbars", "k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "a river with sandbars"
        assert body["latency_ms"] < 1000.0
        assert "results" in body
        if body["results"]:
            first = body["results"][0]
            for key in ("tile_id", "score", "lon", "lat", "acq_date", "sensor", "cloud_fraction", "scene_id"):
                assert key in first

            tile_id = first["tile_id"]
            thumb_resp = client.get(f"/tile/{tile_id}/thumbnail")
            assert thumb_resp.status_code == 200
            assert thumb_resp.headers["content-type"] in ("image/jpeg", "image/png")

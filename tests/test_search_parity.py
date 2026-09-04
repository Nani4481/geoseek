"""Regression guard: the Phase 3.5 seams must not change a single search result.

Compares the live production SearchEngine against tests/fixtures/search_baseline.json,
which was frozen from the pre-refactor engine (see scripts/capture_search_baseline.py).
Skips if the production index / fixture is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine, SearchFilters

FIXTURE = Path(__file__).parent / "fixtures" / "search_baseline.json"
SCORE_TOL = 1e-4  # exact-same FAISS scan; only float re-accumulation noise expected


@pytest.fixture(scope="module")
def baseline_and_engine():
    if not FIXTURE.is_file():
        pytest.skip("no search_baseline.json fixture")
    if not (get_settings().index_dir / "tiles.faiss").is_file():
        pytest.skip("no production faiss index")
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))
    eng = SearchEngine()
    if eng.count() != base.get("index_vectors"):
        eng.close()
        pytest.skip(f"index has {eng.count()} vectors, baseline froze {base.get('index_vectors')}")
    yield base, eng
    eng.close()


def _assert_same(got, expected):
    assert [g.tile_id for g in got] == [e["tile_id"] for e in expected]
    for g, e in zip(got, expected):
        assert abs(float(g.score) - e["score"]) < SCORE_TOL, (g.tile_id, g.score, e["score"])


def test_text_queries_match_frozen_baseline(baseline_and_engine):
    base, eng = baseline_and_engine
    for q, expected in base["text"].items():
        got, _ = eng.search_text(q, k=base["k"])
        _assert_same(got, expected)


def test_filtered_queries_match_frozen_baseline(baseline_and_engine):
    base, eng = baseline_and_engine
    for label, spec in base.get("filtered", {}).items():
        got, _ = eng.search_text(spec["query"], k=base["k"], filters=SearchFilters(**spec["filters"]))
        _assert_same(got, spec["results"])


def test_image_queries_match_frozen_baseline(baseline_and_engine):
    base, eng = baseline_and_engine
    for tile_id, expected in base["image"].items():
        got, _ = eng.search_image(tile_id=tile_id, k=base["k"])
        _assert_same(got, expected)
        assert got[0].tile_id == tile_id and abs(got[0].score - 1.0) < SCORE_TOL

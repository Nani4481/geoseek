"""Full-coverage judge + metrics: the Phase 7a graders unchanged, and the metric definitions pinned by hand."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from geoseek.config import get_settings
from geoseek.eval import full_judge as FJ
from geoseek.eval import judge as J

EVAL_DIR = get_settings().data_dir / "eval_retrieval"
HAVE_PHASE7A = (EVAL_DIR / "judgments.json").is_file() and (EVAL_DIR / "tile_features.json").is_file()


# ------------------------------------------------------------------------------ metrics by hand


def test_precision_ndcg_recall_hand_computed():
    g = np.array([2, 1, 0, 2, 0, 1, 0, 0])                    # corpus of 8 tiles; relevant = 4 (rows 0,1,3,5)
    ranked = [2, 0, 1, 3]                                      # top-2 = [0 (grade 0), 2]
    m = FJ.query_metrics(ranked, g, 2)
    assert m["precision"] == 0.5 and m["n_relevant_in_candidates"] == 4
    assert m["recall"] == pytest.approx(1 / 4) and m["recall_capped"] == pytest.approx(1 / 2)
    dcg = 0 / math.log2(2) + 2 / math.log2(3)
    idcg = 2 / math.log2(2) + 2 / math.log2(3)                 # best two grades in the corpus are 2 and 2
    assert m["ndcg"] == pytest.approx(dcg / idcg) and m["precision_strict"] == 0.5
    perfect = FJ.query_metrics([0, 3, 1, 5], g, 2)
    assert perfect["ndcg"] == pytest.approx(1.0) and perfect["precision"] == 1.0 and perfect["recall_capped"] == 1.0


def test_the_candidate_set_defines_the_ideal_and_the_recall_denominator():
    g = np.array([2, 2, 2, 0, 1, 0])
    cand = np.array([False, False, False, True, True, True])   # e.g. a region filter that excludes all the grade-2 tiles
    m = FJ.query_metrics([4, 3, 5], g, 2, cand)
    assert m["n_relevant_in_candidates"] == 1 and m["recall"] == 1.0
    assert m["ndcg"] == pytest.approx(1.0)                      # ideal is taken INSIDE the candidate set: [1, 0]
    unconstrained = FJ.query_metrics([4, 3, 5], g, 2)
    assert unconstrained["ndcg"] < 1.0 and unconstrained["recall"] == pytest.approx(1 / 4)


def test_a_query_with_no_relevant_tile_scores_zero_not_nan():
    m = FJ.query_metrics([0, 1, 2], np.zeros(5, dtype=int), 3)
    assert (m["precision"], m["recall"], m["ndcg"], m["recall_capped"]) == (0, 0, 0, 0)


def test_macro_is_an_exact_mean_over_the_chosen_queries():
    pq = {q: {5: dict.fromkeys(FJ.METRICS, v)} for q, v in {"a": 0.2, "b": 0.4, "c": 0.9}.items()}
    assert FJ.macro(pq, ["a", "b"], ks=(5,))["k5"]["precision"] == pytest.approx(0.3)
    assert FJ.macro(pq, ["b", "a"], ks=(5,)) == FJ.macro(pq, ["a", "b"], ks=(5,))


def test_cohen_kappa():
    assert FJ.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert FJ.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert FJ.cohen_kappa([1, 1, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.5)
    assert math.isnan(FJ.cohen_kappa([], []))


# ------------------------------------------------------------------------------ the judge


def test_the_phase7a_graders_are_loaded_unchanged_and_cover_all_16_queries():
    g = J.graders()
    assert len(g) == 16 and "a water body" in g
    assert J.low_confidence_queries() == {"a paved road", "a dirt track or unpaved road", "a bridge crossing a river"}


def test_river_relative_queries_are_identified_from_the_grader_functions():
    rr = J.river_relative_queries()
    assert rr == {"newly built structures near a river", "settlement along a riverbank", "riverside construction",
                  "a river with sandbars", "a braided river channel with sandbars", "a bridge crossing a river",
                  "dense vegetation along a riverbank"}
    subsets = FJ.query_subsets(list(J.graders()), J.low_confidence_queries(), rr)
    assert len(subsets["all"]) == 16 and len(subsets["excluding_low_confidence"]) == 13
    assert len(subsets["region_agnostic"]) == 9 and len(subsets["core"]) == 7
    assert not set(subsets["core"]) & (rr | J.low_confidence_queries())


def test_grade_tile_matches_a_hand_applied_rule_and_unusable_tiles_grade_zero():
    water = {"usable": True, "water_frac": 0.5, "dist_water_m": 0.0}
    assert J.grade_tile(water, "a water body") == 2
    assert J.grade_tile({**water, "water_frac": 0.04}, "a water body") == 1
    assert J.grade_tile({**water, "usable": False}, "a water body") == 0
    assert J.grade_tile(None, "a water body") == 0                           # e.g. a Maxar tile: no descriptor at all


def test_river_relative_grading_refuses_tiles_without_river_distance():
    f = {"usable": True, "dist_river_m": None, "dist_water_m": None}
    with pytest.raises(ValueError, match="region context"):
        J.grade_matrix([f], ["a river with sandbars"])
    ok = {"usable": True, "water_frac": 0.5, "dist_water_m": 0.0, "dist_river_m": None}
    J.grade_matrix([ok], ["a water body"])                                         # a non-river query does not need the distance


def test_grade_matrix_shape_and_values():
    feats = [{"usable": True, "water_frac": 0.5, "dist_water_m": 0.0, "bare_frac": 0.0, "ndvi_p50": 0.1, "edge_density": 0.0},
             None]
    gm = J.grade_matrix(feats, ["a water body", "open bare ground"])
    assert gm.shape == (2, 2) and gm.dtype == np.int8 and gm[0, 0] == 2 and gm[1].tolist() == [0, 0]


@pytest.mark.skipif(not HAVE_PHASE7A, reason="Phase 7a artifacts not on this machine")
def test_applied_to_the_stored_ayodhya_features_the_graders_reproduce_every_frozen_judgment():
    feats = json.loads((EVAL_DIR / "tile_features.json").read_text())["features"]
    frozen = json.loads((EVAL_DIR / "judgments.json").read_text())
    checked = 0
    for q, per_tile in frozen.items():
        for tile_id, grade in per_tile.items():
            assert J.grade_tile(feats.get(tile_id), q) == grade, (q, tile_id)
            checked += 1
    assert checked > 800


def test_chance_adjusted_precision_is_precision_over_the_candidate_set_prevalence():
    g = np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0])             # 2 of 10 relevant -> a random top-K scores 0.2
    pq = {"q": {5: FJ.query_metrics([0, 1, 2, 3, 4], g, 5)}}
    c = FJ.chance_adjusted(pq, ["q"], ks=(5,))
    assert c["prevalence"] == pytest.approx(0.2) and c["k5"]["precision"] == pytest.approx(0.4)
    assert c["k5"]["lift_over_chance"] == pytest.approx(2.0)
    cand = np.array([True] * 4 + [False] * 6)                 # restrict to the first 4 tiles: prevalence 2/4
    pq = {"q": {5: FJ.query_metrics([0, 1, 2, 3], g, 4, cand)}}
    assert FJ.chance_adjusted(pq, ["q"], ks=(5,))["prevalence"] == pytest.approx(0.5)

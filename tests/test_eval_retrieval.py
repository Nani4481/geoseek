"""Retrieval-eval library: metric formulas equal the Phase 7a scorer's, ranking is exact + deterministic,
and the reproduction labeller says REPRODUCED / NOT_REPRODUCED / NOT_RUN correctly."""

from __future__ import annotations

import importlib.util
import pathlib
import random

import numpy as np
import pytest

from geoseek.eval import reported as RP
from geoseek.eval import retrieval as RT

SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metric_formulas_equal_the_phase7a_scorer():
    ref = _load("eval_retrieval_score")
    rng = random.Random(3)
    for _ in range(200):
        tiles = [f"t{i}" for i in range(40)]
        judg = {t: rng.choice([0, 0, 1, 2]) for t in rng.sample(tiles, 25)}
        ranked = rng.sample(tiles, 30)
        total = sum(1 for v in judg.values() if v > 0)
        for k in (1, 5, 10, 20):
            assert RT.ndcg_at_k(ranked, judg, k) == ref.ndcg_at_k(ranked, judg, k)
            assert RT.precision_at_k(ranked, judg, k) == ref.precision_at_k(ranked, judg, k)
            assert RT.recall_at_k(ranked, judg, k) == ref.recall_at_k(ranked, judg, k, total)


def test_ranking_is_exact_descending_and_ties_break_by_index():
    vecs = np.array([[1, 0], [0.6, 0.8], [0.6, 0.8], [0, 1], [-1, 0]], dtype=np.float32)
    assert list(RT.rank_query(np.array([0.6, 0.8], np.float32), vecs, depth=5)) == [1, 2, 3, 0, 4]    # tie 1,2 -> index order
    mask = np.array([False, True, False, True, True])
    assert list(RT.rank_query(np.array([0.6, 0.8], np.float32), vecs, mask=mask, depth=2)) == [1, 3]   # indices into the FULL corpus


def _judged():
    q = ["a", "b", "c"]
    j = {"a": {"t0": 2, "t1": 0, "t2": 1}, "b": {"t0": 0, "t3": 2}, "c": {"t1": 1}}
    return RT.Judged(q, j, frozenset({"c"}))


def test_evaluate_system_scores_against_the_judgements_and_is_order_independent():
    vecs = np.eye(4, dtype=np.float32)                       # tile i is the i-th basis vector
    ids = ["t0", "t1", "t2", "t3"]
    enc = {"a": np.array([1, 0.1, 0.05, 0], np.float32), "b": np.array([0, 0, 0, 1], np.float32), "c": np.array([0, 1, 0, 0], np.float32)}
    j = _judged()
    r1 = RT.evaluate_system(enc.__getitem__, vecs, ids, j)
    r2 = RT.evaluate_system(enc.__getitem__, vecs, ids, j, order=["c", "a", "b"])
    assert r1["ranked"] == r2["ranked"] and r1["scores"] == r2["scores"]
    assert r1["ranked"]["a"][:3] == ["t0", "t1", "t2"]
    assert r1["scores"]["b"][1] == {"recall": 1.0, "precision": 1.0, "ndcg": 1.0}
    assert RT.macro(r1["scores"], ["a", "b", "c"]) == RT.macro(r2["scores"], ["c", "a", "b"])     # exact mean: no float drift


def test_summarize_excludes_low_confidence_queries_from_the_second_macro():
    vecs = np.eye(4, dtype=np.float32)
    enc = {"a": vecs[0], "b": vecs[3], "c": vecs[2]}          # c retrieves the wrong tile
    j = _judged()
    s = RT.summarize_system(RT.evaluate_system(enc.__getitem__, vecs, ["t0", "t1", "t2", "t3"], j), j)
    assert s["n_queries"] == 3 and s["n_queries_excluding_low_confidence"] == 2
    assert s["all"]["k1"]["precision"] < s["excluding_low_confidence"]["k1"]["precision"]
    assert [p["query"] for p in s["per_query"]] == ["a", "b", "c"] and s["per_query"][2]["low_confidence"] is True


def test_topk_composition_buckets():
    j = RT.Judged(["q"], {"q": {"S2B_44RPQ_20190330_1_L2A_scaled_r000_c000": 2, "S2B_44RPQ_20190330_1_L2A_scaled_r000_c001": 0}}, frozenset())
    ranked = {"q": ["S2B_44RPQ_20190330_1_L2A_scaled_r000_c000", "S2B_44RPQ_20190330_1_L2A_scaled_r000_c001",
                    "S2B_44RPQ_20190330_1_L2A_scaled_r009_c009", "S2B_43RGP_20231011_0_L2A_r001_c001"]}
    assert RT.topk_composition(ranked, j, k=4) == {"judged_relevant": 0.25, "judged_zero": 0.25, "unjudged_ayodhya": 0.25,
                                                   "unjudged_other_region": 0.25}


# --------------------------------------------------------------------------- reproduction labelling


def test_reproduction_labels_each_reported_number():
    rep = RP.REPORTED
    assert len({r.path for r in rep}) == len(rep), "duplicate reported paths"
    snap = {"suite": {"passed": 450, "failed": 4, "skipped": 3},
            "corpus": {"embedded_tiles_frozen_prefix": 101911, "faiss_vectors": 105245},
            "retrieval": {"ayodhya_3267": {"remoteclip": {"all": {"k10": {"recall": 0.3651, "ndcg": 0.5}}}}}}
    out = RP.reproduction_report(snap)
    by = {e["metric"]: e for e in out["entries"]}
    assert by["corpus.embedded_tiles_frozen_prefix"]["status"] == "REPRODUCED"
    assert by["corpus.faiss_vectors"]["status"] == "NOT_REPRODUCED" and by["corpus.faiss_vectors"]["measured"] == 105245
    assert by["suite.passed"]["status"] == "NOT_REPRODUCED" and by["suite.failed"]["status"] == "NOT_REPRODUCED"
    assert by["suite.skipped"]["status"] == "REPRODUCED"
    assert by["retrieval.ayodhya_3267.remoteclip.all.k10.recall"]["status"] == "REPRODUCED"      # 0.3651 vs reported .365
    assert by["retrieval.ayodhya_3267.remoteclip.all.k10.ndcg"]["status"] == "NOT_REPRODUCED"   # 0.5 vs .419
    assert by["change_detection.oscd_heldout.thr_0p80.pooled.f1"]["status"] == "NOT_RUN"         # section not in this snapshot
    for bad in out["not_reproduced"]:                                                              # the required shape
        assert bad["status"] == "NOT_REPRODUCED" and "reported" in bad and bad["reason"]
    assert out["summary"]["NOT_RUN"] > 100


def test_a_not_reproduced_placeholder_in_the_snapshot_is_reported_not_silently_dropped():
    snap = {"detector": {"dota_val_v15_full_image": {"all_kept_classes": {
        "ap50": {"status": "NOT_REPRODUCED", "reported": 0.817, "reason": "weights unavailable"}}}}}
    out = RP.reproduction_report(snap)
    e = {x["metric"]: x for x in out["entries"]}["detector.dota_val_v15_full_image.all_kept_classes.ap50"]
    assert e["status"] == "NOT_REPRODUCED" and e["measured"] is None


def test_tolerance_semantics():
    r = RP.Reported("x", 0.365, tol_abs=0.0015)
    assert r.within(0.3664) and r.within(0.3636) and not r.within(0.3666)
    rel = RP.Reported("y", 100.0, tol_rel=0.4)
    assert rel.within(140.0) and not rel.within(141.0)
    assert RP.Reported("z", 96997, 0, 0).within(96997) and not RP.Reported("z", 96997).within(96998)

"""Explainability + suppression funnel (geoseek.analyst.explain).

Candidates are built by running the pipeline's OWN pure functions (suppress, classify, confidence) on chosen inputs, so
every stored value in these fixtures is something the pipeline could have written; the assertions then check that the
explanation is derived from those values and reproduces the stored confidence, never from fixed text or numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from geoseek.analyst import explain
from geoseek.change.analyze import radiometric_reliability
from geoseek.change.classify import CandidateSpectra, classify_candidate
from geoseek.change.confidence import compute_confidence, spectral_agreement_for
from geoseek.change.suppress import CandidateFeatures, PairSuppressionContext, suppress_candidate
from geoseek.temporal.persistence import PERSISTENCE_CONFIDENCE, PERSISTENCE_PENALTY

DATES = ["2019-03-30", "2021-03-04", "2024-03-08", "2025-03-08", "2026-03-08"]
SCENE = dict(scene_d_ndvi=0.06, scene_d_ndbi=-0.19, scene_d_ndwi=-0.05)


def _make(cid="c1", *, d_ndvi=-0.03, d_ndbi=0.19, d_ndwi=-0.02, area=900, elong=1.4, prob=0.93, persistence="persistent",
          corr=0.69, sar=None, window=("2019-03-30", "2021-03-04"), notes=None, coreg=0.23):
    """A sidecar-shaped candidate dict produced by the real suppress -> classify -> confidence functions."""
    ctx = PairSuppressionContext(pair_id="p", coreg_residual_px=coreg, coreg_corrected=False,
                                 radiometric_index_band_min_corr=corr, radiometric_low_confidence=False, **SCENE)
    f = CandidateFeatures(candidate_id=cid, area_px=area, bad_scl_fraction_earlier=0.0, bad_scl_fraction_later=0.0,
                          valid_fraction=1.0, d_ndvi=d_ndvi, d_ndbi=d_ndbi, d_ndwi=d_ndwi)
    sup = suppress_candidate(f, ctx).as_dict()
    spectra = CandidateSpectra(candidate_id=cid, d_ndvi=d_ndvi, d_ndbi=d_ndbi, d_ndwi=d_ndwi, area_px=area,
                               elongation=elong, fill_ratio=0.6, **SCENE)
    cl = classify_candidate(spectra).as_dict()
    ev = cl["evidence"]
    sa = spectral_agreement_for(cl["change_type"], ev["ndvi_anomaly"], ev["ndbi_anomaly"], ev["ndwi_anomaly"])
    pconf = PERSISTENCE_CONFIDENCE[persistence]
    sar = sar or {}
    rep = compute_confidence(
        candidate_id=cid, model_prob=prob, bad_scl_fraction=0.0, valid_fraction=1.0, coreg_residual_px=coreg,
        radiometric_reliability=radiometric_reliability(corr), persistence=persistence, persistence_confidence=pconf,
        spectral_agreement=sa, change_type=cl["change_type"], suppression_downweight=sup["combined_downweight"],
        sar_corroboration=sar.get("factor", 1.0), sar_detail=sar.get("verdict"))
    return {
        "candidate_id": cid, "change_type": cl["change_type"], "confidence": round(rep.confidence, 4),
        "significance": 0.5, "queue_score": 0.5, "persistence": persistence, "mean_model_prob": prob,
        "classification": cl, "confidence_breakdown": rep.breakdown, "suppression": sup, "sar": sar, "terrain": {},
        "trajectory": {"persistence": persistence, "persistence_confidence": pconf, "notes": notes or ["note one"],
                       "earliest_supported_change": {"window": list(window)}, "consecutive_pairs": []},
    }


def _svc(*cands, report=None, tmp=None):
    by_id = {c["candidate_id"]: c for c in cands}
    return SimpleNamespace(
        _by_id=by_id, details=list(cands), _observation_dates=lambda: list(DATES), report=report or {},
        span_pair_name="2019-2026", report_mtime="2026-01-01T00:00:00+00:00", out_dir=tmp or Path("."),
        provenance_for=lambda c: {"observations": [{"role": "after", "observation_id": "o26", "acquired_at": "2026-03-08",
                                                    "representative_tile": {"tile_id": "t26"}}]})


# ---------------------------------------------------------------------------- lead sentence


def test_headline_is_built_from_the_stored_values_of_the_winning_rule():
    c = _make(d_ndvi=-0.03, d_ndbi=0.19)               # NDBI up, NDVI flat -> construction
    assert c["classification"]["rule"] == "ndbi_up_ndvi_flat_or_down"
    ev = c["classification"]["evidence"]
    h = explain._headline(c)
    assert f"{ev['ndbi_anomaly']:+.2f}".replace("-", "−") in h and f"{ev['ndvi_anomaly']:+.2f}".replace("-", "−") in h
    assert "new construction" in h and "Built-up index rose" in h


def test_headline_changes_with_the_values_not_just_the_rule():
    strong = explain._headline(_make(d_ndbi=0.19 + 0.30))
    marginal = explain._headline(_make(d_ndbi=-0.19 + 0.06 + 0.0))     # anomaly only just past the 0.05 threshold
    assert "rose sharply" in strong
    assert "only just past" in marginal and "sharply" not in marginal


def test_every_typing_rule_gets_its_own_sentence_and_unclassified_says_so():
    water = _make(d_ndwi=0.55)
    clearance = _make(d_ndvi=-0.40, d_ndbi=-0.19 + 0.0, d_ndwi=-0.05)
    road = _make(elong=4.0, d_ndbi=-0.19 + 0.20, d_ndvi=0.2)
    other = _make(d_ndvi=0.60, d_ndbi=-0.3, d_ndwi=-0.05)
    assert water["classification"]["rule"] == "ndwi_anomaly_up" and "surface water" in explain._headline(water)
    assert clearance["classification"]["rule"] == "ndvi_down_ndbi_flat" and "clearance" in explain._headline(clearance)
    assert road["classification"]["rule"] == "linear_morphology_plus_ndbi" and "road" in explain._headline(road)
    assert other["classification"]["rule"] == "no_rule_matched" and "no spectral rule matched" in explain._headline(other)


def test_persistence_sentence_uses_the_real_counts_and_penalty():
    svc = _svc(_make("a", persistence="persistent"), _make("b", persistence="transient", notes=["it reverted"]))
    a = explain.explain_candidate(svc, "a")["lead"]["persistence"]
    b = explain.explain_candidate(svc, "b")["lead"]["persistence"]
    assert re_count(a) and "since 2021-03-04" in a
    assert "transient" in b and "it reverted" in b and f"×{PERSISTENCE_PENALTY['transient']:.2f}" in b


def re_count(s: str) -> bool:
    import re

    return re.search(r"Present on \d+ of 5 acquisition dates", s) is not None


# ---------------------------------------------------------------------------- confidence decomposition


@pytest.mark.parametrize("kw", [
    {}, {"persistence": "transient"}, {"persistence": "inconsistent", "prob": 0.85}, {"persistence": "none"},
    {"corr": 0.58}, {"coreg": 0.4}, {"sar": {"factor": 1.1, "verdict": "agrees", "available": True, "vv_median_db": -3.2}},
    {"sar": {"factor": 0.8, "verdict": "disagrees", "available": True, "vv_median_db": -9.0}},
    {"d_ndvi": 0.6, "d_ndbi": -0.3},
])
def test_decomposition_reproduces_the_stored_confidence_and_factors_multiply_out(kw):
    c = _make(**kw)
    dec = explain._decompose(c)
    assert dec["reproduces"] and dec["max_abs_error"] < 0.005
    prod = math.prod(t["factor"] for t in dec["terms"]) * math.prod(m["factor"] for m in dec["multipliers"])
    assert min(1.0, prod) == pytest.approx(c["confidence"], abs=0.005)
    assert abs(sum(t["weight_share"] for t in dec["terms"]) - 1.0) < 0.01


def test_terms_are_ordered_by_their_realised_effect_and_flag_weak_evidence():
    c = _make(persistence="transient", corr=0.58)
    ex = explain.explain_candidate(_svc(c), "c1")
    factors = [t["factor"] for t in ex["evidence"]["terms"]]
    assert factors == sorted(factors)
    top = ex["evidence"]["terms"][0]
    assert top["effect_points"] < 0 and top["strength"] == "weak"
    assert any(w["key"] == "persistence" for w in ex["weak_or_absent"])


def test_a_perfect_term_has_no_effect_and_a_down_weight_shows_up_as_a_multiplier():
    c = _make(coreg=0.4)                                # registration residual between trust and max -> down-weight
    ex = explain.explain_candidate(_svc(c), "c1")
    mult = {m["name"]: m for m in ex["evidence"]["multipliers"]}
    assert mult["suppression_downweight"]["factor"] < 1 and mult["suppression_downweight"]["effect_points"] < 0
    reg = next(s for s in ex["trace"] if s["id"] == "registration")
    assert reg["verdict"] == "downweighted" and reg["gate_weight"] < 1


def test_missing_sar_is_stated_plainly_and_a_present_one_is_not_called_missing():
    none = explain.explain_candidate(_svc(_make(), report={"sar_corroboration": {"available": False, "note": "no S1 staged"}}), "c1")
    assert none["weak_or_absent"][0]["key"] == "sar" and none["weak_or_absent"][0]["level"] == "absent"
    assert none["sar"]["available"] is False and none["sar"]["run_note"] == "no S1 staged"
    s = {"factor": 1.1, "verdict": "agrees", "available": True, "vv_median_db": -3.2}
    have = explain.explain_candidate(_svc(_make(sar=s)), "c1")
    assert not any(w["key"] == "sar" for w in have["weak_or_absent"]) and have["sar"]["available"]


def test_spectral_tests_mark_the_decisive_rule_and_whether_each_threshold_was_met():
    ex = explain.explain_candidate(_svc(_make()), "c1")
    dec = [t for t in ex["spectral"]["tests"] if t["decisive"]]
    assert dec and all(t["type"] == "construction" and t["met"] for t in dec)
    assert {a["index"] for a in ex["spectral"]["anomalies"]} == {"ndvi", "ndbi", "ndwi"}
    other = explain.explain_candidate(_svc(_make(d_ndvi=0.6, d_ndbi=-0.3)), "c1")
    assert other["change_type"] == "other" and not any(t["decisive"] for t in other["spectral"]["tests"])
    assert other["overlay"]["focus_indices"] == []      # nothing decided it, so nothing is starred


def test_overlay_points_at_the_after_date_tile_and_carries_the_footprint():
    c = _make(d_ndwi=0.55)
    c["_geometry"] = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    ov = explain.explain_candidate(_svc(c), "c1")["overlay"]
    assert ov["tile_id"] == "t26" and ov["focus_indices"] == ["ndwi"] and ov["geometry"]["type"] == "Polygon"


def test_trace_lists_every_stage_in_pipeline_order_and_marks_demotion():
    ex = explain.explain_candidate(_svc(_make(persistence="transient", prob=0.95)), "c1")
    assert [s["id"] for s in ex["trace"]] == ["morphology", "quality", "registration", "radiometric", "phenology", "typing", "persistence", "sar"]
    pers = next(s for s in ex["trace"] if s["id"] == "persistence")
    assert pers["verdict"] == "demoted" and pers["effect"]["points"] < 0 and "note one" in pers["detail"]
    assert next(s for s in ex["trace"] if s["id"] == "sar")["verdict"] == "unavailable"
    assert "calibrated" in ex["scope"]


def test_unknown_candidate_is_none():
    assert explain.explain_candidate(_svc(_make()), "nope") is None


# ---------------------------------------------------------------------------- funnel


def _report(pairs):
    return {"aoi": "Test AOI", "model": {"threshold": 0.8}, "pairs": pairs,
            "sar_corroboration": {"available": False, "note": "no S1"}}


def _pair(raw, by, survived, comparable=True):
    supp = sum(by.values())
    return {"comparable": comparable, "context": {}, "class_distribution": {"water_gain": survived}, "survivors": survived,
            "suppression": {"raw_candidates": raw, "suppressed": supp, "survived": survived, "suppressed_by_rule": by,
                            "downweighted_survivors": 0}}


def test_funnel_counts_come_from_the_report_and_are_sequential(tmp_path):
    by = {"quality": 2, "registration": 0, "radiometric": 0, "phenology": 5, "morphology": 80}
    svc = _svc(_make("a"), _make("b", persistence="transient"), report=_report({"2019-2026": _pair(100, by, 13)}), tmp=tmp_path)
    f = explain.suppression_funnel(svc)
    p = f["pairs"][0]
    assert (p["raw"], p["survivors"], p["suppressed"], p["consistent"]) == (100, 13, 87, True)
    assert [s["rule"] for s in p["stages"]] == list(explain.GATE_ORDER)
    rem = 100
    for s in p["stages"]:
        assert s["input"] == rem and s["remaining"] == rem - s["removed"]
        rem = s["remaining"]
    morph = p["stages"][0]
    assert morph["removed"] == 80 and morph["share_of_raw"] == 0.8
    assert p["earlier"] == "2019-03-30" and p["later"] == "2026-03-08"


def test_funnel_flags_a_report_whose_gate_counts_do_not_add_up(tmp_path):
    bad = _pair(100, {"quality": 2, "registration": 0, "radiometric": 0, "phenology": 5, "morphology": 80}, 50)
    f = explain.suppression_funnel(_svc(_make("a"), report=_report({"2019-2026": bad}), tmp=tmp_path))
    assert f["pairs"][0]["consistent"] is False


def test_post_gate_stages_are_counted_from_the_retained_candidates(tmp_path):
    cs = [_make("a", persistence="persistent"), _make("b", persistence="transient", prob=0.97),
          _make("c", persistence="inconsistent", prob=0.9), _make("d", persistence="none")]
    by = {"quality": 0, "registration": 0, "radiometric": 0, "phenology": 0, "morphology": 6}
    f = explain.suppression_funnel(_svc(*cs, report=_report({"2019-2026": _pair(10, by, 4)}), tmp=tmp_path))
    pg = f["post_gate"]
    assert pg["survivors"] == 4 and pg["matches_report"] is True
    assert pg["persistence"]["contradicted"] == 2 and pg["persistence"]["supported"] == 1 and pg["persistence"]["no_support"] == 1
    assert pg["sar"]["available"] is False and pg["sar"]["note"] == "no S1"
    assert [d["candidate_id"] for d in f["demoted_sample"]] == ["b", "c"]            # strongest model evidence first
    assert f["demoted_sample"][0]["penalty"] == PERSISTENCE_PENALTY["transient"]
    assert f["demoted_sample"][0]["confidence_without_penalty"] > f["demoted_sample"][0]["confidence"]


def test_rejected_components_are_stated_as_not_retained_with_what_would_be_needed(tmp_path):
    by = {"quality": 0, "registration": 0, "radiometric": 0, "phenology": 0, "morphology": 1}
    f = explain.suppression_funnel(_svc(_make("a"), report=_report({"2019-2026": _pair(2, by, 1)}), tmp=tmp_path))
    r = f["rejected"]
    assert r["retained"] is False and "not kept" in r["text"].lower() and "would have to write" in r["to_emit"]
    assert "calibrated" in f["scope"] and "FIRST gate" in f["attribution_note"]
    assert f["gates"][0]["rule"] == "morphology" and {g["kind"] for g in f["gates"]} == {"reject", "downweight"}


def _ablation(path: Path, cfg_override=None):
    st = lambda p, r, f: {"precision": p, "recall": r, "f1": f}
    cfg = {"max_bad_scl_fraction": 0.05, "min_valid_fraction": 0.8, "radiometric_min_corr": 0.6, "pheno_ndvi_anomaly": 0.1,
           "pheno_ndbi_anomaly": 0.06, "pheno_ndwi_anomaly": 0.08, "morph_min_area_px": 50}
    cfg.update(cfg_override or {})
    path.write_text(json.dumps({
        "generated_at": "2026-01-01T00:00:00Z", "protocol": {"suppression_config": cfg},
        "oscd": {"by_threshold": {"0.80": {
            "aggregate": {"1_raw_model": st(.6, .5, .55), "2_quality_gate": st(.6, .5, .55), "3_radiometric": st(.6, .5, .55)},
            "aggregate_morphology_first": {"2q_5morph": st(.62, .48, .54), "2q_5morph_4phen": st(.62, .25, .36)}}}}}), encoding="utf-8")


def test_labelled_benchmark_deltas_are_computed_from_the_ablation_artifact(tmp_path):
    _ablation(tmp_path / "ablation_study.json")
    by = {"quality": 0, "registration": 0, "radiometric": 0, "phenology": 0, "morphology": 1}
    f = explain.suppression_funnel(_svc(_make("a"), report=_report({"2019-2026": _pair(2, by, 1)}), tmp=tmp_path))
    lb = f["labelled_benchmark"]
    assert lb["available"] and lb["config_matches_current"] is True
    ph = next(s for s in lb["stages"] if s["stage"] == "phenology")
    assert ph["delta_f1"] == pytest.approx(0.36 - 0.54) and ph["delta_recall"] == pytest.approx(0.25 - 0.48)
    assert "Ayodhya" in lb["caveat"]


def test_labelled_benchmark_flags_a_config_mismatch_and_a_missing_artifact(tmp_path):
    _ablation(tmp_path / "ablation_study.json", {"morph_min_area_px": 10})
    by = {"quality": 0, "registration": 0, "radiometric": 0, "phenology": 0, "morphology": 1}
    f = explain.suppression_funnel(_svc(_make("a"), report=_report({"2019-2026": _pair(2, by, 1)}), tmp=tmp_path))
    assert f["labelled_benchmark"]["config_matches_current"] is False
    (tmp_path / "ablation_study.json").unlink()
    f2 = explain.suppression_funnel(_svc(_make("a"), report=_report({"2019-2026": _pair(2, by, 1)}), tmp=tmp_path))
    assert f2["labelled_benchmark"]["available"] is False


# ---------------------------------------------------------------------------- real artefacts + wiring

REAL = Path(__file__).resolve().parents[1] / "data" / "change_model"


@pytest.mark.skipif(not (REAL / "ayodhya_change_ranked_detail.json").is_file(), reason="change pipeline not run")
def test_every_stored_candidate_reproduces_its_confidence_from_the_stored_inputs():
    rows = json.loads((REAL / "ayodhya_change_ranked_detail.json").read_text(encoding="utf-8"))
    assert rows
    worst = max(explain._decompose(c)["max_abs_error"] for c in rows)
    assert worst < 0.005, f"confidence no longer reproducible from stored inputs (max error {worst})"


@pytest.mark.skipif(not (REAL / "ayodhya_change_report.json").is_file(), reason="change pipeline not run")
def test_the_real_report_funnel_is_internally_consistent_for_every_pair():
    report = json.loads((REAL / "ayodhya_change_report.json").read_text(encoding="utf-8"))
    for name, p in report["pairs"].items():
        pf = explain._pair_funnel(name, p, DATES)
        assert pf["consistent"], f"{name}: gate counts do not add up to the survivors"
        assert pf["stages"][-1]["remaining"] == pf["survivors"]


def test_routes_are_registered():
    from geoseek.search.api import app

    paths = {r.path for r in app.routes}
    assert "/ui/candidates/{candidate_id}/explain" in paths and "/ui/pipeline/funnel" in paths

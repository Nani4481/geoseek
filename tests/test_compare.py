"""Snapshot comparison harness: the diff math, noise-band judgement, polarity and placeholder handling."""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoseek.eval import compare as C
from geoseek.eval.compare import DiffRow, diff_snapshots, flatten_metrics, judge, lower_is_better


def snap(metrics: dict, bands: dict | None = None, **extra) -> dict:
    return {"meta": {"x": 1}, "environment": {"gpu": {"name": "g"}}, **metrics,
            "noise_bands": {k: {"band": v} for k, v in (bands or {}).items()}, **extra}


def row(rows, metric) -> DiffRow:
    (r,) = [r for r in rows if r.metric == metric]
    return r


def test_delta_relative_and_band_arithmetic():
    base = snap({"retrieval": {"a": {"ndcg": 0.400, "recall": 0.500}}},
                {"retrieval.a.ndcg": 0.020, "retrieval.a.recall": 0.020})
    cand = snap({"retrieval": {"a": {"ndcg": 0.450, "recall": 0.510}}})
    rows = diff_snapshots(base, cand)
    n = row(rows, "retrieval.a.ndcg")
    assert n.baseline == 0.400 and n.candidate == 0.450
    assert n.delta == pytest.approx(0.050) and n.relative == pytest.approx(0.125)
    assert n.band == 0.020 and n.exceeds_noise is True and n.verdict == "improved"
    r = row(rows, "retrieval.a.recall")
    assert r.delta == pytest.approx(0.010) and r.exceeds_noise is False and r.verdict == "within_noise"


def test_a_delta_exactly_equal_to_the_band_is_noise_and_the_sign_is_respected():
    assert judge(0.02, 0.02, False) == (False, "within_noise")
    assert judge(-0.02, 0.02, False) == (False, "within_noise")
    assert judge(0.0200001, 0.02, False) == (True, "improved")
    assert judge(-0.03, 0.02, False) == (True, "regressed")        # higher-is-better metric went down
    assert judge(-0.03, 0.02, True) == (True, "improved")          # lower-is-better metric went down
    assert judge(0.03, 0.02, True) == (True, "regressed")
    assert judge(0.03, 0.02, None) == (True, "changed")            # no polarity: significant but not "better"
    assert judge(0.5, None, False) == (None, "no_band")


def test_the_larger_of_the_two_snapshots_bands_is_used():
    base = snap({"m": {"f1": 0.5}}, {"m.f1": 0.01})
    cand = snap({"m": {"f1": 0.52}}, {"m.f1": 0.03})
    r = row(diff_snapshots(base, cand), "m.f1")
    assert r.band == 0.03 and r.verdict == "within_noise"          # 0.02 < 0.03


def test_polarity_rules():
    assert lower_is_better("change_detection.x.fpr") is True
    assert lower_is_better("latency.text.median_ms") is True
    assert lower_is_better("latency.text.p95_ms") is True
    assert lower_is_better("footprint.faiss_index_mb") is True
    assert lower_is_better("ingestion.pipeline.tiles_per_s") is False      # *_s must not catch throughput
    assert lower_is_better("retrieval.a.ndcg") is False
    assert lower_is_better("detector.all.AP50") is False
    assert lower_is_better("detector.all.ap50_95") is False
    assert lower_is_better("change_detection.pooled.f1") is False
    assert lower_is_better("retrieval.corpus.n_vectors") is None


def test_only_numeric_metric_leaves_are_compared_and_skip_sections_are_ignored():
    s = snap({"retrieval": {"a": {"ndcg": 0.4, "label": "x", "flag": True, "none": None, "curve": [0.1, 0.2]}}},
             reproduction={"retrieval.a.ndcg": {"reported": 0.5}})
    flat, ph = flatten_metrics(s)
    assert flat == {"retrieval.a.ndcg": 0.4} and ph == []


def test_not_reproduced_placeholders_are_skipped_but_reported():
    base = snap({"detector": {"all": {"ap50": 0.81, "ap50_95": {"status": "NOT_REPRODUCED", "reported": 0.48, "reason": "r"}}}})
    cand = snap({"detector": {"all": {"ap50": 0.83, "ap50_95": 0.50}}})
    rows = diff_snapshots(base, cand)
    assert [r.metric for r in rows] == ["detector.all.ap50", "detector.all.ap50_95"]
    assert row(rows, "detector.all.ap50_95").verdict == "only_in_candidate"
    skipped = C.skipped_placeholders(base, cand)
    assert skipped["baseline"] == ["detector.all.ap50_95"] and skipped["candidate"] == []
    assert "NOT_REPRODUCED" in C.render_markdown(rows, skipped=skipped)


def test_metrics_present_on_one_side_only():
    rows = diff_snapshots(snap({"a": {"f1": 0.5}, "b": {"f1": 0.4}}), snap({"a": {"f1": 0.5}, "c": {"f1": 0.9}}))
    assert row(rows, "b.f1").verdict == "only_in_baseline" and row(rows, "b.f1").candidate is None
    assert row(rows, "c.f1").verdict == "only_in_candidate" and row(rows, "c.f1").baseline is None
    assert row(rows, "a.f1").delta == 0.0


def test_zero_baseline_has_no_relative_delta_and_identical_snapshots_are_all_within_noise():
    rows = diff_snapshots(snap({"m": {"fpr": 0.0, "f1": 0.5}}, {"m.fpr": 0.001, "m.f1": 0.001}),
                          snap({"m": {"fpr": 0.01, "f1": 0.5}}))
    assert row(rows, "m.fpr").relative is None and row(rows, "m.fpr").verdict == "regressed"   # FPR rose
    same = diff_snapshots(snap({"m": {"f1": 0.5}}, {"m.f1": 0.0}), snap({"m": {"f1": 0.5}}))
    assert row(same, "m.f1").delta == 0.0 and row(same, "m.f1").verdict == "within_noise"


def test_only_prefix_and_summary_and_markdown():
    base = snap({"retrieval": {"f1": 0.5}, "latency": {"x_ms": 10.0}}, {"retrieval.f1": 0.01, "latency.x_ms": 1.0})
    cand = snap({"retrieval": {"f1": 0.6}, "latency": {"x_ms": 20.0}})
    rows = diff_snapshots(base, cand)
    assert [r.metric for r in diff_snapshots(base, cand, only_prefix="latency")] == ["latency.x_ms"]
    assert C.summarize(rows) == {"improved": 1, "regressed": 1, "total": 2}
    md = C.render_markdown(rows)
    assert "| `retrieval.f1` |" in md and "improved" in md and "regressed" in md and "+0.1000" in md and "+20.0%" in md
    sig = C.render_markdown([*rows, DiffRow("q", 1, 1, 0, 0, 1, False, "within_noise")], only_significant=True)
    assert "`q`" not in sig


def test_noise_estimators():
    assert C.spread([0.50, 0.52, 0.49]) == pytest.approx(0.03)
    assert C.spread([]) == 0.0 and C.spread([1.0]) == 0.0
    x = np.array([0.0, 1.0] * 8)                       # per-query scores of mean 0.5, sd ~0.5, n=16
    hw = C.bootstrap_halfwidth(x, n_boot=4000, seed=1)
    assert 0.18 < hw < 0.30                            # ~1.96 * 0.5 / sqrt(16) = 0.245
    assert C.bootstrap_halfwidth(x, n_boot=4000, seed=1) == hw          # seeded -> reproducible
    assert C.bootstrap_halfwidth([0.5]) == 0.0 and C.bootstrap_halfwidth(np.full(10, 0.3)) == 0.0


def test_cli_roundtrip(tmp_path, capsys):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("cmp_cli", pathlib.Path(__file__).parents[1] / "scripts" / "compare_to_baseline.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    b.write_text(json.dumps(snap({"m": {"f1": 0.5}}, {"m.f1": 0.01})))
    c.write_text(json.dumps(snap({"m": {"f1": 0.4}})))
    assert cli.main([str(b), str(c), "--json", str(tmp_path / "d.json")]) == 0
    assert "regressed" in capsys.readouterr().out
    assert cli.main([str(b), str(c), "--fail-on-regression"]) == 1
    assert json.loads((tmp_path / "d.json").read_text())["summary"]["regressed"] == 1


# --------------------------------------------------------------------------- paired tests


def _retr_snap(per_query_ndcg: list[float], low=(), system="remoteclip") -> dict:
    qs = [f"q{i}" for i in range(len(per_query_ndcg))]
    pq = [{"query": q, "low_confidence": q in low, "ndcg@10": v} for q, v in zip(qs, per_query_ndcg)]
    keep = [v for q, v in zip(qs, per_query_ndcg) if q not in low]
    return {"retrieval": {"ayodhya_3267": {system: {
        "all": {"k10": {"ndcg": float(np.mean(per_query_ndcg))}},
        "excluding_low_confidence": {"k10": {"ndcg": float(np.mean(keep))}}, "per_query": pq}}}}


def test_paired_ci_excludes_zero_when_every_query_improves_even_though_the_band_says_noise():
    rng = np.random.default_rng(0)
    base_scores = list(rng.uniform(0.2, 0.8, 16))
    cand_scores = [v + 0.05 for v in base_scores]                    # +0.05 on EVERY query
    base, cand = _retr_snap(base_scores), _retr_snap(cand_scores)
    path = "retrieval.ayodhya_3267.remoteclip.all.k10.ndcg"
    base["noise_bands"] = {path: {"band": 0.12}}                     # an independent 16-query band: far wider than +0.05
    (r,) = [r for r in diff_snapshots(base, cand) if r.metric == path]
    assert r.delta == pytest.approx(0.05) and r.verdict == "improved_paired"
    assert r.exceeds_noise is False and r.paired_ci[0] == pytest.approx(0.05) and r.paired_ci[1] == pytest.approx(0.05)
    (r2,) = [r for r in diff_snapshots(base, cand, paired=False) if r.metric == path]
    assert r2.verdict == "within_noise" and r2.paired_ci is None


def test_paired_ci_straddles_zero_for_a_noisy_mixed_change_and_respects_the_low_confidence_filter():
    base = _retr_snap([0.5, 0.5, 0.5, 0.5], low=("q3",))
    cand = _retr_snap([0.7, 0.3, 0.7, 0.3], low=("q3",))             # mean delta 0, mixed signs
    path = "retrieval.ayodhya_3267.remoteclip.all.k10.ndcg"
    base["noise_bands"] = {path: {"band": 0.5}}
    (r,) = [r for r in diff_snapshots(base, cand) if r.metric == path]
    assert r.verdict == "within_noise" and r.paired_ci[0] < 0 < r.paired_ci[1]
    # excluding the low-confidence query q3 must drop it from the paired sample
    pth = "retrieval.ayodhya_3267.remoteclip.excluding_low_confidence.k10.ndcg"
    ci = C.paired_delta_ci(pth, base, cand)
    assert ci is not None and ci[0] < ci[1] + 1e-12
    assert C.paired_delta_ci("retrieval.ayodhya_3267.remoteclip.all.k10.ndcg", base, _retr_snap([0.7, 0.3, 0.7])) is None   # different query sets


def _oscd_snap(per_region: dict) -> dict:
    return {"change_detection": {"oscd_heldout": {"thr_0p80": {
        "pooled": {"f1": 0.5}, "per_region": {r: dict(zip(("tp", "fp", "fn", "tn"), c)) for r, c in per_region.items()}}}}}


def test_paired_oscd_region_bootstrap_and_polarity():
    regions = {f"r{i}": (100 + 10 * i, 50, 60, 5000) for i in range(10)}
    better = {r: (tp + 30, fp - 10, fn - 10, tn) for r, (tp, fp, fn, tn) in regions.items()}    # uniformly better
    path = "change_detection.oscd_heldout.thr_0p80.pooled.f1"
    ci = C.paired_delta_ci(path, _oscd_snap(regions), _oscd_snap(better))
    assert ci is not None and ci[0] > 0
    worse_fpr = {r: (tp, fp + 40, fn, tn - 40) for r, (tp, fp, fn, tn) in regions.items()}
    ci_fpr = C.paired_delta_ci("change_detection.oscd_heldout.thr_0p80.pooled.fpr", _oscd_snap(regions), _oscd_snap(worse_fpr))
    assert ci_fpr[0] > 0                                           # FPR went up in every region
    b, c = _oscd_snap(regions), _oscd_snap(worse_fpr)
    b["change_detection"]["oscd_heldout"]["thr_0p80"]["pooled"]["fpr"] = 0.01
    c["change_detection"]["oscd_heldout"]["thr_0p80"]["pooled"]["fpr"] = 0.02
    b["noise_bands"] = {"change_detection.oscd_heldout.thr_0p80.pooled.fpr": {"band": 0.05}}
    (r,) = [r for r in diff_snapshots(b, c) if r.metric.endswith("pooled.fpr")]
    assert r.verdict == "regressed_paired"                         # lower-is-better metric rose, band alone said noise
    assert C.paired_delta_ci(path, _oscd_snap(regions), _oscd_snap({"r0": (1, 1, 1, 1), "r1": (1, 1, 1, 1)})) is None


def test_paired_bootstrap_ci_basic_properties():
    lo, hi = C.paired_bootstrap_ci([1, 2, 3, 4], [2, 3, 4, 5])
    assert lo == hi == pytest.approx(1.0)
    lo, hi = C.paired_bootstrap_ci([0, 0, 0, 0, 0, 0], [1, -1, 1, -1, 1, -1], seed=3)
    assert lo < 0 < hi
    assert C.paired_bootstrap_ci([5], [7]) == (2.0, 2.0)
    assert C.paired_bootstrap_ci([1, 2], [1, 2], seed=1) == C.paired_bootstrap_ci([1, 2], [1, 2], seed=1)


def test_cli_fail_on_regression_counts_paired_regressions(tmp_path):
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location("cmp_cli2", pathlib.Path(__file__).parents[1] / "scripts" / "compare_to_baseline.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base_scores = [0.8, 0.6, 0.7, 0.5]
    b, c = _retr_snap(base_scores), _retr_snap([v - 0.04 for v in base_scores])
    b["noise_bands"] = {"retrieval.ayodhya_3267.remoteclip.all.k10.ndcg": {"band": 0.2}}
    pb, pc = tmp_path / "b.json", tmp_path / "c.json"
    pb.write_text(json.dumps(b))
    pc.write_text(json.dumps(c))
    assert cli.main([str(pb), str(pc), "--fail-on-regression"]) == 1
    assert cli.main([str(pb), str(pc), "--fail-on-regression", "--no-paired"]) == 0

"""Temporal archive view (geoseek.analyst.temporal): per-interval counts, areas, persistence and SAR from stored candidates.

The synthetic candidates carry hand-chosen values so every expected sum is arithmetic done here, independent of the code under test.
A second test runs the same function against the live report (when present) and checks it conserves the stored candidate list.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geoseek.analyst.temporal import temporal_archive

DATES = ["2019-03-30", "2021-03-04", "2024-03-08", "2025-03-08", "2026-03-08"]


def _c(cid, t, area, pers, window, lon=82.0, sar=False):
    return {"candidate_id": cid, "change_type": t, "area_m2": area, "persistence": pers, "earliest_supported": window,
            "sar": {"available": sar}, "_bounds": (lon, 26.0, lon + 0.01, 26.01)}


def _pair(raw, surv, cd):
    return {"comparable": True, "suppression": {"raw_candidates": raw, "suppressed": raw - surv, "survived": surv,
                                                 "suppressed_by_rule": {"morphology": raw - surv}}, "class_distribution": cd}


def _svc(cands, pairs=None):
    return SimpleNamespace(details=cands, _observation_dates=lambda: list(DATES), report={"aoi": "T", "pairs": pairs or {}, "sar_corroboration": {"available": False, "note": "none staged"}},
                           report_mtime="2026-01-01T00:00:00+00:00")


CANDS = [
    _c("a", "water_gain", 100.0, "persistent", ["2019-03-30", "2021-03-04"]),
    _c("b", "water_gain", 50.0, "transient", ["2019-03-30", "2021-03-04"]),
    _c("c", "construction", 30.0, "progressive", ["2019-03-30", "2021-03-04"], sar=True),
    _c("d", "road", 70.0, "recent", ["2024-03-08", "2025-03-08"]),
    _c("e", "other", 10.0, "inconsistent", ["2025-03-08", "2026-03-08"], lon=83.0),
    _c("f", "water_loss", 5.0, "transient", ["2019-03-30", "2026-03-08"]),          # visible only in the whole-span comparison
    _c("g", "clearance", 20.0, "none", None),
]


def test_candidates_are_bucketed_by_first_detected_interval_with_exact_sums():
    t = temporal_archive(_svc(CANDS))
    iv = {i["id"]: i for i in t["intervals"]}
    assert [i["id"] for i in t["intervals"]] == ["2019-03-30_2021-03-04", "2021-03-04_2024-03-08", "2024-03-08_2025-03-08", "2025-03-08_2026-03-08"]
    first = iv["2019-03-30_2021-03-04"]["stored"]
    assert first["n"] == 3 and first["by_type"] == {"water_gain": 2, "construction": 1}
    assert first["area_m2"] == 180.0 and first["area_m2_by_type"] == {"water_gain": 150.0, "construction": 30.0}
    assert first["held"] == 2 and first["demoted"] == 1 and first["persistence"] == {"persistent": 1, "transient": 1, "progressive": 1}
    assert first["sar_available"] == 1 and first["sar_coverage"] == pytest.approx(1 / 3, abs=1e-4)
    assert iv["2021-03-04_2024-03-08"]["stored"]["n"] == 0 and iv["2021-03-04_2024-03-08"]["stored"]["sar_coverage"] is None
    assert iv["2024-03-08_2025-03-08"]["stored"]["by_type"] == {"road": 1}
    # a bracket wider than one interval is reported on its own, never folded into an interval; no bracket at all is separate too
    assert [(m["id"], m["n"]) for m in t["multi_interval"]] == [("2019-03-30_2026-03-08", 1)]
    assert t["no_interval"]["n"] == 1 and t["no_interval"]["persistence"] == {"none": 1}
    assert t["totals"]["stored_in_scope"] == len(CANDS)
    assert t["sar"]["available_candidates"] == 1


def test_intervals_carry_real_day_counts_so_the_axis_can_be_calendar_not_equal_steps():
    t = temporal_archive(_svc(CANDS))
    assert [i["days"] for i in t["intervals"]] == [705, 1100, 365, 365]
    assert t["dates"] == DATES


def test_cumulative_is_a_running_sum_over_observed_dates_only():
    t = temporal_archive(_svc(CANDS))
    cum = t["cumulative"]
    assert [c["date"] for c in cum] == DATES                           # one point per ACQUISITION date; nothing between them
    assert cum[0]["n"] == 0
    assert cum[1]["area_m2_by_type"] == {"water_gain": 150.0, "construction": 30.0} and cum[1]["n"] == 3
    assert cum[2]["n"] == 3                                            # the 2021-2024 interval holds none
    assert cum[3]["area_m2_by_type"]["road"] == 70.0 and cum[3]["n"] == 4
    assert cum[4]["n"] == 5 and cum[4]["area_m2"] == 180.0 + 70.0 + 10.0   # the span-only and unlocalised candidates are NOT included


def test_region_box_filters_stored_candidates_and_withholds_the_whole_aoi_pair_runs():
    pairs = {"2019-2021": _pair(100, 9, {"water_gain": 9})}
    whole = temporal_archive(_svc(CANDS, pairs))
    assert whole["intervals"][0]["pair_run"] == {"name": "2019-2021", "raw": 100, "survivors": 9, "comparable": True, "by_type": {"water_gain": 9}}
    assert whole["intervals"][1]["pair_run"] is None                     # no run reported for that interval: absent, not zero
    box = temporal_archive(_svc(CANDS, pairs), bbox=(82.99, 26.0, 83.5, 26.5))
    assert box["totals"]["stored_in_scope"] == 1 and box["intervals"][3]["stored"]["n"] == 1
    assert all(i["pair_run"] is None for i in box["intervals"])


def test_span_pair_run_is_reported_apart_from_the_consecutive_intervals():
    pairs = {"2019-2026": _pair(7171, 841, {"water_gain": 5})}
    t = temporal_archive(_svc(CANDS, pairs))
    assert t["span_run"]["raw"] == 7171 and t["span_run"]["survivors"] == 841
    assert all(i["pair_run"] is None for i in t["intervals"])


def test_live_report_conserves_every_stored_candidate_and_matches_the_candidates_endpoint():
    from geoseek.change.analyze import OUT_DIR

    if not (Path(OUT_DIR) / "ayodhya_change_report.json").is_file():
        pytest.skip("no change report on this machine")
    from geoseek.analyst.service import AnalystService

    svc = AnalystService()
    t = temporal_archive(svc)
    total = len(svc.details)
    assert t["totals"]["stored_in_scope"] == total
    for iv in t["intervals"]:                                            # the click-through filter returns exactly the plotted count
        fd = svc.list_candidates(first_detected=iv["id"], limit=5000)
        assert fd["total"] == iv["stored"]["n"], iv["id"]
        assert sum(iv["stored"]["by_type"].values()) == iv["stored"]["n"]
        assert iv["stored"]["held"] + iv["stored"]["demoted"] == sum(v for k, v in iv["stored"]["persistence"].items() if k != "none")
    for m in t["multi_interval"]:
        assert svc.list_candidates(first_detected=m["id"], limit=5000)["total"] == m["n"]
    assert svc.list_candidates(first_detected="none", limit=5000)["total"] == t["no_interval"]["n"]
    two = ",".join(iv["id"] for iv in t["intervals"][:2])                  # several brackets at once (a cumulative point)
    assert svc.list_candidates(first_detected=two, limit=5000)["total"] == sum(iv["stored"]["n"] for iv in t["intervals"][:2])
    assert sum(iv["stored"]["n"] for iv in t["intervals"]) + sum(m["n"] for m in t["multi_interval"]) + t["no_interval"]["n"] == total
    assert t["cumulative"][-1]["area_m2"] == pytest.approx(sum(iv["stored"]["area_m2"] for iv in t["intervals"]), abs=0.5)

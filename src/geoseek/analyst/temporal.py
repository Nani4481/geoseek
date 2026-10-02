"""Temporal view of the change pipeline's own output, per acquisition interval (``/ui/temporal/archive``).

Read-only and derived at request time from the stored ranked candidates and the pipeline report; nothing is recomputed and
nothing is written. Two bases are kept apart because they are different populations:

  * ``stored`` - the candidates the pipeline retained (one row each, from the span-pair run), placed in the interval in which
    the pipeline first saw them (``earliest_supported``, the first-detected bracket). Everything the console can open in
    Changes is here: counts, areas, persistence and SAR corroboration all come from these rows.
  * ``pair_run`` - the report's per-interval survivor counts from the independent consecutive-pair runs. Those components
    are counted in the report but are not retained as candidate rows, so they carry no area / persistence / SAR and cannot be
    opened. It is the pipeline's own per-interval output, shown as a reference series, never mixed into ``stored``.

A candidate whose first-detected bracket is wider than one acquisition interval (e.g. visible only in the 2019-2026 span
pair) is NOT assigned to any single interval - the pipeline could not localise it - and is reported separately.
"""

from __future__ import annotations

from datetime import date

from geoseek.analyst.explain import CONTRADICTED, SUPPORTED, _pair_funnel

TYPE_ORDER = ("water_gain", "construction", "clearance", "road", "other", "water_loss")


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _inside(c: dict, bbox) -> bool:
    if bbox is None:
        return True
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = c.get("_bounds") or (0, 0, 0, 0)
    return not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1)


def _bucket() -> dict:
    return {"n": 0, "area_m2": 0.0, "by_type": {}, "area_m2_by_type": {}, "persistence": {},
            "held": 0, "demoted": 0, "sar_available": 0}


def _add(b: dict, c: dict) -> None:
    t = c.get("change_type") or "other"
    area = float(c.get("area_m2") or 0.0)
    p = c.get("persistence") or "none"
    b["n"] += 1
    b["area_m2"] += area
    b["by_type"][t] = b["by_type"].get(t, 0) + 1
    b["area_m2_by_type"][t] = b["area_m2_by_type"].get(t, 0.0) + area
    b["persistence"][p] = b["persistence"].get(p, 0) + 1
    if p in SUPPORTED:
        b["held"] += 1
    elif p in CONTRADICTED:
        b["demoted"] += 1
    if (c.get("sar") or {}).get("available"):
        b["sar_available"] += 1


def _finish(b: dict) -> dict:
    b["area_m2"] = round(b["area_m2"], 1)
    b["area_m2_by_type"] = {k: round(v, 1) for k, v in b["area_m2_by_type"].items()}
    b["sar_coverage"] = round(b["sar_available"] / b["n"], 4) if b["n"] else None
    return b


def temporal_archive(svc, bbox=None) -> dict:
    dates = svc._observation_dates()
    report = svc.report
    ivs = [{"id": f"{a}_{b}", "from": a, "to": b, "days": _days(a, b)} for a, b in zip(dates, dates[1:])]
    by_id = {iv["id"]: iv for iv in ivs}
    for iv in ivs:
        iv["stored"] = _bucket()
        iv["pair_run"] = None
    multi: dict[str, dict] = {}                   # first-detected brackets wider than one interval
    none = _bucket()

    for c in svc.details:
        if not _inside(c, bbox):
            continue
        w = c.get("earliest_supported")
        if not w:
            _add(none, c)
            continue
        key = f"{w[0]}_{w[1]}"
        if key in by_id:
            _add(by_id[key]["stored"], c)
        else:
            m = multi.setdefault(key, {"id": key, "from": w[0], "to": w[1], "days": _days(w[0], w[1]), **_bucket()})
            _add(m, c)

    # per-interval pair runs from the report (AOI-wide: the report does not store them per component, so a sub-region cannot filter them)
    pair_runs = {}
    if bbox is None:
        for name, p in (report.get("pairs") or {}).items():
            f = _pair_funnel(name, p, dates)
            pair_runs[f"{f['earlier']}_{f['later']}"] = f
    span = next((f for k, f in pair_runs.items() if k not in by_id), None)
    for iv in ivs:
        f = pair_runs.get(iv["id"])
        if f:
            cd = f.get("class_distribution") or {}
            iv["pair_run"] = {"name": f["name"], "raw": f["raw"], "survivors": f["survivors"], "comparable": f.get("comparable"),
                              "by_type": {k: int(v) for k, v in cd.items()}}

    # cumulative, by the date by which the pipeline had first seen each candidate (localised candidates only)
    run_n: dict[str, int] = {}
    run_a: dict[str, float] = {}
    cumulative = [{"date": dates[0] if dates else None, "n_by_type": {}, "area_m2_by_type": {}, "n": 0, "area_m2": 0.0}]
    for iv in ivs:
        for t, n in iv["stored"]["by_type"].items():
            run_n[t] = run_n.get(t, 0) + n
        for t, a in iv["stored"]["area_m2_by_type"].items():
            run_a[t] = run_a.get(t, 0.0) + a
        cumulative.append({"date": iv["to"], "n_by_type": dict(run_n), "area_m2_by_type": {k: round(v, 1) for k, v in run_a.items()},
                           "n": sum(run_n.values()), "area_m2": round(sum(run_a.values()), 1)})
    for iv in ivs:
        _finish(iv["stored"])
    multi_list = [_finish(m) for m in sorted(multi.values(), key=lambda m: (m["from"], m["to"]))]
    sar = report.get("sar_corroboration") or {}
    present = {t for iv in ivs for t in iv["stored"]["by_type"]} | {t for m in multi_list for t in m["by_type"]} | set(none["by_type"])
    return {
        "available": bool(dates and ivs),
        "aoi": report.get("aoi"),
        "bbox": list(bbox) if bbox else None,
        "dates": dates,
        "types": [t for t in TYPE_ORDER if t in present] + sorted(present - set(TYPE_ORDER)),
        "intervals": ivs,
        "multi_interval": multi_list,
        "no_interval": _finish(none),
        "span_run": None if span is None else {"name": span["name"], "from": span["earlier"], "to": span["later"], "raw": span["raw"], "survivors": span["survivors"]},
        "cumulative": cumulative,
        "totals": {"stored_in_scope": sum(iv["stored"]["n"] for iv in ivs) + sum(m["n"] for m in multi_list) + none["n"]},
        "sar": {"available_candidates": sum(iv["stored"]["sar_available"] for iv in ivs) + sum(m["sar_available"] for m in multi_list) + none["sar_available"],
                "note": sar.get("note") or sar.get("reason")},
        "held_classes": list(SUPPORTED), "demoted_classes": list(CONTRADICTED),
        "pair_run_note": ("Per-interval survivor counts from the report's independent consecutive-pair runs. Those components are counted "
                          "but not retained as candidates, so they have no area, persistence or SAR and cannot be opened in Changes. "
                          "They are AOI-wide: unavailable when a sub-region is selected." if bbox is None else
                          "Unavailable for a sub-region: the report stores consecutive-pair counts for the whole AOI only."),
        "scope": ("Stored candidates are placed in the interval in which the pipeline first supported them. A candidate seen only "
                  "in the whole-span comparison is not localised to one interval and is reported separately. Persistence of "
                  "candidates first seen in the last interval cannot yet be tested against later acquisitions."),
        "source": {"report": "data/change_model/ayodhya_change_report.json", "candidates": "data/change_model/ayodhya_change_ranked_detail.json",
                   "report_generated_at": svc.report_mtime},
    }

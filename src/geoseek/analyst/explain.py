"""Explainability and false-alarm-suppression views for the React console (read-only).

Nothing here is a model, a new artefact or new storage. Every figure is read from what the change pipeline already wrote
(``ayodhya_change_report.json`` + the ranked-detail sidecar the analyst service loads, ``ablation_study.json``) or is a
pure re-evaluation of the pipeline's own functions on the inputs it stored:

  * explain_candidate     - "why was this flagged": a lead sentence generated from the winning rule and the stored
                            values, the confidence decomposition, the suppression trace, spectral deltas vs thresholds
  * suppression_funnel    - raw components -> after each gate -> final candidates, per pair, with per-gate counts read
                            from the report, plus what happens to the survivors (typing, persistence, SAR)

The confidence terms are NOT stored per candidate (only a list of text lines is). They are therefore re-derived by calling
:func:`geoseek.change.confidence.compute_confidence` with the inputs the sidecar did store, and the result is compared with
the stored confidence; ``reproduces`` / ``max_abs_error`` say whether that held, so a drift would show instead of hiding.

The gates are Ayodhya-calibrated. Every payload that states a gate's effect carries a ``scope`` string saying so.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from geoseek.analyst.service import _confidence_band
from geoseek.change import classify as K
from geoseek.change.confidence import _EPS, WEIGHTS, compute_confidence, spectral_agreement_for
from geoseek.change.suppress import (
    MORPH_MIN_AREA_PX,
    PIXEL_AREA_M2,
    RADIOMETRIC_LOWCONF_WEIGHT,
    REGISTRATION_FLOOR_WEIGHT,
    SuppressionConfig,
)
from geoseek.temporal.persistence import PERSISTENCE_PENALTY

# The order the Ayodhya pipeline actually applies them in: the 50 px floor is applied while components are extracted
# (before the per-component checks, for tractability); the other four then run in suppress.RULES order.
GATE_ORDER = ("morphology", "quality", "registration", "radiometric", "phenology")

SUPPORTED = ("persistent", "progressive", "recent")          # later observations still show the change
CONTRADICTED = ("transient", "inconsistent")                  # later observations contradict it (penalised)

# a term at or above this reads as supporting evidence; confidence.py uses the same 0.7 cut in its own wording
STRONG = 0.7

SCOPE = ("The gates are calibrated on the Ayodhya AOI and its 2019-2026 Sentinel-2 stack. The counts and effects below "
         "describe this run only; they are not expected to carry over to another region or season.")

TERM_LABEL = {
    "model": "Change model", "persistence": "Repeat observations", "spectral": "Spectral index evidence",
    "quality": "Image quality", "registration": "Alignment of the two dates", "radiometric": "Brightness match between dates",
}
TERM_PLAIN = {
    "model": "how strongly the trained change detector fires over the footprint",
    "persistence": "whether later acquisitions still show the change",
    "spectral": "whether the NDVI / NDBI / NDWI shifts back the assigned change type",
    "quality": "clouds, shadow or missing pixels on either date",
    "registration": "how well the two images line up",
    "radiometric": "how reliably the two dates were brought to a common brightness",
}
INDEX_LABEL = {"ndvi": "Vegetation index (NDVI)", "ndbi": "Built-up index (NDBI)", "ndwi": "Water index (NDWI)"}
RULE_TYPES = ("water_gain", "water_loss", "road", "construction", "clearance")  # classify_candidate's order


# --------------------------------------------------------------------------- small helpers


def _num(x, default=None):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else default


def _sgn(x: float, d: int = 2) -> str:
    return f"{x:+.{d}f}".replace("-", "−")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))


def _adv(frac: float) -> str:
    """Adverb for how far an anomaly sits beyond its decision threshold, using the confidence engine's own ramp widths."""
    return " sharply" if frac >= 1.0 else (" clearly" if frac >= 0.5 else "")


def _val(v: float, thr: float, frac: float) -> str:
    """'+0.38 vs. season', with the margin spelled out when the value is only just past its threshold."""
    return f"{_sgn(v)} vs. season" + (f", only just past the {abs(thr):.2f} threshold" if frac < 0.5 else "")


def _veg_phrase(nv: float) -> str:
    if nv <= K.NDVI_DOWN:
        return "fell"
    return "stayed roughly flat" if nv <= K.NDVI_FLAT_MAX else "rose"


# --------------------------------------------------------------------------- lead sentence


def _headline(c: dict) -> str:
    """One plain sentence built from the winning rule and the stored anomaly values. Every number is a stored value."""
    cl = c.get("classification") or {}
    ev = cl.get("evidence") or {}
    rule, ctype = cl.get("rule"), cl.get("change_type") or c.get("change_type")
    nv, nb, nw = (_num(ev.get(k), 0.0) for k in ("ndvi_anomaly", "ndbi_anomaly", "ndwi_anomaly"))
    d_w, elong = _num(ev.get("d_ndwi"), 0.0), _num(ev.get("elongation"), 0.0)

    if rule == "ndbi_up_ndvi_flat_or_down":
        fr = (nb - K.NDBI_UP) / 0.20
        return (f"Built-up index rose{_adv(fr)} ({_val(nb, K.NDBI_UP, fr)}) while vegetation {_veg_phrase(nv)} "
                f"({_sgn(nv)} vs. season) — consistent with new construction.")
    if rule == "linear_morphology_plus_ndbi":
        return (f"A long, narrow footprint (elongation {elong:.2f}; linear features start at {K.ROAD_ELONGATION:.1f}) with the "
                f"built-up index up {_sgn(nb)} vs. season — consistent with a road or other linear works.")
    if rule == "ndwi_anomaly_up":
        fr = (nw - K.NDWI_CHANGE) / 0.25
        return (f"Water index rose{_adv(fr)} ({_val(nw, K.NDWI_CHANGE, fr)}; raw change {_sgn(d_w)}) — consistent with new "
                f"or expanded surface water.")
    if rule == "ndwi_anomaly_down_not_greening":
        return (f"Water index fell ({_sgn(nw)} vs. season; raw change {_sgn(d_w)}) while vegetation was not greening faster "
                f"than the scene ({_sgn(nv)}) — consistent with surface water drying up.")
    if rule == "ndvi_down_ndbi_flat":
        fr = (-nv + K.NDVI_DOWN) / 0.20
        return (f"Vegetation index fell{_adv(fr)} ({_val(nv, K.NDVI_DOWN, fr)}) while the built-up index barely moved "
                f"({_sgn(nb)}) — consistent with vegetation or land clearance.")
    return (f"The change model flagged this footprint but no spectral rule matched (vegetation {_sgn(nv)}, built-up "
            f"{_sgn(nb)}, water {_sgn(nw)} vs. season) — left unclassified, so the spectral evidence for it is weak.")


def _persistence_sentence(c: dict, tl: dict | None) -> str:
    pers = (c.get("trajectory") or {}).get("persistence") or c.get("persistence") or "none"
    if tl and tl.get("first_detected"):
        first, n, tot = tl["first_detected"]["date"], tl["n_supporting"], tl["n_total"]
        if pers in SUPPORTED:
            return f"Present on {n} of {tot} acquisition dates since {first}."
        if pers in CONTRADICTED:
            notes = (c.get("trajectory") or {}).get("notes") or []
            why = (notes[0] + ". ") if notes else ""
            pen = PERSISTENCE_PENALTY.get(pers, 1.0)
            return (f"Time is against it ({pers}): {why}First flagged {first}; seen on {n} of {tot} dates, so confidence "
                    f"carries a ×{pen:.2f} penalty.")
    if pers == "single_pair":
        return "Only two acquisitions cover this site, so persistence cannot be assessed."
    return "No acquisition interval flags the change at this site, so repeat observation cannot confirm it."


# --------------------------------------------------------------------------- spectral tests


def _spectral_tests(c: dict) -> list[dict]:
    """Every spectral/shape test the typing rules apply, with the candidate's value, the threshold and whether it was met.
    ``decisive`` marks the tests of the type that was actually assigned."""
    cl = c.get("classification") or {}
    ev = cl.get("evidence") or {}
    ctype = cl.get("change_type") or c.get("change_type")
    nv, nb, nw = (_num(ev.get(k), 0.0) for k in ("ndvi_anomaly", "ndbi_anomaly", "ndwi_anomaly"))
    d_w, elong = _num(ev.get("d_ndwi"), 0.0), _num(ev.get("elongation"), 0.0)

    def t(index, typ, label, op, thr, val, met, fmt=_sgn):
        return {"index": index, "type": typ, "test": label, "op": op, "threshold": thr, "value": val, "met": bool(met),
                "decisive": typ == ctype, "text": f"{label} {op} {fmt(thr)}"}

    return [
        t("ndwi", "water_gain", "NDWI anomaly", "≥", K.NDWI_CHANGE, nw, nw >= K.NDWI_CHANGE),
        t("ndwi", "water_gain", "NDWI raw change", "≥", K.NDWI_MIN_ABS, d_w, d_w >= K.NDWI_MIN_ABS),
        t("ndwi", "water_loss", "NDWI anomaly", "≤", -K.NDWI_CHANGE, nw, nw <= -K.NDWI_CHANGE),
        t("ndvi", "water_loss", "NDVI anomaly (not greening)", "≤", K.NDVI_FLAT_MAX, nv, nv <= K.NDVI_FLAT_MAX),
        t("ndbi", "road", "NDBI anomaly", "≥", K.ROAD_NDBI_UP, nb, nb >= K.ROAD_NDBI_UP),
        t("shape", "road", "Elongation", "≥", K.ROAD_ELONGATION, elong, elong >= K.ROAD_ELONGATION,
          fmt=lambda x: f"{x:.1f}"),
        t("ndbi", "construction", "NDBI anomaly", "≥", K.NDBI_UP, nb, nb >= K.NDBI_UP),
        t("ndvi", "construction", "NDVI anomaly (not greening)", "≤", K.NDVI_FLAT_MAX, nv, nv <= K.NDVI_FLAT_MAX),
        t("ndvi", "clearance", "NDVI anomaly", "≤", K.NDVI_DOWN, nv, nv <= K.NDVI_DOWN),
        t("ndbi", "clearance", "NDBI anomaly magnitude", "<", K.NDBI_UP, abs(nb), abs(nb) < K.NDBI_UP,
          fmt=lambda x: f"{x:.2f}"),
    ]


# focus (which index maps to open first) per assigned type
_FOCUS = {"construction": ["ndbi", "ndvi"], "road": ["ndbi"], "water_gain": ["ndwi"], "water_loss": ["ndwi", "ndvi"],
          "clearance": ["ndvi"], "other": []}   # unclassified: no index decided it, so none is starred


# --------------------------------------------------------------------------- confidence decomposition


def _decompose(c: dict) -> dict:
    """Re-derive the six evidence terms + three multipliers from the stored inputs and check them against the stored value."""
    from geoseek.change.analyze import radiometric_reliability

    tr = {t["rule"]: t for t in (c.get("suppression") or {}).get("trace", [])}
    q = (tr.get("quality") or {}).get("values", {})
    rg = (tr.get("registration") or {}).get("values", {})
    ra = (tr.get("radiometric") or {}).get("values", {})
    cl = c.get("classification") or {}
    ev = cl.get("evidence") or {}
    ctype = cl.get("change_type") or c.get("change_type") or "other"
    traj = c.get("trajectory") or {}
    pers = traj.get("persistence") or c.get("persistence") or "none"
    sar = c.get("sar") or {}
    sar_on = bool(sar.get("available"))
    dw = float((c.get("suppression") or {}).get("combined_downweight", 1.0))
    sar_f = float(sar.get("factor", 1.0)) if sar_on else 1.0
    spec_q = spectral_agreement_for(ctype, _num(ev.get("ndvi_anomaly"), 0.0), _num(ev.get("ndbi_anomaly"), 0.0),
                                    _num(ev.get("ndwi_anomaly"), 0.0))
    rep = compute_confidence(
        candidate_id=c["candidate_id"], model_prob=float(c.get("mean_model_prob", 0.0)),
        bad_scl_fraction=max(q.get("bad_scl_fraction_earlier", 0.0), q.get("bad_scl_fraction_later", 0.0)),
        valid_fraction=q.get("valid_fraction", 1.0), coreg_residual_px=rg.get("coreg_residual_px", 0.0),
        radiometric_reliability=radiometric_reliability(ra.get("index_band_min_corr", 1.0)),
        persistence=pers, persistence_confidence=float(traj.get("persistence_confidence", 0.0)),
        spectral_agreement=spec_q, change_type=ctype, suppression_downweight=dw, sar_corroboration=sar_f,
        sar_detail=sar.get("verdict"))

    wsum = sum(t.weight for t in rep.terms)
    pen = PERSISTENCE_PENALTY.get(pers, 1.0)
    sar_applied = float(max(0.5, min(1.15, sar_f)))
    dw_applied = float(max(0.0, min(1.0, dw)))
    factors = {t.name: max(t.value, _EPS) ** (t.weight / wsum) for t in rep.terms}
    raw = math.prod(factors.values()) * dw_applied * pen * sar_applied
    final = _clip01(raw)

    def points(f: float) -> float:       # change in confidence (0-100 scale) attributable to this factor alone
        return round(100.0 * (final - _clip01(raw / f)), 1) if f > 0 else None

    terms = []
    for t in rep.terms:
        f = factors[t.name]
        terms.append({
            "name": t.name, "label": TERM_LABEL[t.name], "plain": TERM_PLAIN[t.name], "value": round(t.value, 3),
            "weight": t.weight, "weight_share": round(t.weight / wsum, 3), "factor": round(f, 4), "effect_points": points(f),
            "strength": "supports" if t.value >= STRONG else "weak", "raw": t.raw,
        })
    # largest realised effect on this candidate's number first; weight breaks ties
    terms.sort(key=lambda x: (x["factor"], -x["weight"]))
    mult = [
        {"name": "suppression_downweight", "label": "Gate down-weights", "factor": round(dw_applied, 4),
         "effect_points": points(dw_applied) if dw_applied < 0.999 else 0.0,
         "plain": "alignment / brightness gates that lowered confidence instead of rejecting"},
        {"name": "persistence_penalty", "label": "Temporal-support penalty", "factor": round(pen, 4),
         "effect_points": points(pen) if pen < 0.999 else 0.0,
         "plain": f"extra cut for a '{pers}' trajectory" if pen < 1 else "none for this trajectory"},
        {"name": "sar", "label": "SAR corroboration", "factor": round(sar_applied, 4),
         "effect_points": points(sar_applied) if abs(sar_applied - 1) > 1e-3 else 0.0,
         "plain": ("Sentinel-1 backscatter " + str(sar.get("verdict") or "")) if sar_on else "no Sentinel-1 evidence — neutral"},
    ]
    stored = _num(c.get("confidence"), 0.0)
    return {
        "terms": terms, "multipliers": mult, "weights_sum": wsum, "spectral_q": round(spec_q, 3),
        "stored_confidence": stored, "recomputed_confidence": round(rep.confidence, 4),
        "max_abs_error": round(abs(rep.confidence - stored), 5), "reproduces": abs(rep.confidence - stored) <= 0.005,
        "method": ("Weighted geometric mean of six terms: confidence = ∏ term^(weight / Σweights), then × gate "
                   "down-weights × temporal penalty × SAR factor. Each term can only pull the number down from 1.0; "
                   "'effect' is how far this term alone moved it."),
        "_rep": rep, "_pen": pen, "_dw": dw_applied, "_sar": sar_applied, "_factors": factors, "_points": points,
    }


# --------------------------------------------------------------------------- trace (2a)


def _trace(c: dict, dec: dict, cfg: SuppressionConfig) -> list[dict]:
    """Every stage the candidate passed through, in the order the pipeline applied them."""
    tr = {t["rule"]: t for t in (c.get("suppression") or {}).get("trace", [])}
    fac = dec["_factors"]
    cl = c.get("classification") or {}
    ctype = cl.get("change_type") or c.get("change_type") or "other"
    traj = c.get("trajectory") or {}
    pers = traj.get("persistence") or "none"
    sar = c.get("sar") or {}
    stages: list[dict] = []

    pts = dec["_points"]

    def effect(factor, text, term):
        return {"factor": round(factor, 4) if factor is not None else None, "term": term, "text": text,
                "points": pts(factor) if factor is not None and factor > 0 else None}

    def gate(rule, term=None, note=None):
        t = tr.get(rule)
        if t is None:
            stages.append({"id": rule, "kind": "gate", "verdict": "unknown", "detail": "no trace stored", "effect": None})
            return
        w = float(t.get("weight", 1.0))
        down = t.get("verdict") == "downweight"
        factor = (fac.get(term) if term else None)
        if factor is not None and down:
            factor *= w                                  # the gate's own down-weight multiplies on top of the term
        stages.append({
            "id": rule, "kind": "gate",
            "verdict": "downweighted" if down else ("passed" if t.get("verdict") == "pass" else "rejected"),
            "detail": t.get("detail", "") + (f" {note}" if note else ""), "values": t.get("values", {}),
            "gate_weight": w if down else None,
            "effect": effect(factor, ("feeds the confidence score via the '" + TERM_LABEL[term] + "' term"
                                      + (f" and a ×{w:.2f} gate down-weight" if down else "")) if term
                             else "pass / reject only — no effect on the confidence number", term),
        })

    gate("morphology", note="Applied at component extraction, before the other gates.")
    gate("quality", "quality")
    gate("registration", "registration")
    gate("radiometric", "radiometric")
    gate("phenology")

    ev = cl.get("evidence") or {}
    stages.append({
        "id": "typing", "kind": "typing", "verdict": "unclassified" if ctype == "other" else "typed",
        "detail": cl.get("detail", ""), "rule": cl.get("rule"), "change_type": ctype, "values": ev,
        "effect": effect(fac["spectral"], "spectral agreement with the assigned type feeds the confidence score", "spectral"),
    })

    pen = dec["_pen"]
    pv = ("demoted" if pers in CONTRADICTED else "no_support" if pers == "none"
          else "not_assessable" if pers == "single_pair" else "supported")
    stages.append({
        "id": "persistence", "kind": "score", "verdict": pv, "persistence": pers,
        "detail": "; ".join(traj.get("notes") or []) or f"persistence class '{pers}'",
        "values": {"persistence_confidence": traj.get("persistence_confidence"), "penalty": pen},
        "effect": effect(fac["persistence"] * pen, "repeat-observation term" + (f" and a ×{pen:.2f} penalty" if pen < 1 else ""), "persistence"),
    })

    sar_on = bool(sar.get("available"))
    stages.append({
        "id": "sar", "kind": "score", "verdict": (str(sar.get("verdict") or "applied") if sar_on else "unavailable"),
        "detail": (f"Sentinel-1 VV median {sar.get('vv_median_db')} dB" if sar_on
                   else "No Sentinel-1 evidence for this candidate — the SAR factor is neutral (×1.00)."),
        "values": {k: sar.get(k) for k in ("vv_median_db", "vh_median_db", "factor")} if sar_on else {},
        "effect": effect(dec["_sar"], "weight only, clamped to [0.50, 1.15]; never overrides the optical detection", None),
    })
    return stages


# --------------------------------------------------------------------------- public: one candidate


def _weak_or_absent(c: dict, dec: dict, tl: dict | None, ov: dict | None) -> list[dict]:
    out: list[dict] = []
    sar = c.get("sar") or {}
    if not sar.get("available"):
        out.append({"key": "sar", "level": "absent", "text": "No Sentinel-1 coverage for this candidate — radar could not corroborate or contradict the optical change."})
    unclassified = (c.get("classification") or {}).get("change_type") == "other"
    for t in dec["terms"]:
        if t["strength"] == "weak" and not (unclassified and t["name"] == "spectral"):   # the typing line below says it
            out.append({"key": t["name"], "level": "weak", "text": f"{t['label']} is weak — {t['raw']}."})
    if unclassified:
        out.append({"key": "typing", "level": "weak", "text": "No change-type rule matched, so the type is 'other' and there is no positive spectral evidence for it."})
    pers = (c.get("trajectory") or {}).get("persistence")
    if pers in CONTRADICTED:
        out.append({"key": "persistence_penalty", "level": "weak", "text": f"Later acquisitions contradict the change ('{pers}'); confidence carries a ×{dec['_pen']:.2f} penalty."})
    if not (c.get("terrain") or {}).get("plain_language"):
        out.append({"key": "terrain", "level": "absent", "text": "No terrain context was sampled for this candidate."})
    if ov is None:
        out.append({"key": "overlay", "level": "absent", "text": "No staged tile covers this candidate on the after date, so the index maps cannot be shown."})
    return out


def _overlay_ref(svc, c: dict) -> dict | None:
    """The catalog tile on the AFTER date that covers the candidate centroid, for the NDVI/NDBI/NDWI index maps."""
    try:
        prov = svc.provenance_for(c)
    except Exception:
        return None
    after = next((o for o in prov["observations"] if o["role"] == "after"), None)
    rt = (after or {}).get("representative_tile")
    if not rt:
        return None
    ctype = (c.get("classification") or {}).get("change_type") or c.get("change_type") or "other"
    return {"tile_id": rt["tile_id"], "observation_id": after["observation_id"], "acquired_at": after.get("acquired_at"),
            "focus_indices": _FOCUS.get(ctype, _FOCUS["other"]), "geometry": c.get("_geometry"),
            "note": "Index maps of the after-date tile. The change itself is the difference between dates; these show what "
                    "the surface looks like at the end, with the candidate's footprint outlined."}


def explain_candidate(svc, candidate_id: str) -> dict | None:
    from geoseek.analyst.ui_support import candidate_timeline

    c = svc._by_id.get(candidate_id)
    if c is None:
        return None
    cfg = SuppressionConfig()
    dec = _decompose(c)
    tl = candidate_timeline(svc, candidate_id)
    ev = (c.get("classification") or {}).get("evidence") or {}
    head, pers = _headline(c), _persistence_sentence(c, tl)
    ov = _overlay_ref(svc, c)
    sar = c.get("sar") or {}
    rep_sar = svc.report.get("sar_corroboration") or {}
    run_note = None if rep_sar.get("available", False) else rep_sar.get("note")
    bands = {k: cfg_v for k, cfg_v in (("ndvi", cfg.pheno_ndvi_anomaly), ("ndbi", cfg.pheno_ndbi_anomaly), ("ndwi", cfg.pheno_ndwi_anomaly))}
    anomalies = []
    for idx in ("ndvi", "ndbi", "ndwi"):
        a = _num(ev.get(f"{idx}_anomaly"))
        anomalies.append({"index": idx, "label": INDEX_LABEL[idx], "delta": ev.get(f"d_{idx}"),
                          "seasonal": ev.get(f"scene_d_{idx}"), "anomaly": a, "season_band": bands[idx],
                          "outside_season_band": a is not None and abs(a) > bands[idx]})
    out_terms = [{k: v for k, v in t.items()} for t in dec["terms"]]
    return {
        "available": True, "candidate_id": candidate_id,
        "change_type": (c.get("classification") or {}).get("change_type") or c.get("change_type"),
        "lead": {"headline": head, "persistence": pers, "text": f"{head} {pers}",
                 "rule": (c.get("classification") or {}).get("rule"),
                 "source": "generated from the winning typing rule and the stored index anomalies and trajectory; no fixed numbers"},
        "evidence": {
            "terms": out_terms, "multipliers": dec["multipliers"], "method": dec["method"],
            "stored_confidence": dec["stored_confidence"], "recomputed_confidence": dec["recomputed_confidence"],
            "max_abs_error": dec["max_abs_error"], "reproduces": dec["reproduces"],
            "breakdown_lines": c.get("confidence_breakdown") or [],
            "significance": c.get("significance"), "queue_score": c.get("queue_score"),
        },
        "spectral": {
            "rule": (c.get("classification") or {}).get("rule"), "rule_detail": (c.get("classification") or {}).get("detail"),
            "anomalies": anomalies, "tests": _spectral_tests(c),
            "note": "An anomaly is the candidate's index change minus the scene-wide seasonal change, so a drought-to-green season does not read as change.",
        },
        "weak_or_absent": _weak_or_absent(c, dec, tl, ov),
        "sar": {"available": bool(sar.get("available")), "verdict": sar.get("verdict"), "factor": sar.get("factor"),
                "vv_median_db": sar.get("vv_median_db"), "run_note": run_note},
        "terrain": ({"plain_language": c["terrain"]["plain_language"], "used_in_confidence": False}
                    if (c.get("terrain") or {}).get("plain_language") else None),
        "trace": _trace(c, dec, cfg),
        "ranking": {"queue_score": (svc.report.get("ranking") or {}).get("queue_score"),
                    "significance": (svc.report.get("ranking") or {}).get("significance")},
        "overlay": ov,
        "scope": SCOPE,
    }


# --------------------------------------------------------------------------- public: the pipeline funnel


def _gate_meta(cfg: SuppressionConfig) -> list[dict]:
    return [
        {"rule": "morphology", "kind": "reject", "thresholds": {"min_area_px": cfg.morph_min_area_px},
         "what": f"Drops connected components under {cfg.morph_min_area_px} px ({cfg.morph_min_area_px * PIXEL_AREA_M2 / 10_000:.1f} ha): "
                 "too small to delineate at 10 m once co-registration jitter is allowed for. Applied first, while components are extracted."},
        {"rule": "quality", "kind": "reject",
         "thresholds": {"max_bad_scl_fraction": cfg.max_bad_scl_fraction, "min_valid_fraction": cfg.min_valid_fraction},
         "what": f"Rejects a component with over {cfg.max_bad_scl_fraction * 100:.0f}% cloud / shadow / snow / saturated pixels on "
                 f"either date, or under {cfg.min_valid_fraction * 100:.0f}% jointly valid pixels."},
        {"rule": "registration", "kind": "downweight",
         "thresholds": {"trust_px": cfg.registration_trust_px, "max_px": cfg.registration_max_px, "floor_weight": REGISTRATION_FLOOR_WEIGHT},
         "what": f"Never rejects. Down-weights confidence when the pair's co-registration residual exceeds {cfg.registration_trust_px} px "
                 f"(weight {REGISTRATION_FLOOR_WEIGHT} from {cfg.registration_max_px} px)."},
        {"rule": "radiometric", "kind": "downweight",
         "thresholds": {"min_corr": cfg.radiometric_min_corr, "weight": RADIOMETRIC_LOWCONF_WEIGHT},
         "what": f"Never rejects. Down-weights (×{RADIOMETRIC_LOWCONF_WEIGHT}) when the worst inter-date correlation of the index bands is under "
                 f"{cfg.radiometric_min_corr} or the brightness normalisation was low-confidence."},
        {"rule": "phenology", "kind": "reject",
         "thresholds": {"ndvi": cfg.pheno_ndvi_anomaly, "ndbi": cfg.pheno_ndbi_anomaly, "ndwi": cfg.pheno_ndwi_anomaly},
         "what": f"Rejects a component whose NDVI, NDBI and NDWI changes all stay within ±{cfg.pheno_ndvi_anomaly} / ±{cfg.pheno_ndbi_anomaly} / "
                 f"±{cfg.pheno_ndwi_anomaly} of the scene-wide seasonal shift: seasonal change alone."},
    ]


def _pair_funnel(name: str, p: dict, dates: list[str]) -> dict:
    s = p.get("suppression") or {}
    by = s.get("suppressed_by_rule") or {}
    raw = int(s.get("raw_candidates", 0))
    stages, remaining = [], raw
    for rule in GATE_ORDER:
        removed = int(by.get(rule, 0))
        before = remaining
        remaining -= removed
        stages.append({"rule": rule, "removed": removed, "remaining": remaining, "input": before,
                       "share_of_raw": round(removed / raw, 4) if raw else None,
                       "share_of_input": round(removed / before, 4) if before else None})
    ey, ly = name.split("-")[0], name.split("-")[-1]
    return {
        "name": name, "earlier": next((d for d in dates if d[:4] == ey), ey), "later": next((d for d in dates if d[:4] == ly), ly),
        "comparable": p.get("comparable"), "raw": raw, "survivors": int(s.get("survived", 0)), "suppressed": int(s.get("suppressed", 0)),
        "consistent": remaining == int(s.get("survived", -1)), "stages": stages,
        "downweighted_survivors": s.get("downweighted_survivors"), "context": p.get("context"),
        "class_distribution": p.get("class_distribution"),
    }


def _labelled_benchmark(svc, cfg: SuppressionConfig) -> dict:
    p = svc.out_dir / "ablation_study.json"
    if not p.is_file():
        return {"available": False, "reason": "ablation_study.json is not present (scripts/ablation_change_pipeline.py)"}
    a = json.loads(p.read_text(encoding="utf-8"))
    thr = f"{float((svc.report.get('model') or {}).get('threshold', 0.80)):.2f}"
    o = (a.get("oscd") or {}).get("by_threshold", {}).get(thr)
    if not o:
        return {"available": False, "reason": f"no OSCD ablation at operating threshold {thr}"}
    agg, mf = o.get("aggregate") or {}, o.get("aggregate_morphology_first") or {}
    chain = [("raw model", agg.get("1_raw_model")), ("quality", agg.get("2_quality_gate")), ("radiometric", agg.get("3_radiometric")),
             ("morphology", mf.get("2q_5morph")), ("phenology", mf.get("2q_5morph_4phen"))]
    stages, prev = [], None
    for label, m in chain:
        if not m:
            continue
        row = {"stage": label, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
               "delta_f1": None if prev is None else round(m["f1"] - prev["f1"], 4),
               "delta_precision": None if prev is None else round(m["precision"] - prev["precision"], 4),
               "delta_recall": None if prev is None else round(m["recall"] - prev["recall"], 4)}
        stages.append(row)
        prev = m
    cfgd = (a.get("protocol") or {}).get("suppression_config") or {}
    now = {"max_bad_scl_fraction": cfg.max_bad_scl_fraction, "min_valid_fraction": cfg.min_valid_fraction,
           "radiometric_min_corr": cfg.radiometric_min_corr, "pheno_ndvi_anomaly": cfg.pheno_ndvi_anomaly,
           "pheno_ndbi_anomaly": cfg.pheno_ndbi_anomaly, "pheno_ndwi_anomaly": cfg.pheno_ndwi_anomaly,
           "morph_min_area_px": cfg.morph_min_area_px}
    return {
        "available": True, "threshold": float(thr), "stages": stages, "generated_at": a.get("generated_at"),
        "dataset": "OSCD held-out test (10 regions, pixel-pooled) — a labelled benchmark, independent of the model",
        "config_matches_current": all(abs(float(cfgd.get(k, math.nan)) - v) < 1e-9 for k, v in now.items() if k in cfgd) and bool(cfgd),
        "order_note": "Stages are cumulative in the order OSCD can measure (registration is not ablated; radiometric only down-weights, so it does not change the mask).",
        "caveat": "OSCD is a bitemporal, optical-only benchmark with different seasons from Ayodhya. A gate that costs F1 here (typically "
                  "phenology) is tuned for Ayodhya's seasonal signal and over-suppresses subtle changes elsewhere.",
        "source": "data/change_model/ablation_study.json#oscd",
    }


def suppression_funnel(svc) -> dict:
    cfg = SuppressionConfig()
    report = svc.report
    dates = svc._observation_dates()
    pairs_raw = report.get("pairs") or {}
    span = svc.span_pair_name
    pairs = [_pair_funnel(n, p, dates) for n, p in pairs_raw.items()]
    ds = svc.details

    # --- what happens to the span-pair survivors (all retained in the ranked sidecar) ---
    by_type: dict[str, int] = {}
    by_pers: dict[str, int] = {}
    downweighted = {"registration": 0, "radiometric": 0}
    bands = {"High": 0, "Medium": 0, "Low": 0}
    sar_up = sar_down = sar_neutral = 0
    for c in ds:
        by_type[c.get("change_type") or "other"] = by_type.get(c.get("change_type") or "other", 0) + 1
        by_pers[c.get("persistence") or "none"] = by_pers.get(c.get("persistence") or "none", 0) + 1
        for t in (c.get("suppression") or {}).get("trace", []):
            if t.get("verdict") == "downweight" and t["rule"] in downweighted:
                downweighted[t["rule"]] += 1
        bands[_confidence_band(c.get("confidence"))] += 1
        s = c.get("sar") or {}
        if s.get("available"):
            f = _num(s.get("factor"), 1.0)
            if f > 1.001:
                sar_up += 1
            elif f < 0.999:
                sar_down += 1
            else:
                sar_neutral += 1
    n = len(ds)
    supported = sum(by_pers.get(k, 0) for k in SUPPORTED)
    contradicted = sum(by_pers.get(k, 0) for k in CONTRADICTED)
    rep_sar = report.get("sar_corroboration") or {}
    demoted = [c for c in ds if (c.get("persistence") in CONTRADICTED)]
    demoted.sort(key=lambda c: (-float(c.get("mean_model_prob") or 0.0), c["candidate_id"]))
    sample = []
    for c in demoted[:8]:
        pen = PERSISTENCE_PENALTY.get(c.get("persistence"), 1.0)
        notes = (c.get("trajectory") or {}).get("notes") or []
        sample.append({
            "candidate_id": c["candidate_id"], "change_type": c.get("change_type"), "persistence": c.get("persistence"),
            "mean_model_prob": c.get("mean_model_prob"), "confidence": c.get("confidence"), "penalty": pen,
            "confidence_without_penalty": round(min(1.0, (c.get("confidence") or 0.0) / pen), 4) if pen else None,
            "reason": notes[0] if notes else f"persistence class '{c.get('persistence')}'",
        })
    span_stats = next((p for p in pairs if p["name"] == span), None)
    return {
        "available": True, "aoi": report.get("aoi"), "span_pair": span, "scope": SCOPE,
        "model_threshold": (report.get("model") or {}).get("threshold"),
        "source": {"report": "data/change_model/ayodhya_change_report.json#pairs",
                   "candidates": "data/change_model/ayodhya_change_ranked_detail.json",
                   "report_generated_at": svc.report_mtime},
        "attribution_note": ("Each rejected component is counted against the FIRST gate that rejected it (morphology, then "
                             "quality, registration, radiometric, phenology). Gates that only down-weight remove nothing."),
        "morphology_note": ("The 'shape & size' count is the area floor applied at extraction plus any component rejected by the "
                            "same check later; the report stores them as one number, so they cannot be separated here. Components "
                            "below the floor were never given the other four checks."),
        "gates": _gate_meta(cfg),
        "pairs": pairs,
        "post_gate": {
            "pair": span, "survivors": n,
            "matches_report": bool(span_stats and span_stats["survivors"] == n),
            "typing": {"by_type": by_type, "unclassified": by_type.get("other", 0)},
            "persistence": {"by_class": by_pers, "supported": supported, "contradicted": contradicted,
                            "no_support": by_pers.get("none", 0),
                            "penalty": {k: v for k, v in PERSISTENCE_PENALTY.items()}},
            "downweighted_by_gate": downweighted,
            "confidence_bands": bands,
            "sar": {"available": bool(rep_sar.get("available", False)), "note": rep_sar.get("note"),
                    "moved_up": sar_up, "moved_down": sar_down, "neutral": sar_neutral},
        },
        "demoted_sample": sample,
        "rejected": {
            "retained": False,
            "text": ("Rejected components are not kept. The pipeline writes only counts per gate (above) and the surviving candidates; "
                     "components under the area floor never received a per-candidate trace at all, and the others were dropped from "
                     "memory once counted. The probability rasters remain on disk, but no list of what was rejected, or why, exists "
                     "to browse."),
            "demoted_instead": ("What is retained and inspectable are the candidates that were demoted rather than dropped: every one is in "
                                "the review queue with its trace. A sample is listed here."),
            "to_emit": ("To make rejections browsable the pipeline would have to write, per suppressed component of each pair: candidate id, "
                        "bounding box, area, the first rejecting gate and its trace values; and it would have to record the area-floor drops "
                        "separately from the later morphology check. That output does not exist and was not added."),
        },
        "labelled_benchmark": _labelled_benchmark(svc, cfg),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

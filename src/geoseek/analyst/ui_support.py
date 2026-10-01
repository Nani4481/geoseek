"""Read-only projections that back the React console (``/ui/*``).

Everything here is derived at request time from data the analyst service, the catalog seam and the stored evaluation
artefacts already hold. There is no new storage and no recompute of any model; nothing here writes. Every figure
carries a ``source`` string naming where it was read from, so the console can show provenance beside the number and
never has to hardcode one.

  * console_metrics   - dashboard stat cards (counts, sensors, change-model F1, detector AP50, findings by region)
  * latency_probe     - search latency MEASURED now on a few fixed queries (not a quoted benchmark)
  * candidate_timeline- acquisition dates + first-detected bracket + persistence for one candidate
  * tile_footprints   - real tile footprints for the fingerprint gallery's footprint axis
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

PROBE_QUERIES = (
    "an open water reservoir or pond",
    "dense urban buildings and rooftops",
    "a river with wide sandbars",
    "trees and dense vegetation",
    "bare dry open ground",
)
_PROBE_TTL_S = 30.0
_probe_cache: dict = {"at": 0.0, "value": None}

# the change pipeline's persistence classes for which the later observations still show the change
_PERSISTING = ("persistent", "progressive", "recent")


# --------------------------------------------------------------------------- metrics


@lru_cache(maxsize=4)
def _model_card_eval(checkpoint: str) -> dict | None:
    """``model_card['eval']`` embedded in the change-model checkpoint (held-out OSCD test metrics)."""
    p = Path(checkpoint)
    if not p.is_file():
        return None
    import torch

    blob = torch.load(p, map_location="cpu", weights_only=False)
    return (blob.get("model_card") or {}).get("eval")


def _change_model_block(svc) -> dict:
    ckpt = Path(svc.model_info.get("checkpoint") or "")
    if not ckpt.is_file():                       # the report may name a path from another machine
        ckpt = svc.out_dir / "fc_siam_diff.pt"
    ev = _model_card_eval(str(ckpt)) if ckpt.is_file() else None
    if not ev or "test_at_precision_favouring" not in ev:
        return {"available": False}
    dep, d50 = ev["test_at_precision_favouring"], ev.get("test_at_0.50", {})
    return {
        "available": True,
        "name": svc.model_info.get("name"),
        "operating_threshold": dep.get("threshold"),
        "f1": dep.get("f1"), "precision": dep.get("precision"), "recall": dep.get("recall"), "iou": dep.get("iou"),
        "f1_at_0_50": d50.get("f1"),
        "validation_f1": (ev.get("validation_at_chosen_threshold") or {}).get("f1"),
        "dataset": "OSCD held-out test (10 regions, pixel-pooled)",
        "caveat": "trained on OSCD (Sentinel-2 L1C); applied here to L2A - the domain gap is stated in the evaluation report",
        "source": "change-model checkpoint model_card.eval (test_at_precision_favouring)",
    }


def _detector_block(svc) -> dict:
    p = svc.settings.project_root / "data" / "detect_eval" / "eval_results.json"
    out: dict = {"available": False}
    if p.is_file():
        e = json.loads(p.read_text(encoding="utf-8"))
        v, boot = e.get("val_full_image_v15"), e.get("val_bootstrap_v15") or {}
        if v:
            small = v["per_class"]["small-vehicle"]["AP50"]
            gv, allc = v["groups"]["ground_vehicles"], v["groups"]["all_kept_classes"]
            boot_g = (boot.get("groups") or {})
            out = {
                "available": True,
                "dota_val": {
                    "small_vehicle_ap50": small,
                    "small_vehicle_ap50_ci": ((boot.get("per_class") or {}).get("small-vehicle") or {}).get("AP50_ci"),
                    "ground_vehicles_ap50": gv["macro_AP50"],
                    "ground_vehicles_ap50_ci": (boot_g.get("ground_vehicles") or {}).get("macro_AP50_ci"),
                    "all_classes_ap50": allc["macro_AP50"],
                    "all_classes_ap50_ci": (boot_g.get("all_kept_classes") or {}).get("macro_AP50_ci"),
                    "n_images": (v.get("counts") or {}).get("n_images"),
                    "dataset": "DOTA v1.5 official val (full-image, DOTA-devkit protocol; never used for checkpoint or threshold selection)",
                    "source": "data/detect_eval/eval_results.json#val_full_image_v15",
                },
            }
    try:                                          # xView independent test half - the harder, transfer-honest figure
        ops = svc.detector_model_info().get("operating_points", {})
        sv, lv = ops.get("small-vehicle", {}), ops.get("large-vehicle", {})
        if sv.get("AP50") is not None:
            out["xview_test"] = {
                "small_vehicle_ap50": sv.get("AP50"), "large_vehicle_ap50": lv.get("AP50"),
                "dataset": "xView independent test half (423 images, HBB-fair protocol)",
                "source": sv.get("source"),
            }
            out["available"] = True
    except Exception:
        pass
    out["caveat"] = ("DOTA-val figures do not transfer to every product: on staged Maxar tiles the detector finds "
                     "aircraft and large objects but only a few percent of visible cars.")
    return out


def _region_counts(svc) -> list[dict]:
    regions = svc.list_regions()
    counts = {r["name"]: 0 for r in regions}
    for c in svc.details:
        lon, lat = c.get("centroid_lonlat") or (None, None)
        if lon is None:
            continue
        for r in regions:                         # first match, like the existing region lookup
            w, s, e, n = r["bbox"]
            if w <= lon <= e and s <= lat <= n:
                counts[r["name"]] += 1
                break
    return [{**r, "candidates": counts[r["name"]],
             "center": [round((r["bbox"][0] + r["bbox"][2]) / 2, 5), round((r["bbox"][1] + r["bbox"][3]) / 2, 5)]}
            for r in regions]


def console_metrics(svc) -> dict:
    repo = svc.repo
    collections = repo.list_collections()
    scenes = repo.list_scenes()
    scenes_by_coll: dict[str, int] = {}
    for s in scenes:
        scenes_by_coll[s.collection_id] = scenes_by_coll.get(s.collection_id, 0) + 1
    sensors = [{"collection_id": c.collection_id, "sensor": c.sensor, "platform": c.platform,
                "native_gsd_m": c.native_gsd_m, "n_scenes": scenes_by_coll.get(c.collection_id, 0)}
               for c in collections]
    by_region = _region_counts(svc)
    dist: dict[str, int] = {}
    for c in svc.details:
        dist[c.get("change_type", "other")] = dist.get(c.get("change_type", "other"), 0) + 1
    engine = svc._engine
    return {
        "counters": {
            "tiles_indexed": repo.count_tiles(),
            "vectors_searchable": engine.count() if engine is not None else None,
            "scenes": len(scenes),
            "regions": len(by_region),
            "sensors": len({s["sensor"] for s in sensors}),
            "collections": len(collections),
            "change_candidates": len(svc.details),
            "analyst_decisions": len(repo.list_analyst_decisions()),
        },
        "sensors": sensors,
        "findings_by_region": by_region,
        "findings_by_type": dist,
        "change_pipeline_aoi": svc.report.get("aoi"),
        "observation_dates": svc._observation_dates(),
        "change_model": _change_model_block(svc),
        "detector": _detector_block(svc),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def latency_probe(engine) -> dict:
    """Search latency measured now: one discarded warm-up run, then each fixed query once. Cached briefly so the
    dashboard polling it cannot itself load the engine."""
    now = time.monotonic()
    if _probe_cache["value"] is not None and now - _probe_cache["at"] < _PROBE_TTL_S:
        return _probe_cache["value"]
    engine.search_text(PROBE_QUERIES[0], k=10)                       # warm-up, not reported
    times = [float(engine.search_text(q, k=10)[1]) for q in PROBE_QUERIES]
    ordered = sorted(times)
    value = {
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 2),
        "min_ms": round(ordered[0], 2), "max_ms": round(ordered[-1], 2),
        "n_queries": len(times), "k": 10, "vectors": engine.count(),
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "source": "measured now: text query -> embed -> exact vector search -> catalog join (engine-reported latency_ms)",
    }
    _probe_cache.update(at=now, value=value)
    return value


# --------------------------------------------------------------------------- timeline


def candidate_timeline(svc, candidate_id: str) -> dict | None:
    """Acquisition dates for a candidate with the first-detected bracket and persistence, built only from dates
    that exist in the catalog and the change pipeline's own trajectory for that candidate."""
    c = svc._by_id.get(candidate_id)
    if c is None:
        return None
    dates = svc._observation_dates()
    tr = c.get("trajectory") or {}
    by_later = {}
    for p in tr.get("consecutive_pairs", []):
        by_later[(p.get("window") or ["", ""])[1]] = p
    es = tr.get("earliest_supported_change") or {}
    window = es.get("window") or None
    persistence = tr.get("persistence") or c.get("persistence") or "none"
    onset = window[1] if window else None
    persists = persistence in _PERSISTING

    points = []
    for i, d in enumerate(dates):
        iv = by_later.get(d)
        if i == 0:
            role = "baseline"
        elif onset is None:
            role = "no_change_seen"
        elif d < onset:
            role = "before_detection"
        elif d == onset:
            role = "first_detected"
        else:
            role = "supported" if persists else "not_supported"
        points.append({
            "date": d, "year": d[:4], "role": role,
            "interval": None if iv is None else {
                "from": iv["window"][0], "to": iv["window"][1], "changed": bool(iv.get("changed")),
                "probability": iv.get("probability"), "comparable": bool(iv.get("comparable"))},
        })
    supporting = [p["date"] for p in points if p["role"] in ("first_detected", "supported")]
    return {
        "candidate_id": candidate_id,
        "dates": dates,
        "points": points,
        "persistence": persistence,
        "persistence_confidence": tr.get("persistence_confidence"),
        "first_detected": None if not window else {
            "date": onset, "bracket": window, "caveat": es.get("caveat"),
            "quality_note": es.get("imagery_quality_note")},
        "supporting_dates": supporting,
        "n_supporting": len(supporting),
        "n_total": len(dates),
        "span": tr.get("span_pair"),
        "notes": tr.get("notes", []),
        "definition": ("n_supporting counts acquisition dates from the first-detected date onward on which the change is "
                       "still supported (persistent / progressive / recent classes); a transient or inconsistent "
                       "candidate counts only its first-detected date. n_total is every acquisition date in the catalog."),
    }


# --------------------------------------------------------------------------- footprints


def _km_per_deg(lat: float) -> tuple[float, float]:
    return 111.32 * math.cos(math.radians(lat)), 110.57


def tile_footprints(repo, tile_ids: list[str]) -> list[dict]:
    from shapely import wkt as shapely_wkt

    out = []
    for tid in tile_ids[:100]:
        t = repo.get_tile(tid)
        if t is None:
            continue
        try:
            w, s, e, n = shapely_wkt.loads(t.geom_wkt_4326).bounds
        except Exception:
            continue
        kx, ky = _km_per_deg((s + n) / 2)
        out.append({
            "tile_id": tid, "bbox": [round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
            "width_km": round((e - w) * kx, 3), "height_km": round((n - s) * ky, 3),
            "area_km2": round((e - w) * kx * (n - s) * ky, 3),
            "cloud_fraction": t.cloud_fraction, "observation_id": t.observation_id,
        })
    return out

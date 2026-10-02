"""Read-only projections that back the React console (``/ui/*``).

Everything here is derived at request time from data the analyst service, the catalog seam and the stored evaluation
artefacts already hold. There is no new storage and no recompute of any model; nothing here writes. Every figure
carries a ``source`` string naming where it was read from, so the console can show provenance beside the number and
never has to hardcode one.

  * console_metrics   - dashboard stat cards (counts, sensors, change-model F1, detector AP50, findings by region)
  * latency_probe     - search latency MEASURED now on a few fixed queries (not a quoted benchmark)
  * candidate_timeline- acquisition dates + first-detected bracket + persistence for one candidate
  * tile_footprints   - real tile footprints for the fingerprint gallery's footprint axis
  * candidate_dossier - catalog-sourced sensor provenance for the two dates a dossier is printed for
  * projection_*      - the precomputed 3-D embedding projection (scripts/compute_projection.py): a display sample + exact lookups
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


# --------------------------------------------------------------------------- dossier sensor provenance

# Standard STAC `view:` fields. They are reported ONLY if some catalogued metadata blob actually carries them: the dossier
# omits a field the catalog does not hold rather than showing a placeholder.
_VIEW_FIELDS = {"sun_elevation_deg": "sun_elevation", "sun_azimuth_deg": "sun_azimuth", "off_nadir_deg": "off_nadir"}


def _find_number(blobs: list, needle: str):
    """First numeric value whose (normalised) key contains ``needle`` anywhere in the nested metadata dicts."""
    stack = [b for b in blobs if isinstance(b, (dict, list))]
    while stack:
        cur = stack.pop(0)
        items = cur.items() if isinstance(cur, dict) else enumerate(cur)
        for k, v in items:
            if isinstance(k, str) and needle in k.lower().replace(":", "_").replace("-", "_") and isinstance(v, (int, float))                     and not isinstance(v, bool):
                return float(v)
            if isinstance(v, (dict, list)):
                stack.append(v)
    return None


def candidate_dossier(svc, candidate_id: str, before: str, after: str) -> dict | None:
    """Sensor provenance for the before / after acquisitions of a candidate, read from the catalog (scene -> observation ->
    collection -> nearest tile). ``before`` / ``after`` are dates or years from the timeline."""
    from geoseek.analyst.service import S2_COLLECTION
    from geoseek.change.analyze import DATE_TO_OBS

    c = svc._by_id.get(candidate_id)
    if c is None:
        return None
    lon, lat = c["centroid_lonlat"]
    out_obs, found = [], {k: False for k in _VIEW_FIELDS}
    for role, when in (("before", before), ("after", after)):
        year = (when or "")[:4]
        obs_id = DATE_TO_OBS.get(year)
        if obs_id is None:
            raise ValueError(f"{when!r} is not one of the catalog's acquisition years {sorted(DATE_TO_OBS)}")
        obs = svc.repo.get_observation(obs_id)
        scene = svc.repo.get_scene(obs.scene_id) if obs else None
        coll = svc.repo.get_collection(scene.collection_id) if scene else None
        tile, eps = None, 0.01
        try:
            recs = [r for r in svc.repo.query_tiles(bbox=(lon - eps, lat - eps, lon + eps, lat + eps), collection=S2_COLLECTION)
                    if r.observation_id == obs_id]
            tile = svc.repo.get_tile(recs[0].tile_id) if recs else None
        except Exception:
            tile = None
        blobs = [getattr(scene, "metadata", None), getattr(obs, "metadata", None), getattr(coll, "metadata", None),
                 getattr(tile, "quality_flags", None)]
        row = {
            "role": role, "requested": when, "observation_id": obs_id,
            "acquired_at": obs.acquired_at if obs else None,
            "acquired_at_precision": ((getattr(scene, "metadata", None) or {}).get("acquired_at_precision")),
            "scene_id": scene.scene_id if scene else None,
            "platform": scene.platform if scene else None,
            "sensor": coll.sensor if coll else None, "collection_id": coll.collection_id if coll else None,
            "native_gsd_m": coll.native_gsd_m if coll else None,
            "processing_baseline": scene.processing_baseline if scene else None,
            "crs": scene.crs if scene else None, "license": scene.license if scene else None,
            "tile_id": tile.tile_id if tile else None,
            "cloud_fraction": tile.cloud_fraction if tile else None,
        }
        for key, needle in _VIEW_FIELDS.items():
            v = _find_number(blobs, needle)
            if v is not None:
                row[key] = v
                found[key] = True
        out_obs.append(row)
    return {"candidate_id": candidate_id, "centroid_lonlat": [lon, lat], "observations": out_obs,
            "view_fields_not_catalogued": sorted(k for k, ok in found.items() if not ok),
            "source": "catalog: scenes / observations / collections / nearest tile to the candidate centroid"}


# --------------------------------------------------------------------------- 3-D embedding projection

PROJECTION_NPZ = "projection_3d.npz"
PROJECTION_META = "projection_3d.meta.json"
DEFAULT_PROJECTION_POINTS = 20_000
MAX_PROJECTION_POINTS = 120_000
_proj_cache: dict = {"key": None, "value": None}


def _load_projection(data_dir: Path) -> dict | None:
    """The finished artifact written by scripts/compute_projection.py, cached until the file changes. None if never computed."""
    import numpy as np

    npz, meta = Path(data_dir) / "discovery" / PROJECTION_NPZ, Path(data_dir) / "discovery" / PROJECTION_META
    if not npz.is_file() or not meta.is_file():
        return None
    key = (str(npz), npz.stat().st_mtime_ns, meta.stat().st_mtime_ns)
    if _proj_cache["key"] != key:
        z = np.load(npz, allow_pickle=False)
        ids = z["tile_ids"].astype(str)
        _proj_cache.update(key=key, value={
            "ids": ids, "xyz": z["xyz"], "lon": z["lon"], "lat": z["lat"], "region": z["region"], "cluster": z["cluster"],
            "regions": [str(r) for r in z["regions"]], "index_of": {t: i for i, t in enumerate(ids.tolist())},
            "meta": json.loads(meta.read_text(encoding="utf-8")),
        })
    return _proj_cache["value"]


def projection_sample(data_dir: Path, current_vectors: int | None, max_points: int = DEFAULT_PROJECTION_POINTS, seed: int = 0) -> dict:
    """A deterministic uniform random sample of the projection for drawing (every point if the artifact is smaller than
    ``max_points``). The caller learns exactly how many points exist and how many are shown - the console prints both."""
    import numpy as np

    p = _load_projection(data_dir)
    if p is None:
        return {"available": False, "reason": "no projection has been computed - run `python scripts/compute_projection.py`"}
    n = int(len(p["ids"]))
    max_points = max(1, min(int(max_points), MAX_PROJECTION_POINTS))
    idx = np.arange(n) if n <= max_points else np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))
    meta = p["meta"]
    clusters = sorted({int(c) for c in p["cluster"][idx].tolist() if c >= 0})
    return {
        "available": True, "n_total": n, "n_shown": int(len(idx)), "sampled": bool(len(idx) < n), "seed": seed,
        "regions": p["regions"], "clusters": clusters,
        "tile_ids": p["ids"][idx].tolist(),
        "xyz": np.round(p["xyz"][idx], 4).reshape(-1).tolist(),
        "lon": np.round(p["lon"][idx], 5).tolist(), "lat": np.round(p["lat"][idx], 5).tolist(),
        "region": p["region"][idx].astype(int).tolist(), "cluster": p["cluster"][idx].astype(int).tolist(),
        "stale": current_vectors is not None and current_vectors != meta.get("n_points"),
        "current_vectors": current_vectors,
        "meta": {k: meta.get(k) for k in ("method", "umap", "pca_components", "libraries", "wall_seconds", "power_source",
                                          "created_at", "caveat", "n_points", "partial")},
    }


def projection_lookup(data_dir: Path, tile_ids: list[str]) -> dict:
    """Exact coordinates for specific tiles (search hits), whether or not they are in the display sample."""
    p = _load_projection(data_dir)
    if p is None:
        return {"available": False, "points": [], "missing": list(tile_ids)}
    pts, missing = [], []
    for t in tile_ids[:500]:
        i = p["index_of"].get(t)
        if i is None:
            missing.append(t)
            continue
        x, y, z = (round(float(v), 4) for v in p["xyz"][i])
        pts.append({"tile_id": t, "xyz": [x, y, z], "lon": round(float(p["lon"][i]), 5), "lat": round(float(p["lat"][i]), 5),
                    "region": p["regions"][int(p["region"][i])], "cluster": int(p["cluster"][i])})
    return {"available": True, "points": pts, "missing": missing}


# --------------------------------------------------------------------------- cluster geography

CLUSTER_CELL_DEG = 0.03          # ~3.3 km: about one 256-px Sentinel-2 tile; tiles of one cluster that share a cell are counted together
_geo_cache: dict = {"key": None, "value": None}


def cluster_geo(svc) -> dict:
    """Where each archive cluster lies: every clustered tile's centre, binned to ``CLUSTER_CELL_DEG`` cells per cluster.

    Cluster membership is read from the same stored clustering run ``/discovery/clusters`` reports (``tile_clusters.json``) and
    each tile's position from the catalog footprint, so ``n_tiles`` per cluster equals that run's own ``sizes`` entry. Cells carry
    their tile count; nothing is smoothed, sampled or estimated.
    """
    import re

    p = Path(svc.settings.index_dir) / "tile_clusters.json"
    if not p.is_file():
        return {"available": False, "note": "run scripts/cluster_at_scale.py"}
    key = (str(p), p.stat().st_mtime_ns, id(svc.repo))
    if _geo_cache["key"] == key:
        return _geo_cache["value"]
    run = json.loads(p.read_text(encoding="utf-8"))
    label = run.get("tile_cluster", {})
    num = re.compile(r"-?\d+(?:\.\d+)?")
    cells: dict[str, dict[tuple[int, int], list]] = {}
    bbox: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    unplaced = 0
    for r in svc.repo.iter_tile_records():
        lab = label.get(r.tile_id)
        if lab is None:
            continue
        v = [float(t) for t in num.findall(r.geom_wkt_4326)]
        if len(v) < 8:
            unplaced += 1
            continue
        xs, ys = v[0:8:2], v[1:8:2]
        lon, lat = sum(xs) / 4.0, sum(ys) / 4.0
        c = str(lab)
        counts[c] = counts.get(c, 0) + 1
        ij = (math.floor(lon / CLUSTER_CELL_DEG), math.floor(lat / CLUSTER_CELL_DEG))
        cell = cells.setdefault(c, {}).get(ij)
        if cell is None:
            cells[c][ij] = [(ij[0] + 0.5) * CLUSTER_CELL_DEG, (ij[1] + 0.5) * CLUSTER_CELL_DEG, 1]
        else:
            cell[2] += 1
        b = bbox.setdefault(c, [min(xs), min(ys), max(xs), max(ys)])
        b[0], b[1], b[2], b[3] = min(b[0], min(xs)), min(b[1], min(ys)), max(b[2], max(xs)), max(b[3], max(ys))
    value = {
        "available": True, "cell_deg": CLUSTER_CELL_DEG, "n_clustered_tiles": sum(counts.values()), "n_unplaced": unplaced,
        "clusters": {c: {"n_tiles": counts[c], "n_cells": len(cells[c]), "bbox": [round(v, 5) for v in bbox[c]],
                         "cells": [[round(x, 4), round(y, 4), n] for x, y, n in cells[c].values()]} for c in sorted(counts, key=int)},
        "source": "tile_clusters.json (the stored clustering run) joined to catalog tile footprints",
    }
    _geo_cache.update(key=key, value=value)
    return value

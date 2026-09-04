"""AnalystService - the read/review layer the analyst UI and API sit on.

Everything the Phase 4/5 pipeline computed, made queryable and joined to its
provenance:

  * the ranked change queue (all 1104 survivors, filterable);
  * per-candidate FULL evidence - model probability, persistence trajectory,
    spectral-index anomalies, quality / registration / radiometric terms, SAR
    corroboration, and the suppression trace (what was checked, what passed);
  * the provenance chain tile -> observation -> scene -> collection, with the
    source COG URL, licence, acquisition dates, checksums, model version +
    weights SHA-256 and git commit;
  * analyst confirm/reject, written through the append-only audit seam;
  * GeoJSON/CSV export carrying that provenance per feature.

Catalog access is only ever through the MetadataRepository seam handed in on
the shared SearchEngine; no sqlite3 / faiss here.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from geoseek.analyst.geo import polygon_geojson, ring_bounds
from geoseek.catalog.entities import AnalystDecision
from geoseek.catalog.naming import base_scene_id
from geoseek.change.analyze import DATE_TO_OBS, OUT_DIR
from geoseek.config import get_settings
from geoseek.staging.manifest import load_manifest

S2_COLLECTION = "sentinel-2-l2a"
OBS_TO_DATE = {v: k for k, v in DATE_TO_OBS.items()}
_SUMMARY_KEYS = (
    "rank", "candidate_id", "pair", "centroid_lonlat", "area_px", "area_m2", "change_type",
    "confidence", "significance", "queue_score", "persistence", "earliest_supported",
    "mean_model_prob",
)


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("geoseek")
    except Exception:
        return "0.1.0"


def _box_around(lonlat, area_m2: float) -> dict:
    """Crude square footprint fallback if the raster grid can't be read."""
    lon, lat = lonlat
    half_m = max((float(area_m2) ** 0.5) / 2.0, 30.0)
    dlat = half_m / 111_320.0
    import math

    dlon = half_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]
    return {"type": "Polygon", "coordinates": [[[round(x, 6), round(y, 6)] for x, y in ring]]}


class AnalystService:
    def __init__(self, engine=None, *, report_path: Path | None = None, repo=None):
        self.settings = get_settings()
        self._engine = engine
        self.repo = repo if repo is not None else (engine.repo if engine is not None else self._own_repo())
        self._owns_repo = repo is None and engine is None
        self.out_dir = OUT_DIR
        self.report_path = Path(report_path) if report_path else self.out_dir / "ayodhya_change_report.json"
        self.reload()

    def _own_repo(self):
        from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
        from geoseek.ingest.store import DB_FILENAME

        return SQLiteMetadataRepository(self.settings.index_dir / DB_FILENAME)

    # -- load -------------------------------------------------------------

    def reload(self) -> None:
        if not self.report_path.is_file():
            raise FileNotFoundError(
                f"no change report at {self.report_path} - run `python -m geoseek.change.analyze` first")
        self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.model_info = self.report.get("model", {})
        self.span_pair_name = self.report.get("span_pair", "2019-2024")
        self.prob_raster_path = self._resolve_prob_raster()

        detail_p = Path(self.report.get("full_ranked_detail_json")
                        or (self.out_dir / "ayodhya_change_ranked_detail.json"))
        if detail_p.is_file():
            self.details = json.loads(detail_p.read_text(encoding="utf-8"))
        else:  # fall back to the diversified top-N carried inline in the report
            self.details = self.report.get("top_candidates", [])
        self._by_id = {c["candidate_id"]: c for c in self.details}

        for c in self.details:
            if "_geometry" in c:
                continue
            try:
                c["_geometry"] = polygon_geojson(self.prob_raster_path, c["bbox_rc"])
            except Exception:
                c["_geometry"] = _box_around(c["centroid_lonlat"], c.get("area_m2", 0.0))
            c["_bounds"] = ring_bounds(c["_geometry"]["coordinates"][0])

        manifest = load_manifest()
        self.git_commit = manifest.get("git", {}).get("commit_short") or manifest.get("git", {}).get("commit", "")
        self.pipeline_version = _pkg_version()
        self.report_mtime = datetime.fromtimestamp(
            self.report_path.stat().st_mtime, tz=timezone.utc).isoformat()

    def _resolve_prob_raster(self) -> Path:
        de, dl = self.span_pair_name.split("-")
        earlier, later = DATE_TO_OBS[de], DATE_TO_OBS[dl]
        return self.out_dir / f"prob_{earlier}__to__{later}.tif"

    # -- queue ----------------------------------------------------------

    def _current_verdicts(self) -> dict:
        return {cid: d.decision for cid, d in self.repo.latest_decision_by_candidate().items()}

    @staticmethod
    def _window_of(c: dict) -> tuple[str, str]:
        w = c.get("earliest_supported") or ["2019-03-30", "2024-03-08"]
        return w[0], w[1]

    def list_candidates(
        self, *, bbox=None, date_start=None, date_end=None, change_type=None, min_confidence=None,
        sensor=None, persistence=None, decision=None, sort="queue_score", limit=100, offset=0,
    ) -> dict:
        verdicts = self._current_verdicts()
        rows = []
        for c in self.details:
            if change_type and c.get("change_type") != change_type:
                continue
            if min_confidence is not None and (c.get("confidence") or 0.0) < float(min_confidence):
                continue
            if persistence and c.get("persistence") != persistence:
                continue
            if sensor:
                s = sensor.lower()
                if s in ("sentinel-1", "s1", "sar", "c-sar"):
                    if not (c.get("sar") or {}).get("available"):
                        continue
                elif s not in ("sentinel-2", "s2", "msi", "sentinel-2a", "sentinel-2b"):
                    continue
            if date_start or date_end:
                w0, w1 = self._window_of(c)
                if date_start and w1 < date_start:
                    continue
                if date_end and w0 > date_end:
                    continue
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                bx0, by0, bx1, by1 = c.get("_bounds") or (0, 0, 0, 0)
                if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                    continue
            v = verdicts.get(c["candidate_id"], "undecided")
            if decision and v != decision:
                continue
            rows.append((c, v))

        reverse = True
        key = sort.lstrip("-")
        if sort.startswith("-") or key == "rank":
            reverse = False
        rows.sort(key=lambda cv: (cv[0].get(key) if cv[0].get(key) is not None else -1), reverse=reverse)

        total = len(rows)
        page = rows[offset: offset + limit]
        return {
            "total": total, "count": len(page), "offset": offset, "limit": limit,
            "sort": sort,
            "filters": {"bbox": bbox, "date_start": date_start, "date_end": date_end,
                        "change_type": change_type, "min_confidence": min_confidence,
                        "sensor": sensor, "persistence": persistence, "decision": decision},
            "candidates": [self._summary(c, v) for c, v in page],
        }

    def _summary(self, c: dict, verdict: str) -> dict:
        out = {k: c.get(k) for k in _SUMMARY_KEYS}
        out["geometry"] = c["_geometry"]
        out["decision"] = verdict
        ev = (c.get("classification") or {}).get("evidence", {})
        out["spectral_anomaly_max"] = round(max(
            abs(ev.get("ndvi_anomaly", 0.0)), abs(ev.get("ndbi_anomaly", 0.0)),
            abs(ev.get("ndwi_anomaly", 0.0))), 3)
        sar = c.get("sar") or {}
        out["sar"] = ({"available": True, "vv_median_db": sar.get("vv_median_db"),
                       "verdict": sar.get("verdict"), "factor": sar.get("factor")}
                      if sar.get("available") else {"available": False})
        return out

    # -- detail + provenance ------------------------------------------

    def get_candidate(self, candidate_id: str) -> dict | None:
        c = self._by_id.get(candidate_id)
        if c is None:
            return None
        decisions = [d.as_dict() for d in self.repo.list_analyst_decisions(candidate_id=candidate_id)]
        base = {k: v for k, v in c.items() if not k.startswith("_")}
        base["geometry"] = c["_geometry"]
        base["provenance"] = self.provenance_for(c)
        base["decisions"] = decisions
        base["current_decision"] = decisions[-1] if decisions else None
        base["temporal_trajectory"] = self._trajectory_view(c)
        earlier_obs, later_obs = (c["pair"].split("->") + ["", ""])[:2]
        after_date = OBS_TO_DATE.get(later_obs)
        # the selector only offers valid BEFORE dates: acquisitions strictly
        # before the pair's later observation (picking the after-date would give
        # "before 2024 | after 2024" - two identical panels).
        before_dates = [d for d in DATE_TO_OBS if after_date is None or d < after_date]
        base["imagery"] = {
            "before_dates": before_dates,
            "after_date": after_date or list(DATE_TO_OBS)[-1],
            "dates": before_dates,  # back-compat alias
            "views": ["rgb", "overlay"],
            "url_template": f"/candidates/{candidate_id}/imagery?date={{date}}&view={{view}}",
        }
        return base

    def _trajectory_view(self, c: dict) -> dict:
        """The change trajectory reduced to what the UI draws: one row per
        interval with a changed/stable verdict + probability."""
        tr = c.get("trajectory") or {}
        intervals = []
        for p in tr.get("consecutive_pairs", []):
            intervals.append({
                "window": p.get("window"), "changed": p.get("changed"),
                "probability": p.get("probability"), "comparable": p.get("comparable"),
                "kind": "consecutive",
            })
        sp = tr.get("span_pair")
        if sp:
            intervals.append({"window": sp.get("window"), "changed": sp.get("changed"),
                              "probability": sp.get("probability"), "comparable": sp.get("comparable"),
                              "kind": "span"})
        return {
            "persistence": tr.get("persistence"),
            "persistence_confidence": tr.get("persistence_confidence"),
            "intervals": intervals,
            "earliest_supported_change": tr.get("earliest_supported_change", {}),
            "notes": tr.get("notes", []),
        }

    def provenance_for(self, c: dict) -> dict:
        earlier_obs, later_obs = (c["pair"].split("->") + ["", ""])[:2]
        lon, lat = c["centroid_lonlat"]
        chain = []
        for role, obs_id in (("before", earlier_obs), ("after", later_obs)):
            obs = self.repo.get_observation(obs_id)
            scene = self.repo.get_scene(obs.scene_id) if obs else None
            coll = self.repo.get_collection(scene.collection_id) if scene else None
            rep_tile = None
            eps = 0.01
            try:
                recs = [r for r in self.repo.query_tiles(
                    bbox=(lon - eps, lat - eps, lon + eps, lat + eps), collection=S2_COLLECTION)
                    if r.observation_id == obs_id]
                if recs:
                    rep_tile = self.repo.get_tile_provenance(recs[0].tile_id).as_dict()
            except Exception:
                rep_tile = None
            chain.append({
                "role": role,
                "observation_id": obs_id,
                "acquired_at": obs.acquired_at if obs else None,
                "scene": None if scene is None else {
                    "scene_id": scene.scene_id, "platform": scene.platform,
                    "source_url": scene.source_url, "license": scene.license,
                    "processing_baseline": scene.processing_baseline, "crs": scene.crs,
                    "checksums": scene.checksums,
                },
                "collection": None if coll is None else {
                    "collection_id": coll.collection_id, "sensor": coll.sensor,
                    "platform": coll.platform, "bands": list(coll.bands),
                    "native_gsd_m": coll.native_gsd_m,
                },
                "representative_tile": rep_tile,
            })

        sar = c.get("sar") or {}
        sar_block = None
        if sar.get("available"):
            sc = self.report.get("sar_corroboration", {})
            sar_block = {
                "s1_pair": sc.get("s1_pair"),
                "speckle_filter": sc.get("speckle_filter"),
                "vv_median_db": sar.get("vv_median_db"), "vh_median_db": sar.get("vh_median_db"),
                "scene_dvv_db": sar.get("scene_dvv_db"), "scene_dvh_db": sar.get("scene_dvh_db"),
                "verdict": sar.get("verdict"), "confidence_factor": sar.get("factor"),
            }

        return {
            "observations": chain,
            "model": {
                "name": self.model_info.get("name"),
                "checkpoint": self.model_info.get("checkpoint"),
                "threshold": self.model_info.get("threshold"),
                "weights_sha256": self.model_info.get("weights_sha256"),
            },
            "code": {"git_commit": self.git_commit, "pipeline_version": self.pipeline_version},
            "probability_raster": str(self.prob_raster_path),
            "sar_corroboration": sar_block,
            "report_generated_at": self.report_mtime,
        }

    # -- decisions ----------------------------------------------------

    def record_decision(self, candidate_id: str, *, decision: str, note: str = "",
                        analyst: str = "") -> dict:
        c = self._by_id.get(candidate_id)
        if c is None:
            raise KeyError(candidate_id)
        snapshot = self.get_candidate(candidate_id)   # freeze the full evidence + provenance
        snapshot.pop("decisions", None)
        snapshot.pop("current_decision", None)
        row = self.repo.record_analyst_decision(AnalystDecision(
            decision_id="", candidate_id=candidate_id, decision=decision,
            analyst_note=note or "", analyst=analyst or "",
            model_version=f"{self.model_info.get('name')}@{self.model_info.get('threshold')}",
            weights_sha256=self.model_info.get("weights_sha256", ""),
            git_commit=self.git_commit, pipeline_version=self.pipeline_version,
            confidence_at_decision=c.get("confidence"),
            evidence_snapshot=snapshot,
        ))
        return row.as_dict()

    def audit(self, *, candidate_id: str | None = None, limit: int | None = None,
              full: bool = False) -> dict:
        rows = self.repo.list_analyst_decisions(candidate_id=candidate_id, limit=limit)
        out = []
        for d in rows:
            r = d.as_dict()
            if not full:  # keep the log view light - snapshot fetched per-decision
                snap = r.pop("evidence_snapshot", {}) or {}
                r["evidence_snapshot_keys"] = sorted(snap.keys())
                r["has_evidence_snapshot"] = bool(snap)
            out.append(r)
        return {
            "count": len(out),
            "append_only": True,
            "note": "decisions are never updated or deleted; re-deciding appends a new row",
            "decisions": out,
        }

    def get_decision(self, decision_id: str) -> dict | None:
        d = self.repo.get_analyst_decision(decision_id)
        return d.as_dict() if d else None

    # -- stats / health --------------------------------------------

    def stats(self) -> dict:
        span = self.report.get("pairs", {}).get(self.span_pair_name, {})
        return {
            "index": {
                "vectors": self._engine.count() if self._engine is not None else None,
                "tiles": self.repo.count_tiles(),
                "collections": [c.collection_id for c in self.repo.list_collections()],
                "scenes": len(self.repo.list_scenes()),
            },
            "change_pipeline": {
                "aoi": self.report.get("aoi"),
                "span_pair": self.span_pair_name,
                "observations": self.report.get("observations"),
                "candidates_ranked": self.report.get("full_ranked_total", len(self.details)),
                "class_distribution": span.get("class_distribution"),
                "ranking": self.report.get("ranking", {}).get("queue_score"),
            },
            "model": {
                "name": self.model_info.get("name"),
                "threshold": self.model_info.get("threshold"),
                "weights_sha256": self.model_info.get("weights_sha256"),
                "checkpoint": self.model_info.get("checkpoint"),
                "trained_on": "OSCD (Sentinel-2 L1C); applied L2A - domain gap stated in the report",
            },
            "audit": {"decisions": len(self.repo.list_analyst_decisions())},
            "sar": self.report.get("sar_corroboration", {}).get("available", False),
            "build": {
                "git_commit": self.git_commit,
                "pipeline_version": self.pipeline_version,
                "report_path": str(self.report_path),
                "report_generated_at": self.report_mtime,
                "probability_raster": str(self.prob_raster_path),
            },
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "report_loaded": bool(self.details),
            "candidates": len(self.details),
            "probability_raster_present": self.prob_raster_path.is_file(),
            "vectors": self._engine.count() if self._engine is not None else None,
            "audit_decisions": len(self.repo.list_analyst_decisions()),
        }

    # -- discovery pass-through -----------------------------------

    def discovery_clusters(self) -> dict:
        p = self.settings.index_dir / "tile_clusters.json"
        if not p.is_file():
            return {"available": False, "note": "run scripts/cluster_tiles.py"}
        d = json.loads(p.read_text(encoding="utf-8"))
        by_obs: dict[str, dict[str, int]] = {}
        for tile_id, lab in d.get("tile_cluster", {}).items():
            obs = tile_id.rsplit("_r", 1)[0]
            by_obs.setdefault(obs, {}).setdefault(str(lab), 0)
            by_obs[obs][str(lab)] += 1
        return {
            "available": True,
            "n_clusters": d.get("n_clusters"), "noise_count": d.get("noise_count"),
            "n_tiles": d.get("n_tiles"), "sizes": d.get("sizes"),
            "cluster_concepts": d.get("cluster_concepts"), "params": d.get("params"),
            "per_observation_counts": by_obs,
            "cluster_map_png": d.get("cluster_map_png"),
        }

    def cluster_for_tile(self, tile_id: str):
        p = self.settings.index_dir / "tile_clusters.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8")).get("tile_cluster", {}).get(tile_id)

    def similar_to_candidate(self, candidate_id: str, *, k: int = 8) -> dict:
        if self._engine is None:
            raise RuntimeError("similarity search needs the SearchEngine")
        from geoseek.discovery.knn import find_more_like_this

        c = self._by_id.get(candidate_id)
        if c is None:
            raise KeyError(candidate_id)
        lon, lat = c["centroid_lonlat"]
        res = find_more_like_this(self._engine, lon=lon, lat=lat, k=k)
        res["seed_candidate_id"] = candidate_id
        res["seed_cluster"] = self.cluster_for_tile(res.get("seed_tile_id", ""))
        for r in res.get("results", []):
            r["cluster"] = self.cluster_for_tile(r["tile_id"])
        return res

    # -- export ----------------------------------------------------

    def _resolve_for_export(self, candidate_ids, filters) -> list[dict]:
        if candidate_ids:
            return [self._by_id[cid] for cid in candidate_ids if cid in self._by_id]
        if filters:
            listing = self.list_candidates(**{**filters, "limit": 100000, "offset": 0})
            ids = [row["candidate_id"] for row in listing["candidates"]]
            return [self._by_id[i] for i in ids]
        return list(self.details)

    @staticmethod
    def _date_from_obs(obs_id: str) -> str | None:
        parts = base_scene_id(obs_id).split("_")
        if len(parts) >= 3 and len(parts[2]) == 8 and parts[2].isdigit():
            return f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}"
        return None

    def _feature(self, c: dict, verdict_row: dict | None) -> dict:
        earlier_obs, later_obs = (c["pair"].split("->") + ["", ""])[:2]
        acq_dates = [self._date_from_obs(earlier_obs), self._date_from_obs(later_obs)]
        cls = c.get("classification") or {}
        ev = cls.get("evidence", {})
        sar = c.get("sar") or {}
        supp = c.get("suppression") or {}
        return {
            "type": "Feature",
            "geometry": c["_geometry"],
            "properties": {
                "candidate_id": c["candidate_id"],
                "change_type": c.get("change_type"),
                "confidence": c.get("confidence"),
                "significance": c.get("significance"),
                "queue_score": c.get("queue_score"),
                "persistence": c.get("persistence"),
                "earliest_supported_change": (c.get("trajectory", {})
                                              .get("earliest_supported_change", {}).get("window")),
                "earliest_supported_caveat": (c.get("trajectory", {})
                                              .get("earliest_supported_change", {}).get("caveat")),
                "area_m2": c.get("area_m2"),
                "centroid_lonlat": c.get("centroid_lonlat"),
                "evidence_summary": " | ".join(c.get("confidence_breakdown", [])),
                "spectral_anomalies": {
                    "ndvi_anomaly": ev.get("ndvi_anomaly"), "ndbi_anomaly": ev.get("ndbi_anomaly"),
                    "ndwi_anomaly": ev.get("ndwi_anomaly"),
                    "framing": "delta minus scene-wide seasonal delta (drought->green makes raw deltas useless)",
                },
                "sar_corroboration": (None if not sar.get("available") else {
                    "vv_median_db": sar.get("vv_median_db"), "verdict": sar.get("verdict"),
                    "confidence_factor": sar.get("factor")}),
                "suppression": {"suppressed": supp.get("suppressed"),
                                "suppressed_by": supp.get("suppressed_by"),
                                "combined_downweight": supp.get("combined_downweight")},
                "source_scene_ids": [base_scene_id(earlier_obs), base_scene_id(later_obs)],
                "source_observation_ids": [earlier_obs, later_obs],
                "acquisition_dates": acq_dates,
                "sensor": "MSI / Sentinel-2 (optical); Sentinel-1 C-SAR corroboration where available",
                # flat, GIS-reader-friendly copies of the key provenance fields
                "model_version": f"{self.model_info.get('name')}@{self.model_info.get('threshold')}",
                "weights_sha256": self.model_info.get("weights_sha256"),
                "git_commit": self.git_commit,
                "pipeline_version": self.pipeline_version,
                "processing": {
                    "model": self.model_info.get("name"),
                    "model_threshold": self.model_info.get("threshold"),
                    "weights_sha256": self.model_info.get("weights_sha256"),
                    "git_commit": self.git_commit,
                    "pipeline_version": self.pipeline_version,
                },
                "analyst_decision": (None if not verdict_row else {
                    "decision": verdict_row["decision"], "analyst": verdict_row["analyst"],
                    "note": verdict_row["analyst_note"], "at": verdict_row["created_at"],
                    "decision_id": verdict_row["decision_id"]}),
            },
        }

    def export(self, *, candidate_ids=None, filters=None, fmt: str = "geojson", write: bool = True) -> dict:
        cands = self._resolve_for_export(candidate_ids, filters)
        latest = {cid: d.as_dict() for cid, d in self.repo.latest_decision_by_candidate().items()}
        features = [self._feature(c, latest.get(c["candidate_id"])) for c in cands]
        fc = {
            "type": "FeatureCollection",
            "name": "geoseek_change_candidates",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "aoi": self.report.get("aoi"),
                "span_pair": self.span_pair_name,
                "model": self.model_info,
                "git_commit": self.git_commit,
                "pipeline_version": self.pipeline_version,
                "count": len(features),
            },
            "features": features,
        }
        out: dict = {"count": len(features), "geojson": fc}

        if fmt in ("csv", "both"):
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["candidate_id", "lon", "lat", "change_type", "confidence", "significance",
                        "queue_score", "persistence", "earliest_supported", "area_m2",
                        "source_scene_ids", "acquisition_dates", "weights_sha256", "git_commit",
                        "pipeline_version", "analyst_decision"])
            for f in features:
                p = f["properties"]
                lon, lat = p["centroid_lonlat"]
                w.writerow([p["candidate_id"], lon, lat, p["change_type"], p["confidence"],
                            p["significance"], p["queue_score"], p["persistence"],
                            p["earliest_supported_change"], p["area_m2"],
                            "|".join(x for x in p["source_scene_ids"] if x),
                            "|".join(str(x) for x in p["acquisition_dates"] if x),
                            p["processing"]["weights_sha256"], p["processing"]["git_commit"],
                            p["processing"]["pipeline_version"],
                            (p["analyst_decision"] or {}).get("decision", "")])
            out["csv"] = buf.getvalue()

        if write:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            exp_dir = self.out_dir / "exports"
            exp_dir.mkdir(parents=True, exist_ok=True)
            gj = exp_dir / f"change_candidates_{ts}.geojson"
            gj.write_text(json.dumps(fc, indent=1), encoding="utf-8")
            out["geojson_path"] = str(gj)
            if "csv" in out:
                cf = exp_dir / f"change_candidates_{ts}.csv"
                cf.write_text(out["csv"], encoding="utf-8")
                out["csv_path"] = str(cf)
        return out

    def close(self) -> None:
        if self._owns_repo:
            self.repo.close()

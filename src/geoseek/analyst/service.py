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
import re
from datetime import datetime, timezone
from pathlib import Path

from geoseek.analyst.geo import polygon_geojson, ring_bounds
from geoseek.catalog.entities import AnalystDecision, WatchArea
from geoseek.catalog.naming import base_scene_id
from geoseek.change.analyze import DATE_TO_OBS, OUT_DIR
from geoseek.config import get_settings
from geoseek.staging.manifest import load_manifest

S2_COLLECTION = "sentinel-2-l2a"
OBS_TO_DATE = {v: k for k, v in DATE_TO_OBS.items()}

# --- plain-language phrasing for the demo / overview presentation layer -----
# These never replace the technical values the analyst UI already shows; they
# sit *beside* them so a non-specialist viewer can read a finding at a glance.
HUMAN_CHANGE_TYPE = {
    "water_gain": "New open water / flooding",
    "water_loss": "Water body shrank or dried",
    "construction": "New built-up surface",
    "clearance": "Vegetation or land cleared",
    "road": "New road / linear corridor",
    "other": "Surface change (unclassified)",
}
_REGION_DROP_TOKENS = {"diverse", "scaled", "82km", ""}


def _region_of(aoi_name: str | None) -> str:
    """Collapse an observation ``aoi_name`` (``kerala_backwaters_43PFL_diverse``)
    to its human region (``kerala_backwaters``) by dropping the MGRS tile token
    and the staging suffixes."""
    toks = [t for t in re.split(r"[_\-]", aoi_name or "")
            if t.lower() not in _REGION_DROP_TOKENS
            and not re.fullmatch(r"\d{2}[a-z]{3}", t.lower())]
    return "_".join(toks) or (aoi_name or "unknown")


def _confidence_band(v: float | None) -> str:
    v = float(v or 0.0)
    return "High" if v >= 0.85 else ("Medium" if v >= 0.60 else "Low")
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
        self.span_pair_name = self.report.get(
            "span_pair", f"{list(DATE_TO_OBS)[0]}-{list(DATE_TO_OBS)[-1]}" if DATE_TO_OBS else "")
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

    def _window_of(self, c: dict) -> tuple[str, str]:
        dates = self._observation_dates()
        fallback = [dates[0], dates[-1]] if dates else ["", ""]
        w = c.get("earliest_supported") or fallback
        return w[0], w[1]

    def list_candidates(
        self, *, bbox=None, date_start=None, date_end=None, change_type=None, min_confidence=None,
        sensor=None, persistence=None, decision=None, sort="queue_score", limit=100, offset=0,
        candidate_ids=None,
    ) -> dict:
        verdicts = self._current_verdicts()
        ids_filter = set(candidate_ids) if candidate_ids else None
        rows = []
        for c in self.details:
            if ids_filter is not None and c["candidate_id"] not in ids_filter:
                continue
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

    # -- presentation layer (demo / overview) ----------------------------
    #
    # A read-only projection for non-specialist viewers. Everything here is
    # derived from data the analyst service already holds - the ranked detail
    # list and the catalog seam - joined to plain-language phrasing. No new
    # storage, no recompute; the real analyst views are untouched.

    def _observation_dates(self) -> list[str]:
        out = []
        for obs_id in self.report.get("observations", []):
            m = re.search(r"_(\d{4})(\d{2})(\d{2})_", obs_id)
            if m:
                out.append("-".join(m.groups()))
        return sorted(set(out))

    def _later_observation_count(self, c: dict) -> int:
        """How many acquisitions sit at or after the end of a candidate's
        earliest supported change window - i.e. how many later looks agree the
        change is still there."""
        win = ((c.get("trajectory") or {}).get("earliest_supported_change") or {}).get("window")
        end = (win or c.get("earliest_supported") or ["", ""])[1]
        dates = self._observation_dates()
        n = sum(1 for d in dates if end and d >= end)
        return n or 1

    def _persistence_human(self, c: dict) -> str:
        n = self._later_observation_count(c)
        p = c.get("persistence")
        obs = "observation" if n == 1 else "observations"
        return {
            "persistent": f"Confirmed across {n} later {obs}",
            "progressive": f"Grew steadily across {n} later {obs}",
            "recent": "Only visible in the most recent interval",
            "transient": "Appeared, then reverted - treat with caution",
            "inconsistent": "Flickers across dates - low trust",
        }.get(p, "Single before / after pair only")

    def _imagery_dates(self, c: dict) -> tuple[str, str]:
        earlier_obs, later_obs = (c["pair"].split("->") + ["", ""])[:2]
        after = OBS_TO_DATE.get(later_obs) or list(DATE_TO_OBS)[-1]
        w0 = (c.get("earliest_supported") or [""])[0][:4]
        before = w0 if w0 in DATE_TO_OBS else list(DATE_TO_OBS)[0]
        if before >= after:  # never show before == after
            before = next((d for d in DATE_TO_OBS if d < after), list(DATE_TO_OBS)[0])
        return before, after

    def _plain_caption(self, c: dict) -> str:
        d0, d1 = c.get("earliest_supported") or ["the first date", "a later date"]
        dates = self._observation_dates()
        last = dates[-1] if dates else d1
        n = self._later_observation_count(c)
        obs = "observation" if n == 1 else "observations"
        ct = c.get("change_type")
        pers = c.get("persistence")
        if ct == "water_gain":
            if pers in ("persistent", "progressive"):
                return (f"Open water appeared here between {d0} and {d1} and was still "
                        f"present in {last} - confirmed across {n} later {obs}.")
            return f"Open water appeared here between {d0} and {d1}."
        if ct == "water_loss":
            return f"An open-water surface shrank or dried between {d0} and {d1}."
        if ct == "construction":
            tail = ("and has stayed built-up since" if pers in ("persistent", "progressive")
                    else "in the most recent interval")
            return f"New built-up surface appeared between {d0} and {d1} {tail}."
        if ct == "road":
            return (f"A new linear cleared corridor - most likely a road or track - "
                    f"appeared between {d0} and {d1}.")
        if ct == "clearance":
            return f"Vegetation or land cover was cleared here between {d0} and {d1}."
        return (f"{HUMAN_CHANGE_TYPE.get(ct, 'A surface change')} was detected between "
                f"{d0} and {d1}; {self._persistence_human(c).lower()}.")

    def _featured_card(self, c: dict) -> dict:
        before, after = self._imagery_dates(c)
        cid = c["candidate_id"]
        return {
            "candidate_id": cid,
            "change_type": c.get("change_type"),
            "change_type_human": HUMAN_CHANGE_TYPE.get(c.get("change_type"), "Surface change"),
            "confidence": c.get("confidence"),
            "confidence_band": _confidence_band(c.get("confidence")),
            "persistence": c.get("persistence"),
            "persistence_human": self._persistence_human(c),
            "later_observations": self._later_observation_count(c),
            "area_m2": c.get("area_m2"),
            "centroid_lonlat": c.get("centroid_lonlat"),
            "queue_score": c.get("queue_score"),
            "caption": self._plain_caption(c),
            "before_date": before,
            "after_date": after,
            "imagery": {
                "before": f"/candidates/{cid}/imagery?date={before}&view=rgb",
                "after": f"/candidates/{cid}/imagery?date={after}&view=rgb",
                "overlay": f"/candidates/{cid}/imagery?date={after}&view=overlay",
            },
        }

    def _pick_featured(self, n: int = 4) -> list[dict]:
        """Best candidates, diversified by change type: highest-queue candidate
        of each distinct type first, then fill by queue score."""
        ranked = sorted(self.details, key=lambda c: c.get("queue_score") or 0.0, reverse=True)
        ranked = [c for c in ranked if (c.get("confidence") or 0.0) >= 0.60]
        picked, seen_types, used = [], set(), set()
        for c in ranked:
            t = c.get("change_type")
            if t not in seen_types:
                picked.append(c); seen_types.add(t); used.add(c["candidate_id"])
            if len(picked) == n:
                break
        for c in ranked:
            if len(picked) == n:
                break
            if c["candidate_id"] not in used:
                picked.append(c); used.add(c["candidate_id"])
        return [self._featured_card(c) for c in picked]

    def list_regions(self) -> list[dict]:
        """The AOI regions already in the catalog (Ayodhya, Kutch, ...) with
        a bbox spanning every observation footprint in that region - used to
        populate a one-click region picker on the watch-area form, the Review
        Queue's AOI filter, and the Search view's AOI filter, instead of
        making an analyst type a raw bbox string."""
        from shapely import wkt as shapely_wkt

        try:
            observations = self.repo.list_observations()
        except Exception:
            return []
        by_region: dict[str, list] = {}
        for o in observations:
            by_region.setdefault(_region_of(o.aoi_name), []).append(o)
        out = []
        for name, obs_list in sorted(by_region.items()):
            w, s, e, n = 180.0, 90.0, -180.0, -90.0
            for o in obs_list:
                try:
                    bx0, by0, bx1, by1 = shapely_wkt.loads(o.footprint_wkt_4326).bounds
                except Exception:
                    continue
                w, s, e, n = min(w, bx0), min(s, by0), max(e, bx1), max(n, by1)
            if not (e > w):
                continue
            out.append({"name": name, "bbox": [round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
                       "n_observations": len(obs_list)})
        return out

    def presentation_summary(self) -> dict:
        try:
            observations = self.repo.list_observations()
        except Exception:
            observations = []
        regions = sorted({_region_of(o.aoi_name) for o in observations}) if observations else []

        dist: dict[str, int] = {}
        conf_vals = []
        for c in self.details:
            dist[c.get("change_type", "other")] = dist.get(c.get("change_type", "other"), 0) + 1
            if c.get("confidence") is not None:
                conf_vals.append(float(c["confidence"]))
        high_conf = sum(1 for v in conf_vals if v >= 0.85)

        featured = self._pick_featured(4)
        by_q = sorted(self.details, key=lambda x: x.get("queue_score") or 0.0, reverse=True)
        water = next((c for c in by_q if c.get("change_type") == "water_gain"), None)
        demo_cid = (water or (by_q[0] if by_q else {})).get("candidate_id")

        return {
            "offline": True,
            "aoi": self.report.get("aoi"),
            "span_pair": self.span_pair_name,
            "observation_dates": self._observation_dates(),
            "counters": {
                "tiles_indexed": self.repo.count_tiles(),
                "regions": len(regions),
                "scenes": len(self.repo.list_scenes()),
                "change_candidates": len(self.details),
                "vectors": self._engine.count() if self._engine is not None else None,
                "high_confidence": high_conf,
                "analyst_decisions": len(self.repo.list_analyst_decisions()),
                "unseen_notifications": self.unseen_notification_count(),
            },
            "latest_alerts": self.list_notifications()[:5],
            "regions": regions,
            "change_type_distribution": dist,
            "change_type_labels": HUMAN_CHANGE_TYPE,
            "featured": featured,
            "demo": {
                "search_query": "an open water reservoir or pond",
                "water_gain_candidate_id": demo_cid,
                "discovery_seed": demo_cid,
                "steps": [
                    "Natural-language search over the tile index",
                    f"Open the strongest water-gain candidate - every observation date "
                    f"({', '.join(self._observation_dates())}) + change overlay",
                    "Find more places that look like it",
                    "Everything ran offline, with full provenance and an append-only audit trail",
                ],
            },
        }

    # -- discovery pass-through -----------------------------------

    def discovery_clusters(self) -> dict:
        p = self.settings.index_dir / "tile_clusters.json"
        if not p.is_file():
            return {"available": False, "note": "run scripts/cluster_at_scale.py"}
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
            "cluster_concepts": d.get("cluster_concepts"),
            "display_labels": d.get("display_labels"),
            "params": d.get("params"),
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

    # -- object detection pass-through (Phase 8F) --------------------------
    # Thin wrappers over geoseek.analyst.detections - real stored oriented-box
    # detections (scripts/detect_maxar.py's output), never re-run inference.

    def detector_model_info(self) -> dict:
        from geoseek.analyst.detections import model_info
        return model_info()

    def list_detection_observations(self) -> list[dict]:
        from geoseek.analyst.detections import list_observations
        return list_observations(self.repo)

    def list_detection_tiles(self, observation_id: str) -> list[dict]:
        from geoseek.analyst.detections import list_tiles_with_detections
        return list_tiles_with_detections(observation_id)

    def tile_detections(self, observation_id: str, row: int, col: int) -> dict:
        from geoseek.analyst.detections import tile_detections
        return tile_detections(self.repo, observation_id, row, col)

    def tile_image_png(self, observation_id: str, row: int, col: int) -> bytes:
        from geoseek.analyst.detections import tile_image_png
        return tile_image_png(self.repo, observation_id, row, col)

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

    # -- standing watch areas (Phase 8 Step C) -----------------------------
    # Thin wrappers over the MetadataRepository seam - see geoseek.watch.evaluator
    # for the matching logic run after each change-pipeline re-run.

    def list_watch_areas(self, *, active_only: bool = False) -> list[dict]:
        return [w.as_dict() for w in self.repo.list_watch_areas(active_only=active_only)]

    def get_watch_area(self, watch_id: str) -> dict | None:
        w = self.repo.get_watch_area(watch_id)
        return w.as_dict() if w else None

    def create_watch_area(
        self, *, name: str, bbox=None, polygon_wkt_4326: str | None = None, text_query: str = "",
        change_types: tuple[str, ...] = (), min_confidence: float | None = None, created_by: str = "",
    ) -> dict:
        w = WatchArea(
            watch_id="", name=name, bbox=tuple(bbox) if bbox else None, polygon_wkt_4326=polygon_wkt_4326,
            text_query=text_query, change_types=tuple(change_types), min_confidence=min_confidence,
            created_by=created_by,
        )
        return self.repo.create_watch_area(w).as_dict()

    def update_watch_area(self, watch_id: str, **fields) -> dict:
        """Partial update: an omitted keyword leaves that field unchanged. For
        ``bbox``/``polygon_wkt_4326``/``min_confidence`` (all optional-by-design
        on a watch area), passing the key with value ``None`` explicitly CLEARS
        it; every other field's ``None`` is treated as "not supplied"."""
        existing = self.repo.get_watch_area(watch_id)
        if existing is None:
            raise KeyError(watch_id)
        if "bbox" in fields and fields["bbox"] is not None:
            fields["bbox"] = tuple(fields["bbox"])
        if "change_types" in fields:
            fields["change_types"] = tuple(fields["change_types"])
        import dataclasses as _dc

        updated = _dc.replace(existing, **{k: v for k, v in fields.items() if v is not None or k in
                                           ("bbox", "polygon_wkt_4326", "min_confidence")})
        return self.repo.update_watch_area(updated).as_dict()

    def delete_watch_area(self, watch_id: str) -> None:
        self.repo.delete_watch_area(watch_id)

    # Severity = max(confidence * significance) over a notification's matched
    # candidates. Both factors already exist and already drive queue_score
    # (confidence in [0,1], significance clipped to [0.10, 1.0]) - this is just
    # "how confident, and how big/anomalous, is the worst-case member of this
    # notification's match set". Thresholds are the empirical 75th/90th
    # percentile of this product over the current candidate population
    # (841 span-survivor candidates: p75=0.26, p90=0.43) rounded to 0.25/0.5 -
    # i.e. HIGH is roughly "top decile" (matching how rare confidence>=0.85
    # alone already is - 5.6% of candidates), MEDIUM the next-highest quartile.
    _SEVERITY_HIGH, _SEVERITY_MEDIUM = 0.5, 0.25

    def _severity(self, candidate_ids) -> tuple[str, float]:
        score = 0.0
        for cid in candidate_ids:
            c = self._by_id.get(cid) or {}
            score = max(score, float(c.get("confidence") or 0.0) * float(c.get("significance") or 0.0))
        if score >= self._SEVERITY_HIGH:
            band = "high"
        elif score >= self._SEVERITY_MEDIUM:
            band = "medium"
        else:
            band = "low"
        return band, round(score, 4)

    def list_notifications(self, *, watch_id: str | None = None, unseen_only: bool = False) -> list[dict]:
        rows = []
        for n in self.repo.list_notifications(watch_id=watch_id, unseen_only=unseen_only):
            d = n.as_dict()
            watch = self.repo.get_watch_area(n.watch_id)
            d["watch_name"] = watch.name if watch else None
            d["observation_date"] = self._date_from_obs(n.observation_id)
            d["severity"], d["severity_score"] = self._severity(n.candidate_ids)
            d["candidates"] = [
                {"candidate_id": cid, "change_type": (self._by_id.get(cid) or {}).get("change_type"),
                 "confidence": (self._by_id.get(cid) or {}).get("confidence"),
                 "link": f"/candidates/{cid}"}
                for cid in n.candidate_ids
            ]
            rows.append(d)
        return rows

    def unseen_notification_count(self) -> int:
        return len(self.repo.list_notifications(unseen_only=True))

    def mark_notification_seen(self, notification_id: str) -> None:
        self.repo.mark_notification_seen(notification_id)

    # -- sector summary brief (Phase 8 Step D) -----------------------------

    def sector_brief(
        self, *, bbox: tuple[float, float, float, float] | None = None,
        date_start: str | None = None, date_end: str | None = None,
    ) -> dict:
        """Plain-language + structured summary for a chosen AOI/date range:
        counts + total area per change type, the most significant candidates,
        the time window covered, and imagery/quality caveats - carrying the
        same provenance (model/weights/git/pipeline) as the GeoJSON export."""
        listing = self.list_candidates(bbox=bbox, date_start=date_start, date_end=date_end, limit=100000)
        rows = listing["candidates"]

        by_type: dict[str, dict] = {}
        for r in rows:
            t = r.get("change_type") or "other"
            d = by_type.setdefault(t, {"change_type": t, "count": 0, "area_m2": 0.0})
            d["count"] += 1
            d["area_m2"] += float(r.get("area_m2") or 0.0)
        for d in by_type.values():
            d["area_m2"] = round(d["area_m2"], 1)
            d["area_ha"] = round(d["area_m2"] / 10_000.0, 2)
        by_type_list = sorted(by_type.values(), key=lambda d: d["area_m2"], reverse=True)

        top = sorted(rows, key=lambda r: r.get("queue_score") or 0.0, reverse=True)[:10]

        obs_dates = sorted(self._observation_dates())
        window = [date_start or (obs_dates[0] if obs_dates else None),
                  date_end or (obs_dates[-1] if obs_dates else None)]

        span = self.report.get("pairs", {}).get(self.span_pair_name, {})
        caveats = [self.report.get("domain_gap_statement", "")]
        radio = span.get("context", {}).get("radiometric_reliability")
        if radio is not None and radio < 0.7:
            caveats.append(f"radiometric reliability for the span pair is {radio:.2f} (of 1.0) - "
                           "some spectral-anomaly evidence should be read with that in mind.")
        if not (self.report.get("sar_corroboration") or {}).get("available"):
            caveats.append("no Sentinel-1 SAR staged for part of this window - SAR corroboration was "
                           "neutral (no effect) for candidates outside SAR coverage.")
        n_no_terrain = sum(1 for r in rows if not (self._by_id.get(r["candidate_id"], {}).get("terrain")))
        if n_no_terrain:
            caveats.append(f"{n_no_terrain} of {len(rows)} candidates have no terrain evidence "
                           "(DEM not staged when the pipeline last ran for them).")

        struct = {
            "aoi": {"bbox": list(bbox) if bbox else None, "name": self.report.get("aoi")},
            "window": window,
            "total_candidates": len(rows),
            "by_change_type": by_type_list,
            "total_area_m2": round(sum(d["area_m2"] for d in by_type_list), 1),
            "most_significant": [
                {"candidate_id": r["candidate_id"], "change_type": r.get("change_type"),
                 "confidence": r.get("confidence"), "area_m2": r.get("area_m2"),
                 "centroid_lonlat": r.get("centroid_lonlat"), "queue_score": r.get("queue_score")}
                for r in top
            ],
            "imagery_and_quality_caveats": caveats,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "model_version": f"{self.model_info.get('name')}@{self.model_info.get('threshold')}",
                "weights_sha256": self.model_info.get("weights_sha256"),
                "git_commit": self.git_commit,
                "pipeline_version": self.pipeline_version,
                "report_generated_at": self.report_mtime,
            },
        }
        struct["text"] = self._sector_brief_text(struct)
        return struct

    @staticmethod
    def _sector_brief_text(s: dict) -> str:
        lines = []
        aoi_label = s["aoi"]["name"] or (f"bbox {s['aoi']['bbox']}" if s["aoi"]["bbox"] else "the full AOI")
        lines.append(f"SECTOR SUMMARY BRIEF - {aoi_label}")
        lines.append(f"Window covered: {s['window'][0]} to {s['window'][1]}")
        lines.append(f"Generated: {s['generated_at']}")
        lines.append("")
        lines.append(f"{s['total_candidates']} change candidate(s), {s['total_area_m2']:,.0f} m^2 total.")
        lines.append("")
        lines.append("By change type:")
        for d in s["by_change_type"]:
            lines.append(f"  - {d['change_type']:<14s} {d['count']:>4d} candidate(s), "
                         f"{d['area_m2']:>10,.0f} m^2 ({d['area_ha']:.1f} ha)")
        lines.append("")
        lines.append("Most significant candidates:")
        for i, r in enumerate(s["most_significant"], 1):
            lon, lat = r["centroid_lonlat"] or (None, None)
            lines.append(f"  {i:>2d}. {r['candidate_id']}  [{r['change_type']}]  "
                         f"confidence {r['confidence']:.2f}  {r['area_m2']:,.0f} m^2  "
                         f"({lon:.5f}, {lat:.5f})" if lon is not None else
                         f"  {i:>2d}. {r['candidate_id']}  [{r['change_type']}]")
        lines.append("")
        lines.append("Imagery / quality caveats:")
        for c in s["imagery_and_quality_caveats"]:
            if c:
                lines.append(f"  - {c}")
        lines.append("")
        p = s["provenance"]
        lines.append(f"Model {p['model_version']}  weights_sha256={p['weights_sha256']}  "
                     f"git_commit={p['git_commit']}  pipeline_version={p['pipeline_version']}")
        return "\n".join(lines)

    def export_sector_brief(self, **kwargs) -> dict:
        """sector_brief() + write both the human-readable .txt and the
        structured .json to disk (same directory pattern as export())."""
        brief = self.sector_brief(**kwargs)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = self.out_dir / "sector_briefs"
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"sector_brief_{ts}.txt"
        json_path = out_dir / f"sector_brief_{ts}.json"
        txt_path.write_text(brief["text"], encoding="utf-8")
        json_path.write_text(json.dumps(brief, indent=2), encoding="utf-8")
        brief["text_path"] = str(txt_path)
        brief["json_path"] = str(json_path)
        return brief

    def close(self) -> None:
        if self._owns_repo:
            self.repo.close()

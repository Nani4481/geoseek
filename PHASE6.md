# Phase 6 — the analyst interface

What a judge touches. Everything the Phase 4/5 pipeline already computes is made
**visible** and **auditable**, and the whole thing runs with the network off.

## Stack (and why)

**FastAPI backend + a static vanilla-JS/CSS single-page frontend served by the
same process.** No React/Vite (needs `npm install` + a build — not offline-
packageable without committing `node_modules`); no Streamlit (not installed,
`pip install` needs the network). The chosen stack has **zero install, zero
build, zero CDN**: every dependency (`fastapi`, `uvicorn`, `jinja2`, `pillow`)
was already in the env. Fonts are the system stack. The **map is an HTML5
`<canvas>` in EPSG:4326** with a lon/lat graticule — candidate footprints are
reprojected server-side and drawn/hit-tested locally. **No web tile server is
ever contacted** — provable by grepping the bundle for `http`/`https` (none) and
by the Playwright run that aborts every off-origin request while the pages still
render.

```
src/geoseek/analyst/
  service.py   AnalystService  — queue, full evidence, provenance, decisions, export
  imagery.py   before/after/overlay PNG crops from the cached prob raster (LRU, offline)
  geo.py       candidate pixel-bbox -> EPSG:4326 polygon
  web/         index.html + style.css + app.js   (the four-view SPA, /app/)
src/geoseek/search/api.py       — routes mounted on the existing app
```

Seams respected: no `sqlite3` / `faiss` outside `catalog/sqlite_repository.py`
and `vectorindex/faiss_flat.py`. Catalog access goes through
`MetadataRepository`; vector search through `SearchEngine`.

---

## Acceptance

### 1. Endpoints + working curl examples

| method | path | purpose |
|---|---|---|
| GET | `/search/text`, `/search/image` | existing semantic search (reused) |
| GET | `/candidates` | ranked change queue — filter `bbox`, `date_start/end`, `change_type`, `min_confidence`, `sensor`, `persistence`, analyst `decision`; `sort` + paginate; each row carries a GeoJSON footprint |
| GET | `/candidates/{id}` | full detail: geometry, type, confidence + **full evidence breakdown**, **suppression trace**, **temporal trajectory**, earliest supported change + caveat, **SAR corroboration**, **provenance chain** |
| GET | `/candidates/{id}/imagery?date=&view=` | before / after / change-overlay PNG (`date` = 2019\|2021\|2024, `view` = rgb\|overlay) |
| POST | `/candidates/{id}/decision` | analyst confirm/reject + note → append-only audit |
| GET | `/audit`, `/audit/{decision_id}` | the decision log (list is light; per-row fetch carries the frozen evidence snapshot) |
| POST | `/export` | GeoJSON (+CSV) of selected or filtered candidates, full provenance per feature |
| GET | `/health`, `/stats` | index size, tile/collection counts, model version + weights SHA-256, git commit, build info |
| GET | `/discovery/clusters`, `/discovery/similar`, `/candidates/{id}/similar` | KNN "more like this" + HDBSCAN clusters |
| — | `/app/` | the offline frontend |

```bash
# ranked queue, filtered
curl 'http://127.0.0.1:8000/candidates?change_type=construction&min_confidence=0.9&sort=queue_score&limit=2'
# -> {"total":30, ... "candidates":[{"rank":7,"candidate_id":"2019_2024_007069",
#     "change_type":"construction","confidence":0.9395,"queue_score":0.8869,
#     "persistence":"persistent","earliest_supported":["2019-03-30","2021-03-04"],
#     "geometry":{"type":"Polygon","coordinates":[[...5 lon/lat positions...]]},
#     "decision":"undecided","sar":{"available":true,"vv_median_db":5.15,
#     "verdict":"VV anomaly +4.5 dB vs expected rise -> strong agreement"}}, ...]}

# full evidence + provenance for one candidate
curl 'http://127.0.0.1:8000/candidates/2019_2024_011912'
# -> confidence_breakdown[8]  (model / persistence / spectral / quality /
#      registration / radiometric / SAR / => confidence 0.99)
#    suppression.trace[5]     (quality, registration, radiometric, phenology,
#      morphology — each with verdict, weight, detail, values)
#    temporal_trajectory.intervals[3] + earliest_supported_change.caveat
#    provenance.observations[before,after] -> scene{source_url(COG), license,
#      crs, checksums}, collection{sensor:"MSI", bands, native_gsd_m},
#      representative_tile{tile_id, checksums}
#    provenance.model{weights_sha256:452ac0...}, provenance.code{git_commit:ecf0943}
```

### 2. Audit schema + demonstrated confirm/reject, append-only

`analyst_decisions` (via `MetadataRepository`, **the only sqlite3 is the seam**):

```sql
CREATE TABLE analyst_decisions (
    decision_id            TEXT PRIMARY KEY,
    candidate_id           TEXT NOT NULL,
    decision               TEXT NOT NULL CHECK (decision IN ('confirm','reject')),
    analyst_note           TEXT NOT NULL DEFAULT '',
    analyst                TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL,
    model_version          TEXT NOT NULL DEFAULT '',
    weights_sha256         TEXT NOT NULL DEFAULT '',
    git_commit             TEXT NOT NULL DEFAULT '',
    pipeline_version       TEXT NOT NULL DEFAULT '',
    confidence_at_decision REAL,
    evidence_snapshot_json TEXT NOT NULL DEFAULT '{}'
);
-- append-only ENFORCED AT THE STORAGE LAYER:
CREATE TRIGGER trg_analyst_decisions_no_update BEFORE UPDATE ON analyst_decisions
  BEGIN SELECT RAISE(ABORT,'analyst_decisions is append-only: UPDATE is not permitted'); END;
CREATE TRIGGER trg_analyst_decisions_no_delete BEFORE DELETE ON analyst_decisions
  BEGIN SELECT RAISE(ABORT,'analyst_decisions is append-only: DELETE is not permitted'); END;
```

`python scripts/demo_audit_trail.py` (throwaway DB):

```
WRITE #1: confirm 2019_2024_011912   -> dec_8135d2c8...
WRITE #2: reject  2019_2024_008471   -> dec_caf6e554...
WRITE #3: re-decide 2019_2024_008471 -> confirm (APPEND, not overwrite) -> dec_09917a1f...

history for 2019_2024_008471:  v1 reject -> v2 confirm   (2 rows kept; current verdict = 'confirm')

UPDATE analyst_decisions ...  -> rejected: analyst_decisions is append-only: UPDATE is not permitted
DELETE FROM analyst_decisions -> rejected: analyst_decisions is append-only: DELETE is not permitted
row count unchanged after the blocked UPDATE/DELETE: 3
```

Live via the UI (`GET /audit`):

```
count 2 | append_only true
  06:48:21  dec_f3f8619127aa...  2019_2024_011912  confirm  conf@0.9873  weights=452ac062e3b3 git=ecf0943 pv=0.1.0
  06:48:24  dec_5cd780052034...  2019_2024_011926   reject  conf@0.7422  weights=452ac062e3b3 git=ecf0943 pv=0.1.0
```

Each row snapshots the confidence **and** the full evidence + provenance blob
*at the time of the decision* (`GET /audit/{decision_id}` returns it).

### 3. Screenshots of all four views

`data/change_model/ui_screens/` (captured with Playwright driving system Chrome,
every off-origin request aborted):

- `01_review_queue.png` — the 1104 ranked candidates, footprints on the canvas map
- `02_review_queue_filtered.png` — construction, confidence ≥ 0.9 (30 match)
- `03_candidate_detail.png` — BEFORE│AFTER│OVERLAY, trajectory, evidence + suppression trace, provenance, decision
- `04_candidate_detail_2021.png` — the date selector on the BEFORE panel (2021)
- `05_candidate_detail_after_decision.png` — a confirm written, audit history shown
- `06_search.png` — natural-language search, result cards + map
- `07_discovery.png` — KNN neighbours + the HDBSCAN cluster map

### 4. Sample exported GeoJSON feature (full provenance)

```json
{
 "type": "Feature",
 "geometry": {"type": "Polygon", "coordinates": [[[82.047798,26.782729],[82.064193,26.782606],
   [82.064131,26.776016],[82.047738,26.776139],[82.047798,26.782729]]]},
 "properties": {
  "candidate_id": "2019_2024_007069", "change_type": "construction", "confidence": 0.9395,
  "significance": 0.797, "queue_score": 0.8869, "persistence": "persistent",
  "earliest_supported_change": ["2019-03-30", "2021-03-04"],
  "earliest_supported_caveat": "cannot claim a change date earlier than the earliest usable observation (2019-03-30); ...",
  "area_m2": 535600.0, "centroid_lonlat": [82.056388, 26.779172],
  "evidence_summary": "model: mean p 0.90 ... | persistence: persistent (0.95) | spectral: ... 'construction' (0.88) | quality: ... (0.94) | registration: 0.15 px (1.00) | radiometric: ... 0.50 | SAR corroboration (VV anomaly +4.5 dB vs expected rise -> strong agreement): x1.10 | => confidence 0.94",
  "spectral_anomalies": {"ndvi_anomaly": -0.2835, "ndbi_anomaly": 0.2125, "ndwi_anomaly": 0.2279,
    "framing": "delta minus scene-wide seasonal delta (drought->green makes raw deltas useless)"},
  "sar_corroboration": {"vv_median_db": 5.15, "verdict": "VV anomaly +4.5 dB vs expected rise -> strong agreement", "confidence_factor": 1.1},
  "suppression": {"suppressed": false, "suppressed_by": null, "combined_downweight": 1.0},
  "source_scene_ids": ["S2B_44RPQ_20190330_1_L2A", "S2A_44RPQ_20240308_0_L2A"],
  "source_observation_ids": ["S2B_44RPQ_20190330_1_L2A_scaled", "S2A_44RPQ_20240308_0_L2A_scaled"],
  "acquisition_dates": ["2019-03-30", "2024-03-08"],
  "sensor": "MSI / Sentinel-2 (optical); Sentinel-1 C-SAR corroboration where available",
  "model_version": "FCSiamDiff@0.8",
  "weights_sha256": "452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab",
  "git_commit": "ecf0943", "pipeline_version": "0.1.0",
  "processing": {"model": "FCSiamDiff", "model_threshold": 0.8, "weights_sha256": "452ac0...", "git_commit": "ecf0943", "pipeline_version": "0.1.0"},
  "analyst_decision": null
 }
}
```

`python scripts/validate_export.py` — structural + provenance-completeness
checks, then **GDAL/OGR (the standard GIS reader) opens it**: GeoJSON driver,
layer SRS = **WGS 84 / EPSG:4326**, geometry **Polygon**, 24 attribute fields,
1104 features, `feature[0]` geometry round-tripped to WKT and `IsValid()`.
`ALL CHECKS PASSED` for the full 1104-feature and the filtered 30-feature export.

### 5. Offline run — every view working, network disabled

`python scripts/verify_offline_perf.py` hard-disables the network **in process**
(`socket.connect` / `connect_ex` / `getaddrinfo` raise for any non-loopback
address, proven with an `8.8.8.8:53` probe) *before* building the app.

```
network block active: outbound connect to 8.8.8.8:53 rejected in process
app booted offline in 2.9s (RemoteCLIP + FAISS + catalog + change report all from local files)

functional checks (every view actually works offline):
  [PASS] health / stats
  [PASS] search returns hits
  [PASS] queue lists candidates + footprints
  [PASS] detail has evidence + suppression trace + provenance
  [PASS] imagery renders PNG
  [PASS] decision written to append-only audit
  [PASS] export carries provenance
  [PASS] discovery KNN + clusters
  [PASS] frontend bundle served
no process made (or attempted) a non-loopback network call
```

The Playwright screenshot run is a second, browser-level proof: it aborts every
request whose origin isn't `127.0.0.1:<port>` and all four views still render.

### 6. Per-view interactive latency (network disabled, p95)

| view / path | p95 | budget |
|---|---:|---|
| SEARCH — `/search/text` | 16 ms | < 1 s ✓ |
| SEARCH — `/search/image` | 17 ms | ✓ |
| QUEUE — `/candidates` (400 rows) | 48 ms | ✓ |
| QUEUE — `/candidates` (filtered) | 7 ms | ✓ |
| DETAIL — `/candidates/{id}` | 155 ms | ✓ |
| DETAIL — `/candidates/{id}/imagery` (cold) | 117 ms | ✓ |
| DETAIL — `/candidates/{id}/imagery` (warm, LRU) | 2 ms | ✓ |
| DECISION — `POST /candidates/{id}/decision` | 169 ms | ✓ |
| `/audit` | 18 ms | ✓ |
| `/export` (filtered ~30) | 34 ms | ✓ |
| `/export` (all 1104 — bulk, not interactive) | 313 ms | ✓ |
| DISCOVERY — `/discovery/clusters` | 4 ms | ✓ |
| DISCOVERY — `/candidates/{id}/similar` | 148 ms | ✓ |

Every interactive path is **well under the 1 s budget**; nothing is over. The
one heavier operation, exporting all 1104 candidates at once, is a bulk action
(not on an interactive path) and still finishes in ~0.3 s.

### 7. Tests + seams

`python -m pytest` — **201 passed, 0 skipped** (186 prior + 15 Phase 6:
`tests/test_phase6.py` covers the append-only audit seam incl. trigger
enforcement, every endpoint, the OGR export load, and the light/full audit
views). Seams grep-clean: the only `sqlite3` is `catalog/sqlite_repository.py`
+ `catalog/migrate.py`; the only `faiss` is `vectorindex/faiss_flat.py`.

---

## Run it

```bash
# 1. produce the change report + full-detail sidecar (offline, cached rasters)
python -m geoseek.change.analyze

# 2. serve the API + UI
uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
#    open http://127.0.0.1:8000/app/

# verification
python scripts/demo_audit_trail.py       # audit schema + append-only demo
python scripts/validate_export.py         # GeoJSON built + validated in GDAL/OGR
python scripts/verify_offline_perf.py     # network disabled + per-view latency
node   scripts/shoot_analyst_ui.mjs       # screenshots of the four views
```

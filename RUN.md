# RUN.md — geoseek quickstart

Everything below assumes staging has already been done once (model weights +
Ayodhya scenes are on disk under `data/`) and is fully **offline**. Commands are
PowerShell, run from the repo root: `C:\Users\Prash\Downloads\SIH 2026\geoseek`.

---

## 0. Or: run it in Docker (skip the conda setup)

If you'd rather not set up conda/GDAL/CUDA by hand, `docker compose up --build`
builds a CPU-only image (torch, faiss-cpu, rasterio, opencv, the `detect`
extra — everything sections 1-9 below need) and serves the app at
`http://localhost:8000/app/`. `data/` is bind-mounted, not baked into the
image, so stage it once first exactly as section 1's prerequisites describe -
an empty `data/` still starts the container, the API just has nothing
indexed yet. A host GPU works too: it needs the NVIDIA Container Toolkit
installed, then uncommenting the `deploy.resources.reservations` block in
`docker-compose.yml` (the app's own device selection, `select_device()` in
`src/geoseek/config.py`, already picks up `cuda` automatically once it's
visible in the container - no image change needed beyond that block).

```bash
docker compose up --build
```

The rest of this file (sections 1-9) is the bare-metal/conda path; skip to
section 3 once the container is up, or read on if you want the process
running directly on your machine instead.

---

## 1. Activate the conda environment

Env name: **`geoseek`** (lives at `C:\AnacondaPython\anaconda3\envs\geoseek`,
Python 3.11, torch 2.2.2 + CUDA, rasterio, faiss-cpu).

```powershell
conda activate geoseek
cd "C:\Users\Prash\Downloads\SIH 2026\geoseek"
```

GDAL/PROJ and console-encoding vars this machine needs (otherwise rasterio warns
`Cannot find gdalvrt.xsd` and non-ASCII prints crash the cp1252 console):

```powershell
$env:GDAL_DATA   = "C:\AnacondaPython\anaconda3\envs\geoseek\Library\share\gdal"
$env:PROJ_LIB    = "C:\AnacondaPython\anaconda3\envs\geoseek\Library\share\proj"
$env:PYTHONIOENCODING = "utf-8"
```

Verify the env:

```powershell
python -c "import torch, rasterio, faiss; print(torch.__version__, torch.cuda.is_available())"
```

Expected output:

```
2.2.2 True
```

---

## 2. Start the FastAPI server

One process serves both the search API and the analyst UI.

```powershell
uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
```

Expected output (last lines):

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Open the **analyst console** in a browser:

```
http://127.0.0.1:8000/app/
```

The console is a React build that is **committed to the repository** (`src/geoseek/analyst/web_react/`), so nothing here needs Node
or npm. `/react/` (its former address) redirects to `/app/`. If that directory is ever missing, `/app/` answers **503** with a page that
names it and the command that rebuilds it (`cd frontend-react && npm ci && npm run build`, which needs Node); the API keeps working
meanwhile. A Docker image ships the committed build as-is (the Dockerfile has no Node step).

Other useful URLs: API docs `http://127.0.0.1:8000/docs`, service index
`http://127.0.0.1:8000/`.

Leave this terminal running; open a second activated terminal for the curl
examples below.

---

## 3. Verify it's running

```powershell
curl.exe -s http://127.0.0.1:8000/health
```

Expected output (JSON; vector count is ~101,911 with the full Phase 7b index,
`3267` if only the Ayodhya baseline is ingested):

```json
{"status":"ok","vectors":101911,"analyst":"ready","report_loaded":true,"candidates":1104,"probability_raster_present":true,"audit_decisions":0}
```

Key fields: `status` = `ok`, `analyst` = `ready`, `report_loaded` = `true`.
If `analyst` shows an error string, the search API still works but the change
report (`data/change_model/ayodhya_change_ranked_detail.json`) wasn't found —
run step 5 first.

---

## 4. Example text search

### via curl

```powershell
curl.exe -s "http://127.0.0.1:8000/search/text?q=a+river+with+sandbars&k=5"
```

Expected output — a JSON array of 5 hits, highest score first, each with a tile
id, cosine score (~0.2–0.35 for a good match), lon/lat centroid and a GeoJSON
footprint:

```json
{"query":"a river with sandbars","k":5,"results":[
  {"tile_id":"S2B_44RPQ_20190330_1_L2A_scaled_r12_c34","score":0.2971,"centroid":[82.19,26.79],"cloud_fraction":0.0,"acquired":"2019-03-30", ...},
  ...
]}
```

### via the browser

Open:

```
http://127.0.0.1:8000/search/text?q=a river with sandbars&k=5
```

or use the **Search** view in the analyst console at `http://127.0.0.1:8000/app/#/search` —
type the query, results render on the map with thumbnails.

---

## 5. Run the change pipeline on Ayodhya

Analyst-grade 3-date change pipeline (2019-03-30 / 2021-03-04 / 2024-03-08).
Offline. Probability rasters are cached under `data/change_model/prob_*.tif`, so
a re-run without `--refresh` is fast.

```powershell
python -m geoseek.change.analyze
```

Add `--refresh` to recompute the FC-Siam-diff probability rasters (~35 s/pair),
`--no-panels` to skip the PNG panels.

Expected output (tail): per-pair flagged-area percentages, a suppression
breakdown (dominated by morphology + phenology), the top-ranked changes
(water/moisture gain and construction), then:

```
wrote data/change_model/ayodhya_change_report.json
wrote data/change_model/ayodhya_change_ranked_detail.json   (1104 candidates)
wrote panels -> data/change_model/panels/
manifest section 'ayodhya_change_pipeline' updated
```

The `ayodhya_change_ranked_detail.json` file is what the analyst UI / `/candidates`
endpoint reads — restart the server (step 2) after a fresh run so it reloads.

---

## 6. Ingest a new scene incrementally

FAISS + SQLite appends only — existing vectors/rows are never rebuilt.

### A staged Sentinel-2 scene directory (same MGRS tile, e.g. Ayodhya)

```powershell
python -m geoseek.ingest.pipeline ingest "data\datasets\S2A_44RPQ_20240308_0_L2A_scaled"
```

Expected output:

```
read S2A_44RPQ_20240308_0_L2A_scaled  -> 1849 tiles (nodata-only skipped)
quality (SCL) ... embed (RemoteCLIP, batch=64, CUDA) ...
append: FAISS 100062 -> 101911   SQLite tiles +1849
mean per-tile embed latency: ~4.8 ms
manifest 'ingest_runs' appended
```

### A brand-new diverse AOI (different MGRS tile / region) — Phase 7b path

Stages one date for a registered region, then ingests it with correct
`aoi_name` / provenance:

```powershell
python scripts\stage_diverse_aois.py --region dehradun
python scripts\ingest_diverse_scene.py "data\datasets\<new_scene_dir>"
```

Verify the append landed:

```powershell
python -c "from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex; from geoseek.config import settings; print(FaissFlatIPIndex(settings().index_dir).count())"
```

Expected: the new total vector count (previous count + tiles added).

---

## 7. Run the full test suite

```powershell
python -m pytest -q
```

Expected output (≈211 tests; a few skipped by design — the frozen-baseline
parity checks that legitimately shift once new regions are indexed):

```
............................................................ [ 30%]
............................................................ [ 60%]
............................................................ [ 90%]
.....................                                         [100%]
208 passed, 3 skipped in <time>
```

Run one module, e.g. the analyst interface:

```powershell
python -m pytest -q tests\test_phase6.py
```

---

## 8. Offline demo (network disabled)

Hard-disables the network **in-process** (every non-loopback
`socket.connect` / `connect_ex` / `getaddrinfo` raises) *before* the app is
built, then exercises search, review queue, candidate detail, imagery, decision,
audit, export and discovery — and measures per-view p95 latency. Audit writes go
to a throwaway copy of the catalog, so `data/index` is untouched.

```powershell
python scripts\verify_offline_perf.py
```

Expected output (an example from one run on the development laptop, trimmed; latencies vary by machine and this is a pass/fail
check, not a benchmark, so the power source was not recorded):

```
app booted offline in 6.3s (RemoteCLIP + FAISS + catalog + change report all from local files)
COLD start (first interaction; encoders + thumbnail path pre-warmed at startup, thumbnail LRU still cold):
  GET /search/text  (1st call)                   26.6 ms
...
functional checks (every view actually works offline):
  [PASS] health / stats
  [PASS] overview summary (counters + featured, offline)
  [PASS] search returns hits
  [PASS] queue lists candidates + footprints
  [PASS] detail has evidence + suppression trace + provenance
  [PASS] imagery renders PNG
  [PASS] decision written to append-only audit
  [PASS] export carries provenance
  [PASS] discovery KNN + clusters
  [PASS] console page served (root element + module script + entry bundle)

LATENCY SUMMARY (p95):
  ok   GET /presentation/summary              38.6 ms
  ok   GET /health                            18.8 ms
  ok   GET /stats                             22.9 ms
  ok   GET /search/text                       28.9 ms
  ok   GET /tile/{id}/thumbnail (warm)         1.4 ms
  ok   POST /search/image                     36.8 ms
  ok   GET /candidates (queue, 400)           60.0 ms
  ok   GET /candidates (filtered)             17.7 ms
  ok   GET /candidates/{id} (detail)          31.5 ms
  ok   GET .../imagery rgb (cold)             55.6 ms
  ok   GET .../imagery rgb (warm/LRU)          0.8 ms
  ok   GET .../imagery overlay (cold)         64.9 ms
  ok   GET .../imagery rgb x2 (cold)         137.8 ms
  ok   POST /candidates/{id}/decision         23.9 ms
  ok   GET /audit                             34.4 ms
  ok   POST /export (filtered ~30)            47.5 ms
  ok   POST /export (all 1104)               212.1 ms
  ok   GET /discovery/clusters                86.6 ms
  ok   GET /candidates/{id}/similar          381.2 ms
  ok   GET /discovery/similar (lon,lat)      382.6 ms
no process made (or attempted) a non-loopback network call
functional checks: ALL PASS
every interactive path is well under the 1 s budget
```

The last functional check is the **console page**: `GET /app/` must return 200 with the root mount element (`<div id="root"></div>`)
and the module `<script>` of the console's own entry bundle (`/app/assets/index-….js`), and that bundle must itself be served as
JavaScript. A different page at `/app/` would fail all three.

To demo the **console** fully offline: disable your network adapter (or pull the
cable / turn off Wi-Fi), then run step 2 and open `http://127.0.0.1:8000/app/` —
the map, thumbnails, imagery and change queue all work with zero outbound
requests (no CDN, no web fonts, no external map tiles: the base map is rendered by
this server from imagery already in the archive). A browser-level check of every
route is `node frontend-react/tools/verify-offline.mjs --base http://127.0.0.1:8000/app/`.

---

## 9. Object detector (Phase 8F-2)

Oriented detection of vehicles / ships / aircraft / tanks / harbors / bridges in
sub-metre RGB imagery (YOLO26s-OBB fine-tuned on DOTA v1.5). It needs the optional
extra (`ultralytics` is **AGPL-3.0**, see `docs/PHASE8F2.md`):

```powershell
pip install -e ".[detect]" --no-deps      # only ultralytics; never let pip touch the CUDA torch
$env:YOLO_OFFLINE = "1"                    # ultralytics must not phone home
```

End to end (each step is resumable and prints what it measured):

```powershell
python scripts\prepare_detect_data.py --plan-only     # measure first: windows, negatives, monitor split, RFS table
python scripts\prepare_detect_data.py                 # convert DOTA -> 1024 px YOLO-OBB chips (~3 min, 12 workers)
python scripts\render_detect_sanity_chips.py          # pre-flight: GT chips, loader round-trip, augmented batch
python scripts\train_detector.py --print-config       # the full planned training config, runs nothing
python scripts\train_detector.py                      # ~3.1 h of compute on an RTX 4060 laptop (measured); re-run the same command to resume
python scripts\finalize_detector.py                   # freeze best.pt, model card, manifest, loss/mAP curves
python scripts\eval_detector.py                       # honest evaluation on the held-out val (~45 min, GPU); add --stages checkpoints for the per-checkpoint val curve
python scripts\finalize_detector.py                   # again: the model card now carries the operating point + evaluation summary
python scripts\eval_size_operating_point.py           # vehicle recall / precision by pixel size at the operating confidence (no GPU)
python scripts\detect_maxar.py                        # run on the staged Maxar tiles (GeoJSON + samples + DerivedProduct)
python scripts\eval_domain_shift_proxy.py             # controlled blur / upscaling on the monitor split (one component of the DOTA -> Maxar gap)
python scripts\audit_maxar_crops.py --observation vannuys --n-tiles 5 --crops-per-tile 2 --min-dets 0   # seeded crops for a MANUAL tally
python scripts\maxar_scale_probe.py --observations vannuys --scales 1.0,1.5,2.0                         # exploratory: test-time upscaling on Maxar
python scripts\make_phase8f2_tables.py > data\detect_eval\phase8f2_tables.md   # every table of docs/PHASE8F2.md from the JSON artifacts
```

Use it from code (RGB `HxWx3 uint8` in, oriented detections out):

```python
from geoseek.models import YoloObbDetectionModel, TileGeoRef
model = YoloObbDetectionModel("data/models/detector/geoseek_obb_v15_yolo26s.pt")   # threshold from the model card
dets = model.detect(rgb_tile, classes=["small-vehicle"], geo=TileGeoRef(transform=(a, b, c, d, e, f), crs="EPSG:32611"))
# opt-in test-time upscaling (default 1.0): resamples the tile, reports boxes in ORIGINAL pixels. Exploratory - see docs/PHASE8F2.md sec. 7.5
model2 = YoloObbDetectionModel("data/models/detector/geoseek_obb_v15_yolo26s.pt", upscale=2.0)
```

What to expect (measured, `docs/PHASE8F2.md`): on the held-out DOTA val the 8-class macro AP50 is 0.817; on the staged Maxar tiles the detector
finds aircraft and large objects but only a few percent of visible cars - do not use it as a car counter there.

Machine notes that cost real time to find (details in `docs/PHASE8F2.md`): the trainer sets a hard torch
VRAM cap because Windows' driver otherwise spills CUDA memory into RAM; do not run other torch processes
while it trains (RAM is ~95 % used); never start two supervisors on one run directory (there is a lock).

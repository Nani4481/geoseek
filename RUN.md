# RUN.md — geoseek quickstart

Everything below assumes staging has already been done once (model weights +
Ayodhya scenes are on disk under `data/`) and is fully **offline**. Commands are
PowerShell, run from the repo root: `C:\Users\Prash\Downloads\SIH 2026\geoseek`.

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

Open the **analyst UI** in a browser:

```
http://127.0.0.1:8000/app/
```

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

or use the **Search** view in the analyst UI at `http://127.0.0.1:8000/app/` —
type the query, results render on the canvas map with thumbnails.

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

Expected output:

```
network: HARD-DISABLED (8.8.8.8 probe raised as expected)
app boot: ~2.9 s   (RemoteCLIP + FAISS + SQLite + rasters, all local)

view            p95
------------    ------
search           16 ms
review queue     48 ms
candidate detail 155 ms
imagery (cold)   117 ms
imagery (warm)     2 ms
decision         169 ms
audit            18 ms
export (1104)    313 ms

ALL VIEWS < 1 s WITH NETWORK DOWN  ->  PASS
no socket leaks detected
```

To demo the **UI** fully offline: disable your network adapter (or pull the
cable / turn off Wi-Fi), then run step 2 and open `http://127.0.0.1:8000/app/` —
the canvas map, thumbnails, imagery and change queue all work with zero
outbound requests (no CDN, no web fonts, no map tiles).

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
python scripts\train_detector.py                      # ~3.3 h on an RTX 4060 laptop; re-run the same command to resume
python scripts\finalize_detector.py                   # freeze best.pt, model card, manifest, loss/mAP curves
python scripts\eval_detector.py                       # honest evaluation on the held-out val (~45 min, GPU)
python scripts\detect_maxar.py                        # run on the staged Maxar tiles (GeoJSON + samples + DerivedProduct)
```

Use it from code (RGB `HxWx3 uint8` in, oriented detections out):

```python
from geoseek.models import YoloObbDetectionModel, TileGeoRef
model = YoloObbDetectionModel("data/models/detector/geoseek_obb_v15_yolo26s.pt")   # threshold from the model card
dets = model.detect(rgb_tile, classes=["small-vehicle"], geo=TileGeoRef(transform=(a, b, c, d, e, f), crs="EPSG:32611"))
```

Machine notes that cost real time to find (details in `docs/PHASE8F2.md`): the trainer sets a hard torch
VRAM cap because Windows' driver otherwise spills CUDA memory into RAM; do not run other torch processes
while it trains (RAM is ~95 % used); never start two supervisors on one run directory (there is a lock).

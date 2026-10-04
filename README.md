# geoseek

Search satellite imagery in plain English, detect what changed, fully offline.

Built for Smart India Hackathon 2026, Problem Statement 26227: analysts need
a way to search large volumes of satellite imagery by describing what they're
looking for, and to be alerted when something in an area of interest changes
between observations — without depending on a live internet connection or a
cloud vendor. geoseek indexes Sentinel-1/2 imagery into a searchable,
offline-capable system with a working analyst interface on top.

## What it does

- **Semantic search** — describe a scene in plain language ("bare dry open
  ground near a river bend") and get back the matching tiles, ranked by
  similarity, using a vision-language model pretrained on satellite imagery.
- **Change detection** — compare two dates of the same area and highlight
  where land cover actually changed, using a model trained specifically for
  this task.
- **False-alarm suppression** — filters out changes caused by seasonal
  vegetation cycles, cloud shadows, or radiometric drift between scenes
  rather than real change on the ground.
- **Discovery** — clusters the whole tile corpus by visual similarity so an
  analyst can browse what kinds of terrain and change exist without a
  starting query.
- **Analyst workflow with audit trail** — a web interface for reviewing
  candidates, marking decisions, and exporting findings, with every decision
  logged to an append-only record.
- **Incremental offline ingestion** — new imagery can be added to the index
  at any time; every module after the one-time staging step runs with no
  network access.

## Key numbers

| | |
|---|---|
| Tiles catalogued | 107,168 total (256×256 px each) across both collections — `data/index/tiles.sqlite` |
| Tiles searchable | 104,089 Sentinel-2 tiles embedded in the FAISS search index (the other 3,079 are Sentinel-1 SAR, catalogued but not embedded for semantic search) — same database, joined against `scenes.collection_id` |
| Regions covered | 8 diverse regions across India (Himalayan foothills, Thar desert, Sundarbans, Delhi NCR, Kanha, Kerala backwaters, Kutch, Deccan) + the primary Ayodhya AOI |
| Observation dates | 2019-03-29 to 2026-03-08 (77 Sentinel-1/2 scenes; Ayodhya alone has 5 dates: 2019, 2021, 2024, 2025, 2026) |
| Search latency (warm) | ~17 ms median / 26 ms p99 per query on the full corpus |
| Change detection | F1 56.0% on the held-out OSCD test split (vs. 52.8% published for the same architecture) |
| Tests | 351 passed, 3 skipped (`pytest`, `tests/`; the 275 tests that predate Phase 8F-2 are unchanged, +76 for the detector) |

## Quick start

```powershell
conda activate geoseek
cd path\to\geoseek
$env:GDAL_DATA = "$env:CONDA_PREFIX\Library\share\gdal"
$env:PROJ_LIB  = "$env:CONDA_PREFIX\Library\share\proj"

uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
```

Linux/WSL: `export GDAL_DATA=$CONDA_PREFIX/share/gdal PROJ_LIB=$CONDA_PREFIX/share/proj`.

Then open `http://127.0.0.1:8000/app/` for the analyst console (the root URL returns a JSON service index). This assumes imagery
and model weights are already staged (see **Data and models** below); for the
full setup path, environment creation, and every other command, see
[RUN.md](RUN.md).

## How it works

```
   imagery (Sentinel-1/2)
        │
        ▼
   ingest → tile (256x256) → embed (RemoteCLIP)  ──┐
        │                                          ▼
        │                                    vector index (FAISS + R*Tree)
        │                                          │
        └─► co-register + normalize                ▼
              same-location date pairs      text query → embed → nearest tiles
                    │                              (semantic SEARCH)
                    ▼
          FC-Siam-diff change model
                    │
                    ▼
      suppress seasonal/radiometric noise
                    │
                    ▼
        ranked candidates → analyst UI
        (review, decide, export, audit trail)
```

Two different models do two different jobs. **RemoteCLIP** is a pretrained
vision-language model (never fine-tuned here) that turns a tile or a text
query into a vector — nearby vectors mean visually/semantically similar
scenes, which is what powers search. **FC-Siam-diff** is a small
change-detection network trained from scratch on the OSCD dataset to compare
two co-registered images of the same location and predict which pixels
changed — this is what powers change detection. The two are independent and
serve different queries: one finds *what looks like X*, the other finds
*what's different here now*.

## Tech stack

| Technology | What it does | Why chosen |
|---|---|---|
| RemoteCLIP (OpenCLIP ViT-B/32) | Embeds imagery and text into a shared vector space | Pretrained specifically on remote-sensing imagery; no fine-tuning needed for search to work well |
| FC-Siam-diff (PyTorch) | Pixel-level change detection between two dates | Small (1.09M params), fast on a laptop GPU, standard architecture with a public benchmark to compare against |
| FAISS + R*Tree | Nearest-neighbor and spatial (bbox/point) indexing | Sub-20ms warm search over 100k+ vectors; R*Tree gives ~837x speedup on bounding-box queries over a linear scan |
| rasterio / GDAL / pyproj | Reading, reprojecting, and co-registering satellite imagery | Standard geospatial stack with correct CRS/warp handling |
| FastAPI + SQLite | Analyst API and audit-trail storage | Lightweight, offline-capable, no external database server needed |
| React 19 + TypeScript, built with Vite | Analyst console (`/app/`, source in `frontend-react/`) | Runs fully offline with no external script dependency. There **is** a build step, but its output is **committed** (`src/geoseek/analyst/web_react/`, every file SHA-256-pinned), so running the app needs Python only: no Node, no npm. Changing the frontend does need Node (`cd frontend-react && npm ci && npm run build`); the Docker image has no Node stage and ships the committed bundle, so rebuild and commit it together with the source (details in `docs/FRONTEND_REACT.md`) |
| Leaflet | Interactive maps in the console: detection polygons, real evidence imagery draped at its true bounds, restricted-zone rectangles, and one same-origin base-map tile layer (`/ui/basemap/{z}/{x}/{y}`) that this server renders from imagery already in the archive | Self-hostable (`scripts/stage_leaflet.py`) and bundled into the console build; no external tile service, so nothing reaches out |
| Hand-rolled SVG/CSS charts | Evidence-gate, count and persistence charts in the console | No chart library is shipped (Chart.js belonged to the retired interface and was removed), so there is nothing extra to audit; charts read the same real numbers the UI states in text |
| Docker / docker-compose | Optional containerized deployment | `Dockerfile` builds a CPU-only image with every dependency including the `detect` extra; `data/` stays a bind mount since it's machine-specific and gigabytes in size |

## Data and models

- **Imagery**: Sentinel-2 L2A (10m, optical) and Sentinel-1 GRD (SAR), fetched
  once via Earth Search STAC (Element 84, backed by the public AWS Open Data
  Sentinel-2 bucket) under the Copernicus Sentinel Data license (free and
  open, EU Copernicus Data Policy).
- **Sub-metre imagery** (object-detection track, Phase 8 Step A): Maxar Open
  Data Program ARD tiles (`maxar-opendata` collection), 0.3–0.6 m visual
  (pansharpened RGB), staged for the North India Floods (Sikkim glacial lake
  outburst, Oct 2023) event — see `data/provenance_manifest.json` under
  `maxar_opendata` for the full per-tile record.
  **Maxar Open Data Program was accessed on 2026-09-13 from
  https://registry.opendata.aws/maxar-open-data.** Licensed
  CC BY-NC 4.0 (non-commercial) — satellite imagery courtesy of Maxar
  Technologies.
- **Search model**: RemoteCLIP ViT-B-32 weights from Hugging Face
  (`chendelong/RemoteCLIP`), Apache-2.0 licensed, used as published with no
  fine-tuning.
- **Change model**: FC-Siam-diff, trained from scratch here on OSCD
  (Onera Satellite Change Detection dataset). The architecture and training
  code are Apache-2.0; the trained weights are a derivative of OSCD, which is
  CC-BY-NC-SA-4.0 (non-commercial, share-alike) — the weights inherit that
  restriction.
- **Object-detection training data** (Phase 8F-1/8F-2): DOTA v1.5 oriented-box
  annotations on the DOTA-v1.0 images (train 1,411 img / val 458 img,
  academic-use-only license), staged via `geoseek.staging.download_dota` and
  converted into 1024x1024 YOLO-OBB chips (200 px overlap) by
  `scripts/prepare_detect_data.py`. Three splits, disjoint by source image:
  `train`, a class-stratified `monitor` holdout carved out of *train* (the only
  thing per-epoch curves, checkpoint choice and the confidence threshold ever
  see), and the official `val` (touched only by `scripts/eval_detector.py`,
  after training). 8 of the 16 classes are trained (vehicles, ships, aircraft,
  storage tanks, harbors, bridges); the reasoning, the imbalance strategy
  (repeat-factor sampling, hard-negative chips) and the label-source decision
  (v1.5 OBB, because v1.0's vehicle labels are incomplete) are in
  `geoseek.detect.classes`. A staging bug that silently overwrote the v1.5
  oriented labels with axis-aligned ones was found and fixed in this phase —
  see `docs/PHASE8F2.md`.
- **Object-detection model**: YOLO26s-OBB (Ultralytics), fine-tuned from its
  DOTAv1-pretrained weights onto those 8 classes (class-head rows transplanted
  from the pretrained head by name). **Licence: ultralytics and its weights are
  AGPL-3.0** (including the network-use clause) while geoseek is Apache-2.0, so
  it is an *optional* extra (`pip install geoseek[detect]`), imported only inside
  `geoseek.models.yolo_obb` behind the dependency-free `ObjectDetectionModel`
  interface; the fine-tuned weights are also non-commercial (DOTA is
  academic-use-only). Chosen over Oriented R-CNN/MMRotate on measured VRAM and
  a stale dependency lattice — see `geoseek.staging.download_yolo` and
  `docs/PHASE8F2.md` for the comparison and the numbers. **Measured result**
  (official DOTA val, v1.5 labels, full-image DOTA protocol, evaluated once after
  training): macro AP50 0.817 / AP50:95 0.482 over the 8 classes; ground vehicles
  0.854 / 0.471, ships + aircraft 0.859 / 0.557, infrastructure 0.751 / 0.414.
  On the staged Maxar tiles it finds aircraft and other large objects but only a
  few percent of visible cars (limitations below).
- Every download, checksum, and derivation is recorded in
  `data/provenance_manifest.json`, generated automatically by the staging
  scripts — see that file for the authoritative, machine-checked record.

## Limitations

- **Resolution**: Sentinel-2's 10m pixels cannot resolve individual vehicles
  or small structures — this system finds land-cover-scale change, not
  object-level detail.
- **Domain gap**: OSCD (the change-detection training/eval set) is Sentinel-2
  L1C imagery over mostly urban European scenes; our imagery is L2A over
  semi-rural and mixed Indian terrain. The reported F1 is measured on OSCD's
  own held-out split, not on our target domain, and should be read with that
  gap in mind.
- **Relevance judgements**: search-quality metrics are built from
  constructed/synthetic relevance judgements, not expert-annotated ground
  truth.
- **SAR corroboration**: Sentinel-1 change signal is used as a confidence
  weight on optical change candidates, not as an independently validated
  detector — it hasn't been checked against ground-truth SAR change labels.
- **Vehicle detection on Maxar imagery**: the detector's DOTA-val vehicle
  numbers do not transfer to the staged Maxar tiles. There is no ground truth
  there; a small manual audit found only a few percent of visible cars at the
  operating confidence (aircraft, a bridge and round tanks were found correctly),
  and a controlled-blur experiment explains only ~4 points of the gap. Test-time
  upscaling (`upscale=2`, opt-in) recovers about a third of the cars but multiplies
  `ship` false positives ~9x. See `docs/PHASE8F2.md` sections 7-8.
- **Deployment**: designed and tested as a single-node, single-machine
  system; no distributed or multi-user concurrency story.

## Documentation

- [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) — full evaluation
  methodology and results, including honest negative results.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system design and the
  seams the codebase is built around.
- [RUN.md](RUN.md) — complete setup and run instructions.
- `PHASE*.md` files — detailed engineering notes from each stage of
  development, kept for historical/technical reference.

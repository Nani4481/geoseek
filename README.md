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
| Tests | 275 passed, 3 skipped (`pytest`, `tests/`) |

## Quick start

```powershell
conda activate geoseek
cd path\to\geoseek
$env:GDAL_DATA = "$env:CONDA_PREFIX\Library\share\gdal"
$env:PROJ_LIB  = "$env:CONDA_PREFIX\Library\share\proj"

uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
```

Linux/WSL: `export GDAL_DATA=$CONDA_PREFIX/share/gdal PROJ_LIB=$CONDA_PREFIX/share/proj`.

Then open `http://127.0.0.1:8000` for the analyst UI. This assumes imagery
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
| Vanilla JS (no framework, no CDN) | Analyst web UI | Runs fully offline with zero build step or external script dependency |

## Data and models

- **Imagery**: Sentinel-2 L2A (10m, optical) and Sentinel-1 GRD (SAR), fetched
  once via Earth Search STAC (Element 84, backed by the public AWS Open Data
  Sentinel-2 bucket) under the Copernicus Sentinel Data license (free and
  open, EU Copernicus Data Policy).
- **Search model**: RemoteCLIP ViT-B-32 weights from Hugging Face
  (`chendelong/RemoteCLIP`), Apache-2.0 licensed, used as published with no
  fine-tuning.
- **Change model**: FC-Siam-diff, trained from scratch here on OSCD
  (Onera Satellite Change Detection dataset). The architecture and training
  code are Apache-2.0; the trained weights are a derivative of OSCD, which is
  CC-BY-NC-SA-4.0 (non-commercial, share-alike) — the weights inherit that
  restriction.
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

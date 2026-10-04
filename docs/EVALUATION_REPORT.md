# geoseek — Evaluation Report (PS 2.3)

**Problem statement:** *"A searchable, browsable index over a large satellite-image
archive, with semantic text/image retrieval, temporal change analysis, and a
reproducible evaluation of retrieval quality, change-detection quality, index
build/query performance at scale, and clustering/discovery."*

Every number in this report comes from a **measured run on one machine** (spec in
§4). Nothing is projected except where a row is explicitly marked *estimated* or
*extrapolated*. Each section names the script that produced it and the on-disk
artifact that holds the raw output, so the whole report is reproducible from a
clean checkout after a one-time data staging step (§13).

The build is a **single-node, offline** system. It has been exercised at
**101,911 embedded tiles** across **8 + 1 regions**; the design path past that
(PostGIS, HNSW, distributed workers) is in `docs/ARCHITECTURE.md` §"Future work"
and is **not** claimed here.

---

## Table of contents

1. [Indexed archive — area, scenes, observations, tiles](#1-indexed-archive)
2. [Index build time, incremental ingestion, storage footprint](#2-build-ingestion-storage)
3. [Query latency — every query type, cold and warm](#3-query-latency)
4. [Hardware, and the AC-vs-battery caveat](#4-hardware)
5. [Model provenance](#5-model-provenance)
6. [Dataset provenance](#6-dataset-provenance)
7. [Retrieval metrics — Phase 7a and the corrected at-scale result](#7-retrieval-metrics)
8. [Change-detection metrics — OSCD held-out, vs published baselines](#8-change-detection-metrics)
9. [Ablation study — what each pipeline stage contributes](#9-ablation-study)
10. [Scale benchmark — FlatIP scaling and the HNSW crossover](#10-scale-benchmark)
11. [Clustering / discovery — 12k and 100k, with ARI validation](#11-clustering-discovery)
12. [SAR corroboration — 51% agreement and the physical reason](#12-sar-corroboration)
13. [Reproduction — clean checkout, offline after staging](#13-reproduction)
14. [Limitations](#14-limitations)
15. [Object detector (Phase 8F-2) — a separate track, summarised here](#15-object-detector)
16. [Offline guarantee — what is claimed, and how it is checked](#16-offline-guarantee--what-is-claimed-and-how-it-is-checked)

---

## 1. Indexed archive

*Source: the production catalog `data/index/tiles.sqlite` + the FAISS index
`data/index/tiles.faiss`; provenance manifest sections `oscd`, `third_date`,
`sentinel1`, `diverse_aois`, `ingest_runs`.*

### 1.1 Totals

| quantity | value |
|---|--:|
| collections | **2** (`sentinel-2-l2a`, `sentinel-1-grd`) |
| scenes | **75** |
| observations (scene ∩ AOI at one date) | **75** |
| catalog tiles | **104,990** |
| ├ Sentinel-2 tiles, embedded (have a FAISS vector) | **101,911** |
| └ Sentinel-1 SAR tiles, **not** embedded (`faiss_id IS NULL`) | **3,079** |
| FAISS vectors (RemoteCLIP ViT-B/32, 512-d, `IndexFlatIP`) | **101,911** |
| analyst decisions recorded (append-only audit table) | 10 |
| derived products registered (NDVI/NDWI/NDBI rasters, per-tile index CSV) | 8 |

Tile grid: **256 px** (2.56 km at Sentinel-2's 10 m GSD → **6.55 km² per tile**).
Partial edge tiles are kept; a tile is skipped only when *every* pixel is nodata.

### 1.2 Coverage

Bounding box of every indexed tile (EPSG:4326): **`[69.985, 8.964, 89.993, 30.718]`**
— i.e. from the Rann of Kutch to the Sundarbans delta to the Himalayan foothills.

* **Tile-area sum** (counts multi-date repeats of the same ground):
  101,911 × 6.55 km² ≈ **667,900 km²**.
* **Approximate unique ground footprint**: 14 distinct MGRS tiles, each a
  near-full ~110 × 110 km tile minus a 1.5 km buffer, with some regions spanning
  two adjacent tiles → on the order of **150,000–180,000 km²** of distinct
  terrain (the tiles overlap slightly and several regions were imaged on 5–11
  dates).

### 1.3 Per region

| region | category | MGRS tile(s) | scenes | AOI bbox (EPSG:4326) | seasons | max cloud |
|---|---|---|--:|---|---|--:|
| **Ayodhya** (reference AOI) | Gangetic plain, semi-rural + town | 44RPQ | 3 | `[82.01, 26.35, 82.86, 27.12]` (~82 × 82 km) | pre-monsoon March ×3 (2019 drought / 2021 / 2024 green) | <0.01% |
| Dehradun / Rishikesh | Himalayan foothills | 43RGP, 44RKU | 11 | `[77.08, 29.72, 79.02, 30.70]` | winter, post-monsoon | 0.25% |
| Jaisalmer | Thar desert edge | 42RXQ, 42RYQ, 42RYR | 10 | `[70.01, 26.11, 72.14, 28.00]` | post-monsoon, winter, summer | <0.01% |
| Sundarbans | coastal delta (Bengal) | 45QXE, 45QYE | 9 | `[87.98, 21.60, 89.98, 22.59]` | winter, summer | 0.07% |
| Kanha | Central India forest / plateau | 44QMK | 9 | `[80.04, 21.63, 81.08, 22.59]` | summer, winter | 8.18% |
| Delhi NCR | dense urban | 43RFM, 43RGM | 8 | `[76.03, 27.92, 78.16, 28.91]` | post-monsoon, winter, summer | 0.03% |
| Deccan plateau | dryland cropland | 43QGU | 8 | `[76.88, 16.19, 77.90, 17.16]` | winter, post-monsoon, summer | 0.16% |
| Kerala backwaters | tropical coast / wetland | 43PFL, 43PFM | 7 | `[75.92, 8.97, 76.90, 10.84]` | winter | 10.75% |
| Kutch | salt marsh / Rann | 42QXM | 7 | `[69.99, 23.42, 71.05, 24.40]` | winter | 34.78% |

Ayodhya is the AOI for temporal change analysis (§8, §9, §12) and the retrieval
evaluation (§7); the 8 diverse regions are the scale + generalization corpus
(§10, §11). Full per-scene provenance (per-band SHA256, source URL, acquisition
date, cloud %, local-window nodata fraction) is in the manifest `diverse_aois`
section, one entry per scene.

---

## 2. Build, ingestion, storage

*Source: `scripts/measure_tier.py` → `data/eval_retrieval/scale_report_tier{1,2,3}.json`;
per-scene ledger `data/eval_retrieval/diverse_ingest_ledger.json`.*

### 2.1 Ingestion throughput (pipeline only — tile + quality + embed + FAISS append + SQLite insert; excludes network fetch)

| tier | n vectors at measurement | scenes | tiles added | wall (pipeline) | end-to-end tiles/s | pure embed (GPU forward) tiles/s | peak RSS | peak VRAM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Tier 1 | 11,394 | 4 | 6,278 | 30.6 s | **205.4** | 393.2 | 2,452.8 MB | 739.0 MB |
| Tier 2 | 50,126 | 27 | 37,708 | 185.3 s | **203.5** | 375.9 | 2,468.7 MB | 739.0 MB |
| Tier 3 | 100,887 | 34 | 49,705 | 345.0 s | 144.1 † | 293.4 † | 2,533.5 MB | 739.0 MB |

† **Tier-3 ingestion rows 48–64 were measured on battery power** (see §4). On AC —
Phase 7a, Tier 1, Tier 2, and Tier-3 rows 0–47 — the sustained rate is
**~205–210 tiles/s** with pure-embed ~375–393 tiles/s, and it is **flat across an
11× index growth within the AC segment** (6,965 → 75,843 vectors, 210.5 tiles/s
±6%). Ingestion cost in this architecture is strictly per-tile and **independent
of how many vectors already exist**. Root cause of the battery step-change:
Windows `Kernel-Power` event 105 at 2026-09-06 11:10:29 UTC, exactly between
ledger rows 47 and 48; evidence in
`data/eval_retrieval/throughput_diag/ingestion_throughput_diagnosis.json`.

**Network fetch, not the pipeline, dominates wall-clock time.** A full ~1,850-tile
MGRS-tile scene takes **~7–8 minutes** to stage (streaming HTTP GET of the whole
COG per band, 4 concurrent, then a local crop) versus **seconds** to ingest.
Naive windowed `/vsicurl/` reads over a near-full-tile extent measured ~300 KB/s
(9 min/band) on this connection — a request-pattern pathology, fixed by bulk
fetch (`scripts/stage_diverse_aois.py`).

### 2.2 Incremental ingestion — append-only, byte-identical, no rebuild

Measured once per tier: stage a brand-new ~1,000-tile scene, ingest it, then
sample 25 pre-existing FAISS vectors uniformly at random and compare
`reconstruct(id).tobytes()` before vs after.

| tier | scene | tiles added | ingest time | tiles/s | 25 sampled vectors byte-identical? | max abs diff | rebuild? |
|---|---|--:|--:|--:|:--:|--:|:--:|
| Tier 1 | `S2A_44QMK_20240308_0_L2A` (Kanha, 81 km crop) | 1,024 | 4.85 s | 211.3 | **yes** | 0.0 | no |
| Tier 2 | `S2B_43QGU_20231227_0_L2A` (Deccan, 81 km crop) | 1,056 | 5.66 s | 186.5 | **yes** | 0.0 | no |
| Tier 3 | `S2B_44QMK_20240103_0_L2A` (Kanha, 81 km crop) | 1,024 | 11.62 s † | 88.1 † | **yes** | 0.0 | no |

† battery power (§4). Append speed does not degrade with existing corpus size
(Tier-1 and Tier-2 rows, both on AC, are ~186–211 tiles/s at 11k and 50k
vectors). The seam contract (`VectorIndex` is positional / append-only;
`delete()` raises `NotImplementedError`) is what makes this a hard guarantee
rather than an observation.

### 2.3 Storage footprint (Tier 3, n = 100,887)

| component | size | notes |
|---|--:|---|
| FAISS index (`tiles.faiss`) | **206.62 MB** | exact: `n × 512 × 4 bytes` |
| SQLite catalog (`tiles.sqlite`) | **90.82 MB** | collections/scenes/observations/tiles/derived/analyst_decisions + R\*Tree |
| raw staged imagery (`data/datasets/`) | 32,396.4 MB | Sentinel-2 L2A COG crops (5 bands + SCL) + Sentinel-1 geocoded VV/VH |
| derived products (`data/change_model/`, `data/discovery/`, `data/eval_retrieval/`) | 531.4 MB | NDVI/NDWI/NDBI rasters, change probability rasters, eval JSON/PNG |
| models (`data/models/`) | 1,210.4 MB | RemoteCLIP (605 MB) + vanilla OpenCLIP (605 MB) checkpoints |
| **total `data/`** | **34,449.9 MB** (≈ 33.6 GiB) | |

The **index proper** (FAISS + SQLite) is **297 MB for 101k tiles** — ~2.9 KB/tile
(2.0 KB vector + ~0.9 KB catalog row incl. the R\*Tree entry). Raw imagery is
99% of the footprint and is not needed to serve queries once tiles are embedded
(it is kept only for the change pipeline and thumbnail rendering).

### 2.4 FAISS full-file persist time — the one clear scaling problem

`FaissFlatIPIndex.persist()` rewrites the **entire** index file on every save:

| n vectors | persist time | ratio vs previous |
|--:|--:|--:|
| 11,394 | 18.6 ms | — |
| 50,126 | 117.7 ms | 6.3× (for 4.4× more vectors) |
| 100,887 | **1,357.6 ms** | **11.5× (for 2.0× more vectors)** |

O(n) by construction, but the observed growth is well beyond O(n) — plausibly OS
file-cache effects once the file exceeds what sits comfortably in page cache. At
the current append cadence (one scene per several minutes, fetch-bound) 1.4 s of
persist time is invisible, but a continuously-ingesting production system at this
size would want batched saves or an id-mapped / incremental-write index format.
Reported plainly; not hidden.

---

## 3. Query latency

*Source: Phase 7a `scripts/eval_retrieval_prepare.py` →
`data/eval_retrieval/latency.json` + `report.json` (cold/warm text search);
Phase 7b `scripts/measure_tier.py` → `scale_report_tier{1,2,3}.json` (all query
types, per tier). Same timing harness throughout.*

### 3.1 Cold vs warm — `search_text(query, k=20)`, 3,267-vector index

| | n | median | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| **cold** — fresh Python process, `SearchEngine._prewarm` suppressed, first query timed, 12 processes | 12 | **235.4 ms** | 390.9 ms | 434.5 ms | 434.5 ms |
| **warm** — one pre-warmed process (prewarm + keepwarm as in production), 3 discarded warm-ups, then 6 × 16 queries | 96 | **17.4 ms** | 21.8 ms | 26.3 ms | 72.0 ms |

Production serves the warm path (prewarm at boot, keepwarm every 20 s), so real
user queries are ~17 ms median / 26 ms p99. Even a genuine cold encoder is a few
hundred ms, not the multi-second CUDA-wake cliff that Phase 6 removed.

### 3.2 All query types, by index size (median / p95, warm)

| operation | 3,267 *(Phase 7a)* | 11,394 | 50,126 | 100,887 |
|---|--:|--:|--:|--:|
| **text search** `search_text(k=20)` (incl. text encode) | 17.4 / 21.8 ms | 8.4 / 10.4 ms | 14.4 / 17.1 ms | **27.8 / 38.3 ms** |
| **image search** `search_image(k=20)` (pure FAISS scan) | ~1.0 / 1.5 ms | 1.5 / 3.0 ms | 7.2 / 10.8 ms | **21.7 / 29.8 ms** |
| **point-seeded KNN** (`find_more_like_this(lon,lat)`) | 1.00 / 2.09 ms | 2.0 / 3.0 ms | 7.9 / 11.6 ms | **20.0 / 28.9 ms** |
| **tile-seeded KNN** (`find_more_like_this(tile_id)`) | 0.99 / 1.57 ms | 1.6 / 2.9 ms | 7.3 / 10.9 ms | **18.0 / 24.5 ms** |
| **bbox filter** `query_tiles(bbox=…)` (SQLite R\*Tree, no FAISS) | 0.28 / 0.47 ms | 0.21 / 0.39 ms | 0.23 / 0.32 ms | **0.29 / 0.54 ms** |
| p99 (worst case, text search) | 26.3 ms | — | — | ~45 ms *(from p95 38.3 / max 45.6)* |
| peak RSS / VRAM during a query burst | — | 1,829 / 620 MB | 1,896 / 620 MB | 2,099 / 620 MB |

**Spatial-index fix (Phase 7a).** Before an R\*Tree was added behind the
`MetadataRepository` seam, `query_tiles(bbox=…)` was an O(N) shapely
`.intersects()` scan over every footprint — 230.9 ms median on the 6,346-tile
catalog. After: **0.28 ms** (≈ 837× faster); point-seeded KNN, which resolves its
seed tile through that path, went 237.7 → 1.00 ms (≈ 238×). Result sets are
byte-identical (the exact shapely post-filter still runs on the R\*Tree's
candidate rowids). Bench: `scripts/bench_spatial_index.py` →
`data/eval_retrieval/spatial_index_bench_{before,after}.json`.

Every FAISS-touching operation is **exact** (brute-force inner product,
`IndexFlatIP`) — no recall/latency trade-off, no tuning. The bbox filter never
touches FAISS and stays flat with scale. §10 analyses where the O(n) FAISS scan
would eventually need an approximate index.

---

## 4. Hardware

*Source: manifest `oscd_change_model.training_run`, Phase 7b measurement rows,
`ingestion_throughput_diagnosis.json`.*

| component | spec |
|---|---|
| GPU | **NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB**, driver 592.82, CUDA 12.x (torch 2.2.2) |
| CPU | 16 logical cores |
| RAM | **16.4 GB** total (≈ 4–6 GB typically free during this work) |
| OS | Windows 11 Home Single Language 26200; power scheme "Balanced" |
| Python env | conda `geoseek`, Python 3.11.16, rasterio 1.4.4, faiss-cpu, numpy 1.26.4, hdbscan, scikit-learn |

**AC-vs-battery caveat (affects every ingestion / training row measured on
battery, and possibly the Tier-2→Tier-3 latency slope).**
On this class of laptop, unplugging from AC cuts **both** the RTX 4060 Laptop
GPU's power/clock budget **and** the CPU package power limit, roughly halving
sustained preprocessing + embedding throughput:

| segment | scenes | index vectors | big-scene tiles/s | pure-embed ms/tile |
|---|--:|--:|--:|--:|
| **AC** (Phase 7a, Tier 1, Tier 2, Tier-3 rows 0–47) | 48 | 6,965 → 75,843 | **210.5** (±6%) | **2.48** |
| **battery** (Tier-3 rows 48–64) | 17 | 77,692 → 100,887 | 123.7 | 4.60 |

Capability benchmarks must be run on AC and the power source recorded.
The Tier-3 ingestion cells in §2.1 marked † and the Tier-3 incremental-proof row
are **partly battery-measured and understate AC capability**. All query-latency,
storage, retrieval-accuracy, change-accuracy and ablation numbers in this report
are unaffected (they were measured on AC, or are power-independent).

---

## 5. Model provenance

*Source: manifest `artifacts` list + `oscd_change_model` card. Every checkpoint is
SHA256-pinned; the pipeline refuses to run against a mismatched hash where it
records one.*

### 5.1 Retrieval embedding model — RemoteCLIP ViT-B/32 (production)

| field | value |
|---|---|
| model | **RemoteCLIP ViT-B/32** (CLIP fine-tuned on remote-sensing image–text pairs; Liu et al., 2023) |
| role | the **only** production embedding model — text queries and image tiles both go through it |
| source | Hugging Face `chendelong/RemoteCLIP` → `RemoteCLIP-ViT-B-32.pt` |
| weights SHA256 | `60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4` |
| size | 605,208,421 bytes |
| licence | **Apache-2.0** |
| output | 512-d L2-normalized embedding; cosine similarity = inner product on the unit sphere = `IndexFlatIP` |
| input | true-colour RGB only (B04/B03/B02, fixed `[0, 0.3]` reflectance stretch) — the model never sees NIR/SWIR/SCL |

### 5.2 Retrieval control model — vanilla OpenAI CLIP ViT-B/32 (evaluation only)

| field | value |
|---|---|
| model | OpenAI CLIP ViT-B/32, served via `open_clip` `pretrained='openai'` |
| role | **control** for the retrieval evaluation (§7) — never in the serving path |
| source | `open_clip` pretrained cache (`models--timm--vit_base_patch32_clip_224.openai`) |
| weights SHA256 (`open_clip_model.safetensors`) | `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31` |
| licence | MIT (OpenAI CLIP ViT-B/32 weights) |
| note | loaded with `force_quick_gelu=True` so the control is not handicapped by a QuickGELU/GELU activation mismatch |

### 5.3 Change-detection model — FC-Siam-diff (trained in-repo)

| field | value |
|---|---|
| architecture | **FC-Siam-diff** (Daudt et al., IGARSS 2018): shared Siamese encoder, `|f₁ − f₂|` skip + bottleneck differencing (order-invariant), U-Net transposed-conv decoder, 1×1 change-logit head |
| config | `in_channels=5`, `base_channels=24`, `depth=4`, channels `[24, 48, 96, 192]` |
| parameters | **1,085,113** (1.085 M) |
| bands | B02, B03, B04, B08, B11 — exactly geoseek's production bands; input is reflectance (DN / 10000), per-band standardized (stats fit on the 11 fit regions, `data/change_model/norm_stats.json`) |
| training data | OSCD 11-region *fit* subset (see §6.2); 3 further OSCD-train regions held out for validation + threshold selection; 10 OSCD-test regions never read until evaluation |
| training run | 21 epochs, **2.06 min**, peak VRAM **0.905 GB**, best epoch 6, on the RTX 4060 Laptop GPU |
| loss | `0.5 · weighted-BCE(pos_weight ≤ 10) + 1.0 · soft-Dice`; 40% change-oversampled 96 px patches; dihedral-D4 augmentation; AMP; Adam + cosine |
| weights SHA256 | `452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab` |
| size | 4,384,546 bytes |
| licence | **Apache-2.0** (geoseek code) for the architecture + training script; the **weights are a derivative of OSCD** (CC-BY-NC-SA-4.0 — non-commercial, ShareAlike) |
| deployed operating point | **threshold 0.80** (precision-favouring, chosen on the validation regions, then frozen — §8), embedded in the checkpoint's eval card |
| provenance anchor | trained at git `ecf0943` (`scripts/train_change.py`); source files SHA256-listed in the model card |

RemoteCLIP is used **as-is, zero fine-tuning**, for retrieval and for the
discovery clustering. The FC-Siam-diff weights are the project's only trained
model.

---

## 6. Dataset provenance

*Source: manifest sections `oscd`, `diverse_aois`, `third_date`, `sentinel1`,
`extra_bands`.*

### 6.1 Sentinel-2 L2A (retrieval corpus + change-detection subject imagery)

| field | value |
|---|---|
| product | Sentinel-2 **L2A** (bottom-of-atmosphere surface reflectance), bands B04/B03/B02/B08/B11 + SCL |
| access | Earth Search STAC API `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`, bucket `s3://sentinel-cogs/` (public, no login) |
| licence | **Copernicus Sentinel Data** — free & open (EU Copernicus Data Policy) |
| radiometry | Earth Search baseline ≥ 05.00, `earthsearch:boa_offset_applied=true`, encoding reflectance × 10000 (scale 1e-4, offset −0.1); verified against DN distributions over stable surfaces — no raw-DN dark-floor correction needed |
| scenes | **3 Ayodhya dates** (2019-03-30 `S2B_44RPQ_20190330`, 2021-03-04 `S2A_44RPQ_20210304`, 2024-03-08 `S2A_44RPQ_20240308`) + **69 diverse-region scenes** across 14 MGRS tiles |
| per-scene provenance | **per-band SHA256**, source URL, acquisition date, cloud %, scene + local-window nodata fraction — one entry per scene in manifest `diverse_aois` (69 entries) and `third_date` / `extra_bands` (Ayodhya) |
| Ayodhya AOI | MGRS 44RPQ clipped to `[82.01, 26.35, 82.86, 27.12]` (~82 × 82 km, 8384 × 8384 px @ 10 m); B11 resampled 20 m → 10 m bilinear; B08 + B11 for 2019/2024 staged via `download_datasets.py --extra-bands` |

### 6.2 OSCD — Onera Satellite Change Detection (change-model training + evaluation)

| field | value |
|---|---|
| dataset | OSCD (Daudt, Le Saux, Boulch, Gousseau — *"Urban Change Detection for Multispectral Earth Observation Using CNNs"*, IGARSS 2018, doi:10.1109/IGARSS.2018.8518015) |
| canonical home | IEEE DataPort, doi:10.21227/asqe-7s69 (subscription-gated) |
| download mirror (pinned) | Hugging Face `hkristen/oscd` @ commit `4958d786c1389ede1511d91a6ecf1a75c4074933` — the mirror the torchgeo OSCD loader resolves against |
| archives (SHA256) | Images `940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656` (512,716,711 B) · Train Labels `89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5` · Test Labels `2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d` |
| licence | **CC-BY-NC-SA-4.0** — non-commercial, ShareAlike (this is why the FC-Siam-diff weights inherit a non-commercial clause) |
| imagery | Sentinel-2 **L1C** (top-of-atmosphere), 13 bands in `imgs_{1,2}_rect/` (resampled to 10 m, no CRS); we train on B02/B03/B04/B08/B11 |
| labels | `<region>-cm.tif`, single band, **1 = no-change, 2 = change**; overall change fraction ~3% (some regions <1%) |
| split (from OSCD's own `train.txt` / `test.txt`, cross-checked against the label folders) | **14 train** = 11 *fit* (abudhabi aguasclaras beihai bercy hongkong mumbai nantes paris pisa rennes saclay_e) + **3 validation** (bordeaux cupertino beirut, held out of gradient updates; used for the loss curve + threshold pick). **10 test** (brasilia chongqing dubai lasvegas milano montpellier norcia rio saclay_w valencia) — **never read** until `evaluate()`; `assert` guards make a train/val/test overlap a hard error. |
| domain gap vs Ayodhya | OSCD is **L1C TOA, urban, mostly same-season pairs**; Ayodhya is **L2A BOA, semi-rural, a drought March → a green March**. Out-of-distribution transfer — see §8 and §14. |

### 6.3 Sentinel-1 GRD (SAR corroboration, §12)

| field | value |
|---|---|
| product | **S1A IW GRDH 1SDV** (dual-pol VV+VH), **relative orbit 56, ascending** |
| access | Earth Search STAC `sentinel-1-grd`, bucket `s3://sentinel-s1-l1c/` (public, no login) |
| licence | Copernicus Sentinel Data — free & open (Earth Search tags the STAC collection `proprietary` generically; the data is Copernicus free & open) |
| scenes | 3, nearest to the 3 Sentinel-2 Ayodhya dates: `…20190329…`, `…20210306…`, `…20240302…` |
| geocoding | `gdal.Warp` with the product's ~210 GCPs onto the exact Sentinel-2 10 m grid (rasterio `WarpedVRT(src_gcps=…)` flattens the image — a real gotcha; `gdal.Warp` gives real data at ~93% AOI coverage; the E strip beyond the GCP grid = nodata = "SAR unavailable" = neutral) |
| caveat | GRD is ESA L1 — detected, multilooked, ground-range; **not** radiometrically-terrain-corrected / speckle-filtered. geoseek applies an adaptive Lee 7×7 filter (ENL 4.4) in the intensity domain before the dB ratio. These tiles are catalogued but **not embedded** (`faiss_id IS NULL`). |

---

## 7. Retrieval metrics

*Source: Phase 7a `scripts/eval_retrieval_{prepare,features,judge,score}.py` →
`data/eval_retrieval/{queries,pools,tile_features,judgments,judgments_rationale,report}.json`;
manifest `retrieval_evaluation`. At-scale: `scripts/eval_precision_at_scale.py`,
`scripts/eval_retrieval_at_scale.py`, `scripts/diagnose_judge_transfer.py` →
`precision_at_scale.json`, `retrieval_at_scale_tier3.json`,
`judge_diagnosis/`.*

### 7.1 Method — judgements independent of the model under test

RemoteCLIP is the system under test. Judging its results with RemoteCLIP (or the
vanilla CLIP control, or any learned embedding) makes the evaluation circular and
forces every metric to ≈ 1.0. So relevance is graded from a signal **neither
retrieval model can see**: the models embed only true-colour RGB; the judge uses
**per-tile NDVI / NDWI / NDBI / SCL statistics and an NDWI-derived river mask**,
computed from bands the models never receive (`eval_retrieval_features.py` →
`tile_features.json` for all 3,267 Ayodhya S2 tiles). One fixed rule set per
query, grades **2** (clear physical match) / **1** (partial) / **0** (no match),
each grade stored with its criterion in `judgments_rationale.json`. **No embedding
is used to assign any grade.**

**16 natural-language queries** over the AOI's real content (river/sandbar,
settlement, cropland, bare ground, water body, riverside construction, roads,
bridge, riverbank vegetation), including the PS's own example phrasings
*"newly built structures near a river"* and *"settlement along a riverbank"*.
No vehicle-scale queries — 10 m GSD cannot resolve them.

**Pool** per query = RemoteCLIP top-20 ∪ vanilla top-20 ∪ a fixed random sample
of 15 corpus tiles (seed `20260905:<query>`). Depth 20 = the max reported K, so
no unjudged item can enter a metric. Both systems pooled symmetrically. Realised:
mean pool **53.2** tiles, mean RC∩VA top-20 overlap **1.6 / 20**, and **30
relevant tiles across the 16 queries were found only via the random draw**.

### 7.2 Results — macro-averaged, 3,267-tile corpus, identical judgements for both systems

**All 16 queries:**

| K | RC Recall | RC Prec | RC NDCG | VA Recall | VA Prec | VA NDCG | ΔNDCG (RC − VA) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 0.040 | 0.563 | 0.438 | 0.019 | 0.313 | 0.281 | **+0.156** |
| 5  | 0.228 | 0.425 | 0.397 | 0.062 | 0.263 | 0.231 | **+0.165** |
| 10 | 0.365 | 0.381 | 0.419 | 0.115 | 0.238 | 0.223 | **+0.195** |
| 20 | 0.705 | 0.356 | 0.526 | 0.227 | 0.231 | 0.254 | **+0.272** |

**Excluding the 3 low-confidence road / track / bridge queries (13 queries):**

| K | RC Recall | RC Prec | RC NDCG | VA Recall | VA Prec | VA NDCG |
|--:|--:|--:|--:|--:|--:|--:|
| 5  | 0.244 | 0.492 | 0.454 | 0.067 | 0.308 | 0.257 |
| 10 | 0.403 | 0.446 | 0.480 | 0.122 | 0.277 | 0.244 |
| 20 | 0.667 | 0.381 | 0.547 | 0.261 | 0.277 | 0.282 |

RemoteCLIP beats the vanilla control at **every K on every metric**. Absolute
numbers are "good, not perfect" (RC P@1 ≈ 0.56–0.69, R@20 ≈ 0.70) — exactly what a
non-circular evaluation should show; ≈ 1.0 would mean the judge was leaking the
model. Honest nuance from `report.json → per_query`:

* RC's edge is largest on **specific / rarer** concepts (*"newly built structures
  near a river"* NDCG@10 0.76 vs 0.00; *"settlement along a riverbank"* 0.65 vs
  0.00; *"open bare ground"* 0.50 vs 0.00).
* On **high-prevalence** classes RC can *lose* to vanilla: *"agricultural fields"*
  / *"cropland with visible field boundaries"* have 30–37 relevant tiles in a
  ~53-tile pool, so anything green scores well and vanilla's generic-greenery
  bias is rewarded — the metric is near its ceiling for both and not
  discriminating.
* *"dense urban buildings"* / *"an urban residential neighborhood"*: ≈ 0 for
  **both** — neither surfaced the AOI's small built-up core for those phrasings
  (RC did for *"settlement along a riverbank"*). A pooling limitation, reported.
* The 3 low-confidence road/bridge queries are near-zero and noisy for both —
  which is why they are split out.

### 7.3 At scale — the corrected result, and the judge-coverage artifact

Re-running the **same 16 queries** against the grown **101,911-tile** index,
scored against the **frozen, unchanged** Phase 7a judgements:

| condition | R@10 | P@10 | NDCG@10 | R@20 | P@20 | NDCG@20 |
|---|--:|--:|--:|--:|--:|--:|
| **Phase 7a (3,267-tile corpus)** | 0.365 | 0.381 | 0.418 | 0.705 | 0.356 | 0.526 |
| Global 101,911-tile (raw) | 0.027 | 0.031 | 0.026 | 0.106 | 0.028 | 0.048 |
| **101,911-tile + Ayodhya region pre-filter (corrected)** | **0.365** | **0.381** | **0.418** | **0.705** | **0.356** | **0.526** |

The corrected row **reproduces Phase 7a to within rounding** after a 31× corpus
growth across 8 new biomes — expected, because the incremental-ingest proofs
(§2.2) show the Ayodhya vectors are byte-identical at every tier, so the same
deterministic RemoteCLIP encoder ranking the same 3,267 Ayodhya tiles yields the
same top-K. **Retrieval quality inside the judgeable domain is unchanged at
scale.**

**Why the raw global number collapses — it is ~78% a judge-coverage artifact, not
retrieval failure.** The Phase 7a judge is a set of per-tile spectral criteria
calibrated on 3,267 **Ayodhya** tiles (5 of the 16 queries are relative to the
Ayodhya river mask specifically), and the 8 new regions were staged **RGB + SCL
only — no NIR/SWIR → no NDVI/NDWI/NDBI → they cannot be judged at all**. Global
top-20 composition, averaged over the 16 queries:

| bucket | fraction of global top-20 |
|---|--:|
| judged relevant (grade > 0) | 2.8% |
| judged 0 (judge saw it, said no) — genuine retrieval error | **6.6%** |
| unjudged, Ayodhya | 0.0% |
| **unjudged, other region** | **90.6%** |

A blind visual pass over 187 of the un-positively-judged high-rank tiles
(`judge_diagnosis/ANNOTATIONS.json`) found **81% of the unjudged-other-region
tiles are on-target for their query text** (desert for *"open bare ground"*,
tidal channels for *"a water body"*, Delhi fabric for *"dense urban buildings"*).
Scoring those correctly-retrieved-but-unjudgeable tiles as "irrelevant" is the
entire collapse. The **8 `judged_zero` tiles** are genuine imperfections — a
small recurring set of haze / false-colour Ayodhya tiles that RemoteCLIP embeds
~0.32–0.34 cosine to many text queries. That is ~6.6% of the top-20, not 90%.

**Precision levers, measured independently** (`eval_precision_at_scale.py`):

| lever | effect at K ≤ 20 |
|---|---|
| **region metadata pre-filter** | the entire recoverable win — P@20 0.028 → 0.356, NDCG@20 0.048 → 0.526, R@20 0.106 → 0.705. Same operation as the domain restriction. |
| score threshold τ\* = 0.29 (picked by macro-F1 on the disjoint 3,267-corpus rankings) | **no-op at K ≤ 20** on a 100k corpus (every top-20 already > 0.29); trims ~0.6 tiles/query on the region-filtered corpus (Pj@20 0.356 → 0.366). Thresholding is the wrong tool for an over-supplied corpus. |
| near-duplicate suppression τ_dup = 0.97 | marginal positive on the global corpus (R@20 0.106 → 0.112); ~neutral on the region-filtered corpus. A display-time refinement, not a precision lever. |

**The region pre-filter delivers 100% of the recoverable precision; the other two
are second-order polish.** This is the architectural answer — *filter cheap
metadata first, then semantic-rank* — and it is how an analyst actually searches
(a sector, not the whole archive).

### 7.4 Full judge coverage — what the §7.3 collapse really was (Phase 10, Block A)

*Source: `scripts/stage_nir_swir_and_describe.py` → catalog table `tile_spectral` (104,089 Sentinel-2 tiles);
`scripts/eval_judge_coverage.py` → `data/eval/phase10_blockA.json`; corpus = the **frozen** 101,911-tile corpus
(`faiss_id < 101911`).*

§7.3 attributed the global-precision collapse to unjudged tiles but could not score them. Block A can: B08/B11 were
staged for the 8 diverse regions (74 scenes, 15.6 GB downloaded, grid-identical to the existing crops) and a per-tile
spectral descriptor was computed in the same pass; the **unchanged Phase 7a graders** were then applied to every tile.
The mixed judge reproduces all 852 frozen Phase 7a judgements on the pool (852/852), so on Ayodhya it *is* the frozen
judge. Per-query precision under the two judges:

| RemoteCLIP, 16 queries | global, Phase 9 (unjudged = 0) | global, **full coverage** | Ayodhya-filtered |
|---|--:|--:|--:|
| P@5  | 0.038 | **0.287** | 0.425 |
| P@10 | 0.031 | **0.300** | 0.381 |
| P@20 | 0.028 | **0.306** | 0.356 |

**Claim 1 — "~78% of the collapse is a judge-coverage artifact": confirmed, and as a lower bound.** Full coverage
recovers 64.5% (K=5), 76.8% (K=10) and 84.8% (K=20) of the P@K gap between the raw global number and the region-filtered
one for RemoteCLIP (vanilla CLIP: 86% / 95% / 104%). Every disagreement between this judge and the blind visual pass
runs towards MORE relevance (below), so the true figure is at least this large.

**Claim 2 — "81% of the unjudged high-rank tiles are on-target": not reproduced by the spectral judge.** On the same
179 other-region tiles the visual pass called 81.0% on-target, the spectral judge calls **28.5%** (all 187 annotated
pairs: 77.5% vs 27.3%); across all unjudged top-20 slots it grades 30.7% relevant (20.0% at grade 2). The two agree where
the criterion is physical (water body 11/11, open bare ground 11/12, dense urban 9/11) and disagree where the spectral
criterion is narrower than the query text: "agricultural fields" 0/12 and "irrigated farmland" 0/12 — the visually
accepted tiles are Deccan/Thar fields in Oct–Mar with median NDVI 0.12–0.15 (fallow, spectrally bare; the Phase 7a rule
means *green* cropland at Ayodhya's peak-rabi phenology) — and the river-relative queries (0/10 for "settlement along a
riverbank"), where "the largest water component of the region" is not the river (median 90 km away). The judge has no
false positives against the visual negatives (0 of 10) but recalls only 33% of the visual positives. So **81% is neither
confirmed nor refuted**: it is a single-labeller, RGB-only call (the labeller sees what the model sees), and the spectral
31% is a conservative floor. The corrected global P@20 lies above 0.306; pinning it down needs a stratified random sample
labelled by independent annotators, which this evaluation does not have.

**Claim 3 — "the region pre-filter is a 12.7× precision lever (P@20 0.028 → 0.356)": the numbers reproduce, the reading
is withdrawn.** With every tile judged, global P@20 is 0.306 against 0.356 filtered: **1.16×, not 12.7×**. Against chance
(a random top-K scores the candidate set's prevalence of relevant tiles: 15.3% over the corpus, 13.6% inside Ayodhya)
RemoteCLIP lifts 2.01× globally and 2.62× inside the sector (1.3× apart). On the 7 "core" queries (no river-relative or
sub-pixel grader) the filter is *worse*: P@20 0.507 unfiltered vs 0.429 filtered (lift 1.79× vs 1.58×). Per query it can
destroy relevance ("dense urban buildings" 0.70 → 0.05, "open bare ground" 0.95 → 0.20, "an urban residential
neighborhood" 0.60 → 0.05: Ayodhya has none to retrieve). A region filter is a **scoping tool** (answer the question
*inside this sector*), not a way to make the same question more precise.

**Where the judge itself is weak (read every number above with this).** Thresholds were calibrated on Ayodhya's cropland;
outside it they are applied as-is. The "built-up" signature (NDBI > −0.05, NDVI < 0.30, NDWI < 0) fires on arid bare soil,
so "dense urban buildings" and "residential" are graded relevant for 44% and 48% of the corpus — precision for those queries is
mostly base rate. Raw-band descriptors reproduce the stored Ayodhya features (correlation 0.93–0.999; 96.6% grade
agreement over all queries) except for a threshold-sensitive shift in three queries (dense urban, residential, irrigated:
±10 points) because the Phase 7a features come from radiometrically *normalized* index rasters; the "uniform" judge variant
(descriptor everywhere, Ayodhya included) moves global P@20 by < 0.001 and the filtered figure from 0.356 to 0.353. Metric
definitions change under full coverage: precision keeps the baseline definition (so the coverage effect is isolated), while
NDCG's ideal and Recall's denominator are now the whole candidate set (see `geoseek.eval.full_judge`); only precision is
comparable with the Phase 9 baseline row by row.

---

## 8. Change-detection metrics

*Source: `scripts/train_change.py --eval-only` → manifest `oscd_change_model_eval`;
`data/change_model/pr_curve_{val,test}.png`, `qual_{lasvegas,montpellier,chongqing}.png`.*

**Protocol.** Full-image inference (reflection-padded to /8, cropped back),
sigmoid, per-pixel. Threshold selected on the **3 validation regions** by argmax
F0.5 (β = 0.5, precision weighted 2× recall — PS 2.2.3 favours precision) with a
recall floor of 0.15, then **frozen** and applied **once** to the 10 held-out
test regions. Leakage is asserted-against in code: the test regions are not read
during training, norm-stat fitting, or threshold selection.

### 8.1 Held-out test metrics (pixel-pooled over all 10 test regions)

| operating point | P | R | F1 | IoU | FPR | TP | FP | FN |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| default **0.50** | 51.7% | 61.0% | **56.0%** | 38.8% | 3.10% | 96,997 | 90,601 | 62,080 |
| **precision-favouring 0.80** (deployed) | **60.3%** | 51.0% | 55.3% | 38.2% | **1.83%** | 81,146 | 53,342 | 77,931 |
| *(diagnostic, NOT used)* test-set oracle max-F1 @ 0.61 | 54.7% | 57.9% | 56.2% | — | — | | | |

Change-pixel fraction of the test set = **5.17%** (159,077 / 3,077,936 px).
Validation set at the chosen 0.80: P 53.1% / R 49.2% / F1 51.1% / F0.5 52.3% —
the frozen point transfers to the test set with the intended precision bias
(P 60.3% vs the default's 51.7%) at ~10 pts of recall.

### 8.2 vs published OSCD baselines (change class)

| model | P | R | F1 |
|---|--:|--:|--:|
| FC-EF (Daudt 2018) | 44.9% | 51.7% | 48.0% |
| FC-Siam-conc (Daudt 2018) | 42.1% | 62.2% | 50.2% |
| FC-Siam-diff (Daudt 2018) | 48.8% | 57.5% | 52.8% |
| **geoseek FC-Siam-diff @ 0.50** | 51.7% | 61.0% | **56.0%** |
| **geoseek FC-Siam-diff @ 0.80** | 60.3% | 51.0% | **55.3%** |

Our F1 (~56%) sits **just above** the published FC-Siam-diff (52.8%) — attributable
to the 5-band input (RGB+NIR+SWIR vs RGB) and a modern training recipe
(Dice + weighted-BCE, D4 augmentation, cosine LR, early stopping). It is **not**
above by a margin that flags leakage (the code checks: > +8 pts F1 would trigger
a "TREAT AS LEAKAGE RED FLAG" verdict; it reads "in range … expected").

### 8.3 Per-region test F1 @ 0.80 — the spread is large and honest

| region | P | R | F1 | | region | P | R | F1 |
|---|--:|--:|--:|---|---|--:|--:|--:|
| lasvegas | 74.3% | 74.3% | **74.3%** | | rio | 50.2% | 34.8% | 41.1% |
| montpellier | 84.2% | 58.3% | 68.9% | | milano | 77.0% | 22.4% | 34.8% |
| brasilia | 53.9% | 62.6% | 57.9% | | norcia | 20.8% | 46.5% | 28.7% |
| chongqing | 74.5% | 45.6% | 56.6% | | saclay_w | 17.2% | 41.1% | 24.3% |
| dubai | 59.1% | 35.4% | 44.3% | | valencia | 2.2% | 13.8% | **3.8%** |

Regions with <1% change (valencia 0.44%, saclay_w 1.14%, milano 0.80%) are
dominated by false positives — a single-detector ceiling on very-low-prevalence
scenes. `norcia` is the 2016 earthquake (building collapse), a genuinely
different change type from the "new construction" the model mostly saw.

### 8.4 On Ayodhya (out-of-distribution, no labels)

The domain gap (L1C→L2A, urban→semi-rural, same-season→drought-to-green) means
the *effective* Ayodhya precision/recall is **materially worse than the 55% OSCD
F1** — stated in the pipeline's own report as a `domain_gap_statement`. The model
flags **1.33%** of the valid AOI per pair (814,127 changed px of the 2019→2024
span). This is why the suppression + persistence + confidence stages exist; §9
measures what each one buys.

---

## 9. Ablation study

*Source: `scripts/ablation_change_pipeline.py` →
`data/change_model/ablation_study.json`; manifest `change_pipeline_ablation`.*

Cumulative, one variable at a time, in the order the pipeline applies them. Two
domains, and **the metric is different in each — never conflated**:

* **OSCD held-out (10 pixel-labelled regions)** — stages 1–5 give real
  **P / R / F1 / IoU / FPR** vs the change mask (pixel-pooled). Stage 8 gives a
  **component-level Average Precision**. Stages 6 (persistence) and 7 (SAR) are
  **definitionally inapplicable** — OSCD is bitemporal and has no Sentinel-1.
* **Ayodhya (2019→2024 span, NO labels)** — **candidate counts and qualitative
  effects only.** These are **not** accuracy numbers. Stages 6/7/8 are assessed
  here because this is the only domain where they run.

The gates are the shipped code — the script builds real
`geoseek.change.suppress` objects and calls `suppress_candidate`; it does not
re-implement any rule. Stage-1 numbers reproduce §8 exactly (a self-check).

### 9.1 OSCD — cumulative pixel metrics @ the deployed operating point (0.80)

> **How to read this table — the two columns measure different regimes and must be
> read together.** The **OSCD column** scores each stage on a benchmark of
> **same-season, mostly urban, bitemporal pairs**. The suppression stages (quality,
> radiometric, phenology) are calibrated for the **deployment regime** — Ayodhya's
> **multi-year interval with a strong seasonal confounder** (a drought March in
> 2019 vs a green March in 2024) — a confounder that OSCD, by construction, does
> **not** contain. So a stage that scores **negative on OSCD is not evidence it
> fails**; it is evidence the benchmark lacks the very confounder that stage
> exists to remove. The OSCD column is a *safety check* (does the stage break the
> raw detector on labelled data?) and a *domain-transfer probe*; the **Ayodhya
> column (§9.3–9.4) is the deployment-regime evidence** and is where each
> suppression stage earns or fails to earn its place. Concretely: **phenology
> removes 110 seasonal false positives on Ayodhya** (§9.3) **while costing recall
> on OSCD, which has no seasonal gap to suppress.**
>
> **What ships by default.** `python -m geoseek.change.analyze` runs the **full
> pipeline — all five suppression gates + persistence + SAR + the 6-term
> confidence score + `queue_score` ranking + diversify — at the frozen
> precision-favouring operating point 0.80** (the threshold embedded in the
> checkpoint's eval card; §8). No gate is disabled. The rationale is PS 2.2.3's
> precision bias: the analyst queue must **lead with trustworthy, temporally-
> confirmed detections**, and on the deployment data every stage 4–8 measurably
> serves that goal (§9.3–9.4). The OSCD-negative stages are kept on because the
> benchmark they under-perform on is not the regime the system runs in.

| # | stage | P | R | F1 | IoU | FPR | ΔF1 |
|---|---|--:|--:|--:|--:|--:|--:|
| 1 | FC-Siam-diff raw output | 0.6034 | 0.5101 | **0.5528** | 0.3820 | 0.0183 | — |
| 2 | + quality / SCL gating | 0.6034 | 0.5101 | 0.5528 | 0.3820 | 0.0183 | **±0.0000** |
| 3 | + radiometric normalization | 0.6034 | 0.5101 | 0.5528 | 0.3820 | 0.0183 | **±0.0000** |
| 4 | + phenology suppression | 0.5910 | 0.2736 | 0.3741 | 0.2301 | 0.0103 | **−0.1787** |
| 5 | + morphology (50 px floor) | 0.6278 | 0.2541 | **0.3618** | 0.2208 | 0.0082 | −0.0123 |

*(At threshold 0.50 the pattern is the same: 1 → 0.5596, 2 → 0.5596, 3 → 0.5596,
4 → 0.3731, 5 → 0.3699.)*

**Stage 2 — quality / SCL gating: exact no-op on OSCD.** OSCD is L1C — **there is
no SCL** — and the rectified crops have **0% nodata** in all 10 test regions, so
neither half of the gate can fire (measured: **0 components removed**, all
regions, both thresholds). Not a failure of the gate — its input does not exist
on this benchmark. Its contribution is only demonstrable on Ayodhya (§9.3).

**Stage 3 — radiometric normalization: a confidence DOWN-WEIGHT, not a hard gate.**
It cannot remove a candidate, so it **cannot change the detection mask** — ΔF1 = 0
by construction. It *does* fire: **521 components across 3 regions**
(norcia min inter-date band corr 0.25 → 66, saclay_w 0.20 → 204, valencia 0.48 →
251) drop to a ×0.6 confidence weight. That effect surfaces only in ranking
(stage 8), where it is part of why confidence ordering loses on those regions.

**Stage 4 — phenology suppression: the headline negative result. −0.18 F1 on
OSCD, almost entirely recall (0.51 → 0.27) for essentially zero precision gain
(0.603 → 0.591).** The stage removes **more true-positive pixels than
false-positive pixels** (−37,615 TP vs −23,219 FP). Under the Ayodhya pipeline's
own morphology-first ordering the marginal effect is the same size (−0.19 F1;
recall 0.485 → 0.254 after the 50 px floor). **Why:** the phenology gate's
anomaly bands (±0.10 NDVI, ±0.06 NDBI, ±0.08 NDWI) are calibrated for the Ayodhya
scene, where the drought→green shift is huge (scene ΔNDVI +0.30) and a structural
change breaks sharply from it. On OSCD the scene ΔNDVI is tiny (−0.01 to +0.06),
so those bands are wide *relative to the actual anomalies*, and OSCD's labelled
changes — spectrally-subtle **bare-soil → concrete/rooftop** transitions at 10 m
mixed pixels — fall inside them and get suppressed as "seasonal". This is the
**expected failure of transplanting a scene-calibrated rule across a domain
gap**, not evidence the rule is wrong for its intended use. Per-region: it is
catastrophic on high-recall regions (dubai raw F1 0.443 → 0.081; montpellier
0.689 → 0.126) and slightly *helpful* on the FP-dominated <1%-change regions
(norcia 0.287 → 0.289, valencia 0.038 → 0.043).

**Stage 5 — morphology (50 px / 0.5 ha floor): ~F1-neutral on OSCD.** In isolation
(quality → morphology, no phenology): F1 0.5528 → 0.5480 @ 0.80 (recall −0.025 for
precision +0.026, IoU 0.382 → 0.377) and 0.5596 → **0.5663** @ 0.50 (the floor
removes tiny FP specks and *helps* at the lower threshold). It does exactly its
designed recall↔precision trade and is close to break-even on the benchmark. Its
real value is on Ayodhya (§9.3).

### 9.2 OSCD — stage 8 (confidence-weighted ranking): a negative result

Component-level Average Precision over the stage-5 survivors, ordered by **raw
mean model probability** vs the **full confidence score** (a component is a TP if
it overlaps ≥ 1 GT change pixel; a stricter ≥ 25%-overlap variant agrees):

| operating point | mean AP by raw prob | mean AP by confidence | Δ | regions confidence wins / loses / ties |
|---|--:|--:|--:|--:|
| 0.50 | 0.680 | 0.648 | **−0.033** | 4 / 5 / 1 |
| 0.80 | 0.715 | 0.671 | **−0.045** | 3 / 6 / 1 |

**On a bitemporal, optical-only benchmark the confidence engine slightly *hurts*
the ranking.** Only 2 of its 6 terms carry signal on OSCD (model, spectral) —
persistence is the constant `single_pair` case, SAR is absent, quality and
registration are constant — and the spectral term is derived from the **same
Ayodhya-calibrated index anomalies that misfire in stage 4**. Where spectral
evidence is clean it wins (montpellier +0.128, chongqing +0.028); where the
Ayodhya-tuned spectral / radiometric terms misfire it loses hard (norcia −0.352,
saclay_w −0.118). **The confidence engine's value depends on the persistence and
SAR terms that only exist in the 3-date + SAR Ayodhya deployment.**

### 9.3 Ayodhya — candidate counts (2019→2024 span, NOT accuracy)

Independent recompute on the cached probability raster **reproduces the committed
Phase 4/5 report exactly**:

| stage | candidates | Δ | what it removed |
|---|--:|--:|---|
| 1 · raw model (`p ≥ 0.80` connected components) | **13,563** | — | — |
| 5 · morphology 50 px floor *(applied first on Ayodhya, for tractability)* | **1,215** | −12,348 | sub-0.5-ha specks — 91% of raw components, visually indistinguishable from noise |
| 2 · quality / SCL gate | **1,214** | −1 | one cloud/shadow-contaminated component (the AOI is near cloud-free) |
| 3 · radiometric | **1,214** | −0 (0 down-weighted) | min inter-date band corr 0.687 > 0.60 trust line; but the span pair's radiometric *reliability* is still only 0.50 and feeds the confidence score as a weak term |
| 4 · phenology | **1,104** | −110 | genuine seasonal components (~9% of the ≥50 px set) — here the gate earns its place: scene ΔNDVI +0.30, so real seasonal change tracks it and structural change breaks from it |
| 5 · morphology (residual) | **1,104** | −0 | no ≥50 px component is below the floor — expected |

**On Ayodhya the roles invert relative to OSCD:** morphology is the workhorse
(−91%), phenology genuinely removes seasonal false alarms (−110), quality catches
the one cloud component, radiometric down-weights confidence without removing
anything.

### 9.4 Ayodhya — stages 6, 7, 8 (qualitative, from the committed Phase 4/5 report)

**Stage 6 — temporal persistence** (needs the 3-date stack; OSCD cannot do this).
Of the 1,104 span survivors:

| persistence class | count | confidence treatment |
|---|--:|---|
| persistent / progressive / recent (temporally **supported**) | **601** | full confidence |
| transient / inconsistent (temporally **contradicted**) | **420** (38%) | penalty ×0.50 / ×0.45 → pushed below actionable confidence |
| none (span-only support) | 83 | |

The **601** survivors that keep confidence ≥ 0.5 are *exactly* the
persistent + progressive + recent set — **persistence is the lever that splits the
actionable queue from the rest.** A one-shot high-probability blob that reverts by
the next date cannot score "medium" no matter how strong the model / spectral /
quality evidence.

**Stage 7 — SAR corroboration** (needs Sentinel-1; OSCD has none). Confidence
factor distribution over the 1,104 survivors: ×0.80 → 31, ×0.93 → 80, ×1.00 → 613
(281 with no C-band expectation [road/other] + 2 with no usable co-located SAR +
330 within speckle noise), ×1.04 → 178, ×1.10 → 202. **491 of 1,104 candidates
(44%) are moved — 380 up, 111 down.**
Weight-only, clamped to [0.80, 1.10]; SAR never overrides the optical detection.
The water-gain agreement rate is **193 / 376 = 51.3%** — see §12 for the physical
reason it is 51% and not higher.

**Stage 8 — confidence-weighted ranking + diversify.** Contrast the shipped
headline top-10 with a naive raw-mean-probability top-10:

| | full pipeline (`queue_score`, then diversify) | naive raw-prob top-10 |
|---|---|---|
| comes from full-queue ranks | 1–10 | 2, 13, 30, 55, 81, 96, 121, 197, 213, **578** |
| median area | **260,450 m²** | 55,150 m² (4.7× smaller) |
| persistent / progressive | **10 / 10** | 2 / 10 (7 are single-interval "recent", one is **transient**, confidence 0.36) |
| composition | 3 water_gain / 3 construction / 3 road / 1 other (≤ 3 per type, ≥ 1.5 km spacing) | whatever is brightest |

Raw probability surfaces the model's loudest pixels regardless of trust or size.
The full `queue_score` = `confidence^0.65 · significance^0.35` (weighted geometric
mean), then diversify, yields **large, temporally-confirmed, type-diverse changes
an analyst can action**.

### 9.5 Ablation — bottom line

| stage | OSCD (labelled) | Ayodhya (deployment) |
|---|---|---|
| 1 raw FC-Siam-diff | F1 0.553 @ 0.80 | flags 1.33% of the AOI |
| 2 quality / SCL | **±0.000** (no SCL in L1C, 0% nodata) — inert on this benchmark | −1 component (cloud); low-value on a cloud-free AOI |
| 3 radiometric | **±0.000** to the mask (it's a down-weight); fires on 3/10 regions → ranking only | −0 removed; reliability 0.50 feeds confidence |
| 4 phenology | **−0.18 F1** — Ayodhya-calibrated bands over-suppress OSCD's subtle bare→built changes | **−110** genuine seasonal FPs (earns its place — scene ΔNDVI +0.30) |
| 5 morphology 50 px | ~F1-neutral (−0.012 @ 0.80, +0.007 @ 0.50) | **−12,348** specks (91%) — the workhorse for a usable queue |
| 6 persistence | **N/A** (bitemporal) | demotes **420 / 1,104** to non-actionable |
| 7 SAR | **N/A** (no S1) | moves 491 / 1,104 (weight-only); water-gain 51% agreement |
| 8 confidence ranking | **−0.04 AP** — needs its temporal + SAR terms to help | top-10 → 10/10 persistent, 4.7× larger, type-diverse |

**Every stage that under-performs is a stage tuned for a different regime than the
one it was tested in.** Quality needs clouds/nodata to matter; phenology and the
confidence spectral term need Ayodhya's large seasonal signal; persistence and
SAR need the 3-date + Sentinel-1 deployment. On the deployment they were built
for, each of 4–8 measurably shapes the analyst queue; on OSCD, 2–3 are inert and
4/8 are Ayodhya-calibration mismatches. That distinction is the finding.

---

## 10. Scale benchmark

*Source: `scripts/measure_tier.py` (per tier), `scripts/plot_scale.py` →
`data/eval_retrieval/plot_latency_vs_scale.png`, `plot_storage_vs_scale.png`.*

### 10.1 Latency and storage vs index size

| metric | 3,267 *(7a)* | 11,394 | 50,126 | 100,887 | 11,394 → 50,126 ratio | 50,126 → 100,887 ratio |
|---|--:|--:|--:|--:|--:|--:|
| n ratio | — | — | — | — | 4.40× | 2.01× |
| FAISS index size | 6.69 MB | 23.33 MB | 102.66 MB | 206.62 MB | 4.40× (exact `n·512·4`) | 2.01× |
| SQLite catalog | ~4.6 MB *(est.)* | 14.69 MB | 47.59 MB | 90.82 MB | 3.24× | 1.91× |
| total `data/` | — | 9,370 MB | 20,251 MB | 34,450 MB | 2.16× | 1.70× |
| FAISS persist (full rewrite) | — | 18.6 ms | 117.7 ms | 1,357.6 ms | 6.3× | **11.5×** |
| text search median | 17.4 ms | 8.4 ms | 14.4 ms | 27.8 ms | 1.71× | 1.93× |
| image search median | ~1.0 ms | 1.5 ms | 7.2 ms | 21.7 ms | **4.76×** | 3.02× |
| point-KNN median | 1.00 ms | 2.0 ms | 7.9 ms | 20.0 ms | 3.97× | 2.53× |
| tile-KNN median | 0.99 ms | 1.6 ms | 7.3 ms | 18.0 ms | 4.64× | 2.45× |
| bbox filter median | 0.28 ms | 0.21 ms | 0.23 ms | 0.29 ms | flat (R\*Tree) | flat |
| peak RSS / VRAM (query burst) | — | 1,829 / 620 MB | 1,896 / 620 MB | 2,099 / 620 MB | flat | +11% RSS |

### 10.2 The FlatIP scaling signature

Fitting `latency ∝ n^k` to `image_search` median (the cleanest FAISS-scan-only
signal — no text-encode overhead):

| segment | n ratio | latency ratio | implied exponent *k* |
|---|--:|--:|--:|
| baseline → Tier 1 | 3.5× | ~1.5× | ~0.36 (still in the sub-noise-floor at these small n) |
| Tier 1 → Tier 2 | 4.40× | 4.76× | **1.05** — textbook linear, exactly as expected for `IndexFlatIP` |
| Tier 2 → Tier 3 | 2.01× | 3.02× | **1.58** — clearly super-linear |

Every full-scan operation (`search_text`, `search_image`, both KNN — the ranker
always asks FAISS for `n` results, not `k`) scaled 3.6–4.8× over the 4.40× Tier
1→2 growth (linear); the R\*Tree bbox filter and every RAM/VRAM/ingest number
stayed flat. The **Tier 2→3 segment is genuinely steeper** across all three
scan-only operations — likely a real effect once the 206 MB vector array stops
fitting comfortably in CPU cache, possibly confounded by the same host-level
slowdown noted in §4 (though the latency run was its own fresh process).

### 10.3 HNSW crossover — an estimate with an honest range

FlatIP is **comfortably fine at 100k** (21.7 ms image search, 27.8 ms text — far
under any interactivity threshold) and remains a defensible exact zero-tuning
choice well past it. The crossover to where an approximate index (HNSW) is
*needed* for a snappy (< 100–200 ms) interactive UI:

| trend assumed | reaches ~100 ms at | reaches ~1 s at |
|---|--:|--:|
| **conservative** (Tier 1→2's near-perfect linear rate) | ~700k vectors | ~7 M vectors |
| **observed** (Tier 2→3's k ≈ 1.58 rate, at face value) | ~300k vectors | ~1.1–1.2 M vectors |

The crossover sits in the **low hundreds of thousands to low millions of
vectors**; by ~10⁷ (tens of millions) FlatIP would clearly need replacing
regardless of which extrapolation holds. Text search inherits a fixed per-query
encode cost on top of the same scan, so its absolute numbers run a little higher,
but the same logic applies once the O(n) term dominates (already true by Tier 2).

---

## 11. Clustering / discovery

*Source: `scripts/cluster_tiles.py` (Tier 1), `scripts/cluster_at_scale.py`
(100k); `data/index/tile_clusters.json`,
`data/discovery/cluster_at_scale_100k.json`,
`cluster_at_scale_region_heatmap.png`.*

Method: HDBSCAN over the L2-normalized 512-d RemoteCLIP embeddings (euclidean on
unit-norm = cosine), `min_cluster_size = 40`, `cluster_selection_method = "eom"`;
clusters labelled by the nearest of ~15 RemoteCLIP text concepts.

### 11.1 At 12,418 tiles (5 regions)

| | EOM (shipped default) | leaf mode (labelled secondary run) |
|---|---|---|
| clusters | **3** | **9** |
| noise | 26.9% (3,338 tiles) | 72.9% (9,047 tiles) |
| cluster sizes | 7,754 / 983 / 343 | — |
| top concepts | bare dry open ground · trees/vegetation · sandy braided riverbed | — |

**EOM did not increase the cluster count** across five new biomes — it selects
the most *stable* cut, and a coarse bare/vegetation/water 3-way split stays most
stable. What changed: noise tripled (9% → 27%), itself a diversity signal. **Leaf
mode** (a documented HDBSCAN mode for exactly the "EOM over-merges" case) recovers
far more structure at a steep coverage cost — and **7 of its 9 clusters are >96%
single-region with zero access to location metadata**:

| cluster | size | region purity | concept |
|--:|--:|---|---|
| 4 | 730 | 100% jaisalmer | bare dry open ground / quarry |
| 1 | 632 | 100% ayodhya | trees and dense vegetation |
| 5 | 496 | 100% kanha | bare dry open ground |
| 7 | 441 | 100% ayodhya | bare dry open ground |
| 8 | 437 | 96.6% ayodhya | rural village settlement |
| 2 | 287 | 99.7% sundarbans | sandy braided riverbed / waterlogged |
| 6 | 154 | 100% dehradun | rural village + cropland |
| 3 | 112 | 100% dehradun | bare ground / riverbed |
| 0 | 82 | 100% ayodhya | irrigated cropland |

RemoteCLIP clusters the Thar desert, the Sundarbans delta and the Himalayan
foothill villages into their own tight near-pure groups purely from visual
similarity — **the generalization claim made concrete: the same frozen model
separates terrain it was never trained or tuned on.**

### 11.2 At 101,911 tiles — a real scalability wall, and the fix

**Single-pass HDBSCAN over 101,911 raw 512-d embeddings does not finish in
practical batch time.** Two attempts, both killed: 49+ min single-threaded,
**60 m 50 s** with all 16 cores. The MST / cluster-hierarchy-extraction step
(which HDBSCAN does not parallelize) is the wall; tree-based nearest-neighbour
acceleration loses its advantage in 512-d. This is the one Phase-5 capability that
does **not** hold up unmodified at 100k on this hardware — distinct from and
unrelated to the FAISS search-latency scaling, which stays healthy (§10).

**Fix — stratified sample + nearest-centroid assign** (`cluster_at_scale.py`):
HDBSCAN on a 20,000-tile stratified sample (9 regions, 200-tile floor, seeded),
then assign all 101,911 tiles to the nearest sample-cluster centroid (one dense
cosine matmul on unit-norm vectors).

| step | time |
|---|--:|
| bulk vector load (`FaissFlatIPIndex.reconstruct_all`) | < 0.1 s |
| sample HDBSCAN (20k) + assignment (101,911) | **218 s (3.6 min)** |

6 clusters; 4 are region-concentrated (kutch 91%, kerala 63%, sundarbans 59%,
ayodhya 75%) and one "generic land" cluster absorbs 85% — the same coarsening EOM
showed at Tier 1. **Agreement with the Tier-1 full clustering** on the 12,418
common tiles:

| comparison | n | ARI | AMI | homogeneity | completeness |
|---|--:|--:|--:|--:|--:|
| **non-noise in both** | 9,080 | **0.959** | 0.905 | 0.925 | 0.886 |
| all tiles (incl. Tier-1 noise) | 12,418 | 0.412 | 0.463 | 0.376 | 0.602 |

The **cluster geometry transfers** (non-noise ARI 0.96 — the 3 Tier-1 EOM
clusters map almost 1:1 onto 3 of the 6 at-scale clusters). The all-tiles ARI is
lower for one honest reason: **hard centroid assignment labels every tile**, so it
necessarily disagrees with density-based HDBSCAN on exactly the 26.9% of tiles
Tier 1 calls noise. An abstention sweep confirms it — all-tiles ARI peaks at ≈
0.53 around an abstain-cosine of 0.82 (≈ 21% noise) and never fully reconciles.
PCA-to-32/64-d before clustering and a leaf-mode sample run are the open paths for
finer structure at scale — flagged as follow-up, not substituted in.

---

## 12. SAR corroboration

*Source: `geoseek.sar.evidence` in `python -m geoseek.change.analyze`;
`data/change_model/ayodhya_change_report.json → sar_corroboration`.*

Co-located Sentinel-1 dB change is turned into a **confidence factor in
[0.80, 1.10]** per candidate — a **weight, never an override** (there is no ground
truth to validate an override). Expected C-band signatures, scene-detrended:
water_gain → VV **drop** (specular off open water); water_loss / construction →
VV rise; clearance → VH drop; road / other → no confident expectation → neutral.

**Result — water_gain agreement: 193 / 376 = 51.3%** (median VV anomaly
**−1.09 dB**, agreement threshold ≤ −1.0 dB; scene VV trend +0.69 dB, VH +1.44 dB;
speckle: adaptive Lee 7×7, ENL 4.4). SAR coverage 92.9% of the AOI.

**Why 51% and not higher — the physical explanation.** In a drought-recovery March
(2019 severe drought → 2024 normal), a large share of what the optical pipeline
labels *"water_gain"* is actually **soil-moisture and vegetation recovery**, not
new open water. SAR C-band backscatter **rises** with soil moisture and biomass —
so only *true* open-water gain shows the expected specular **drop**; moisture /
greening water_gain candidates show the opposite. The 51% is the honest split
between the two. **The top-ranked water_gain candidates do show the VV drop** and
receive the +10% factor (`ayodhya_change_report.json` diverse top-3 are all
persistent water_gain with strong SAR agreement).

**Cloud-penetration value is real but not demonstrable on this AOI.** The 3
Sentinel-2 dates are near cloud-free (bad-SCL << 1%); only **4** optical
components across all pairs were cloud-suppressed — too few to show SAR filling a
gap. Stated as a limitation, not claimed.

---

## 13. Reproduction

Everything below runs **offline** after a one-time staging step (which needs
network for the public STAC / Hugging Face endpoints). Use the conda `geoseek`
env (Python 3.11, torch 2.2.2, rasterio 1.4.4, faiss-cpu, hdbscan). Set
`GDAL_DATA` / `PROJ_LIB` to the env's `Library/share/{gdal,proj}` and
`PYTHONIOENCODING=utf-8`.

```bash
# ── 0. install ────────────────────────────────────────────────────────────
conda env create -f environment.yml        # or: pip install -e .
pytest -q                                   # 228 passed, 3 skipped (parity guards)

# ── 1. stage data (network; one time) ─────────────────────────────────────
python -m geoseek.staging.download_models          # RemoteCLIP + vanilla CLIP (SHA256-checked)
python -m geoseek.staging.download_datasets        # Ayodhya S2 L2A: 2019 + 2024, 5 bands + SCL
python -m geoseek.staging.download_datasets --extra-bands   # B08 + B11 for 2019/2024
python -m geoseek.staging.download_datasets --third-date    # Ayodhya 2021-03-04
python -m geoseek.staging.download_oscd             # OSCD (3 archives, SHA256-checked)
python -m geoseek.staging.download_sentinel1        # 3 S1 GRD scenes, geocoded to the S2 grid
python scripts/stage_diverse_aois.py --list        # then run_diverse_ingest.py to grow the corpus

# ── 2. build the index (offline) ─────────────────────────────────────────
python -m geoseek.ingest.pipeline                   # tile + embed + index the 3 Ayodhya dates
python -m geoseek.catalog.migrate                   # flat table -> collections/scenes/observations/tiles
python scripts/align_third_date.py                  # co-register + PIF-offset the 2021 date
python -m geoseek.change.prep                       # Phase 3a: co-reg + radiometric norm + indices
python scripts/rebuild_index.py                     # (if needed) rebuild FAISS from the catalog

# ── 3. train + evaluate the change model (offline; ~2 min on the RTX 4060) ─
python scripts/train_change.py                      # train on OSCD 11-region fit subset + eval on 10 test
python scripts/train_change.py --eval-only          # re-evaluate the delivered checkpoint (§8)

# ── 4. run the change pipeline on Ayodhya (offline) ──────────────────────
python -m geoseek.change.analyze                    # 3-date pipeline -> ayodhya_change_report.json (§9, §12)

# ── 5. the evaluations in this report ───────────────────────────────────
python scripts/bench_spatial_index.py --report                          # §3.2 spatial index before/after
python scripts/eval_retrieval_prepare.py && python scripts/eval_retrieval_features.py \
  && python scripts/eval_retrieval_judge.py && python scripts/eval_retrieval_score.py   # §7.2
python scripts/eval_precision_at_scale.py                               # §7.3 at-scale + levers
python scripts/measure_tier.py --tier tier3                             # §2, §3, §10 (needs the grown corpus)
python scripts/plot_scale.py                                            # §10 latency/storage plots
python scripts/cluster_tiles.py --min-cluster-size 40                   # §11.1 (12k)
python scripts/cluster_at_scale.py --sample-size 20000 --n-jobs -1      # §11.2 (100k)
python scripts/ablation_change_pipeline.py                              # §9  -> data/change_model/ablation_study.json

pytest -q                                                              # still 228 passed, 3 skipped
```

Determinism: the RemoteCLIP encoder and the FC-Siam-diff inference are
deterministic; the retrieval judgements, the τ\* threshold, the random pool draw,
and the stratified cluster sample are all seeded. The incremental-ingest proofs
(§2.2) assert byte-identical vectors, so a rebuild from a clean checkout produces
the same top-K and the same metrics.

**Seam check** (an acceptance criterion — no app code imports `sqlite3` / `faiss`
outside the seam modules):

```bash
grep -rn "import sqlite3\|import faiss" src/
# -> only catalog/{sqlite_repository,migrate,embedding_map}.py and vectorindex/faiss_flat.py
```

`catalog/embedding_map.py` (added in Phase 9) is the third `sqlite3` importer: it holds the `faiss_id ↔ tile_id`
mapping database that ships next to a *re-embedded candidate* index (`scripts/reembed.py --finalize`), so the production
catalog is never touched by a re-embed. It lives in the catalog package and does not use the production schema.

---

## 14. Limitations

Stated explicitly, because several of them bound how far the numbers above can be
pushed.

1. **10 m GSD cannot resolve vehicles, individual small structures, road width, or
   bridge decks.** A road is 1–2 px in a 256 px tile. The retrieval evaluation has
   **no vehicle-scale queries**, and its road / track / bridge queries are flagged
   `low_confidence` with metrics reported both with and without them (§7.2). The
   change model's per-region F1 is weakest exactly where changes are sub-parcel
   (§8.3).

2. **OSCD → Ayodhya is a genuine domain gap.** OSCD is Sentinel-2 **L1C
   top-of-atmosphere**, **urban**, mostly **same-season** pairs; Ayodhya is
   **L2A bottom-of-atmosphere**, **semi-rural**, and spans a **drought March
   (2019) to a green March (2024)**. The 55% OSCD held-out F1 is an **upper bound**
   on Ayodhya change accuracy — the pipeline's own report says so, and §9.1 shows
   the Ayodhya-calibrated phenology gate *loses* 18 F1 points when run back on
   OSCD. Change accuracy on Ayodhya itself is **not measured** (no labels there).

3. **The retrieval relevance judgements are constructed, not expert, and
   Ayodhya-calibrated.** They come from per-tile spectral proxies (NDVI/NDWI/NDBI/
   SCL + an NDWI river mask), not photo-interpretation or field data; they are
   single-date (*"newly built"* is judged as *"built-up near a river"*, not
   verified as recent); NDBI is only weakly positive over this AOI's low-rise
   tree-mixed built-up; there is one fixed threshold set with no sensitivity
   sweep. Crucially, **the judge only covers the 3,267 Ayodhya tiles** — the 8
   diverse regions were staged RGB + SCL only, so the "corrected" at-scale
   retrieval metric (§7.3) is a **region-restricted** measurement, and the
   global-corpus retrieval quality on the new regions is assessed only by a blind
   visual pass (81% on-target), not a metric.

4. **SAR corroboration is a weight only, and is not validated against ground
   truth.** The [0.80, 1.10] factor never overrides the optical detection; the
   51% water-gain agreement rate (§12) is interpreted physically but there is no
   independent water-extent reference to check it against. Cloud-penetration
   value is argued, not demonstrated (the AOI is cloud-free).

5. **Single-node deployment.** One process, one machine. No PostGIS (the
   `MetadataRepository` seam is shaped for it but not wired), no HNSW (FlatIP is
   exact and fine to ~10⁵–10⁶ vectors — §10.3 — but not beyond), no distributed
   ingestion workers. The FAISS full-file-rewrite persist (§2.4) is O(n) and
   scaling worse than O(n); a continuously-ingesting system at this size would
   need batched or incremental-write saves.

6. **HDBSCAN at scale uses sample + assign, not full clustering.** Single-pass
   HDBSCAN over 101,911 raw 512-d embeddings does not finish in practical batch
   time on this hardware (§11.2). The 100k clustering is a 20k stratified-sample
   HDBSCAN + hard nearest-centroid assignment: the cluster *geometry* transfers
   (non-noise ARI 0.96 vs the Tier-1 full run) but hard assignment labels every
   tile, so it does **not** reproduce HDBSCAN's ~27% density-based noise set
   (all-tiles ARI ~0.5). PCA-first / leaf-mode remain unexplored.

7. **Measurement environment.** The RTX 4060 Laptop GPU throttles on battery
   power; some Tier-3 ingestion rows are battery-measured and **understate**
   sustained AC capability (§4). The Tier 2 → Tier 3 latency slope (§10.2) may be
   partly a host-level slowdown rather than a pure FlatIP effect — reported as a
   range, not a point.

8. **Corpus scale.** "At scale" here means **101,911 tiles / ~34 GB / 14 MGRS
   tiles**. That is enough to expose the FlatIP O(n) signature, the HDBSCAN wall,
   and the persist-time growth — but it is not a national or global archive, and
   the crossover estimates in §10.3 are extrapolations, explicitly ranged.

---

## 15. Object detector

*A separate, later track (sub-metre imagery, not the Sentinel-2 archive above); everything below is summarised from the
full report in `docs/PHASE8F2.md`, which names the script and artifact behind each number.*

A **YOLO26s-OBB** oriented detector (10.5 M parameters; DOTA-pretrained, fine-tuned on DOTA v1.5 for 8 classes: small / large vehicle, ship,
plane, helicopter, storage tank, harbor, bridge) sits behind the dependency-free `ObjectDetectionModel` interface. Trained 20 epochs in
3.1 h of compute on the RTX 4060 (peak reserved VRAM 6.87 GB = the hard cap), it was evaluated **once, after training, on the official DOTA val** (458 images; disjoint source images,
never used for checkpoint or threshold selection — `docs/PHASE8F2.md` §6.1).

| DOTA official val, v1.5 labels, full-image DOTA-devkit protocol (95 % bootstrap CI over images) | AP50 | AP50:95 |
|---|---:|---:|
| ground vehicles (small + large) | 0.854 [0.786–0.891] | 0.471 |
| ships + aircraft | 0.859 [0.735–0.911] | 0.557 |
| infrastructure (tanks, harbors, bridges) | 0.751 [0.704–0.784] | 0.414 |
| all 8 classes | 0.817 [0.764–0.845] | 0.482 |

Reported as measured: vehicles do **not** score lower than infrastructure at IoU 0.5, but small vehicles are the weakest at strict localisation
(AP50:95 0.413), have the lowest precision at the operating point (0.597) and collapse below ~16 px (AP50 0.42 for 10–16 px cars). The
fine-tuning gain is concentrated in one epoch and one class (small-vehicle AP50 0.606 → 0.845, the effect of correcting the labels from v1.0 to v1.5).
On the staged **Maxar** tiles (no ground truth) the detector reliably finds aircraft and other large objects but only a few percent of visible cars at its
DOTA-calibrated operating point; a controlled blur explains only ~4 points of that gap, so the vehicle numbers above do **not** transfer to that product
(`docs/PHASE8F2.md` §7). The evaluator's own disagreement with Ultralytics' validator was measured and attributed (§6.8).
Licence: `ultralytics` and the weights are AGPL-3.0 (optional extra, isolated behind the interface); the fine-tuned weights are non-commercial.

---

## 16. Offline guarantee — what is claimed, and how it is checked

**What is being audited.** The analyst interface is a React 19 + TypeScript single-page console built with Vite. **It has a build
step** (`cd frontend-react && npm ci && npm run build`) and is served at `/app/` by the same FastAPI process as the API. It replaced an
earlier hand-written vanilla-JavaScript interface, which has been deleted. The built output is **committed**
(`src/geoseek/analyst/web_react/`), so running the system needs no Node, and every emitted file is byte-pinned by SHA-256
(`frontend-react/build-pins.json`). The Docker image has no Node stage and ships the committed bundle. The third-party files the
build bundles live in `src/geoseek/analyst/vendor/`. Details: `docs/FRONTEND_REACT.md`.

**Claim: zero external network requests at runtime.** It is *not* "zero external URL strings in shipped assets":
vendored third-party libraries (and the bundle built from them) legitimately contain attribution banners, XML namespace identifiers,
browser-bug citations and diagnostic text. A reader who greps `vendor/` or the built bundle will find them; none is ever fetched.

**What is in the vendored assets** (`src/geoseek/analyst/vendor/`; found by the offline guard, classified one by one,
10 flagged occurrences = 6 distinct URLs, plus 1 relative `sourceMappingURL` comment and one regex-source fragment):

| file | URL string | class |
|---|---|---|
| `leaflet.css` | `bugs.chromium.org/p/chromium/issues/detail?id=600120`; `bugzilla.mozilla.org/show_bug.cgi?id=888319` | other — comments citing browser bugs |
| `leaflet.js` | `https://leafletjs.com` (header) | attribution comment |
| `leaflet.js` | `https://leafletjs.com` (inside a string) | other — `<a href>` in the default attribution-control HTML; never rendered, the app sets `attributionControl: false` |
| `leaflet.js` | `http://www.w3.org/2000/svg` (×3) | other — XML namespace identifier, never dereferenced |
| `leaflet.js` | `sourceMappingURL=leaflet.js.map` | sourcemap comment (as above) |
| `three.module.min.js` | `http://www.w3.org/1999/xhtml` | other — XML namespace identifier |
| `three.module.min.js` | `https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733` (×2) | other — text inside a `console.warn` |
| `three.module.min.js` | `https?://` (regex source, ×2) | other — URL-detection regex text, not a URL |

**What is in the built bundle** (`src/geoseek/analyst/web_react/`): nine distinct inert URL strings (ten file-level occurrences), each
classified in `geoseek.staging.react_build_pins.ALLOWED_URLS` and enumerated per file in `build-pins.json`: five W3C XML-namespace
identifiers, a React error-documentation string, a three.js console-warning string, a geotiff.js error-message string and Leaflet's
attribution `href`. None is dereferenced.

**Live runtime fetches: none.** This was established by tracing network *sinks*, not by grepping for `http`, and then measured. Static:
the console's own source (`frontend-react/src`) contains no external URL at all (scanned with no allowlist); the page's
Content-Security-Policy is `default-src 'self'` with `connect-src 'self'`, so the browser itself refuses any other origin; Leaflet is
constructed with `attributionControl: false`. The console has **exactly one Leaflet tile layer and it is same-origin**:
`/ui/basemap/{z}/{x}/{y}`, rendered by this server (`src/geoseek/analyst/basemap.py`) from imagery already in the archive. It is not a
tile service, and the test fails if a second tile layer or any external template appears. Candidate imagery is drawn as same-origin PNGs,
and the three.js textures are bundled same-origin assets (`/app/assets/day-*.jpg`, `night-*.jpg`). The generic loaders inside the
libraries (`fetch(`, `.src =`) only ever receive URLs the app passes them. Measured: headless Chrome over the DevTools protocol
(`frontend-react/tools/verify-offline.mjs`) visited 16 routes (every screen, deep links to a candidate, the Brief alias and an unknown
route) and logged 196 requests, **0 external**, 0 CSP blocks; the 98-step e2e run made 2,495 requests, **0 external**; and
`scripts/verify_offline_perf.py` runs the whole API with non-loopback sockets hard-disabled in process (see `RUN.md`). Figures are from
the run that retired the original interface and vary with the run.

**How it is kept true.** `tests/test_frontend_offline.py` scans every text file under `analyst/vendor/`, requires each vendored file's
URL strings to be enumerated in the provenance manifest, and (since Phase 10) requires **every** vendored file other than the fonts —
URL-bearing or not — to be pinned by SHA256 (`OrbitControls.js` previously passed a hash check it was never subjected to). For the
React build it byte-pins every emitted file, classifies every URL string, checks the CSP, scans the app source, locks the toolchain and
requires `index.html` to reference only `/app/…`. `tests/test_console_mount.py` guards what `/app/` can serve. The allowlist content for
the vendored files is committed in `geoseek.staging.vendor_provenance` and regenerated offline with
`python -m geoseek.staging.vendor_provenance`, because the manifest itself is git-ignored.

**A portability trap found on the way.** With `core.autocrlf=true` (the Windows default) a checkout rewrites the vendored
JavaScript with CRLF line endings: `three.module.min.js` hashed `8acd07f8…` on disk against `3e690ac7…` for the committed
(upstream) bytes. A hash allowlist recorded from such a working tree fails on every LF checkout (Linux CI, a fresh clone,
a Docker build). `.gitattributes` now marks `src/geoseek/analyst/vendor/** -text` (the directory was `analyst/web/vendor/` when this was found; it
was moved when the original interface was retired, byte-identically), so vendored bytes are identical on every platform and the pinned
hashes are the upstream/blob hashes.

**Known gap — font provenance.** The seven web fonts the console ships (Inter ×4, JetBrains Mono ×3) are **not in the provenance
manifest**: `scripts/stage_fonts.py` is written to record them, but the manifest holds no entry for any font, so there is no recorded
source URL or licence line tied to their hashes. They are pinned only through the React build pins, which proves the shipped bytes cannot
change unnoticed but not where they came from. The source and licence the script states (Inter v4.1, JetBrains Mono v2.304, SIL OFL 1.1)
have not been re-verified against these bytes, and no manifest entries were generated to hide that. Two further fonts (Cinzel), with no
recorded source at all, were unused by every interface and were removed.

---

*Generated for Phase 7c. Artifacts referenced: `data/eval_retrieval/*.json`,
`data/change_model/ablation_study.json`, `data/change_model/ayodhya_change_report.json`,
`data/discovery/*.json`, `data/provenance_manifest.json` (sections named per
section above). Companion document: `docs/ARCHITECTURE.md`.*

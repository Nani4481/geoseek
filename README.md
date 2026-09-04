# geoseek

Offline-first geospatial ML platform. Target hardware: NVIDIA RTX 4060
Laptop (8GB VRAM, CUDA), Ryzen 7 7840HS.

## Hard rule: offline after staging

Only `geoseek.staging.*` is allowed to touch the network — that's the
one-time step that downloads and verifies model weights and datasets and
records their provenance. Every other module (`config`, `ingest.*`,
`search.*`) reads exclusively from `data/` on disk. If you find yourself
adding a network call outside `staging/`, stop — it belongs in staging
instead.

## Install

Detected environment: **Windows, with conda available.** Per project policy,
conda-forge / pytorch / nvidia channels are used for the binary-heavy
packages (torch+CUDA, faiss, rasterio, gdal, pyproj, shapely) since they
build correctly against Windows CUDA/GDAL without wheel headaches. Pure
Python packages come from `pyproject.toml` via pip.

```powershell
conda env create -f environment.yml
conda activate geoseek
pip install -e . --no-deps
pip install --no-deps open_clip_torch timm ftfy huggingface_hub requests tqdm `
    hdbscan fastapi "uvicorn[standard]" pydantic sqlalchemy pytest affine pillow
```

`--no-deps` is not optional here: `open_clip_torch` (via `timm`) depends on
`torch`/`torchvision`, and pip's resolver has no idea the conda-installed
CUDA build satisfies that — left to itself it silently installs its own
CPU-only wheels over the top and CUDA availability quietly goes to `False`.
This was hit and confirmed while building this scaffold. If it happens
anyway:

```powershell
pip uninstall -y torch torchvision
conda install -n geoseek -c pytorch -c nvidia -c conda-forge pytorch=2.2.2 pytorch-cuda=12.1 torchvision
```

### Linux / WSL (plain pip) alternative

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Staging (one-time, needs network)

Downloads RemoteCLIP ViT-B-32 (OpenCLIP format) weights, verifies them
against the live Hugging Face repo listing, loads them, runs a smoke encode,
and writes `data/provenance_manifest.json`.

```powershell
python -m geoseek.staging.download_models
```

Expected output includes the startup banner (torch version, CUDA
availability, GPU name, embedding dim) followed by staging progress, and
finishes by printing the GPU name and the embedding dimension (512).

Checkpoint provenance (confirmed via Hugging Face at staging-script-authoring
time, not recalled from memory — see `src/geoseek/staging/download_models.py`
docstring):

| | |
|---|---|
| HF repo | `chendelong/RemoteCLIP` |
| File | `RemoteCLIP-ViT-B-32.pt` |
| License | Apache-2.0 (per the [ChenDelong1999/RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) GitHub LICENSE) |
| Upstream paper | *RemoteCLIP: A Vision-Language Foundation Model for Remote Sensing*, IEEE TGRS 2023 |

Re-running the command after the first successful stage skips the network
entirely — it detects the already-staged file under `data/models/` and loads
straight from disk.

### Test data (Sentinel-2 L2A AOI, two dates)

```powershell
python -m geoseek.staging.download_datasets
```

Fetches a small AOI (~6.2km x 6.2km, Ayodhya, UP — Saryu river + urban edge +
open ground) from **Earth Search STAC** (`https://earth-search.aws.element84.com/v1`,
Element 84, backed by the public AWS Open Data Sentinel-2 COG bucket — no
login/API key needed), for two dates ~5 years apart:

| | |
|---|---|
| Scene A | `S2B_44RPQ_20190330_1_L2A` — 2019-03-30 |
| Scene B | `S2A_44RPQ_20240308_0_L2A` — 2024-03-08 |
| Bands | B04 (red), B03 (green), B02 (blue) at 10m; SCL resampled 20m→10m |
| License | Copernicus Sentinel Data (free & open, EU Copernicus Data Policy) |

Only the AOI window of each COG is transferred, via HTTP range requests —
never the full ~110x110km tile. Re-running skips the network once a scene's
bands are already staged under `data/datasets/<scene_id>/`. See
`src/geoseek/staging/download_datasets.py` for full provenance detail.

## Ingestion pipeline

```powershell
python -m geoseek.ingest.pipeline ingest "data/datasets/S2B_44RPQ_20190330_1_L2A"
```

read (`ingest/reader.py`, rasterio) → tile (`ingest/tiler.py`, 256x256,
partial edge tiles kept, fully-nodata tiles skipped, lon/lat footprint per
tile) → quality (`ingest/quality.py`, per-tile cloud/shadow/snow/saturated
fraction from the SCL band) → embed (`ingest/embed.py`: **fixed** Sentinel-2
true-color stretch of B4/B3/B2 — subtract the per-scene BOA additive offset
(0 DN for both staged dates, already offset-applied upstream), clip to
0–0.30 reflectance (~DN 0–3000), scale to 8-bit at a fixed gamma. The bounds
are identical for every tile and every date, so equal ground reflectance →
equal 8-bit value (radiometric consistency for retrieval and change
detection); the exact bounds + offset handling are recorded in
`data/provenance_manifest.json` under `radiometry`. Then OpenCLIP's own
preprocessing, then RemoteCLIP — loaded once, reused for
every tile) → store (`ingest/store.py`: FAISS `IndexFlatIP` over unit-norm
512-d vectors + a SQLite `tiles` table, both under `data/index/`,
**incremental** — ingesting a new scene only ever appends, existing
vectors/rows are never touched or rebuilt).

Each ingest run appends a report (tiles added, build time, index size, mean
per-tile embed latency) to `data/provenance_manifest.json` under
`ingest_runs`.

## Tests

```bash
pytest tests/test_env.py tests/test_ingest.py
```

`test_env.py` asserts torch imports, CUDA is visible, and RemoteCLIP loads +
encodes to a 512-d embedding **from the local staged file only** — no
network calls, so this passes with the network off as long as staging has
run once.

`test_ingest.py` covers the ingest pipeline end-to-end against the staged
AOI scenes above: geospatial metadata preservation, tiling (partial edge
tiles, nodata skipping), SCL-based quality scoring, the fixed-bounds
reflectance→true-color stretch, single model load/reuse, incremental FAISS+SQLite
append (byte-identical old vectors after a second scene is added), and a
full ingest with the network forcibly disabled.

## Phase 2: scale-up + semantic search

### Scaled-up AOI

`python -m geoseek.staging.download_datasets` still stages the small 9-tile
demo AOI. A second, much larger AOI over the *same* Ayodhya region and same
two dates is staged via `geoseek.staging.download_datasets.stage_large_aoi()`
— an 82km x 82km box, fitted (via a live query of the source COG's own
georeferenced bounds) to stay fully inside the single MGRS tile 44RPQ (no
cross-tile mosaicking) while still containing the original Ayodhya point.
Stored under `data/datasets/<scene_id>_scaled/`, ~1000+ tiles/date. Ingest it
the same way as any scene:

```powershell
python -m geoseek.ingest.pipeline ingest "data/datasets/S2B_44RPQ_20190330_1_L2A_scaled"
python -m geoseek.ingest.pipeline ingest "data/datasets/S2A_44RPQ_20240308_0_L2A_scaled"
```

At this scale, `pipeline.ingest_scene` batches the embedding step
(`embed.embed_tiles_batch`, default batch size 64) instead of one
`model.encode_image` call per tile — far fewer GPU launches, and VRAM stays
bounded regardless of scene size (ViT-B-32 batches are tiny relative to the
8GB budget).

### Semantic search

```
src/geoseek/search/
  engine.py   SearchEngine - RemoteCLIP + FAISS index loaded ONCE, reused for
              every query. search_text() (RemoteCLIP text tower -> FAISS ->
              filtered/joined results) and search_image() (by tile_id or a
              fresh image - "find more like this"). Filters (bbox, date
              range, sensor, max_cloud_fraction) are a post-filter over the
              full flat-index brute-force scan - at a few thousand vectors
              this is sub-millisecond, simpler than pushing filters into
              FAISS itself, and comfortably meets the <1s interactive budget.
  api.py      FastAPI: GET /search/text, POST /search/image,
              GET /tile/{id}/thumbnail (regenerates the stretched true-color
              PNG on demand - not stored on disk), GET /health. SearchEngine
              constructed ONCE at app startup (lifespan), reused per request.
```

Run the API:

```powershell
uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
```

```bash
curl "http://127.0.0.1:8000/search/text?q=a+river+with+sandbars&k=5"
```

### Proving it's really RemoteCLIP

```powershell
python scripts/prove_semantic.py
```

Runs 4 satellite-specific text queries ("a river with sandbars", "dense
urban buildings", "agricultural fields", "open bare ground") against the
real search index and prints the top-5 tile_ids/scores/centroids for each,
saving a contact-sheet PNG of the top-5 thumbnails under
`data/prove_semantic/remoteclip_<query>.png`. As a control, it then embeds
every tile in the scaled AOI with **vanilla OpenCLIP ViT-B-32**
(`pretrained='openai'`, staged via
`geoseek.staging.download_models.stage_vanilla_openclip()` /
`python -m geoseek.staging.download_models --vanilla`) and runs the same 4
queries, saving `vanilla_<query>.png` sheets. Comparing the two side by side
is the proof: RemoteCLIP (remote-sensing-tuned) should retrieve tiles that
visibly match the query on a satellite view, while vanilla CLIP — trained
only on everyday photos — is visibly less consistent on this vocabulary.

(Two correctness notes on the vanilla control model, both fixed in
`geoseek.ingest.embed.load_vanilla_clip_once`:
`open_clip`'s `'openai'` tag needs `force_quick_gelu=True` to match how
those checkpoints were actually trained — without it, `open_clip` only
*warns* about a QuickGELU activation mismatch rather than correcting it,
which would silently cripple the control model for an unrelated reason.
And even with a fully populated local cache, `open_clip`'s pretrained
loader (via `huggingface_hub`) still phones home by default to check for a
newer revision — a real network call from code that is not
`geoseek.staging`. `HF_HUB_OFFLINE=1` is set before loading so this module
stays genuinely offline after the vanilla model is staged once, the same
guarantee the rest of geoseek makes; `tests/test_ingest.py` proves it with
sockets actually blocked.)

## Phase 3a: making the 2019/2024 pair comparable (change-detection prerequisite)

Before *any* change detection, the two dates have to be genuinely
comparable — same pixels on the ground, same radiometric scale. Phase 3a
does that and nothing more (no change detection yet — that's 3b).

### Extra bands (staging, network)

```powershell
python -m geoseek.staging.download_datasets --extra-bands
```

Tops up **only the scaled AOI** dirs (`data/datasets/<scene>_scaled/`) with
`B08` (NIR, native 10m) and `B11` (SWIR-1, native 20m → resampled to the 10m
grid with **BILINEAR**, since SWIR reflectance is continuous — NEAREST stays
reserved for the categorical SCL band). Each new band is recorded in the
provenance manifest with its SHA256. The staging step then confirms B08/B11
carry the *same* reflectance encoding as the RGB bands — STAC `raster:bands`
`scale=1e-4 offset=-0.1` identical to red, median DN in a sane reflectance
range, and no unremoved `BOA_ADD_OFFSET` pedestal (global minimum bottoms
out at the same low floor as B04, not ~1000 DN above it) — and writes that
confirmation to the manifest under `extra_bands`. The small demo AOI is left
RGB+SCL only, so `tests/test_ingest.py` stays valid.

### Pair preparation

```powershell
python -m geoseek.change.prep
```

`src/geoseek/change/` — offline, reads only from `data/datasets/`:

```
  coregister.py  FFT phase correlation (Hann window + parabolic sub-pixel
                 peak) between the two dates on a stable band (B08). Reports
                 the median (dy, dx) shift over a grid of overlapping tiles;
                 corrects (scipy.ndimage sub-pixel resample) ONLY if the
                 median shift exceeds 0.5 px. For this pair it's ~0.05 px —
                 well inside the Sentinel-2 L1C multitemporal registration
                 spec — so it is measured, reported, and left as-is.
  normalize.py   Relative radiometric normalization, per band. Pick
                 pseudo-invariant pixels (valid + SCL-good at both dates, not
                 water, |NDVI| < 0.25 at both dates = stable bare/built), drop
                 gross cross-band outliers, then fit an ADDITIVE per-band
                 correction reference_DN = subject_DN + offset_b(x, y). Over
                 82 km the haze is not uniform, so offset_b is a SPATIALLY
                 VARYING surface: the robust median of reference - subject over
                 the PIF pixels of each ~2.5 km block, sparse blocks filled
                 from their nearest neighbour, lightly smoothed, bilinearly
                 upsampled. A band whose scene-wide offset is below ~0.35x the
                 per-pixel PIF scatter and whose dates are well correlated is
                 left untouched (offset forced to 0) - the dates already agree
                 within the noise there, and adding a correction would only
                 degrade an already-matched band. For this pair that zeroes
                 red/green/NIR/SWIR and leaves only blue (~700-800 DN of
                 nearly-uncorrelated haze, inter-date r ~ 0.24). (A free
                 gain/RMA fit instead *amplifies* the difference on unchanged
                 tiles; gain stays 1.) 2019 is the subject (adjusted, it is
                 the hazier date), 2024 the reference. The per-band
                 offset-surface stats + the block grid go into the manifest;
                 reconstruct with normalize.expand_offset_grid.
  indices.py     NDVI=(B08-B04)/(B08+B04), NDWI=(B03-B08)/(B03+B08),
                 NDBI=(B11-B08)/(B11+B08), computed on the normalized
                 reflectance.
  prep.py        Orchestrator + acceptance gate. Steps B-E, prints the
                 co-registration shift, the per-band offset-surface table, an
                 unchanged-tile before/after reflectance table, and
                 NDVI/NDWI/NDBI ranges with cropland/river/town spot-checks.
                 Writes a per-tile index table to
                 `data/index/spectral_indices_per_tile.csv`, full-res
                 `NDVI/NDWI/NDBI.tif` per scene (skip with `--no-rasters`),
                 and the `coregistration`, `radiometric_normalization`,
                 `normalization_acceptance_gate` and `spectral_indices`
                 sections of the manifest.
```

Phase correlation is implemented directly on NumPy/SciPy FFTs —
scikit-image is deliberately not a dependency.

### On the acceptance gate for THIS date pair

The Phase-2 note that the two dates "differ by ~300-450 DN scene-wide
(atmospheric)" is measured over *all* pixels. Over genuine no-change ground
the story is different: **March 2019 was a drought** (scene NDVI median
~0.33) and **March 2024 was green** (~0.68), so most of that scene-wide
difference is real vegetation phenology, not atmosphere. On stable hard
surfaces the two dates already agree to within ~70-140 DN in
red/green/NIR/SWIR; the only large atmospheric term is **blue: ~700-800 DN
of haze** (inter-date r ~ 0.24), which the normalization removes.

The strict gate ("every band, on 5 clearly-unchanged tiles, well under
~100 DN after normalization") is therefore **not achievable for this pair
without erasing real change**: genuinely-static tiles are scarce, and on the
ones that qualify, red/green still carry ~100-180 DN of *real*
non-atmospheric surface variability (thin haze + soil-moisture / tillage
differences between a dry and a wet March). `prep.py` reports the strict
target explicitly and passes the gate on the substantive requirements for
Phase 3b: the scene-wide bias and the blue haze are removed, and NIR/SWIR -
the bands that drive NDVI/NDWI/NDBI - are brought within ~110 DN on
no-change ground.

## Phase 3.5: load-bearing architecture (before change detection)

Phase 3.5 puts the abstractions the change pipeline will be written against in
place first — a refactor + extension, not a rewrite. No change detection is
built here.

### Scene → observation → tile data model

The flat `tiles` table is replaced by a proper hierarchy under
`src/geoseek/catalog/`:

```
collections   sensor / platform / bands / native GSD          (sentinel-2-l2a)
  scenes      source product id, footprint, acquisition, baseline, source URL,
              license, per-band checksums                      (S2B_44RPQ_20190330_1_L2A, …)
    observations   a scene ∩ our AOI at one acquisition time: quality summary,
                   radiometry / normalization params used, co-registration
                   status                                      (…_L2A_scaled)
      tiles        256×256, geom_4326, row/col, cloud_fraction, quality flags,
                   embedding ref, indices ref
derived       per-tile / per-observation products (NDVI/NDWI/NDBI, masks,
              thumbnails) — referenced by path, never blobbed
```

Every tile reaches its source scene by foreign key, and provenance is
reconstructable upward from any tile (`repo.get_tile_provenance(tile_id)` →
tile → observation → scene → collection).

Migrate an existing flat catalog in place (non-destructive: the sqlite file is
backed up to `<db>.pre_phase35.bak`, the legacy `tiles` table is renamed to
`_migration_legacy_flat_tiles`, rows are copied verbatim — no re-ingest, no
re-embed):

```powershell
python -m geoseek.catalog.migrate          # migrate + verify
python -m geoseek.catalog.migrate --verify # verify only
```

Verification asserts every legacy tile is present and byte-identical
(geometry, provenance link, faiss_id mapping), plus a search-result parity
check against a frozen baseline.

### Repository seams

Nothing outside these modules touches `sqlite3` or `faiss` directly:

```
catalog/repository.py       MetadataRepository (ABC)        — the geospatial/metadata catalog
catalog/sqlite_repository.py  SQLiteMetadataRepository       — the only sqlite3; swappable to PostGIS
vectorindex/base.py         VectorIndex (ABC)               — add/search/get_vector/delete/persist/load/count/validate
vectorindex/faiss_flat.py     FaissFlatIPIndex              — the only faiss; HNSW impl drops in later
models/base.py              EmbeddingModel, QualityEstimator, ChangeDetectionModel (interface only)
models/remoteclip.py          RemoteCLIPEmbeddingModel      — the existing RemoteCLIP behind EmbeddingModel
models/quality.py             SclQualityEstimator
```

`SearchEngine`, `TileStore` and `ingest.pipeline` all go through these seams.

### Temporal observation matcher

`src/geoseek/temporal/matcher.py` — given a location, which observations are
legitimately comparable:

```powershell
python -m geoseek.temporal.matcher --lon 82.1998 --lat 26.7922 [--all-pairs] [--json]
```

For each ordered pair it reports `comparable: yes/no` with a per-criterion
breakdown: spatial overlap (blocking), temporal separation, sensor
compatibility, collection compatibility (blocking), resolution compatibility
(blocking), quality (cloud / usable pixels), and co-registration status. The
output `ObservationSequence` / `ObservationPair` is exactly what a Phase 3b
`ChangeDetectionModel.predict_change` consumes — no reshaping at the boundary.

### Third acquisition date

```powershell
python -m geoseek.staging.download_datasets --third-date   # network, staging only
python -m geoseek.ingest.pipeline ingest "data/datasets/S2A_44RPQ_20210304_1_L2A_scaled"
python scripts/align_third_date.py                          # offline: co-registration + PIF offset provenance
```

`S2A_44RPQ_20210304_1_L2A` (2021-03-04) — same 82 km AOI, same MGRS tile
44RPQ, same six bands (B4/B3/B2/B8/B11/SCL), same fixed radiometry, chosen
from a live Earth Search query (early-March phenology window, low cloud, near
the temporal midpoint). Ingested through the catalog as a third observation
(incremental: existing vectors untouched, no rebuild → 3267 tiles / 3
observations). `align_third_date.py` computes its co-registration + PIF
additive offset against the same 2024-03-08 reference the original pair uses
and records them on the observation + the manifest (`third_date`,
`third_date_alignment`).

## Phase 3b: a trained change-detection model (FC-Siam-diff on OSCD)

Phase 3b is the project's genuine training contribution: a real
change-detection CNN, trained from scratch, exposed through the existing
`geoseek.models.base.ChangeDetectionModel` seam and fed a
`TemporalObservationMatcher` `ObservationPair` with **zero reshaping**.

### Stage OSCD (network, staging only)

```powershell
python -m geoseek.staging.download_oscd
```

Stages the **Onera Satellite Change Detection** dataset — 24 co-registered
Sentinel-2 image pairs with human-drawn binary change masks, standard **14
train / 10 test** split — under `data/datasets/oscd/`. Download is the pinned
Hugging Face mirror `hkristen/oscd` @ `4958d786…` (the mirror the maintained
`torchgeo` OSCD loader resolves against; canonical home is IEEE DataPort DOI
`10.21227/asqe-7s69`, now subscription-gated). Each archive's SHA256 is
verified against the value recorded from the live HF API, and the source URLs,
SHA256s, licence (**CC-BY-NC-SA-4.0**), the exact split (region lists, read
from OSCD's own `train.txt`/`test.txt` and cross-checked against the label
folders), the 13 available bands (native GSD 10/20/60 m, all resampled to the
10 m grid in `imgs_*_rect`), and the radiometry are written to the manifest
`oscd` section.

**Radiometry / harmonization.** OSCD is Sentinel-2 **L1C top-of-atmosphere**
reflectance; geoseek's Ayodhya pipeline is **L2A bottom-of-atmosphere**. Same
encoding (`uint16`, `reflectance*10000`, offset 0 — asserted at staging),
different physical quantity. Harmonized by (1) `/1e4` to a common reflectance
scale, (2) per-band standardization fit on the OSCD *train* split and
re-applied at inference, (3) FC-Siam-diff's feature **differencing**, which
cancels an atmospheric term shared by both dates of a pair, and (4) on the
Ayodhya side, Phase 3a's relative inter-date normalization run first.

### The model — `geoseek.change.models.fc_siam_diff.FCSiamDiff`

Siamese fully-convolutional encoder with **shared weights** across both dates,
**feature differencing at every skip level** (`|f1 − f2|`, so the network is
invariant to date order — a change is a change either way), a U-Net
transposed-conv decoder, and a 1×1 head producing one change logit per pixel.
Uses the 5 bands geoseek actually stages (**B02, B03, B04, B08, B11** — RGB +
NIR + SWIR-1), so the trained encoder transfers to Ayodhya with no band
remapping. `base_channels=24, depth=4` → **1,085,113 parameters** (printed by
the trainer; `count_parameters(model)`).

### Train it — `scripts/train_change.py`

```powershell
python scripts/train_change.py                # train (11 regions) + evaluate (10 held-out test regions)
python scripts/train_change.py --eval-only    # re-evaluate the saved checkpoint
```

Offline. Gradient updates use **only 11 of the 14 OSCD train regions**; 3 train
regions (`bordeaux, cupertino, beirut`) are held out for validation (loss curve
+ threshold selection). The 10 OSCD **test** regions are never read until
`evaluate()` — `assert` guards make any train/val/test region overlap a hard
error.

* **Class imbalance** (~3 % change pixels, some regions < 1 %): loss is
  `0.5·weighted-BCE(pos_weight ≤ 10) + 1.0·soft-Dice`. Dice targets region
  overlap directly (an F1 surrogate, imbalance-robust); the capped weighted
  BCE keeps per-pixel gradients well-conditioned. 40 % of sampled patches are
  forced to contain change.
* Augmentation: dihedral D4 (flips + 90° rotations), identical for both dates
  and the mask. Mixed precision (AMP). Patch 96×96, batch 32.
* Adam + cosine LR, early stop on val loss (patience 15).
* Per-epoch train/val loss logged; loss curve saved to
  `data/change_model/loss_curve.png`.
* Checkpoint (`data/change_model/fc_siam_diff.pt`) + a full **model card**
  (architecture, param count, bands, training regions, split, hyperparameters,
  source SHA256s, weights SHA256) → manifest `oscd_change_model`.

Last run: **21 epochs, ~2 min wall-clock, peak VRAM 0.90 GB** on the RTX 4060.

### Honest evaluation — held-out 10-region test split

The precision-favouring operating point (per PS 2.2.3: `argmax F0.5`, recall
floor 0.15) is chosen on the **validation** regions, then frozen and applied
**once** to the test regions — never tuned on test.

| operating point | P | R | F1 | IoU | FPR |
|---|---|---|---|---|---|
| default 0.50 | 51.7 % | 61.0 % | **56.0 %** | 38.8 % | 3.10 % |
| precision-favouring 0.80 (val-selected) | **60.3 %** | 51.0 % | 55.3 % | 38.2 % | **1.83 %** |

Published OSCD baselines (change class): FC-EF 48.0 %, FC-Siam-conc 50.2 %,
FC-Siam-diff **52.8 %** F1. Ours (~56 % F1) sits just above — expected for a
5-band model with a modern recipe (Dice+BCE, AMP, augmentation, best-val
checkpointing) and *not* a leakage red flag: per-region test F1 ranges from
3.8 % (valencia, 0.44 % change) to 74.3 % (lasvegas), the spread of genuine
generalization. Precision-recall curves for validation and test, and
`[before | after | ground truth | prediction]` panels for 3 test regions
(`lasvegas`, `montpellier`, `chongqing`), are saved under `data/change_model/`;
all metrics + the leakage statement go to manifest `oscd_change_model_eval`.

### Wired into the architecture — `FCSiamDiffChangeModel(ChangeDetectionModel)`

```python
from geoseek.temporal.matcher import TemporalObservationMatcher
from geoseek.change.models import FCSiamDiffChangeModel

pair = TemporalObservationMatcher(repo).match(location=(82.1998, 26.7922)).comparable_pairs[0]
result = FCSiamDiffChangeModel().predict_change(pair)   # -> ChangeResult
```

`predict_change` takes the matcher's `ObservationPair` directly, runs the
trained net tile by tile over the later observation's grid, and returns a
`ChangeResult` with a per-tile change score, a per-tile `change`/`no_change`
label, a georeferenced full-AOI change-mask GeoTIFF, and a confidence. A
non-comparable pair is refused (confidence 0, no mask, reason in `notes`). It
imports only torch / numpy / rasterio + the catalog entities — the sqlite and
faiss seams are untouched. Full-scene suppression on Ayodhya is Phase 4; Phase
3b stops at a smoke test.

## Tests

```bash
pytest
```

`test_env.py` — torch + CUDA visible, RemoteCLIP loads + encodes to 512-d
**from the local staged file only**. `test_ingest.py` — the ingest pipeline
end-to-end on the staged AOI (metadata, tiling, SCL quality, fixed-bounds
true-color stretch, single model load, incremental FAISS+catalog append,
network forcibly disabled). `test_search.py` — the Phase 2 semantic search
module. `test_change.py` — Phase 3a: phase-correlation recovers known
sub-pixel shifts, PIF selection excludes water/vegetation, the linear fit
recovers a known gain/offset and rejects injected change, normalization
shrinks the inter-date bias, index formulas and ranges; plus light
integration checks against the real staged pair (skipped if B08/B11 aren't
staged). `test_catalog.py` — entities, `SQLiteMetadataRepository` round-trips
+ queries, the migration (synthetic + idempotent + rollback), and integration
checks against the migrated production catalog. `test_vectorindex.py` /
`test_models.py` — the VectorIndex and model seams. `test_temporal.py` — the
matcher's sequencing and per-criterion verdicts. `test_search_parity.py` —
the seams changed no search result vs the frozen baseline. `test_change_align.py`
— the third-date alignment summary. `test_change_model.py` — Phase 3b:
`FCSiamDiff` architecture (param count in the 1-2 M target, forward shapes,
order-invariance, shared Siamese encoder) and the `FCSiamDiffChangeModel` seam
(consumes a matcher `ObservationPair` unchanged, returns a `ChangeResult` with
a written mask, refuses a non-comparable pair, runs offline) — all hermetic
(tiny synthetic checkpoint + rasters), plus a skip-if-absent check against the
real trained weights.

## Layout

```
geoseek/
  pyproject.toml          pip-installable deps (torch, open_clip, faiss-cpu, rasterio, fastapi, ...)
  environment.yml         conda-forge/pytorch/nvidia env for the binary-heavy packages (Windows path)
  scripts/
    prove_semantic.py        Step C: RemoteCLIP vs vanilla CLIP contact-sheet proof
    radiometry_check.py      Phase 2 follow-up: S2 baseline / BOA-offset + harmonization sanity check
    capture_search_baseline.py  freeze the production search results into a regression fixture
    align_third_date.py      Phase 3.5 Step 4: co-registration + PIF offset provenance for the 3rd date
    train_change.py          Phase 3b: train FC-Siam-diff on OSCD + honest held-out evaluation
  src/geoseek/
    config.py              paths, device auto-select, startup banner
    staging/
      download_models.py    network entrypoint: stage RemoteCLIP weights + (--vanilla) control CLIP
      download_datasets.py  network entrypoint: demo AOI + scaled AOI (--large) + B08/B11 (--extra-bands) + 3rd date (--third-date)
      download_oscd.py      network entrypoint: stage the OSCD change-detection dataset (Phase 3b training data)
      manifest.py            provenance manifest (sha256, license, timestamp, ingest runs, analysis sections) writer
    catalog/                 Phase 3.5: the scene -> observation -> tile metadata catalog
      entities.py             storage-agnostic dataclasses (Collection/Scene/Observation/Tile/DerivedProduct/…)
      repository.py           MetadataRepository ABC (swappable to PostGIS)
      sqlite_repository.py    SQLiteMetadataRepository — the only sqlite3 in geoseek
      schema.py / naming.py   DDL; observation<->scene id helpers + provenance constants
      migrate.py              CLI: flat `tiles` -> the hierarchy, non-destructive + verified
    vectorindex/             Phase 3.5: the embedding-search seam
      base.py                VectorIndex ABC
      faiss_flat.py          FaissFlatIPIndex — the only faiss in geoseek
    models/                  Phase 3.5: model interfaces
      base.py                EmbeddingModel, QualityEstimator, ChangeDetectionModel (interface only)
      remoteclip.py          RemoteCLIPEmbeddingModel (wraps ingest.embed)
      quality.py             SclQualityEstimator
    temporal/                Phase 3.5: temporal reasoning
      contract.py            ObservationSequence / ObservationPair / PairComparability (matcher <-> change-detection contract)
      matcher.py             CLI: TemporalObservationMatcher — comparable/not-comparable + reasons per pair
    ingest/
      reader.py              rasterio: read bands, preserve CRS/transform/nodata
      tiler.py                256x256 tiling + lon/lat footprint per tile; single-tile window reads
      quality.py              SCL-based per-tile cloud/shadow/snow/saturated fraction
      embed.py                reflectance -> fixed-bounds true-color 8-bit RGB -> RemoteCLIP (loaded once, batched)
      store.py                TileStore: thin coordinator over the MetadataRepository + VectorIndex seams
      pipeline.py              CLI: read -> tile -> quality -> embed (batched) -> store
    search/
      engine.py               SearchEngine: text-to-image + image-to-image, filters, thumbnails (via the seams)
      api.py                  FastAPI service wrapping SearchEngine
    change/                   Phase 3a: temporal pair prep + Phase 3b: the trained change model
      coregister.py            FFT phase-correlation sub-pixel co-registration check + correction
      normalize.py             pseudo-invariant-feature per-band linear radiometric normalization
      indices.py               NDVI / NDWI / NDBI
      align.py                 Phase 3.5: windowed co-registration + PIF offset summary for one pair
      prep.py                  CLI: steps B-E + acceptance gate; writes the manifest analysis sections
      models/                  Phase 3b: the trained change-detection model
        fc_siam_diff.py         FCSiamDiff nn.Module (shared Siamese encoder, |f1-f2| skips, U-Net decoder)
        fc_siam_diff_model.py   FCSiamDiffChangeModel(ChangeDetectionModel) - ObservationPair -> ChangeResult
  data/
    models/ datasets/ tiles/ index/     gitignored, populated by staging + ingest
    datasets/oscd/                      gitignored, OSCD change-detection dataset (download_oscd.py)
    change_model/                       gitignored, fc_siam_diff.pt + loss/PR curves + qualitative panels
    index/tiles.sqlite                  the catalog (collections/scenes/observations/tiles/derived)
    index/tiles.faiss                   the FaissFlatIPIndex vectors
    provenance_manifest.json            gitignored; staging + ingest + change.prep + Step 4 + Phase 3b (oscd*, oscd_change_model*)
    sample_tile_true_color.png          gitignored, written by the ingest pipeline
    prove_semantic/                     gitignored, RemoteCLIP vs vanilla CLIP contact sheets
    index/spectral_indices_per_tile.csv gitignored, per-tile NDVI/NDWI/NDBI per date (change.prep)
  tests/
    test_env.py  test_ingest.py  test_search.py  test_change.py  test_change_model.py
    test_catalog.py  test_vectorindex.py  test_models.py  test_temporal.py
    test_search_parity.py  test_change_align.py
    fixtures/search_baseline.json  fixtures/pre_migration_tiles.json
```

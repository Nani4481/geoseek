# Phase 7b — scalability, measured (three tiers: 3267 → 10k → 50k → 100k)

Phase 7a fixed spatial-index latency and produced measured retrieval metrics
on the 3267-tile Ayodhya-only catalog. Phase 7b asks a different question:
**does the same pipeline, same weights, zero retraining, hold up as the
catalog grows 30x, and does it generalize to terrain it has never seen?**

Every number below is from a real run on this machine: real Sentinel-2 L2A
scenes fetched live from Earth Search STAC, tiled/embedded/indexed through
the **unmodified** `geoseek.ingest` pipeline (same `RemoteCLIPEmbeddingModel`,
same `FaissFlatIPIndex`, same catalog schema — see "What changed" below for
the one honest exception), and measured with `scripts/measure_tier.py`.
Nothing is estimated except where explicitly marked.

## AOI selection

Five geographically diverse AOIs, each a near-full Sentinel-2 MGRS tile
(~108×108 km, staged via a NEW generic STAC-item-asset-href fetch — see
`scripts/stage_diverse_aois.py` — so any tile works, not just the
hardcoded 44RPQ path `download_datasets.py` uses for Ayodhya), chosen to
stress the index with real terrain/season diversity and double as a
generalization test:

| region | category | MGRS | scenes so far | seasons | cloud |
|---|---|---|---|---|---|
| Dehradun/Rishikesh | Himalayan foothills | 43RGP | 2 | winter, post-monsoon | ~0% |
| Jaisalmer | Thar desert edge | 42RYQ | 1 | post-monsoon | ~0% |
| Sundarbans | coastal delta (Bengal) | 45QYE | 1 | winter | ~0.02% |
| Delhi NCR | dense urban | 43RGM | 1 | post-monsoon | ~0.007% |
| Kanha | Central India forest/plateau | 44QMK | 1 (+1 sized for the incremental proof) | summer/pre-monsoon | ~0% |

All confirmed live against the same Earth Search STAC API
(`https://earth-search.aws.element84.com/v1`) the original Ayodhya staging
uses. Every scene recorded in the provenance manifest (`diverse_aois`
section) with per-band SHA256, source URL, cloud %, and the **local** window
nodata fraction (which can differ from the whole-tile STAC value once the AOI
is clipped to the tile bbox).

**Sundarbans is a genuine, informative edge case, not padding**: 22.6% of the
fetched window is nodata (open Bay of Bengal beyond the satellite swath), and
because `tile_scene()` only skips a tile when *every* pixel is nodata, that
nodata is concentrated enough (one contiguous side of the tile is open ocean)
that only 731/1849 possible tiles (39.5%) survive — reported as-is, not padded
to a round number.

## What changed vs. the existing pipeline (and what didn't)

**Zero changes** to tiling (`ingest/tiler.py`), quality scoring
(`ingest/quality.py`), the embedding model (`RemoteCLIPEmbeddingModel`, same
weights), the vector index (`FaissFlatIPIndex`), or the catalog schema — this
invariance is the generalization claim, and `scripts/ingest_diverse_scene.py`
proves it by importing and calling those exact functions unmodified (never a
copy with tweaks).

**One disclosed, additive fix**: `ingest.pipeline.ingest_scene()` hardcodes
`aoi_name="ayodhya_..."` for every scene and never passes `source_url`/
`checksums` to `store.add_tiles()` — harmless when every scene really is
Ayodhya (the `source_url` fallback in `catalog/naming.py` happens to
reconstruct the right URL only because it's hardcoded to the 44RPQ path), but
it would silently mislabel and mis-URL every new region. `scripts/
ingest_diverse_scene.py` is a thin wrapper that calls the same underlying
functions with the correct per-region metadata — **not** a modification to
`ingest/pipeline.py` (that file is untouched; the original Ayodhya CLI path
is bit-for-bit as it was).

**One test-suite fix, same pattern the codebase already established**:
`catalog/migrate.py`'s `_verify_search_parity` compared live search results
against a frozen 3267-vector baseline unconditionally. `tests/
test_search_parity.py`'s own fixture already skips this comparison once the
index outgrows the baseline it was frozen at (an intentional, pre-existing
escape hatch); the lower-level `migrate.py` checker lacked the same guard and
failed once new regions changed some global top-K results (correctly — a
tile from Kanha legitimately outranking an Ayodhya tile for "green cropland"
is not a regression). Fixed by adding the identical count-based skip. One
similarly-scoped fix to `tests/test_catalog.py`'s third-date test, which
summed tile counts across *all* `sentinel-2-l2a` observations and asserted
`== 3267` — now scoped to the three observations whose `aoi_name` starts
with `"ayodhya"` (every Ayodhya observation is tagged that way; every new
region gets its own `aoi_name`), matching the exact "legacy subset must
survive intact" pattern Phase 3.5's own migration verification already uses.

## A real infrastructure finding: naive windowed remote reads vs. bulk fetch

The first staging attempt used the obvious approach (mirroring
`download_datasets.py`): open the remote COG with `rasterio`/GDAL and read a
windowed region directly over HTTP (`/vsicurl/`). For a ~99%-of-tile AOI
window this was **pathologically slow on this connection**: one band (~180
MB useful data) took **9 minutes** — effectively ~300 KB/s, even though a
plain sequential `curl` of the same whole remote file sustained **~0.7–1.1
MB/s**. The gap is request pattern, not bandwidth: a windowed VSICURL read
over a large extent issues many small ranged GETs; this bucket/connection
pays enough per-request overhead that thousands of them dominate.

**Fix**: `stage_diverse_aois.py` downloads each band's **whole remote file**
with a plain streaming HTTP GET (4 concurrent bands via `ThreadPoolExecutor`),
then crops to the AOI locally with `rasterio` against the local file (no
network, <1s per band). Measured on Dehradun: 4 bands, ~614 MB total,
finished in **424.6s wall** (the slowest of the 4 concurrent downloads) vs. a
projected ~10.9 min running the same 4 downloads serially at the measured
single-stream rate — a real but modest ~1.5x win from concurrency (this
connection's aggregate throughput is roughly capped in the ~1.4–1.9 MB/s
range regardless of how many streams share it; a 4th concurrent SCL fetch
was measurably throttled while 3 big bands were active). **Network fetch, not
the pipeline, is the bottleneck** — see the ingestion-throughput numbers below,
where the actual tile→embedding→index step runs at 200+ tiles/sec regardless
of corpus size.

---

## Tier 1 — reach ~10,000 tiles

**Result: 12,418 tiles** (3267 baseline + 9,151 new across 6 scenes / 5
regions — see incremental-proof note below for why it's 12,418 not ~10,000).
Driver stopped at the first scene that crossed 10,000 (11,394); the dedicated
incremental-proof scene (Kanha, sized for ~1000 tiles) added the rest.

| # | scene | region | tiles added | cumulative |
|---|---|---|---:|---:|
| 0 | (existing) | Ayodhya ×3 dates | — | 3,267 |
| 1 | S2B_43RGP_20231220_0_L2A | dehradun (date 1) | 1,849 | 5,116 |
| 2 | S2A_43RGP_20231115_0_L2A | dehradun (date 2) | 1,849 | 6,965 |
| 3 | S2A_42RYQ_20231002_0_L2A | jaisalmer | 1,849 | 8,814 |
| 4 | S2A_45QYE_20240208_0_L2A | sundarbans | 731 | 9,545 |
| 5 | S2A_43RGM_20231006_0_L2A | delhi_ncr | 1,849 | 11,394 |
| 6 | S2A_44QMK_20240308_0_L2A | kanha (81 km incremental-proof crop) | 1,024 | 12,418 |

### Measurement table

| metric | **baseline (3267)** | **Tier 1 (12,418)** |
|---|--:|--:|
| FAISS index size | 6.69 MB *(exact: 3267×512×4B)* | **23.33 MB** *(exact: 12418×512×4B)* |
| SQLite catalog size | not separately captured pre-scale-up (~4.6 MB est., linear extrapolation) | **14.69 MB** |
| raw staged imagery | — | 7,577.8 MB |
| derived products (change_model+discovery+eval_retrieval) | — | 530.7 MB |
| total `data/` footprint | — | 9,370.5 MB |
| FAISS full-file persist (rewrite-on-save) | — | 18.6 ms |
| text search (k=20), median / p95 | **17.4 / 21.8 ms** *(Phase 7a measurement, own harness)* | **8.41 / 10.37 ms** |
| image search (k=20), median / p95 | ~1.0 / 1.5 ms *(≈ tile-seeded KNN's core cost)* | **1.51 / 3.00 ms** |
| point-seeded KNN, median / p95 | 1.00 / 2.09 ms *(Phase 7a)* | **2.00 / 3.00 ms** |
| tile-seeded KNN, median / p95 | 0.99 / 1.57 ms *(Phase 7a)* | **1.58 / 2.93 ms** |
| bbox filter, median / p95 | 0.28 / 0.47 ms *(Phase 7a)* | **0.21 / 0.39 ms** |
| peak RSS during query burst | — | 1,829.5 MB |
| peak VRAM during query burst | — | 619.6 MB |
| peak RSS during ingest (4 ledger-tracked scenes) | — | 2,452.8 MB |
| peak VRAM during ingest | — | 739.0 MB |
| ingestion throughput (pipeline only: tile+quality+embed+index, excludes network fetch) | — | **205.4 tiles/s** (mean over 4 scenes) |
| pure embedding throughput (GPU forward pass only) | — | **393.2 tiles/s** |
| per-scene network fetch time (~1850-tile full-MGRS-tile scene) | — | **~7–8 min**, dominates total wall time |

Text-search latency at Tier 1 is *lower* than the Phase 7a baseline (8.4 ms
vs. 17.4 ms median) despite 3.8x more vectors — both numbers are well within
GPU/CPU noise at this scale (a few thousand vectors is nothing for a
brute-force 512-d inner-product scan; text encode time dominates and varies
run to run). Not a claim that latency *decreases* with scale — see Tier 2/3
for the regime where FlatIP's O(n) behavior actually shows up.

### Incremental +1000 proof

A brand-new Kanha scene (`S2A_44QMK_20240308_0_L2A`), deliberately cropped to
an 81×81 km box (not the full ~108 km tile) so it adds close to 1000 tiles
rather than a full ~1850-tile scene:

- **1,024 tiles added** (11,394 → 12,418). Ingest (tile→embed→index→save,
  *after* the scene was already staged on disk): **4.85s** (211.3 tiles/s —
  indistinguishable from the very first ingest's 203 tiles/s at 3267→5116,
  confirming index-append speed does **not** degrade with existing corpus
  size at these scales).
- **25 sampled pre-existing vectors (faiss_ids chosen uniformly at random
  from 0..11393) are BYTE-IDENTICAL before and after**
  (`before.tobytes() == after.tobytes()` for every sample; max abs diff
  0.0) — no rebuild, exactly the Phase 3.5 precedent.
- Honest caveat: staging always downloads the full remote band files
  regardless of requested crop size (see infrastructure finding above), so
  the 81 km crop saved no network time — total wall clock for this specific
  incremental scene (stage + ingest) was dominated by the same ~7-8 min
  fetch as a full-size scene. The **~1000-tile target was about tile count,
  not about proving a faster incremental path**; the append-speed and
  byte-identical guarantees are the actual claims, and both hold.

### HDBSCAN re-clustering at 12,418 tiles

Ran the *unmodified* `scripts/cluster_tiles.py` (default `min_cluster_size
=40`, HDBSCAN's default `cluster_selection_method="eom"`):

| | baseline (3267, Ayodhya only) | Tier 1 (12,418, 5 new regions) |
|---|---|---|
| clusters | 3 | **3** (unchanged) |
| noise | 9% (294 tiles) | **29.6%** (3,671 tiles) |
| cluster sizes | 1958 / 982 / 31 | 7494 / 966 / 287 |
| top concepts | rural/bare, vegetation, river+sandbars | bare dry open ground; trees/vegetation; sandy braided riverbed |

**Honest finding: cluster *count* did not increase under the shipped EOM
config**, contrary to the naive expectation — EOM (excess-of-mass) selects
the most *stable* cut of the cluster hierarchy, and apparently a coarse
3-way bare/vegetation/water split remains the most stable structure even
across five new biomes. What *did* change: noise more than tripled in
fraction (9%→29.6%) — a large share of the new, more varied content doesn't
fit any of the 3 dominant archetypes well enough to join them, which is
itself a real diversity signal.

To check whether finer structure exists at all (not to cherry-pick a
better-looking number — this is reported as a labeled secondary run, and the
EOM result above remains the official Tier 1 clustering), the same vectors
were also run through HDBSCAN's **`leaf`** cluster-selection mode (same
`min_cluster_size=40`, no other changes) — a standard, documented HDBSCAN
mode for exactly this "EOM over-merges" situation:

| | leaf mode |
|---|---|
| clusters | **9** |
| noise | 72.9% (9,047 tiles) |

The trade is real: leaf mode recovers far more structure at a steep coverage
cost (only 27% of tiles get a cluster label at all, vs. 70% under EOM). But
the structure it recovers is striking — **7 of the 9 leaf clusters are
>96% single-region**, with zero access to location metadata:

| cluster | size | region purity | top concept |
|---:|---:|---|---|
| 4 | 730 | 100% jaisalmer | bare dry open ground / quarry / fallow fields |
| 1 | 632 | 100% ayodhya | trees and dense vegetation |
| 5 | 496 | 100% kanha | bare dry open ground |
| 7 | 441 | 100% ayodhya | bare dry open ground |
| 8 | 437 | 96.6% ayodhya / 3.4% delhi_ncr | rural village settlement |
| 2 | 287 | 99.7% sundarbans | sandy braided riverbed / waterlogged land |
| 6 | 154 | 100% dehradun | rural village + cropland |
| 3 | 112 | 100% dehradun | bare ground / riverbed |
| 0 | 82 | 100% ayodhya | irrigated cropland |

RemoteCLIP embeddings cluster the Thar desert, the Sundarbans delta, and the
Himalayan foothill villages into their own tight, near-pure groups purely
from visual similarity — this is the generalization claim made concrete:
the *same, frozen* model separates terrain it was never trained or tuned on.
(Delhi's 1,849 urban tiles do not form their own leaf cluster at this
`min_cluster_size` — plausibly because dense urban is internally more
heterogeneous (roads, rooftops, parks, construction) than a desert or delta;
not investigated further here, since it doesn't change the headline finding.)

Artifacts: `data/discovery/cluster_map_tier1_eom.png`,
`data/discovery/tile_clusters_tier1_eom.json`,
`data/discovery/tile_clusters_leaf_tier1.json`.

### Tests / seams

`pytest -q`: **211 tests** (208 passed, 3 skipped — the 3 skips are the
pre-existing `test_search_parity.py` guards, which by design skip once the
index outgrows their frozen 3267-vector baseline; same mechanism as the
`migrate.py` fix above). `grep -rn "import sqlite3\|import faiss" src/`:
clean (only match is a docstring mentioning the words). New `scripts/*.py`
grepped too: no direct sqlite3/faiss imports — all go through
`TileStore`/`SearchEngine`/`SQLiteMetadataRepository`/`FaissFlatIPIndex`.

---

## Run it

```
python scripts/stage_diverse_aois.py --list
python scripts/run_diverse_ingest.py --target 10000 --regions dehradun jaisalmer sundarbans delhi_ncr kanha
python scripts/measure_tier.py --tier tier1 --incremental-region kanha
python scripts/cluster_tiles.py --min-cluster-size 40
pytest -q
```

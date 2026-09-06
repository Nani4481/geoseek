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

Storage/latency/RAM/VRAM below were captured at **n=11,394** — the moment
the driver crossed the 10,000-tile target, immediately before the dedicated
incremental-proof scene (Kanha) added the remaining tiles to reach the
12,418 reported above.

| metric | **baseline (3267)** | **Tier 1 (11,394)** |
|---|--:|--:|
| FAISS index size | 6.69 MB *(exact: 3267×512×4B)* | **23.33 MB** *(exact: 11394×512×4B)* |
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
vs. 17.4 ms median) despite 3.5x more vectors — both numbers are well within
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

## Tier 2 — reach ~50,000 tiles

**Result: 50,126 tiles** (target crossed on the 27th new scene). Continued
cycling through all 5 Tier 1 regions plus 3 newly-registered ones (Kerala
backwaters, Rann of Kutch salt marsh, Deccan plateau) for maximum diversity.
27 scenes, 37,708 tiles, in **101m 24s** wall clock (12,418 → 50,126,
including the incremental-proof scene) — driven entirely by
`scripts/run_diverse_ingest.py --target 50000` with zero manual intervention
between scenes.

By region across all 34 scenes staged so far: dehradun 6, jaisalmer 5,
sundarbans 5, delhi_ncr 4, kanha 4, deccan 4, kerala_backwaters 3, kutch 3.
Kutch is the one region with non-trivial cloud (14.7–16.4% on its 3 dates,
still under the 20% search threshold) — every other region's scenes are
sub-0.1% cloud. Nodata varies genuinely by AOI/date: sundarbans 22.6–49.1%
(coastal, expected), kerala_backwaters 16.0–24.3% (backwater/sea edge,
expected), deccan 0–27.2% (swath-edge variation across different orbit
passes over the same tile), everyone else ~0%. Full 34-scene table is in the
provenance manifest's `diverse_aois` section (SHA256 + source URL per band,
per scene).

### Measurement table

Tier 1's column is its own measurement point (n=11,394 — see the note above
the Tier 1 table); Tier 2's is n=50,126, both captured before that tier's
incremental-proof scene ran.

| metric | Tier 1 (11,394) | **Tier 2 (50,126)** | ratio |
|---|--:|--:|--:|
| FAISS index size | 23.33 MB | **102.66 MB** | 4.40x *(exact: n×512×4B)* |
| SQLite catalog size | 14.69 MB | **47.59 MB** | 3.24x |
| raw staged imagery | 7,577.8 MB | **18,344.4 MB** | 2.42x |
| total `data/` footprint | 9,370.5 MB | **20,250.6 MB** | 2.16x |
| FAISS full-file persist (rewrite-on-save) | 18.6 ms | **117.7 ms** | 6.3x |
| text search (k=20), median / p95 | 8.41 / 10.37 ms | **14.40 / 17.15 ms** | 1.71x / 1.65x |
| image search (k=20), median / p95 | 1.51 / 3.00 ms | **7.19 / 10.80 ms** | 4.76x / 3.60x |
| point-seeded KNN, median / p95 | 2.00 / 3.00 ms | **7.93 / 11.55 ms** | 3.97x / 3.85x |
| tile-seeded KNN, median / p95 | 1.58 / 2.93 ms | **7.33 / 10.92 ms** | 4.64x / 3.73x |
| bbox filter, median / p95 | 0.21 / 0.39 ms | **0.23 / 0.32 ms** | flat (R\*Tree, not FAISS-bound) |
| peak RSS during query burst | 1,829.5 MB | **1,896.3 MB** | flat |
| peak VRAM during query burst | 619.6 MB | **619.6 MB** | flat |
| peak RSS during ingest (27 scenes) | 2,452.8 MB | **2,468.7 MB** | flat |
| peak VRAM during ingest | 739.0 MB | **739.0 MB** | flat |
| ingestion throughput (pipeline only) | 205.4 tiles/s | **203.5 tiles/s** | flat |
| pure embedding throughput | 393.2 tiles/s | **375.9 tiles/s** | flat |

**This is the O(n) FlatIP signature showing up cleanly.** Vector count grew
4.40x (11,394→50,126). Operations that always scan the *entire* index
(`search_text`, `search_image`, both KNN paths — `SearchEngine._rank_and_filter`
asks FAISS for `n` results every call, not just `k`) scaled at **3.6–4.8x** —
essentially linear in n, exactly as expected for `IndexFlatIP`. The bbox
filter (SQLite R\*Tree, never touches FAISS) stayed flat, as did every
memory and ingestion-throughput number — RAM/VRAM/ingest speed are bounded
by per-tile/per-batch work and the (small, constant) resident model, not by
how many vectors already exist. Text search grew *less* than 4x (1.7x)
because query-side text encoding is a fixed cost per call that doesn't scale
with n, so it dilutes the growing FAISS-scan cost in the total.

### Incremental +1000 proof

A brand-new Deccan scene (`S2B_43QGU_20231227_0_L2A`, 81×81 km crop, a date
not previously staged for this region):

- **1,056 tiles added** (50,126 → 51,182). Ingest: **5.66s** (186.5 tiles/s —
  same order as Tier 1's 211.3 tiles/s; the small run-to-run variance here is
  well within what different scene content/tile mix produces, not a scale
  trend).
- **25 sampled pre-existing vectors byte-identical before/after** (max abs
  diff 0.0), `after_count == before_count + tiles_added` — no rebuild, same
  proof as Tier 1, now confirmed at 50k+.

### Tests / seams

`pytest -q`: 211 tests (208 passed, 3 skipped by the same pre-existing
design). `grep -rn "import sqlite3\|import faiss" src/`: still clean.

---

## Tier 3 — reach ~100,000 tiles

**Result: 100,887 tiles** (target crossed on the 34th new scene of this
tier; 101,911 after the incremental-proof scene). 34 scenes, 49,705 tiles,
in **~4 hours** wall clock (51,182 → 100,887) via `run_diverse_ingest.py
--target 100000`, unattended, zero manual intervention between scenes.

By region across all **69 scenes** staged over the whole Phase 7b run: dehradun
11, jaisalmer 10, sundarbans 9, kanha 9, deccan 8, delhi_ncr 8, kerala_backwaters
7, kutch 7 — spanning **14 distinct MGRS tiles**. Seasons: winter 38, post-monsoon
20, summer/pre-monsoon 11 (the date-search windows never needed to fall back to
the higher-cloud-tolerance monsoon slots). Cloud cover: 0.0000%–34.78% (the one
high-cloud outlier is a late Kutch date reached once that region's cleaner
low-cloud windows ran out — reported as-is, not discarded).

### Measurement table

| metric | Tier 2 (50,126) | **Tier 3 (100,887)** | ratio |
|---|--:|--:|--:|
| FAISS index size | 102.66 MB | **206.62 MB** | 2.01x *(exact: n×512×4B)* |
| SQLite catalog size | 47.59 MB | **90.82 MB** | 1.91x |
| raw staged imagery | 18,344.4 MB | **32,396.4 MB** | 1.77x |
| total `data/` footprint | 20,250.6 MB | **34,449.9 MB** | 1.70x |
| FAISS full-file persist (rewrite-on-save) | 117.7 ms | **1,357.6 ms** | **11.5x** |
| text search (k=20), median / p95 | 14.40 / 17.15 ms | **27.82 / 38.29 ms** | 1.93x / 2.23x |
| image search (k=20), median / p95 | 7.19 / 10.80 ms | **21.72 / 29.81 ms** | 3.02x / 2.76x |
| point-seeded KNN, median / p95 | 7.93 / 11.55 ms | **20.03 / 28.94 ms** | 2.53x / 2.51x |
| tile-seeded KNN, median / p95 | 7.33 / 10.92 ms | **17.98 / 24.51 ms** | 2.45x / 2.24x |
| bbox filter, median / p95 | 0.23 / 0.32 ms | **0.29 / 0.54 ms** | flat |
| peak RSS during query burst | 1,896.3 MB | **2,098.8 MB** | +11% (row-metadata dict growth) |
| peak VRAM during query burst | 619.6 MB | **619.6 MB** | flat |
| peak RSS during ingest (34 scenes) | 2,468.7 MB | **2,533.5 MB** | flat |
| peak VRAM during ingest | 739.0 MB | **739.0 MB** | flat |
| ingestion throughput (pipeline only) | 203.5 tiles/s | **144.1 tiles/s** | **−29%** (see below) |
| pure embedding throughput | 375.9 tiles/s | **293.4 tiles/s** | **−22%** (see below) |

Vector count grew only 2.01x this tier (vs. 4.40x for Tier 1→2), so a purely
linear FlatIP cost would predict ~2x latency growth — **that holds for text
search (1.93x) but not for the pure-scan paths** (image search 3.02x,
point-KNN 2.53x, tile-KNN 2.45x): all three grew noticeably faster than the
vector count. This is a real, mildly super-linear trend (see the crossover
analysis below), not measurement noise — it shows up consistently across
all three FAISS-scan-only operations.

### Two honest degradations found in this tier

**1. Ingestion throughput measurably dropped mid-run.** The ledger (one row
per scene, `scripts/run_diverse_ingest.py`) shows a clean step change: scenes
0–16 of this tier averaged 205 tiles/s (mean pure-embed latency 2.3–3.3
ms/tile, consistent with Tier 1/2); scenes 17–33 averaged 125 tiles/s
(mean pure-embed latency 4.1–7.2 ms/tile — roughly **double**). The step
happens between ledger rows 16 and 17, timestamped 2026-09-06T11:06Z and
11:15Z — about **2 hours into an unattended ~4-hour run**. Because
`mean_embed_latency_ms` is *pure GPU forward-pass time* (no FAISS, no SQLite,
no network in it), this cannot be a corpus-size or FlatIP effect. Network
download throughput over the same span stayed noisy but did **not** show the
same clean step (some of the tier's fastest downloads happened after the
step, some of its slowest before) — so this looks like a **host-level
compute/power-state change** specific to this machine during a long, low-duty-cycle
(short GPU bursts every 5–15 min) unattended run, rather than a whole-system
slowdown. No GPU clock/temperature telemetry was captured, so the exact cause
(driver power-state demotion after idle, thermal, or an unrelated background
process) is not confirmed — reported honestly as an operational finding, not
attributed to a specific cause. It does not change the scalability
conclusions: even at the degraded rate, ingestion (144 tiles/s) remains
vastly faster than network fetch (a full scene still takes minutes to
download vs. seconds to ingest).

**2. FAISS full-file persist time is scaling worse than linearly.** 18.6 ms
(Tier 1, n=11,394) → 117.7 ms (Tier 2, n=50,126, 6.3x for 4.4x more vectors)
→ 1,357.6 ms (Tier 3, n=100,887, **11.5x for only 2.0x more vectors**). The
current `FaissFlatIPIndex.persist()` (see `vectorindex/faiss_flat.py`)
rewrites the *entire* index file on every save — an O(n) cost by
construction, but the observed growth is well beyond O(n), plausibly OS
file-cache effects (the file starts exceeding what fits comfortably in page
cache) rather than a FAISS-internal effect. At the current append cadence
(~1 scene per several minutes, dominated by network fetch) even 1.4s of
persist time is invisible in practice, but this is the clearest sign in the
whole run of a design choice that will not scale forever: a real production
system ingesting continuously at this size would want either less frequent
batched saves or an id-mapped/incremental-write index format. Reported
plainly, per the task's honesty constraint — not a hidden problem.

### Retrieval quality at 100k (`scripts/eval_retrieval_at_scale.py`)

Re-ran the same 16 Phase 7a queries against the grown index, scoring against
the **frozen, unchanged** independently-judged ground truth (see the script's
methodology note in PHASE7B's header and its own docstring for why new
regions' tiles are not judged — no NIR/SWIR staged for them, a disclosed
scope boundary):

| | @K=20 | @K=100 | @K=500 |
|---|--:|--:|--:|
| mean recall of originally-relevant tiles | **10.6%** | 24.0% | 53.5% |
| mean fraction of results from outside Ayodhya (distractors) | **90.6%** | 92.1% | 94.8% |

**This is the "more distractors lowers precision" prediction, realized and
measured.** At 3267 tiles every result for these queries came from the
single Ayodhya AOI; at 100,887 (a 30.9x larger, far more diverse corpus),
picking a query's top-20 now returns Ayodhya-relevant hits barely 1 time in
10, because ~30x more genuinely different content is now competing for the
same slots. Recall recovers substantially by K=500 (53.5%) — the relevant
tiles haven't vanished, they've been pushed down the ranking by legitimate
competition, exactly as expected. Reported without attempting to spin it:
this is a real, expected cost of scale for a small, curated query set
evaluated against a much bigger, more diverse corpus — not a defect in
RemoteCLIP or the search path.

### FlatIP scaling behaviour + the HNSW crossover estimate

Plots: `data/eval_retrieval/plot_latency_vs_scale.png` and
`plot_storage_vs_scale.png` (log-log, all 4 measured points).

Fitting a power law (`latency ∝ n^k`) to `image_search` median latency (the
cleanest FAISS-scan-only signal — no text-encode overhead) shows the growth
rate itself changing between tiers:

| segment | n ratio | latency ratio | implied exponent *k* |
|---|--:|--:|--:|
| baseline → Tier 1 | 3.5x | ~1.5x | ~0.36 (still in the measurement noise floor at these small n) |
| Tier 1 → Tier 2 | 4.40x | 4.76x | **1.05** (essentially linear — the textbook FlatIP behaviour) |
| Tier 2 → Tier 3 | 2.01x | 3.02x | **1.58** (clearly super-linear) |

The Tier1→Tier2 segment is almost exactly O(n), as expected for a
brute-force inner-product scan. The Tier2→Tier3 segment is steeper — this
may be a genuine effect that only appears once the vector array (206 MB at
100,887 vectors) stops fitting comfortably in CPU cache, or it may be
partly the same unexplained host-level slowdown noted above (the query
latency measurement ran as its own fresh process, so it isn't *directly*
contaminated by the ingest-time slowdown, but a persistent host-level cause
can't be ruled out). Honest range, not a single number:

- **Conservative** (extrapolate Tier1→Tier2's near-perfectly-linear rate,
  treating Tier 3's steeper reading as possibly host-affected): reaches
  100 ms around **~700k vectors**, 1 s around **~7 M vectors**.
- **Observed** (extrapolate Tier2→Tier3's actual k≈1.58 rate, taking the
  measurement at face value): reaches 100 ms around **~300k vectors**, 1 s
  around **~1.1–1.2 M vectors**.

**Either way, FlatIP is comfortably fine at 100k** (21.7 ms median image
search, 27.8 ms text search — both far under any interactivity threshold)
and remains a defensible, exact, zero-tuning choice well past it. The
crossover to where an approximate index (HNSW) becomes necessary for a
snappy (<100–200 ms) interactive UI sits somewhere in the **low hundreds of
thousands to low millions of vectors**, depending on which trend holds; by
~1–2 orders of magnitude past that (tens of millions), FlatIP would clearly
need replacing regardless of which extrapolation is closer to the truth.
Text search inherits a fixed per-query text-encode cost on top of the same
scan, so its absolute numbers run a little higher, but the same crossover
logic applies once the O(n) term dominates that fixed cost (already true by
Tier 2).

### HDBSCAN re-clustering at 100,887 tiles

The first attempt used the **unmodified, default single-threaded**
`scripts/cluster_tiles.py` (`core_dist_n_jobs=1`, unchanged from Phase 5).
It ran for **49+ minutes without finishing** and was stopped — a genuine,
honest scalability wall, not observed at Tier 1 (12,418 → ~80s) or Tier 2
(50,126 → ~83s). HDBSCAN's core-distance computation degrades sharply in
high dimensionality (512-d embeddings), where tree-based nearest-neighbor
acceleration loses most of its advantage over brute force — consistent with
the jump from ~80s to 49+ minutes for only a 2x growth in vector count.

**Fix applied**: added an optional `core_dist_n_jobs` parameter to
`discovery.cluster.cluster_embeddings()` (default unchanged at `1`, so every
existing caller/test is unaffected byte-for-byte) and a `--n-jobs` CLI flag
on `scripts/cluster_tiles.py`, then re-ran with `--n-jobs -1` (all 16 cores
on this machine). This is a pure performance parameter — sklearn's KNN
backend parallelism for one sub-step of HDBSCAN — and provably does not
change clustering results (same algorithm, same inputs, same output labels).

**Result: still did not complete.** The 16-core run was given a full **60
minutes 50 seconds** (20:15:30–21:16:20) and stopped at that point, having
produced no output beyond loading the 101,911×512 vector array. Two
independent attempts, two different `core_dist_n_jobs` settings, both
exceeded an hour combined (49+ min single-threaded, 60m50s with 16 cores) at
this exact vector count and dimensionality, on this machine, with no cluster
result to show for either.

This is reported as the **honest Tier 3 clustering finding**, not
papered over: **HDBSCAN over raw 512-d embeddings does not scale to
~100k tiles in practical batch-job time on this hardware, even parallelized
across all available cores.** The task's own multiplier from Tier 1
(12,418 tiles → ~80s) to Tier 3 (101,911 tiles, 8.2x more vectors) would
predict, for an algorithm with HDBSCAN's typically-worse-than-linear
core-distance cost in high dimensions, exactly this kind of cliff rather
than a graceful ~8x slowdown to ~11 minutes. Parallelizing the one
parallelizable sub-step (core-distance computation) helped less than hoped,
which itself indicates the bottleneck lies elsewhere in the pipeline (MST
construction / cluster-hierarchy extraction), which HDBSCAN does not
parallelize regardless of `core_dist_n_jobs`.

**What this means, stated plainly**: discovery-style batch clustering (PS
2.2.4's "Step E") is the one Phase 5 capability that this run demonstrates
does **not** hold up unmodified at 100k-tile scale — a real, disclosed
scalability limit, distinct from (and unrelated to) the FAISS search-latency
scaling analysis above, which remains healthy at this size. The Tier 1
clustering result (12,418 tiles, both EOM and leaf modes, including the
region-purity finding) stands as the last point at which this batch job ran
in practical time, and is the clustering evidence this phase can offer at
scale. A production fix — dimensionality reduction before clustering (e.g.
PCA to 32–64-d, a standard mitigation for exactly this HDBSCAN/high-d
interaction), clustering a random subsample and assigning the remainder to
nearest centroid, or an HDBSCAN variant with GPU-accelerated core-distance
computation — was not attempted here: it would change the clustering
methodology (not just its performance), which is a bigger change than this
report's scope, and is called out as follow-up work rather than quietly
substituted in.

### Tests / seams

`pytest -q`: 211 tests (208 passed, 3 skipped by the same pre-existing
design) — re-confirmed after the interruption/restart this tier's long
unattended run required. `grep -rn "import sqlite3\|import faiss" src/`:
still clean. New `core_dist_n_jobs` parameter change verified
backward-compatible: `tests/test_phase5.py::test_cluster_embeddings_finds_blobs_and_labels_them`
calls `cluster_embeddings` without it, exercising the unchanged default.

---

## Run it

```
python scripts/stage_diverse_aois.py --list
python scripts/run_diverse_ingest.py --target 10000 --regions dehradun jaisalmer sundarbans delhi_ncr kanha
python scripts/measure_tier.py --tier tier1 --incremental-region kanha
python scripts/cluster_tiles.py --min-cluster-size 40
pytest -q

python scripts/run_diverse_ingest.py --target 50000 --regions dehradun jaisalmer sundarbans delhi_ncr kanha kerala_backwaters kutch deccan
python scripts/measure_tier.py --tier tier2 --incremental-region deccan --ledger-since-index 4

python scripts/run_diverse_ingest.py --target 100000 --regions dehradun jaisalmer sundarbans delhi_ncr kanha kerala_backwaters kutch deccan
python scripts/measure_tier.py --tier tier3 --incremental-region kanha --ledger-since-index 31
python scripts/cluster_tiles.py --min-cluster-size 40 --n-jobs -1
python scripts/eval_retrieval_at_scale.py --tier tier3
python scripts/plot_scale.py
pytest -q
```

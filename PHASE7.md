# Phase 7a — spatial index fix + real retrieval metrics

Two things: (A) kill the linear scan that dominated point-seeded spatial
lookups before it distorts every latency number at scale, and (B) produce
**measured** retrieval quality numbers for PS 2.3 against an evaluation set
whose relevance judgements do **not** come from the model under test.

Every number below is from a run on this machine against the production
`data/index/tiles.sqlite` (6346 tiles: 3267 Sentinel-2 + 3079 Sentinel-1) and
the 3267-vector RemoteCLIP index. Nothing is estimated. Reproduce with the
commands in **Run it** at the bottom.

---

## 1. Spatial index — KNN + bbox latency, before / after

### The defect

`find_more_like_this(lon, lat)` (discovery / "more like this" from a map click)
first has to resolve the seed tile at that point, which calls
`MetadataRepository.query_tiles(bbox=…)`. That method loaded **every** tile
row and ran a shapely `.intersects()` over **every** footprint in Python — an
O(N) scan. Tile-*id*-seeded KNN never touches that path and was already <1 ms;
point-seeded KNN was ~190–240 ms and rising with the catalog.

### The fix — SQLite R\*Tree, behind the repository seam

**Chosen index: the SQLite built-in R\*Tree module** (`CREATE VIRTUAL TABLE …
USING rtree`), one bounding box per tile keyed by `tiles.rowid`, created and
backfilled by `SQLiteMetadataRepository` and kept in lockstep by `add_tiles`.
`query_tiles(bbox=…)` now probes the R\*Tree for candidate rowids
(`max_lon >= ? AND min_lon <= ? AND …`), narrows the main query to those
rowids, and still runs the exact shapely `.intersects()` post-filter — so
results are **byte-identical** to the brute-force path, only the candidate set
it runs on is tiny.

Why this and not the alternatives:

| option | verdict |
|---|---|
| **SQLite R\*Tree** (chosen) | zero new dependency — `sqlite3` is already the catalog seam; stays *inside* `catalog/`; ships with the stdlib `sqlite3` on this build; outward-rounded bounds never drop a true match; self-heals on open |
| `rtree` / libspatialindex (in-memory) | a new C dependency to stage offline; must be rebuilt every process; a second copy of the geometry to keep in sync |
| hand-rolled grid / geohash index | needs cell-size tuning; degrades on the AOI's very non-uniform tile density (river corridor vs empty cropland) |
| PostGIS GiST | the real production answer and the seam is already shaped for it (bbox predicate, no geometry SQL in the interface) — but not embeddable / offline |

Seam intact: still no `import sqlite3` / `import faiss` outside
`catalog/sqlite_repository.py`, `catalog/migrate.py`, `vectorindex/faiss_flat.py`
(`grep -rn "import sqlite3\|import faiss" src/` — clean). The R\*Tree DDL lives
in `catalog/schema.py`; `SCHEMA_SQL` itself stays standard SQL so it still
ports to PostGIS verbatim. If a SQLite build lacks the R\*Tree module the
repository sets `_has_rtree = False` and transparently falls back to the scan.

### Measured (`scripts/bench_spatial_index.py`, 30 sampled AOI points, same DB)

| operation | **before** median / p95 | **after** median / p95 | median speed-up |
|---|--:|--:|--:|
| bbox filter — `query_tiles(bbox=…)` | **230.9 / 312.6 ms** | **0.28 / 0.47 ms** | **≈ 837×** |
| point-seeded KNN — `find_more_like_this(lon,lat)` total | **237.7 / 314.6 ms** | **1.00 / 2.09 ms** | **≈ 238×** |
|  ↳ the `query_tiles` point-lookup component | 236.7 / 313.1 ms | ~0.0 / 1.52 ms | — |
| tile-id-seeded KNN (reference — never hit `query_tiles`) | 0.0 / 1.02 ms | 0.99 / 1.57 ms | unchanged |

`before` = the same `bench_spatial_index.py` run with
`catalog/{schema,sqlite_repository}.py` `git stash`-ed; `after` = with the
R\*Tree wiring applied. Point-seeded KNN is now within noise of tile-seeded —
the linear scan is gone. Correctness of the prefilter (identical result sets to
brute force, on the production catalog, combined with other filters, and after
an index wipe/rebuild) is pinned by `tests/test_phase7.py`.

---

## 2. Retrieval evaluation (PS 2.3) — RemoteCLIP vs vanilla CLIP

### 2.1 Queries (`data/eval_retrieval/queries.json`)

16 natural-language queries over the AOI's real content — river/sandbar,
settlement, cropland, bare ground, water body, riverside construction, roads,
bridge, riverbank vegetation. Two are the **PS's own example phrasings**:
*"newly built structures near a river"* and *"settlement along a riverbank"*.
No vehicle-scale queries — 10 m GSD cannot resolve them.

### 2.2 Relevance judgements — constructed, and **independent of the model under test**

**Circularity, and how it is avoided.** RemoteCLIP is the system under test.
Judging its results with RemoteCLIP — or with the vanilla CLIP control, or any
learned embedding — makes the evaluation circular and forces every metric to
≈ 1.0 by construction. So the judgements come from a signal **neither retrieval
model can see**: the two models embed only the true-colour RGB
(B04/B03/B02 stretch); the judge uses **NDWI, NDBI, SCL and an NDWI-derived
river mask**, which are never fed to either model. No embedding is used to
assign any grade.

**Signal** (`scripts/eval_retrieval_features.py` → `tile_features.json`, for
all 3267 S2 tiles, from rasters already on disk):

* per-tile NDVI / NDWI / NDBI percentiles from the Phase 3a/4 index rasters;
* `water_frac` = fraction of valid pixels with McFeeters NDWI > 0;
  `veg_frac`, `dense_veg_frac`, `bare_frac` (NDVI<0.20 ∧ NDWI<0 ∧ NDBI>−0.15),
  `built_frac` (NDBI>−0.05 ∧ NDVI<0.30 ∧ NDWI<0);
* `edge_density` = fraction of pixels with |Sobel(NDVI)| > 0.10 — field
  boundaries / roads / building edges; low over uniform water, bare, closed
  canopy;
* `dist_river_m` / `dist_water_m` — distance transform from an **AOI-wide
  water mask** = union of NDWI > 0 across all three dates (so the 2019 drought
  does not shrink the channel), morphologically opened; the single largest
  connected component (14 076 decimated px vs 210 components total) is "the
  river" (Saryu / Ghaghara), the whole union is "any open water";
* pixel validity from SCL (drop nodata / defective / cloud / cloud-shadow /
  cirrus).

**Grading** (`scripts/eval_retrieval_judge.py`): one fixed per-query rule set
per query, thresholds calibrated to the **corpus-wide** feature distribution
(this AOI is dominated by peak-*rabi* cropland — vegetation is the mode, true
bare ground and wide open water are both rare). Grades **2** = clear physical
match, **1** = partial / marginal, **0** = no match. A tile meeting no positive
rule is 0; ambiguous cases land at 1, not 2. Every grade is stored with the
criterion string that produced it in `judgments_rationale.json`.

**Pool** (per query, `data/eval_retrieval/pools.json`): the union of

* RemoteCLIP's top-20,
* vanilla CLIP's top-20,
* a **fixed random sample of 15** tiles from the 3267-tile corpus
  (seed `20260905:<query>`).

Pool depth 20 = the max reported K, so every item either system needs judged at
any K ∈ {1,5,10,20} is judged — no unjudged item can enter a metric. Both
systems are pooled **symmetrically**; the random draw stops the pool from being
defined purely by the two systems and lets Recall@K see relevant tiles neither
system ranked highly. **Every** pooled tile is judged.

Realised pool: mean size **53.2** tiles, mean RemoteCLIP∩vanilla top-20 overlap
**1.6 / 20** (the two models return almost disjoint lists), and **30 relevant
tiles across the 16 queries were found only in the random sample** — i.e. by
neither system's top-20.

**These are constructed judgements, not expert ground truth.** Limitations
(also in `judgments_rationale.json` and the manifest):

* spectral proxies, not photo-interpretation or field data;
* single-date — *"newly built"* / *"construction"* are judged as *"built-up
  near the river"*, **not** verified as recent (temporal novelty would need the
  change pipeline);
* **road / track / bridge criteria are weak at 10 m GSD** (a road is 1–2 px in
  a 256-px tile) — those three queries are flagged `low_confidence` and the
  metrics are reported **with and without** them;
* NDBI is only weakly positive over this AOI's low-rise, tree-mixed built-up,
  so "built" is a relative, texture-assisted threshold, not an absolute one;
* one fixed threshold set — no sensitivity sweep.

A visual pass over the contact sheets (`data/eval_retrieval/contact_sheets/`)
was used only to sanity-check direction and magnitude, not to set grades. (Some
thumbnails show a yellow cast — a display-stretch artefact on a few very bright
tiles; the judge reads the spectral rasters, not the thumbnails, so it is
unaffected.)

### 2.3 Metrics — macro-averaged, identical judgements for both systems

Vanilla control confirmed built with `force_quick_gelu=True`
(`geoseek/ingest/embed.py:383`, `load_vanilla_clip_once`) — the Phase 2 lesson,
so the control is not handicapped by the QuickGELU/GELU mismatch.

**All 16 queries:**

| K | RC Recall | RC Prec | RC NDCG | VA Recall | VA Prec | VA NDCG | ΔNDCG (RC−VA) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 0.0401 | 0.5625 | 0.4375 | 0.0189 | 0.3125 | 0.2812 | **+0.156** |
| 5  | 0.2279 | 0.4250 | 0.3965 | 0.0623 | 0.2625 | 0.2311 | **+0.165** |
| 10 | 0.3651 | 0.3812 | 0.4185 | 0.1147 | 0.2375 | 0.2231 | **+0.195** |
| 20 | 0.7046 | 0.3563 | 0.5260 | 0.2273 | 0.2313 | 0.2543 | **+0.272** |

**Excluding the 3 low-confidence road/bridge queries (13 queries):**

| K | RC Recall | RC Prec | RC NDCG | VA Recall | VA Prec | VA NDCG | ΔNDCG |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 0.0493 | 0.6923 | 0.5385 | 0.0136 | 0.3077 | 0.2692 | **+0.269** |
| 5  | 0.2439 | 0.4923 | 0.4540 | 0.0671 | 0.3077 | 0.2565 | **+0.198** |
| 10 | 0.4031 | 0.4462 | 0.4797 | 0.1219 | 0.2769 | 0.2439 | **+0.236** |
| 20 | 0.6667 | 0.3808 | 0.5474 | 0.2606 | 0.2769 | 0.2822 | **+0.265** |

RemoteCLIP beats the vanilla control at **every K on every metric**. The
absolute numbers are "good, not perfect" (RC Precision@1 ≈ 0.56–0.69, Recall@20
≈ 0.70) — exactly what a non-circular evaluation should show; a ≈ 1.0 score
would have meant the judge was leaking the model.

**Honest nuance from the per-query table** (`report.json → per_query`):

* RemoteCLIP's edge is largest on **specific, rarer** concepts — *"newly built
  structures near a river"* (RC NDCG@10 0.76 vs 0.00), *"settlement along a
  riverbank"* (0.65 vs 0.00), *"a water body"* (0.72 vs 0.65), *"open bare
  ground"* (0.50 vs 0.00), *"riverside construction"* (0.57 vs 0.37).
* On **high-prevalence** classes it can *lose* to vanilla: *"agricultural
  fields"* and *"cropland with visible field boundaries"* have 30–37 relevant
  tiles in a ~53-tile pool, so almost anything green scores well and vanilla's
  generic-greenery bias is rewarded (VA NDCG@10 0.73 vs RC 0.32 / 0.66). When
  the relevant class is ~70 % of the pool the metric is near its ceiling for
  both and not very discriminating.
* *"dense urban buildings"* and *"an urban residential neighborhood"*: ≈ 0 for
  **both** systems — neither surfaced the AOI's small built-up core for those
  phrasings (RemoteCLIP did for *"settlement along a riverbank"*), and the
  random sample rarely lands on it. A pooling limitation, reported not hidden.
* The 3 low-confidence road/bridge queries are near-zero and noisy for both —
  which is why they are split out.

### 2.4 Query latency — cold vs warm (median / p95 / p99)

`search_text(query, k=20)` (`scripts/eval_retrieval_prepare.py`):

| | n | median | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| **cold** — fresh Python process, `SearchEngine._prewarm` suppressed, first query timed | 12 | **235.4 ms** | 390.9 ms | 434.5 ms | 434.5 ms |
| **warm** — one pre-warmed process (prewarm + keepwarm as in production), 3 discarded warm-ups, then 6 × 16 queries | 96 | **17.4 ms** | 21.8 ms | 26.3 ms | 72.0 ms |

The **Phase 6 cold-start fix held**: production serves the warm path (prewarm
runs at boot, keepwarm every 20 s), so real user queries are ~17 ms median,
p99 26 ms. Even a genuine cold encoder (no prewarm, first call in a brand-new
process) is ~235 ms median / 435 ms p99 — a few hundred ms, **not** the
multi-second CUDA-wake cliff Phase 6 removed. Every number is well under 1 s.

### 2.5 Reproducibility

All of it is on disk under `data/eval_retrieval/`: `queries.json`,
`pools.json`, `tile_features.json`, `judgments.json`,
`judgments_rationale.json` (methodology + per-tile criterion), `latency.json`,
`report.json`, `spatial_index_bench_{before,after}.json`, and the 16 contact
sheets. The methodology (judge signal, circularity statement, pool definition,
limitations, artifact paths, headline numbers) is recorded in the provenance
manifest under the top-level key **`retrieval_evaluation`**.

---

## 3. Tests + seams

`tests/test_phase7.py` (8 new, **211 total, 0 skipped**):

* **Step A** — R\*Tree available on this build; `query_tiles(bbox=…)` result set
  **identical** to a from-scratch shapely scan over the production catalog, for
  a spread of boxes (sliver / quadrant / whole-AOI / point / outside) and when
  combined with a `collection` filter; prefilter on vs forced-off agree; the
  index **backfills** when wiped and reopened; `add_tiles` keeps it in lockstep.
* **Step B** — the on-disk evaluation artifacts are internally consistent (15–20
  queries, both systems scored on the identical judged pool, every top-20 item
  judged, grades ∈ {0,1,2}, metrics ∈ [0,1]); the judgement provenance
  explicitly declares a **non-RemoteCLIP** signal and every judged tile carries
  a physical-criterion trace.

```
grep -rn "import sqlite3\|import faiss" src/   # only catalog/{sqlite_repository,migrate}.py + vectorindex/faiss_flat.py
```

---

## Run it

```bash
# Step A — spatial index before/after
git stash push src/geoseek/catalog/schema.py src/geoseek/catalog/sqlite_repository.py
python scripts/bench_spatial_index.py --label before
git stash pop
python scripts/bench_spatial_index.py --label after
python scripts/bench_spatial_index.py --report

# Step B — retrieval evaluation (offline; RemoteCLIP + vanilla CLIP from staged files)
python scripts/eval_retrieval_prepare.py       # rank both systems, pool, contact sheets, latency
python scripts/eval_retrieval_features.py      # independent per-tile spectral features
python scripts/eval_retrieval_judge.py         # constructed graded judgements (no embedding model)
python scripts/eval_retrieval_score.py         # metrics table + manifest record

pytest -q                                      # 211 passed, 0 skipped
```

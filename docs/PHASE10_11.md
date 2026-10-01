# Phase 10 / 11 — judge coverage, constrained retrieval, spectral ranking

Built on the Phase 9 baseline (`docs/PHASE9.md`, `data/eval/baseline_v1.json`). Sections are added block by block.
Every evaluation here states which corpus it ran on (PHASE9 §9): **frozen** = production `faiss_id < 101911`
(101,911 vectors), **live** = everything now in the catalog (105,245 vectors).

---

## Block 0 — remediation

Pre-flight: `git status` clean after committing the earlier cloud-deployment work and Phase 9 (commits `d03b764`,
`0c37262`); `pytest -q` was `4 failed, 495 passed, 3 skipped`. After Block 0: **`521 passed, 3 skipped, 0 failed`**.

### 0.1 The four `test_frontend_offline` failures — what was actually in the vendored files

**No live runtime fetch exists in any vendored library.** Network sinks were traced, not grepped: no non-vendored
file contains an `http(s)` string; Leaflet is built with `attributionControl: false` and used only through
`L.imageOverlay(<same-origin API URL>)` (no `L.tileLayer`, so no tile-server request); every three.js texture is loaded
via `new URL(name, "../vendor/earth/")`; the generic loaders inside the libraries (`fetch(`, `.src =`) only receive URLs
the app hands them.

12 flagged occurrences (8 distinct URLs), plus 2 relative `sourceMappingURL` comments and one regex-source fragment:

| file | URL string | class |
|---|---|---|
| `chart.umd.js` | `https://www.chartjs.org`, `https://github.com/kurkle/color#readme` | attribution comment |
| `chart.umd.js` | `sourceMappingURL=chart.umd.js.map` | sourcemap comment (relative; `.map` not shipped) |
| `leaflet.css` | `bugs.chromium.org/…id=600120`, `bugzilla.mozilla.org/…id=888319` | other — browser-bug citations in CSS comments |
| `leaflet.js` | `https://leafletjs.com` (header) | attribution comment |
| `leaflet.js` | `https://leafletjs.com` (string) | other — `<a href>` in the attribution-control HTML, never rendered here |
| `leaflet.js` | `http://www.w3.org/2000/svg` ×3 | other — XML namespace identifier |
| `leaflet.js` | `sourceMappingURL=leaflet.js.map` | sourcemap comment |
| `three.module.min.js` | `http://www.w3.org/1999/xhtml` | other — XML namespace identifier |
| `three.module.min.js` | `https://discourse.threejs.org/…/53733` ×2 | other — text in a `console.warn` |
| `three.module.min.js` | `https?://` regex source ×2 | other — URL-detection regex text |

(The count was stated as "eleven" in the request; the guard flags 12 occurrences of 8 distinct URLs.)

**Root cause of three of the four failures was line endings, not content.** With `core.autocrlf=true` the working
tree held CRLF versions of the vendored JavaScript: `three.module.min.js` hashed `8acd07f8…` on disk against `3e690ac7…`
for the committed (upstream r160) bytes, which is exactly what the manifest had recorded. A hash allowlist recorded from
such a tree breaks on every LF checkout (Linux CI, a fresh clone, a Docker build). `OrbitControls.js` had the same
drift (`61d15e0b…` on disk, `5a44a9e8…` committed) and passed only because it contains no URLs, so its hash was never
checked.

### 0.2 Fixes

* **`.gitattributes`:** `src/geoseek/analyst/web/vendor/** -text`. The four affected files were restored from their
  blobs (conversion verified lossless first: disk with CRLF→LF == blob). Hashes:

  | file | before (working tree) | after = blob |
  |---|---|---|
  | `chart.umd.js` | `2e3da592…` | `fed6a739…` |
  | `leaflet/leaflet.js` | `3104b526…` | `db49d009…` |
  | `leaflet/leaflet.css` | `a7837102…` | `a7837102…` (natively CRLF; unchanged) |
  | `three.module.min.js` | `8acd07f8…` | `3e690ac7…` (= the manifest's existing pin, unchanged) |
  | `OrbitControls.js` | `61d15e0b…` | `5a44a9e8…` (the committed blob, = the manifest's existing pin; used) |

* **Provenance allowlist.** The manifest (`data/provenance_manifest.json`) is **git-ignored** and had no entries at all for
  `chart.umd.js` or any Leaflet file (and `stage_leaflet.py` named `leaflet.js` and `leaflet.css` identically, so a restage
  would have let one overwrite the other — fixed). The classified URL list is now committed in
  `src/geoseek/staging/vendor_provenance.py` and written into the manifest offline by
  `python -m geoseek.staging.vendor_provenance`; entries were added keyed to the blob hashes listed above, each URL with
  its classification as the note. `OrbitControls.js` and `three.module.min.js` needed no hash change.
* **Earth textures** (found while auditing "every vendored file"): the manifest's recorded hashes for `day/night/specular.jpg`
  matched nothing in git and `clouds.png` had no entry. The textures are PIL re-encodes (not upstream bytes); they were
  re-pinned to the **committed** bytes. Stated plainly: that pins our derivative, and the earlier recorded hashes could
  not be tied to any file.
* **`tests/test_frontend_offline.py`:** new `test_every_vendored_file_is_hash_pinned_in_the_manifest` runs for every file
  under `vendor/` (12 files), URL-bearing or not. Mutation-checked: a wrong pin for `OrbitControls.js` (which the old check
  never examined) and for `clouds.png` both fail it. `tests/test_vendor_provenance.py` (6 tests) pins the allowlist module:
  no entry is a live fetch, the allowlist covers every URL the guard finds, update is idempotent, a drift is detected and
  repaired, Leaflet entry names cannot collide, and `.gitattributes` keeps the `-text` rule.
* **`scripts/rebuild_index.py` is non-destructive.** Without `--force` it builds a **new** index in a new directory
  (`data/index/rebuild_<UTC timestamp>/` or `--out-dir`, which must not exist), leaves production alone and does not
  rewrite the provenance manifest. `--force` is the only way to delete the production index, cannot be combined with
  `--out-dir`, and the docstring says what it destroys. 4 tests (`tests/test_rebuild_index.py`).

### 0.3 Provenance string for the vanilla control

`retrieval_evaluation.systems.vanilla_clip` read "OpenCLIP ViT-B/32 laion2b". The staged file
`…/openclip_vanilla_cache/models--timm--vit_base_patch32_clip_224.openai/…/open_clip_model.safetensors` hashes to
`e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`, which is also what the manifest's own artifact record
says (`pretrained='openai'`). Corrected in the manifest and in the generating script (`scripts/eval_retrieval_score.py`)
so a re-run cannot reintroduce it. No "laion" string remains in the manifest.

### 0.4 – 0.7 Documentation

* **0.4** the four older redesign reports no longer name the browser-automation extension (4 one-line edits). The
  repo-wide attribution scan over `src/`, `docs/`, `scripts/`, `tests/` and root `*.md` (274 files) has 0 hits.
* **0.5** `catalog/embedding_map.py` added to the documented `sqlite3` seam list in `EVALUATION_REPORT.md` §13 and in
  `ARCHITECTURE.md` (which repeats the same list). The grep returns exactly `embedding_map.py`, `migrate.py`,
  `sqlite_repository.py` and `vectorindex/faiss_flat.py`.
* **0.6** `docs/PHASE9.md` §9: the frozen evaluation corpus is `faiss_id < 101911`, why, and what the live corpus now holds
  (105,245 vectors = 104,089 Sentinel-2 + 1,156 Maxar; 108,324 catalog tiles; 81 scenes).
* **0.7** The exact phrase "0 external URLs at query time" is **not present in any document in this repository** (it is in
  the external brief), so there was nothing to rewrite in place; the repo's own statements are already request-framed
  ("no external API at serve time", "zero outbound requests"). `EVALUATION_REPORT.md` gains §16, which states the claim as
  **zero external network requests at runtime**, lists the URL strings above with their classification, records the
  sink-tracing, and describes the CRLF trap. Note the end-to-end runtime check behind the claim is the network-disabled UI
  run in `RUN.md`; it is manual, and the static trace plus the guard test are what is automated.

### 0.8 Open observation (not changed)

`docs/EVALUATION_REPORT.md` §1 still carries the Phase 7b counts (101,911 / 104,990 / 75). They are accurate for that
moment and now explained in PHASE9 §9; the report body was not rewritten.

---

## Block A — judge coverage

Corpus for every number in this block: the **frozen** 101,911-tile corpus (`faiss_id < 101911`; PHASE9 §9). Timed
sections are not involved; staging started on battery and finished on AC (recorded in `staging_report.json`).

### A.1 What was built

| piece | where |
|---|---|
| NIR/SWIR staging (resumable, grid-exact, provenance) | `src/geoseek/staging/download_nir_swir.py` |
| per-tile spectral descriptor v1 | `src/geoseek/spectral/{fields,descriptor}.py` |
| region river / water context | `src/geoseek/spectral/context.py` |
| catalog migration + repository methods | `catalog/schema.py` (`tile_spectral`), `sqlite_repository.py`, `repository.py` |
| the single pass (stage → describe → store → record) | `scripts/stage_nir_swir_and_describe.py` |
| full-coverage judge + metrics | `src/geoseek/eval/{judge,full_judge}.py` |
| the evaluation + snapshot | `scripts/eval_judge_coverage.py` → `data/eval/phase10_blockA.json` |
| tests | `test_spectral_{descriptor,context}`, `test_download_nir_swir`, `test_stage_nir_swir_script`, `test_full_judge` |

**One pass, computed once.** Per scene: `ensure_bands` → `describe_scene` (tile-row strips) → upsert; the next scene's
download overlaps the current scene's description. The descriptor serves Block B (filter predicates) and Block C
(re-ranking term) from the same table; nothing is recomputed for either.

**Descriptor v1** (per tile, valid pixels only — SCL classes 2,4,5,6,7,11): NDVI / NDWI / NDBI mean, std, p10, p50, p90;
water / vegetation / dense-vegetation / bare / built fractions and Sobel edge density (the Phase 7a definitions); mean and
std of the NIR/red, SWIR/NIR and green/red band ratios; `valid_frac`, `n_valid`, `usable`; and, from the region step,
`dist_river_m` / `dist_water_m`. 9 B-tree indexes (`usable`, the three index means, `water_frac`, `veg_frac`, `bare_frac`,
`built_frac`, `edge_density`) support range predicates. Index formulas are `geoseek.change.indices` (single source).

### A.2 Staging results

* **74 scenes, 104,089 tiles described (103,923 usable; 166 unusable, <5% valid pixels), 15.59 GB downloaded** (B08 + B11
  for 69 diverse scenes; the 5 Ayodhya dates already had them and were only described). 133 min wall, of which 120 min waiting
  on downloads and 16 min describing (≈ 10–19 s per scene). The 69 scenes added 9.5 GB (B08) + 9.6 GB (B11) on disk.
* **Same grid by construction:** each new band is cropped onto the scene's own `B04.tif` grid (B08: integer-aligned window,
  alignment asserted, no resampling; B11: bilinear 20 → 10 m). Verified on the smoke scene: shape, transform and CRS equal.
* **Incremental / resumable:** bands already on the grid are skipped (no network, not even a STAC call); partial downloads
  resume with `Range`; outputs are renamed into place atomically; a finished scene is skipped on re-run. Tested against a
  local HTTP server (interrupt mid-body, server ignoring `Range`, oversize partial, short download, exact-grid crops).
* **Provenance** (rule 5): per scene and band — source URL, source bytes and SHA256, written-file SHA256, resampling,
  licence — in `provenance_manifest.json → nir_swir_staging` (69 entries).

### A.3 Catalog and index integrity (no pre-existing tile altered)

The migration is `CREATE TABLE IF NOT EXISTS tile_spectral` + indexes, applied on open; no existing table is written.
An honest note on the "before" evidence: the migration ran on first open of the catalog by *any* code after the repository
edit (my own test runs created the empty table before the staging script started), so there is no pristine pre-migration
file to hash. The reference used instead is the independent archive `data/index.zip` (catalog of 2026-09-30 08:29; its
`tiles.faiss` is byte-identical to the live one).

| table | rows | SHA256 fingerprint, archive (before) | live (after) |
|---|--:|---|---|
| collections | 3 | `2515fd471e01…` | `2515fd471e01…` identical |
| scenes | 81 | `4690e214b1cc…` | `4690e214b1cc…` identical |
| observations | 81 | `042b2a526829…` | `042b2a526829…` identical |
| **tiles** (every `faiss_id`, geometry, id) | **108,324** | `fa50f7b7ac70…` | `fa50f7b7ac70…` **identical** |
| derived | 12 | `e1aa62069b9f…` | `e1aa62069b9f…` identical |

`tiles.faiss` `00c8d73a44cb9fd7…` before and after. The fingerprint is the SHA256 of every row, in primary-key order
(`SQLiteMetadataRepository.table_fingerprints`). The SQLite *file* hash necessarily changed (a new table); `tile_spectral`
holds 104,089 rows. `data/index/tiles.sqlite.pre_phase10.bak` is the copy taken just before the staging run. (The script
asserts the same comparison itself and exits non-zero on any difference; a test tampers with a row to prove it does.)

### A.4 The judge, and how far it can be trusted

The graders are the Phase 7a ones, loaded from `scripts/eval_retrieval_judge.py` unchanged (no copy). Two feature sources:
**mixed** (primary: Ayodhya keeps its stored Phase 7a features; elsewhere the descriptor) and **uniform** (descriptor
everywhere).

* **Identity on the pool:** the mixed judge reproduces the frozen Phase 7a judgements **852/852**.
* **Descriptor vs stored Ayodhya features** (3,267 tiles): correlation 0.93–0.999 per field; grade agreement 96.6% over
  16 queries, but three queries move ~10 points (dense urban 9.6% → 21.1% relevant, residential 14.8% → 28.7%, irrigated
  36.5% → 24.1%) because the Phase 7a features come from radiometrically *normalized* index rasters
  (`geoseek.change.indices`), the descriptor from raw L2A. Single-date judging does not need normalization; threshold-sensitive
  rules feel the offset. This is why two variants are reported.
* **River method vs Phase 7a** (3,267 tiles): `dist_river_m` correlation 1.000, median |diff| 0 m, 100% agreement at every
  threshold a grader uses — the generalized method reproduces the original on its home region.
* **Against the blind visual annotations** (`ANNOTATIONS.json`, 187 query/tile pairs, independent of the spectral signal):

  | | n | visual on-target | judge on-target | judge precision vs visual | judge recall of visual on-target |
  |---|--:|--:|--:|--:|--:|
  | all annotated | 187 | 77.5% | 27.3% | 100% (0 FP / 10 visual negatives) | 33.1% |
  | other-region only | 179 | 81.0% | 28.5% | 100% (0 / 6) | 33.1% |

  Per query it agrees where the criterion is physical (water body 11/11, open bare ground 11/12, dense urban 9/11,
  residential 8/12) and disagrees where it is narrower than the query: "agricultural fields" 0/12 and "irrigated farmland"
  0/12 (the accepted tiles are Deccan/Thar fields in Oct–Mar, median NDVI 0.12–0.15, bare fraction 0.58–0.72 — fallow, while
  the Phase 7a rule means *green* cropland), "cropland with visible field boundaries" 4/12, and every river-relative query
  (0/10 for "settlement along a riverbank"; median distance to "the river" 90 km). In Kerala, Kutch and the Sundarbans "the
  largest water component" is the sea, the Rann flats or the tidal network (10⁶ cells), not a river.
* **Judge base rates** (share of the 101,911 corpus graded relevant): "an urban residential neighborhood" 48%, "dense urban
  buildings" 44%, "open bare ground" 34%, "agricultural fields" 22%, "irrigated farmland" 19%, "a water body" 17%, …,
  river-relative 1.9–10%. The "built-up" signature fires on arid bare soil, so the urban queries are mostly base rate.

### A.5 Result — corrected retrieval metrics (mixed judge, 16 queries, frozen corpus)

| system | condition | K | P: baseline → full | NDCG (full ideal) | Recall (whole set) | Recall, capped | P, grade 2 only |
|---|---|--:|--:|--:|--:|--:|--:|
| RemoteCLIP | global | 5 | 0.038 → **0.287** | 0.247 | 0.0001 | 0.287 | 0.225 |
| RemoteCLIP | global | 10 | 0.031 → **0.300** | 0.250 | 0.0002 | 0.300 | 0.212 |
| RemoteCLIP | global | 20 | 0.028 → **0.306** | 0.252 | 0.0006 | 0.306 | 0.209 |
| RemoteCLIP | Ayodhya-filtered | 5 / 10 / 20 | 0.425 / 0.381 / 0.356 (unchanged) | 0.352 / 0.328 / 0.297 | 0.015 / 0.029 / 0.050 | 0.425 / 0.381 / 0.356 | 0.225 / 0.219 / 0.169 |
| vanilla CLIP | global | 5 | 0.000 → **0.225** | 0.190 | 0.0001 | 0.225 | 0.175 |
| vanilla CLIP | global | 10 | 0.000 → **0.225** | 0.191 | 0.0002 | 0.225 | 0.169 |
| vanilla CLIP | global | 20 | 0.006 → **0.241** | 0.200 | 0.0004 | 0.241 | 0.175 |
| vanilla CLIP | Ayodhya-filtered | 5 / 10 / 20 | 0.263 / 0.238 / 0.231 (unchanged) | 0.230 / 0.207 / 0.197 | 0.006 / 0.010 / 0.020 | 0.263 / 0.238 / 0.231 | 0.175 / 0.144 / 0.138 |

Metric definitions (also in `geoseek.eval.full_judge`): precision keeps the baseline definition; NDCG's ideal and Recall's
denominator are now the whole candidate set (the baseline's were the ~55-tile judged pool), so **only precision is
comparable row-for-row with the baseline**; Recall over a 100k corpus is bounded by K / n_relevant, hence "capped".

**Diff against `baseline_v1.json`** (`scripts/compare_to_baseline.py`, paired bootstrap over the 16 queries; full table in
`data/eval/phase10_blockA_diff.md`):

| metric | baseline | full coverage | Δ | paired 95% CI | verdict |
|---|--:|--:|--:|---|---|
| RemoteCLIP global P@5 | 0.038 | 0.287 | +0.250 | [+0.100, +0.425] | improved |
| RemoteCLIP global P@10 | 0.031 | 0.300 | +0.269 | [+0.106, +0.450] | improved |
| RemoteCLIP global P@20 | 0.028 | 0.306 | +0.278 | [+0.125, +0.447] | improved |
| vanilla global P@5 / P@10 / P@20 | 0.000 / 0.000 / 0.006 | 0.225 / 0.225 / 0.241 | +0.225 / +0.225 / +0.234 | all exclude 0 | improved |
| RemoteCLIP and vanilla Ayodhya-filtered P@5/10/20 | — | identical | +0.000 | [0.000, 0.000] | within noise (the control: already fully judged) |

The recall and NDCG rows in that diff also move, but those deltas mix coverage with the definition change above and are not
evidence about coverage.

### A.6 What the three claims come to

1. **"~78% of the collapse is a judge-coverage artifact" — confirmed, as a lower bound.** The spectral judge alone
   recovers 64.5% / 76.8% / 84.8% of the RemoteCLIP P@5/10/20 gap (vanilla 86% / 95% / 104%).
2. **"81% of the unjudged top-ranked tiles are on-target" — not reproduced; neither confirmed nor refuted.** The spectral judge
   gives 28.5% on the same tiles (30.7% of all unjudged top-20 slots; 20.0% at grade 2). The disagreement is systematic and
   explained above (fallow cropland, arid soil, river-relative queries) but the 81% rests on one labeller looking at the
   RGB tile the model also saw. The corrected P@20 lies above 0.306; settling it needs a stratified random sample labelled
   independently (not available here). **This is recorded in `EVALUATION_REPORT.md` §7.4.**
3. **"The region pre-filter is a 12.7× precision lever (0.028 → 0.356)" — the numbers reproduce, the interpretation does not.**
   Under full coverage the lever is **1.16×** (0.306 → 0.356). Against chance (prevalence of relevant tiles in the candidate
   set: 15.3% corpus-wide, 13.6% inside Ayodhya) RemoteCLIP lifts 2.01× globally vs 2.62× filtered (≈1.3×; vanilla 1.58× vs
   1.70×). On the 7 core queries the filter is worse (0.507 vs 0.429; lift 1.79× vs 1.58×); per query it can erase relevance
   ("dense urban buildings" 0.70 → 0.05, "open bare ground" 0.95 → 0.20). **The §7.3 result P@20 0.028 → 0.356 reproduces
   exactly (Phase 9 baseline) — but it is a measurement of judge coverage, not of retrieval precision.** Block B therefore
   reads the pre-filter as a *scoping* mechanism and measures it against chance-adjusted precision, not against the 12.7× figure.

| P@20, by query subset | nq | RC global (full) | RC Ayodhya-filtered | VA global | VA filtered |
|---|--:|--:|--:|--:|--:|
| all | 16 | 0.306 | 0.356 | 0.241 | 0.231 |
| excluding low-confidence | 13 | 0.373 | 0.381 | 0.292 | 0.277 |
| region-agnostic (no river-relative grader) | 9 | 0.400 | 0.400 | 0.317 | 0.306 |
| core (region-agnostic, not low-confidence) | 7 | 0.507 | 0.429 | 0.407 | 0.379 |

### A.7 Limits of Block A

The judge is a spectral proxy calibrated on one region; its recall against visual labels is 33%, and its river-relative
graders are not valid outside Ayodhya (reported separately, never in the headline subsets). The visual annotations are
single-labeller. The 4-query "validated vs visual" subset used in analysis (water body, bare ground, dense urban, residential;
global P@20 0.81 vs 0.33 filtered) was chosen from the annotation agreement after the fact and is not a headline. Descriptor
coverage is complete for Sentinel-2 (104,089 / 104,089); Maxar tiles (RGB only) have none by construction. Battery→AC: staging
began on battery; no throughput figure from it is claimed.

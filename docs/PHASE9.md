# Phase 9 — measurement and re-embedding infrastructure

No model was changed. Nothing in retrieval, change detection or detection changed behaviour; every
production file (`tiles.faiss`, `tiles.sqlite`, all four weights files) is byte-identical to the start of the
phase (SHA256 checked at the end, §9). What this phase adds is the ground that every later model migration
stands on: a frozen, re-measured baseline; a harness that says whether a candidate beat it by more than noise;
a resumable re-embedding pipeline that never touches the production index; and a registry that is the only
place a weights path, hash, licence and preprocessing config live.

## 0. Headline

**Every number that is a pure function of frozen inputs reproduced.** 124 retrieval, OSCD and detector numbers
were re-measured from scratch and all match the reports to the reports' own rounding (largest difference
5.2e-4); the detector AP/CI values and the OSCD TP/FP/FN counts match to the last digit.

**9 of 152 reported numbers did not reproduce — all 9 are inventory, footprint or test-suite counts that went
stale because the repository changed after the report, not measurements that failed to repeat** (§3.2). They
are the ones to read first:

| reported | measured now | why |
|---|--:|---|
| FAISS vectors / Sentinel-2 embedded tiles: 101,911 | 105,245 / 104,089 | Phase 8 appended 2,178 Ayodhya 2025/2026 tiles and 1,156 Maxar tiles. The report's corpus survives exactly as the prefix `faiss_id < 101911` (all Sentinel-2) and is what every "full corpus" number below is measured on. |
| catalog tiles 104,990 / scenes 75 | 108,324 / 81 | same growth (+3,334 tiles = 2,178 + 1,156; +6 scenes = 2 Ayodhya + 4 Maxar observations) |
| `data/` 34,450 MB, `datasets/` 32,396 MB | 329,847 MB, 145,898 MB | Phase 8F staged xView (~38 GB), three DOTA variants (~64 GB) and detector evaluation artifacts (~39 GB) |
| `models/` 1,210 MB | 1,299 MB | detector weights and YOLO initialisation checkpoints staged in Phase 8F |
| test suite: 228 passed, 0 failed | 495 passed, **4 failed**, 3 skipped | the suite grew; and 4 tests in `test_frontend_offline.py` fail on the unmodified tree (§7) |

Also found while reproducing (not number-reproduction failures, but wrong or stale statements):

* The provenance manifest's `retrieval_evaluation.systems.vanilla_clip` says **"laion2b"**. The control model is
  OpenAI CLIP (`pretrained='openai'`, SHA256 `e6d1bd77…`); a fresh re-embed with those weights matches the stored
  Phase 7a vanilla vectors at cosine ≥ 0.9999995, which settles it. The string comes from
  `scripts/eval_retrieval_score.py:176`. Not edited here.
* `docs/EVALUATION_REPORT.md` §13 and the project rules still say the suite is "228 passed / 3 skipped".
* The seam grep in §13 ("only `catalog/{sqlite_repository,migrate}.py` and `vectorindex/faiss_flat.py`") now also
  lists `catalog/embedding_map.py`, added in this phase (§5.4).
* `scripts/rebuild_index.py` **deletes `tiles.faiss` and `tiles.sqlite`** before rebuilding, with no prompt. It
  was not run. `scripts/reembed.py` is the safe alternative: it only ever writes to new paths.
* Text-search latency is slower and noisier than reported: median 31.6 ms (reported 27.8), p95 51.3 ms
  (reported 38.3), per-pass medians 24.7 / 33.4 / 34.2 ms. It is inside the 40% tolerance used for latency
  (the live index is also ~4% larger) so it is labelled reproduced, but the p95 is 34% above the report.

## 1. Pre-flight

The project rules file was read in full, then `EVALUATION_REPORT.md`, `PHASE8F2.md`, `PHASE8F3A.md`.

* **`git status` before any write:** branch `main` at `3b7ee24`, **dirty before this phase began**: 14 modified
  tracked files (cloud-deployment path overrides in `config.py`, `ingest/{embed,store}.py`, `search/engine.py`,
  `catalog/{migrate,sqlite_repository}.py`, Docker files, …) plus untracked `deploy/`, `.env.example`,
  `.gcloudignore`, `.gitattributes`, `tests/test_cloud_config.py`. They are not mine and I did not touch them
  (the diff stat — 14 files, +73/−32 — was identical at the end). The reviewed diffs only add environment-variable
  path overrides; defaults are unchanged. Every write in this phase is a **new file**.
* **`pytest -q` before:** `4 failed, 450 passed, 3 skipped` in 120 s. The 4 failures are
  `test_frontend_offline.py::test_no_external_urls_in_text_asset[vendor/{chart.umd.js, leaflet/leaflet.css,
  leaflet/leaflet.js, three.module.min.js}]` — vendored libraries whose comments contain URLs and which are not in
  the vendor allowlist in `data/provenance_manifest.json`.
* **`pytest -q` after:** `4 failed, 495 passed, 3 skipped` in 110 s — the same 4, **+45 new passing tests**.
* Power: AC at the start of every timed section and at its end (recorded per section in the snapshot).

## 2. What was built

| file | purpose |
|---|---|
| `src/geoseek/models/registry.py` | declarative `key → ModelSpec`; SHA256-verifying `load_model` |
| `src/geoseek/models/vanilla_clip.py` | `EmbeddingModel` adapter for the vanilla CLIP control (needed so it can be re-embedded like any model) |
| `src/geoseek/ingest/reembed.py` | catalog tile selection, tile readers, checkpointed shards, manifest, resume, finalize, fidelity comparison |
| `src/geoseek/catalog/embedding_map.py` | the `faiss_id ↔ tile_id` mapping database for a candidate index |
| `scripts/reembed.py` | CLI for the above |
| `src/geoseek/eval/env.py` | environment capture (GPU, VRAM, driver, torch, **AC/battery**, git SHA + dirty fingerprint, timestamp) |
| `src/geoseek/eval/retrieval.py` | retrieval metrics against the frozen Phase 7a judgements, for any embedding system |
| `src/geoseek/eval/compare.py` | snapshot diff: delta, noise band, paired test, polarity |
| `src/geoseek/eval/reported.py` | the numbers the reports claim (used **only** to label results REPRODUCED / NOT_REPRODUCED) |
| `scripts/snapshot_baseline.py` | runs every evaluation, writes `data/eval/baseline_v1.json` |
| `scripts/compare_to_baseline.py` | diff table for two snapshots |
| `tests/test_{registry,reembed,compare,eval_retrieval}.py` | 10 + 11 + 16 + 8 tests |

Artifacts (git-ignored under `data/`): `data/eval/baseline_v1.json` (the deliverable, 0.21 MB),
`data/eval/baseline_parts/*.json` (per-section parts), `data/eval/logs/`, `data/index/shards/baseline_v1/` and
`…_uninterrupted/` (shards + manifests), `data/index/candidates/` (finalized RemoteCLIP and vanilla indexes over the
frozen corpus + mapping databases).

## 3. Deliverable 1 — the frozen baseline

`python scripts/snapshot_baseline.py` re-runs everything and writes one JSON. Sections are independent (each writes
its own part file as soon as it finishes) and the timed ones refuse to run on battery. Nothing is copied from a
markdown report: the reported values live in `geoseek.eval.reported` and are used only to compute the
`reproduction` block. The production index, catalog and weights are never written — latency runs against a scratch
copy, ingestion into a scratch index, the detector evaluation into a scratch directory, FAISS persist time to a
scratch path. `train_change.evaluate()` was deliberately **not** called: it re-saves the production checkpoint and
rewrites the manifest.

Contents: `environment`, `corpus`, `retrieval`, `change_detection`, `detector`, `latency`, `footprint`,
`ingestion`, `suite`, `noise_bands` (248 entries, with `noise_detail`), `reproduction`, `meta` (SHA256 of every
input: production index and catalog, judgements, all four weights, both candidate indexes).

### 3.1 Measured values

**Retrieval — Ayodhya corpus (3,267 tiles, 16 queries, frozen Phase 7a judgements).** Rankings are computed fresh
(exact inner product); vanilla vectors come from a fresh re-embed. All 16 queries' top-20 lists are identical to the
stored Phase 7a pools for both systems.

| K | RC R | RC P | RC NDCG | VA R | VA P | VA NDCG | ΔNDCG |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.040 | 0.562 | 0.438 | 0.019 | 0.312 | 0.281 | +0.156 |
| 5 | 0.228 | 0.425 | 0.397 | 0.062 | 0.263 | 0.231 | +0.165 |
| 10 | 0.365 | 0.381 | 0.419 | 0.115 | 0.238 | 0.223 | +0.195 |
| 20 | 0.705 | 0.356 | 0.526 | 0.227 | 0.231 | 0.254 | +0.272 |

Excluding the 3 low-confidence road/bridge queries (13 queries): RC NDCG@5/10/20 0.454 / 0.480 / 0.547,
VA 0.257 / 0.244 / 0.282 (full table in the JSON).

**Retrieval — frozen 101,911-tile corpus** (= production `faiss_id < 101911`):

| condition | system | K=5 R / P / NDCG | K=10 R / P / NDCG | K=20 R / P / NDCG |
|---|---|---|---|---|
| global (no filter) | RemoteCLIP | .016 / .038 / .026 | .027 / .031 / .026 | .106 / .028 / .047 |
| global (no filter) | vanilla | .000 / .000 / .000 | .000 / .000 / .000 | .010 / .006 / .006 |
| Ayodhya region-filtered | RemoteCLIP | .228 / .425 / .397 | .365 / .381 / .419 | .705 / .356 / .526 |
| Ayodhya region-filtered | vanilla | .062 / .263 / .231 | .115 / .238 / .223 | .227 / .231 / .254 |

The global rows are dominated by the known judge-coverage artifact (§7.3 of the evaluation report): top-20
composition for RemoteCLIP is 2.8% judged-relevant / 6.6% judged-zero / 90.6% unjudged other-region (reproduces the
report exactly). **The vanilla at-scale rows are new — the report never measured them** — and show the same
artifact in a stronger form (96.3% of vanilla's global top-20 is unjudged other-region). The region-filtered rows are
numerically identical to the Ayodhya-corpus table above, as they must be (the Ayodhya region mask equals the judged
3,267 tiles; recorded as `checks.region_filter_mask_equals_judged_ayodhya_mask = 1`).

Cross-checks recorded in `retrieval.checks`: fresh vanilla vectors vs the stored Phase 7a vectors, n = 3,267, min
cosine 0.9999995, max |diff| 3.6e-7, tile order identical; fresh RemoteCLIP re-embed vs production, n = 101,911 (§6.3).

**Change detection — FC-Siam-diff, 10 held-out OSCD regions** (threshold 0.80 re-derived from the 3 validation
regions; the checkpoint's stored value is also 0.80):

| thr | P | R | F1 | IoU | FPR | TP | FP | FN |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.50 | 51.7% | 61.0% | 56.0% | 38.8% | 3.10% | 96,997 | 90,601 | 62,080 |
| 0.80 | 60.3% | 51.0% | 55.3% | 38.2% | 1.83% | 81,146 | 53,342 | 77,931 |

Per region (P/R/F1/IoU/FPR/counts, both thresholds) is in `change_detection.oscd_heldout.thr_*.per_region.*`.
Test pixels 3,077,936, change fraction 5.17%.

**Detector — DOTA official val, v1.5 labels, full-image protocol, 458 images** (deployed weights, SHA verified;
bootstrap 100 resamples over images):

| group | AP50 [95% CI] | AP50:95 [95% CI] | GT |
|---|--:|--:|--:|
| ground vehicles | 0.854 [0.786–0.891] | 0.471 [0.424–0.508] | 14,902 |
| ships + aircraft | 0.859 [0.735–0.911] | 0.557 [0.477–0.598] | 12,752 |
| infrastructure | 0.751 [0.704–0.784] | 0.414 [0.390–0.441] | 4,398 |
| all 8 classes | 0.817 [0.764–0.845] | 0.482 [0.447–0.498] | 32,052 |

Operating confidence 0.525 (re-derived on the monitor split), per-class AP and operating-point P/R/F1 are in the JSON.
Fresh results equal the stored `data/detect_eval/eval_results.json` to 0.0 for every group AP and every CI endpoint.

**Latency** (live index, 105,245 vectors, scratch copy, AC, `measure_tier.py` query sets and seeds, 3 discarded
warm-ups, 3 passes pooled):

| operation | n | median | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|
| text search k=20 | 120 | 31.62 ms | 51.27 | 52.71 | 56.05 |
| image search k=20 | 360 | 16.71 ms | 21.83 | 23.57 | 25.93 |
| point-seeded KNN | 450 | 17.51 ms | 23.59 | 27.67 | 33.34 |
| tile-seeded KNN | 450 | 17.03 ms | 22.34 | 25.49 | 28.08 |
| bbox filter (R*Tree) | 720 | 0.28 ms | 0.43 | 0.64 | 1.86 |

**Footprint** (live): `tiles.faiss` 215,541,805 B = 105,245 × 2,048 B + 45 B header; `tiles.sqlite` 95,887,360 B
(885 B per catalog tile); index (FAISS + catalog) 2,959 B per embedded tile; FAISS full-file persist to scratch
154 ms. The frozen-corpus RemoteCLIP candidate index is 208,713,773 B (2,048.0004 B/vector).

**Ingestion** (scratch index, Kanha scene, 1,024 tiles, AC, median of 3 warm runs after 1 cold): **204.9 tiles/s
end-to-end** (cold first run 196.5), **389.6 tiles/s pure embed**, peak VRAM 739 MB. Append-only proof in
scratch: ingest scene A (1,024 tiles) then scene B (1,089): all 1,024 of A's vectors byte-identical afterwards,
no rebuild (`incremental_append_byte_identical_fraction = 1.0`).

**Environment** (`environment`): RTX 4060 Laptop GPU, 8,188 MiB, driver 592.82, torch 2.2.2 (CUDA 12.x), AC power
(98–99% battery) for every section, HEAD `3b7ee24` **dirty** (the 14 pre-existing modified files; their diff
SHA256 is recorded), timestamp, versions of numpy / faiss / open_clip / rasterio / ultralytics.

### 3.2 Reproduction status

152 reported numbers are checked (`reproduction.entries`); `reproduction.not_reproduced` lists the failures in the
required `{"status": "NOT_REPRODUCED", "reported", "reason", …}` shape (with the measured value and source).

| area | reported numbers checked | REPRODUCED | NOT_REPRODUCED | tolerance |
|---|--:|--:|--:|---|
| retrieval, Ayodhya corpus (RC + VA, all K, both query subsets) | 42 | 42 | 0 | 0.0015 abs (3-decimal rounding) |
| retrieval, frozen 101,911 corpus (global, region-filtered, composition) | 16 | 16 | 0 | 0.0015 abs |
| OSCD pooled + per-region + threshold + pixel counts | 49 | 49 | 0 | 0.0006 abs on fractions; counts exact |
| detector AP50 / AP50:95 / CI endpoints | 17 | 17 | 0 | 0.0015 abs |
| latency (median, p95, five operations) | 10 | 10 | 0 | ±40% (+0.5 ms for the sub-ms bbox filter); report was at 100,887 vectors |
| ingestion throughput + append proof | 3 | 3 | 0 | ±20%; append exact |
| footprint | 6 | 3 | 3 | bytes/vector ±1 B; per-tile ±20–25%; sizes ±5% |
| corpus counts | 6 | 2 | 4 | exact |
| test suite | 3 | 1 | 2 | exact |
| **total** | **152** | **143** | **9** | |

The 9 are listed in §0 with verified causes. Largest difference among the 124 retrieval/OSCD/detector numbers:
`retrieval.full_101911.remoteclip.global.k20.ndcg`, 5.2e-4 (report prints 0.048, measured 0.0475).
No measurement was impossible, so no metric slot holds a `NOT_REPRODUCED` placeholder; the comparison harness
nevertheless skips and lists any such placeholder if a future snapshot contains one.

## 4. Deliverable 2 — comparison harness

`python scripts/compare_to_baseline.py data/eval/baseline_v1.json candidate.json [--only PREFIX] [--significant-only]
[--md out.md] [--json out.json] [--fail-on-regression] [--no-paired]` prints, per numeric metric: baseline,
candidate, delta, relative delta, noise band, exceeds-noise, paired 95% CI, verdict. `delta = candidate − baseline`;
a delta exceeds the band only if `|delta| > band` (strict). Direction uses the metric's polarity (lower is better
for FPR, latency, sizes; `tiles_per_s` is higher-is-better); a metric with no polarity rule reports `changed`.
Metrics present on one side only, and `NOT_REPRODUCED` placeholders, are reported, never silently dropped. **Any
later phase that claims an improvement must attach this diff.**

### 4.1 Noise bands — how they were estimated, and what the estimate says

Retrieval was re-run **3 times with different query orderings** (seeds 11/12/13; text re-encoded and ranked in each
order), as specified. **The spread is exactly 0.0 for every retrieval metric**: ranking is exact and the macro mean is
order-independent (rankings are identical across orderings: `checks.deterministic_rankings_across_orderings = 1`). A
zero band would make every difference "significant", so each recorded band is
`max(ordering spread, query-bootstrap 95% CI half-width)` — the bootstrap over the 16 queries (2,000 draws) is the
real uncertainty of a 16-query macro mean. The components are stored in `noise_detail`. This is a deviation from the
literal instruction, made because the literal estimate degenerates; it is conservative.

| metric family | band | method |
|---|---|---|
| retrieval (all) | e.g. NDCG@10 ±0.120, R@20 (global) ±0.125 | max(3-ordering spread = 0, query bootstrap) |
| OSCD pooled P/R/F1/IoU/FPR | e.g. F1@0.80 ±0.132 | max(3-pass inference spread = 0, region bootstrap over 10 regions) |
| OSCD per-region | 0.0 | spread over 3 inference passes (a fixed region has no sampling component) |
| detector AP50 / AP50:95 | e.g. all-class AP50 ±0.041 | bootstrap CI half-width over val images (100 resamples) |
| latency | e.g. text median ±9.5 ms, image ±0.47 ms, bbox ±0.03 ms | spread of the per-pass statistic over 3 passes |
| ingestion throughput | ±3.0 tiles/s | spread over 3 warm runs |

**The independent-sample bands are wide for 16 queries / 10 regions.** A real +0.07 pooled-F1 gain would read as
"within noise". So when both snapshots scored the *same* queries or regions (both carry per-query and per-region
data), the diff also computes a **paired bootstrap** of the delta; if the band says noise but the paired 95% CI
excludes zero, the verdict is `improved_paired` / `regressed_paired` (never plain `improved`). This is an addition
beyond the request; `--no-paired` gives the band-only behaviour. Detector and latency have no paired data and use the
band alone.

### 4.2 Harness self-check

Re-running the retrieval and OSCD sections into a separate parts directory and diffing against the baseline gives a
maximum |delta| of **exactly 0.0 across 376 metric rows** (151 retrieval, 225 change detection), and
`baseline_v1.json` is byte-identical afterwards. The candidate workflow was also exercised end to end on real data:
a snapshot built with `--retrieval-system remoteclip=remoteclip-vitb32=<the fresh re-embed index>` (i.e. "a candidate
model" whose vectors come from `scripts/reembed.py`) diffed against the baseline gives 0.0 on all 33 RemoteCLIP
Ayodhya metrics — expected, since the fresh vectors give identical rankings. The tests pin the arithmetic with hand-computed values (band strictness,
sign handling for both polarities, larger-of-two-bands, relative delta with a zero baseline, one-sided metrics,
placeholder skipping, paired CI behaviour, CLI exit codes).

## 5. Deliverable 3 — re-embedding pipeline

```
python scripts/reembed.py --model openclip-vitb32-openai --index-out data/index/candidates/x.faiss --batch-size 64
                          [--shard-size 5000] [--resume] [--finalize] [--collection C] [--max-faiss-id N]
                          [--verify-against data/index/tiles.faiss] [--stop-after-shards N] [--shards-root DIR]
```

* **Tile list from the SQLite catalog** (through the `MetadataRepository` seam), every tile with a `faiss_id`,
  ordered by production `faiss_id`. Pixels are read through a per-collection reader (Sentinel-2 L2A → the same
  `make_true_color_uint8` fixed stretch the production embedder uses; Maxar → delivered RGB). SAR tiles have no
  vectors and are never selected.
* **Checkpointed shards** (`--shard-size`, default 5,000) at `data/index/shards/<model-key>/shard_NNNNN.npy`, each
  written to a temp file, fsynced and renamed, then recorded with its SHA256 in `manifest.json` (replaced atomically).
  The manifest holds the model key, **verified** weights SHA256, preprocessing config, band selection per collection,
  batch size, shard size, tile-list SHA256, selection, source catalog, and per-shard `{start, count, sha256, bytes,
  seconds, tiles_per_s, peak VRAM}`.
* **Resume** (`--resume`) re-hashes every recorded shard, drops any missing or corrupt one, and continues from the
  first missing shard. An existing run without `--resume` is an error; resuming under a different model, weights,
  preprocessing, batch size, shard size or tile list is an error (batch size is pinned because it changes the float
  bytes).
* **`--finalize`** assembles the shards into a FAISS `IndexFlatIP` built at `<name>.partial`, verified (count, dim,
  byte round-trip of every vector), then renamed to `--index-out`; writes `<stem>.mapping.sqlite`
  (`new_id, tile_id, source_faiss_id, observation_id` + meta) and `<name>.finalize.json`. It **refuses** to write
  if the target exists or is the production index path. The production `tiles.faiss` is never opened for writing.
* **Logging** per shard: tiles/s, peak VRAM (allocated/reserved), ETA. The torch VRAM cap is set to 0.80 of the
  device (the cap used by `detect/train.py` and `detect_bench.py`).
* Pixels stream in `batch_size` chunks with one chunk of read-ahead, so a shard of 1,024 px Maxar tiles never sits
  in memory.

### 5.1 Measured throughput (AC, RTX 4060 Laptop, batch 64, shard 5,000)

| run | tiles | shard time | tiles/s | per-shard range | peak VRAM alloc / reserved |
|---|--:|--:|--:|--:|--:|
| RemoteCLIP, uninterrupted | 101,911 | 356.6 s | **285.8** | — | 739 / 818 MB |
| RemoteCLIP, killed + resumed (the 21 shards together) | 101,911 | 345.1 s | 295.3 | 271–340 | 739 / 818 MB |
| vanilla CLIP, uninterrupted (wall 367.3 s) | 101,911 | 367.2 s | **277.6** | 258–319 | 777 / 856 MB |
| RemoteCLIP, Maxar 1,024 px tiles (batch 32) | 1,156 | 17.2 s | 67.0 | 62–80 | 662 / 734 MB |

The VRAM cap is 0.80 × 8,187.5 MiB = 6,550 MiB; the highest reserved value observed is 856 MiB (13% of the cap).
Re-embedding is faster than the 205 tiles/s ingestion pipeline because it skips quality scoring and the SQLite/FAISS
appends; pure model throughput matches the 386–390 tiles/s measured by the ingestion section.

**Projected wall time:** the frozen 101,911-tile corpus ≈ **6.0 min** (measured 5.9–6.1 min of shard time per model,
plus model load and finalize); the whole live index (104,089 Sentinel-2 at ~286 tiles/s + 1,156 Maxar at ~67 tiles/s)
≈ **6.4 min**. On battery these would roughly double (the evaluation report's measured ~2× battery penalty) — that
figure is an estimate, not measured here.

### 5.2 Resumability, measured on the real corpus

* The RemoteCLIP run over the 101,911 tiles was **hard-killed (process force-stopped) 45 s in**: manifest recorded 2
  complete shards, **0 `.tmp` files** left. `--resume` computed the remaining 19 shards (91,911 tiles, 313 s) and
  finalized.
* A second, uninterrupted RemoteCLIP run was then made into a separate directory. **All 21 shard SHA256s are
  identical** between the killed-and-resumed run and the uninterrupted run (21/21).
* The unit tests prove the same property with deterministic fakes (shard bytes, manifest hashes and the finalized
  index all equal) and exercise the failure paths: crash mid-shard, silent corruption of a shard, a vanished shard,
  resume without the flag, config drift, non-unit-norm output, finalize guards. Two mutations (skipping the shard
  hash re-check; ignoring `batch_size` drift) were each caught by a test.

### 5.3 Fidelity to the production index

Re-embedding the 101,911 tiles with the *same* model and comparing to the production vectors: 101,054 rows
(99.16%) byte-identical; max |diff| 7.5e-7; mean top-10 neighbour overlap 1.0 and top-1 agreement 1.0 over 200
sampled queries. The non-identical rows are a batch-composition effect, not a preprocessing difference: re-embedding
exactly the first scene with the original batching (one scene, batch 64) gives **1,089 / 1,089 byte-identical, max
|diff| 0.0**. Maxar tiles through the Maxar reader: min cosine 0.9999995 vs production, neighbour overlap 1.0.
The `WARNING … No pretrained weights loaded for model 'ViT-B-32'` printed at load is from the existing
`open_clip.create_model_and_transforms(..., pretrained=None)` call used only to obtain the preprocessing transform;
the real weights are then loaded and the log shows `All keys matched successfully`.

### 5.4 Seam note

`catalog/embedding_map.py` imports `sqlite3`. It lives in the catalog package, is the only new importer, and never
touches the production schema. FAISS access goes through `FaissFlatIPIndex`; catalog reads go through
`SQLiteMetadataRepository`.

## 6. Deliverable 4 — model registry

`src/geoseek/models/registry.py`. Entries (key — kind — implementation — dim — licence):

| key | kind | implementation | dim | weights SHA256 (prefix) | licence |
|---|---|---|--:|---|---|
| `remoteclip-vitb32` | embedding | `RemoteCLIPEmbeddingModel` | 512 | `60014e39…` | Apache-2.0 |
| `openclip-vitb32-openai` (control) | embedding | `VanillaClipEmbeddingModel` | 512 | `e6d1bd77…` | MIT |
| `fc-siam-diff-oscd` | change | `FCSiamDiffChangeModel` | — | `452ac062…` | Apache-2.0 code / CC-BY-NC-SA-4.0 weights |
| `yolo26s-obb-dota15` | detector | `YoloObbDetectionModel` | — | `67f61717…` | AGPL-3.0 + academic/non-commercial |

Each spec also carries the weights path (relative to the models or data root, with the `MODEL_PATH` override the
settings already honour, and one glob for the Hugging Face snapshot directory), the preprocessing config, the source
URL and notes. `load_model(key)` resolves the path, **hashes the file, and raises `WeightsIntegrityError` on any
mismatch before constructing anything** (a missing file raises `WeightsMissingError`); for RemoteCLIP it also refuses
if the path it verified is not the path the implementation will open. `ultralytics` is only imported lazily by the
detector implementation, never by the registry. Existing call sites (`ingest.embed`, `FCSiamDiffChangeModel`,
`analyst.detections`) were not rewired — "from now on" applies to new code. A test asserts the registry's
embedding preprocessing equals `RADIOMETRY_CONFIG`, so the two cannot drift silently. The vanilla control is a
fourth entry beyond the three requested, because the baseline and the re-embed drill both needed it.

## 7. Constraints and open items

* **Additive only.** 13 new source / script / doc files plus 4 new test files; no existing file edited. `git diff --stat` identical to the
  start (14 pre-existing modified files).
* **Production untouched.** `tiles.faiss`, `tiles.sqlite`, `RemoteCLIP-ViT-B-32.pt`, `fc_siam_diff.pt` SHA256 compared
  to the values taken before any work: identical (`00c8d73a…`, `ee5c25e3…`, `60014e39…`, `452ac062…`); detector
  weights hash equals its model card.
* **Offline.** No network access; no URL added to any shipped asset (source URLs in the registry are metadata strings).
* **Attribution grep** (project rule 3: no AI-tool or assistant name anywhere) over `src/`, `docs/` and every new
  script and test: clean.
* **Not committed.** The tree contains uncommitted changes that are not part of this phase, and no commit was
  requested; nothing was staged.
* **`pytest -q` is not green, and not because of this phase.** The 4 failures are the vendored-asset URL guard. Its
  allowlist exists so a human re-reviews each library's external references before recording them in the
  provenance manifest; filling it in to turn the suite green would defeat the guard, so it was left alone. They need
  a deliberate decision (add the four files with their URLs and SHA256 to the allowlist, or strip the URL comments
  from the vendored copies), and the project rule "never commit red" applies until then.
* **Limits.** Battery-power throughput was not measured. The kill/resume drill was run on the Sentinel-2 corpus, not
  on Maxar. Detector rerun-to-rerun spread was not measured (no GPU-nondeterminism repeats; the band is the bootstrap).
  Latency used 3 passes and a 40% reproduction tolerance. The snapshot depends on the candidate indexes under
  `data/index/candidates/` (git-ignored) — regenerate them with the commands below. The baseline was captured with the
  tree dirty; the dirty-diff SHA256 is in `environment.git`.

## 8. Reproduce

```
export GDAL_DATA=…/geoseek/Library/share/gdal PROJ_LIB=…/Library/share/proj PYTHONIOENCODING=utf-8
# 1. re-embed the frozen corpus with both systems (resumable; ~6 min each on AC)
python scripts/reembed.py --model remoteclip-vitb32       --max-faiss-id 101911 --shards-root data/index/shards/baseline_v1 \
       --index-out data/index/candidates/remoteclip_frozen101911.faiss --resume --finalize --verify-against data/index/tiles.faiss
python scripts/reembed.py --model openclip-vitb32-openai  --max-faiss-id 101911 --shards-root data/index/shards/baseline_v1 \
       --index-out data/index/candidates/vanilla_frozen101911.faiss   --resume --finalize
# 2. snapshot (sections can be run separately; timed ones need AC power)
python scripts/snapshot_baseline.py                     # -> data/eval/baseline_v1.json
# 3. a later phase: write its own snapshot elsewhere, then diff
python scripts/snapshot_baseline.py --parts-dir data/eval/cand_parts --out data/eval/candidate.json --snapshot-id cand \n       --retrieval-system remoteclip=<new-model-key>=data/index/candidates/<new>_frozen101911.faiss
python scripts/compare_to_baseline.py data/eval/baseline_v1.json data/eval/candidate.json --significant-only
pytest -q                                               # 495 passed, 4 failed (pre-existing), 3 skipped
```

To evaluate a *different* embedding model, register it in the registry, re-embed the frozen corpus with
`scripts/reembed.py`, then name it in the snapshot — reusing the baseline's system label so the metric paths line up
for the diff:

```
python scripts/snapshot_baseline.py --sections env,corpus,retrieval --parts-dir data/eval/cand_parts --out data/eval/candidate.json        --snapshot-id cand --retrieval-system remoteclip=<new-model-key>=data/index/candidates/<new>_frozen101911.faiss
```

(`INDEX` may also be the word `production`.) The OSCD and detector sections evaluate the registry's change and detector
entries; a candidate change or detection model of a different architecture needs those two sections adapted
(they use `FCSiamDiff` / `eval_detector.py` directly) — not done here.

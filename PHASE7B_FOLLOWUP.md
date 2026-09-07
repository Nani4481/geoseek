# Phase 7b follow-up — precision diagnosis, precision fixes, throughput root cause, clustering at 100k

Four investigations opened by the Phase 7b Tier-3 run (101,911 tiles). Every
number below is from a real run on this machine against the live 101,911-vector
index; scripts are named at each section and the raw outputs are the committed
JSON under `data/eval_retrieval/` and `data/discovery/`.

| # | question | answer |
|---|---|---|
| 1 | Is the Tier-3 "precision drop" real, or a judgement artifact? | **~78% artifact.** Of the high-ranked tiles the frozen judge can't score, 81% are visually on-target for their query; they're unjudged (out of the Ayodhya judge's domain), not judged-bad. Restricting the evaluation to the Ayodhya sub-corpus reproduces the Phase 7a numbers exactly. |
| 2 | Which precision improvements actually help? | **Region metadata pre-filter: the whole story** (P@20 0.028→0.356, NDCG@20 0.048→0.526 — full Phase 7a recovery). Near-duplicate suppression: marginal positive. Score thresholding: a no-op at K≤20 on a corpus this size. |
| 3 | Why did ingestion throughput ~halve mid-run? | **The laptop was unplugged from AC at 2026-09-06 11:10:29 UTC**, between ledger rows 47 and 48. Battery power caps the RTX 4060 Laptop GPU and CPU. A measurement-environment artifact, not a corpus-size or thermal effect. |
| 4 | HDBSCAN doesn't finish at 100k. | **Sample (20k, stratified) + nearest-centroid assignment finishes in 3.6 min** and recovers the Tier-1 cluster structure (non-noise ARI 0.96). Hard assignment labels every tile, so it does not reproduce HDBSCAN's noise set (all-tiles ARI ~0.5). |

---

## Part 1 — is the precision drop real, or a judgement artifact?

`scripts/diagnose_judge_transfer.py` (visual), `scripts/eval_precision_at_scale.py` (quantitative).
Artifacts: `data/eval_retrieval/judge_diagnosis/` (16 contact sheets + `manifest.json` +
`ANNOTATIONS.json` + `diagnosis_summary.json`), `data/eval_retrieval/precision_at_scale.json`.

### The problem with the Tier-3 number

Tier 3 reported **recall@20 of the originally-relevant set = 10.6%** and a
**90.6% "distractor" fraction**. The Phase 7a relevance judgements are physical
facts about 3,267 specific **Ayodhya** tiles — per-tile NDVI/NDWI/NDBI/SCL
thresholds plus distance to a mask of the *Ayodhya* river. At 101,911 tiles the
top-K for these queries is dominated by tiles from 8 other regions that were
never judged and **cannot** be with the frozen judge: the new regions were
staged RGB + SCL only (no NIR/SWIR → no NDVI/NDWI/NDBI), and 5 of the 16
queries are relative to Ayodhya's river specifically. Scoring those tiles as
"irrelevant" is what produces the collapse.

### Global top-20 composition (`eval_precision_at_scale.py`)

Averaged over the 16 queries, where RemoteCLIP's current global top-20 comes from:

| bucket | fraction of top-20 |
|---|--:|
| judged relevant (grade > 0) | 2.8% |
| judged 0 (judge saw it, said no) | 6.6% |
| unjudged, Ayodhya | 0.0% |
| **unjudged, other region** | **90.6%** |

So of the "distractors," only **6.6%** are Ayodhya tiles the judge actually
graded 0 (retrieval putting a known-bad tile high). The other **90.6%** are
unjudged, not judged-bad.

### Visual inspection of the un-scoreable tiles

For every query, the global top-12 tiles with **no positive judgement**
(judged-0 + unjudged) were rendered to a contact sheet and inspected by eye
(`ANNOTATIONS.json`: 1 = plausibly matches the query text, 0 = clearly not,
? = can't tell at 10 m GSD). 187 tiles across 16 queries:

| bucket | n | fraction plausibly relevant (visual) |
|---|--:|--:|
| unjudged, other region | 179 | **81.0%** |
| judged 0 (Ayodhya) | 8 | 0.0% |
| **overall** | **187** | **77.5%** |

- **Specific concepts transfer cleanly.** "dense urban buildings" → Delhi NCR
  urban fabric (11/12). "open bare ground" → Thar/Rann desert (12/12, and the
  highest similarity scores of any query). "a water body" → Sundarbans tidal
  channels + a Deccan reservoir (11/12). "a braided river channel with
  sandbars" → Dehradun Himalayan-foothill rivers (10/11). "agricultural
  fields", "cropland with visible field boundaries", "an urban residential
  neighborhood", "irrigated farmland" → 12/12 each. These are unjudged only
  because they're not in Ayodhya, and the Ayodhya corpus barely contains true
  bare ground or wide open water (Phase 7a noted both are "rare" there) — so at
  scale the model *correctly* surfaces the desert and delta regions and the
  judge simply cannot see them.
- **The weak queries are still weak, but not wrong.** "a bridge crossing a
  river" (3/12 confident, 8 ambiguous) and "a dirt track or unpaved road"
  (6/12, 6 ambiguous) — a bridge or track is sub-pixel-to-few-pixel at 10 m;
  the model retrieves river-with-crossing and desert-with-track scenes and
  most calls are "can't confirm", not "clearly irrelevant".
- **The 8 `judged_zero` tiles are genuine retrieval imperfections.** They are
  a small, recurring set of Ayodhya tiles with degenerate true-colour
  rendering (heavy haze/cloud; blue/yellow false-colour), which RemoteCLIP
  embeds moderately close to many *text* queries (score ~0.32–0.34). The
  judge correctly grades them 0. This is real, and it is ~6.6% of the top-20,
  not 90%.

**Conclusion.** The Tier-3 "precision drop" is roughly **78% a judge-coverage
artifact** — relevant, correctly-retrieved tiles the frozen judge is not
allowed to score — and only a few percent genuine retrieval error.

### The correction: restrict the evaluation domain

**Chosen: restrict the retrieval evaluation to the Ayodhya sub-corpus** — the
terrain the independent spectral judge was calibrated on, and the way an
analyst actually searches (a sector, not the whole archive). Rejected: per-region
judge recalibration — it needs NIR/SWIR re-staging for 8 regions (hours of
network fetch, out of scope for a follow-up) **and** the 5 river-relative
criteria are physically tied to the Ayodhya river mask, so they cannot be
made region-general at all.

### Corrected precision, alongside the originals (`eval_precision_at_scale.py`)

RemoteCLIP, macro-averaged over 16 queries, scored against the **frozen,
unchanged** Phase 7a judgements. `R` = recall, `P` = precision@K (padded to K,
Phase 7a convention), `N` = NDCG@K.

| condition | R@5 | P@5 | N@5 | R@10 | P@10 | N@10 | R@20 | P@20 | N@20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **Phase 7a (3,267-tile corpus)** | 0.228 | 0.425 | 0.397 | 0.365 | 0.381 | 0.419 | 0.705 | 0.356 | 0.526 |
| Tier 3 global, 101,911 tiles (original) | 0.016 | 0.037 | 0.025 | 0.027 | 0.031 | 0.026 | 0.106 | 0.028 | 0.048 |
| **Tier 3 + Ayodhya domain restriction (corrected)** | **0.228** | **0.425** | **0.397** | **0.365** | **0.381** | **0.418** | **0.705** | **0.356** | **0.526** |

The corrected row **reproduces Phase 7a to within rounding.** This is expected
and it is the point: the incremental-ingest proofs already showed the Ayodhya
vectors are byte-identical at every tier, so ranking the same 3,267 Ayodhya
tiles with the same (deterministic) RemoteCLIP text encoder gives the same
top-K and the same metrics. **Retrieval quality inside the judgeable domain is
unchanged after a 31× corpus growth across 8 new biomes.** The Tier-3 collapse
was entirely the artifact of scoring 90% unjudged out-of-domain tiles as
irrelevant.

---

## Part 2 — precision improvements, measured independently and combined

`scripts/eval_precision_at_scale.py`; helpers in `src/geoseek/search/rerank.py`
(pure, unit-tested). All rows scored against the frozen Ayodhya judgements.
`Pj` = precision over the *judged* subset only (denominator = retrieved tiles
that carry a judgement — honest about not knowing the rest); `sz` = mean
result-set size (< K only when a cut fires).

| condition | R@10 | P@10 | N@10 | R@20 | P@20 | Pj@20 | N@20 | sz@20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  global baseline (no fix) | 0.027 | 0.031 | 0.026 | 0.106 | 0.028 | 0.250 | 0.048 | 20.0 |
| **2a region metadata pre-filter** | **0.365** | **0.381** | **0.418** | **0.705** | **0.356** | 0.356 | **0.526** | 20.0 |
| 2b score threshold (τ\*=0.29) | 0.027 | 0.031 | 0.026 | 0.106 | 0.028 | 0.250 | 0.048 | 20.0 |
| 2c near-duplicate suppression (τ_dup=0.97) | 0.027 | 0.031 | 0.026 | 0.112 | 0.031 | 0.265 | 0.051 | 20.0 |
| 3  region + threshold | 0.365 | 0.381 | 0.418 | 0.694 | 0.353 | 0.366 | 0.521 | 19.4 |
| 3  region + dedup | 0.365 | 0.381 | 0.418 | 0.705 | 0.356 | 0.365 | 0.527 | 20.0 |
| 3  region + threshold + dedup | 0.365 | 0.381 | 0.418 | 0.694 | 0.353 | **0.375** | 0.522 | 19.4 |

### (a) Region metadata pre-filter — the entire win

Filter the candidate set to the analyst's region *before* semantic ranking
(for an exact flat index, pre-filter and post-filter give the identical top-K,
so it's applied as a post-filter here). **P@20 0.028 → 0.356 (12.7×), NDCG@20
0.048 → 0.526 (11×), R@20 0.106 → 0.705.** This is the same operation as the
Part-1 domain restriction; it is listed here too because it is *the*
architectural answer — "filter cheap first" — and the Tier-3 benchmark simply
didn't use it. Top-20 composition goes from 90.6% unjudged-other-region to
0%.

### (b) Score thresholding — a no-op at this scale

The threshold **τ\* = 0.29** was chosen on a **validation set disjoint from the
test rankings**: macro-F1@20 over the Phase 7a 3,267-corpus RemoteCLIP
rankings (see `precision_at_scale.json → validation.tau_curve`). Applied to the
100k rankings it changes **nothing** at K ≤ 20: with 101,911 tiles competing,
every query's global top-20 already sits above 0.29, so there is no weak tail
to trim inside the top-K. On the *region-filtered* corpus it trims ~0.6
tiles/query from the top-20 (sz 20 → 19.4), nudging Pj@20 0.356 → 0.366 at a
small recall cost (0.705 → 0.694). **Thresholding is the wrong tool for an
over-supplied corpus** — it helps a *small* index where top-K includes
sub-threshold padding (as some Phase 7a queries did), not a large one where the
problem is too many plausible matches.

### (c) Near-duplicate suppression — marginal positive

Greedy collapse in rank order: drop a tile whose cosine to an already-kept tile
≥ τ_dup. Suppressed in the **global top-100**, summed over 16 queries:

| τ_dup | 0.95 | 0.96 | 0.97 | 0.98 | 0.99 |
|---|--:|--:|--:|--:|--:|
| tiles suppressed | 443 | 297 | **175** | 109 | 24 |

At τ_dup = 0.97 (~11 per query in the top-100 — same place another date, or an
adjacent tile), the effect on the global metrics is small but positive: R@20
0.106 → 0.112, NDCG@20 0.048 → 0.051. On the region-filtered corpus it is
essentially neutral (NDCG@20 0.526 → 0.527). Worth keeping as a display-time
refinement; not a precision lever on its own.

### Combined

Stacking threshold + dedup on top of the region filter gives **region-filter
performance** (the K≤10 rows are identical) with a slightly tighter top-20
(Pj@20 0.356 → 0.375, R@20 0.705 → 0.694). **The region pre-filter delivers
100% of the recoverable precision; the other two are second-order polish.**

---

## Part 3 — ingestion throughput root cause

`data/eval_retrieval/throughput_diag/ingestion_throughput_diagnosis.json`.

### Finding

The Tier-3 ledger shows a clean step-change in ingestion throughput partway
through the ~4-hour unattended run. **It is a measurement-environment artifact:
the laptop was disconnected from AC power mid-run.** It is **not** a corpus-size
/ index-scale degradation and **not** thermal throttling.

### Evidence

**1. OS power-state logs place the transition inside the ledger gap.**

| source | event |
|---|---|
| Windows `System` log, `Kernel-Power` **event ID 105 "Power source change"** | 2026-09-06 **16:40:27 IST = 11:10:27 UTC** |
| `powercfg /batteryreport` "Recent usage" | `2026-09-06 11:06:45  Active  AC   60%` → `2026-09-06 16:40:29  Active  Battery 100%` (unplugged at full charge = **11:10:29 UTC**) → next AC reconnect `18:34:07 IST = 13:04:07 UTC`, well after the run ended (~12:35 UTC) |
| `diverse_ingest_ledger.json` row 47 (last AC scene) | finished **2026-09-06T11:06:25Z**, 203.6 tiles/s |
| `diverse_ingest_ledger.json` row 48 (first battery scene) | finished **2026-09-06T11:15:32Z**, 93.4 tiles/s |

The AC→battery switch at 11:10:29 UTC falls exactly between ledger rows 47 and
48. The machine then stayed on battery for the remaining 17 scenes of the run.

**2. Both halves of the ingest step doubled together — a whole-machine power
envelope change, not a GPU-only effect.** `mean_embed_latency_ms` is the wall
time of `encode_images` per tile (CPU PIL resize + GPU forward pass +
device→host copy). On battery the RTX 4060 Laptop GPU's power/clock budget
**and** the CPU package power limit are both cut:

| ledger segment | scenes | index vectors | big-scene tiles/s | big-scene embed ms/tile |
|---|--:|--:|--:|--:|
| **AC** (rows 0–47) | 48 | 6,965 → 75,843 | **210.5** (193.9–218.8, ±6%) | **2.48** |
| **Battery** (rows 48–64) | 17 | 77,692 → 100,887 | **123.7** (88.7–155.9) | **4.60** |

**3. Corpus size is ruled out.** Within the AC segment the index grew **~11×**
(6,965 → 75,843 vectors) while big-scene throughput held at 210.5 tiles/s ±6% —
flat. Ingestion cost in this architecture is strictly per-tile (preprocess +
forward pass + FAISS append + one SQLite insert) and does not scale with how
many vectors already exist; that was the expectation and it held on AC. The
drop to 123.7 tiles/s tracks the power-source change, not any index-size
threshold.

**4. Thermal is ruled out.** A thermal throttle relaxes during the multi-minute
network-fetch gap between scenes (GPU idle, short duty cycle). The degradation
instead persisted uniformly across all 17 post-switch scenes and every
inter-scene gap — the signature of a sustained power-envelope change, not heat.

### What this means for the Phase 7b numbers

- The Tier-3 measurement table cells **"ingestion throughput 144.1 tiles/s
  (−29%)"** and **"pure embedding 293.4 tiles/s (−22%)"** are computed over
  ledger rows that are **partly on battery** (rows 48–64 of that tier). They
  **understate** the machine's sustained capability.
- The AC-power rate — ledger rows 0–47, and Phase 7a / Tier 1 / Tier 2, all on
  AC — is **~205–210 tiles/s** and is the number that reflects sustained
  capability. No re-run is needed to establish this; the OS power-state log is
  authoritative.
- **Ingestion throughput is independent of index size in this architecture**
  (the flat AC segment across an 11× vector growth), so the slowdown says
  nothing about scalability.

### Hardware (for PS 2.3)

RTX 4060 Laptop GPU (8 GB), driver 592.82; 16 logical CPUs; Windows 11; power
scheme "Balanced". **On this class of machine AC vs. battery materially changes
sustained GPU/CPU throughput** — capability benchmarks must be run on AC and the
power source recorded. All Phase 7a / Tier 1 / Tier 2 numbers and Tier-3 ledger
rows 0–47 were on AC; Tier-3 ledger rows 48–64 were on battery.

---

## Part 4 — HDBSCAN clustering at 100k

`scripts/cluster_at_scale.py`; `cluster_sample_and_assign` in
`src/geoseek/discovery/cluster.py`. Artifacts: `data/discovery/cluster_at_scale_100k.json`,
`tile_clusters_at_scale_100k.json`, `cluster_at_scale_region_heatmap.png`.

Phase 7b Tier 3 established that a single-pass HDBSCAN over 101,911 raw 512-d
embeddings does not finish in practical batch time (>60 min, killed — the MST /
hierarchy-extraction step, which HDBSCAN does not parallelize, is the wall).

### The standard fix: cluster a stratified sample, assign the rest

1. Stratified sample of **20,000 tiles** across the 9 regions (proportional,
   with a 200-tile floor per region so small regions aren't washed out;
   `stratified_sample_indices`, seeded).
2. HDBSCAN on the sample (`min_cluster_size=40`, `cluster_selection_method="eom"`
   — same config as Tier-1's official clustering; `core_dist_n_jobs=-1`).
3. Assign **all 101,911 tiles** to the nearest sample-cluster centroid — one
   dense cosine matmul on unit-norm vectors.

### Runtime

| step | time |
|---|--:|
| bulk vector load (`FaissFlatIPIndex.reconstruct_all`, new seam method) | **< 0.1 s** |
| sample HDBSCAN (20k) + assignment (101,911) | **218.2 s (3.6 min)** |

vs. Tier-3's full HDBSCAN that did not complete in 60+ minutes.

### Clusters (full corpus, 6 clusters, hard assignment labels every tile)

| cluster | full size | sample size | nearest concept | dominant region (purity) |
|--:|--:|--:|---|---|
| 4 | 86,578 | 8,154 | bare dry open ground / rural village / quarry | jaisalmer (21%) — the "generic land" cluster |
| 3 | 5,761 | 56 | sandy braided riverbed / river with sandbars | sundarbans (59%) |
| 2 | 5,238 | 805 | open water reservoir / waterlogged land | kerala_backwaters (63%) |
| 5 | 2,167 | 132 | bare dry open ground | **kutch (91%)** |
| 0 | 1,410 | 190 | trees and dense vegetation | ayodhya (75%) |
| 1 | 757 | 61 | river with sandbars / braided riverbed | kanha (39%) |

4 of the 6 clusters (0, 2, 3, 5) are region-concentrated — the distinctive
water / delta / desert / vegetation terrain separates out — while a coarse
"generic land" cluster (c4) absorbs 85% of the corpus. This is the same
behaviour Tier-1's EOM run showed (3 clusters, 60% in the largest): EOM on this
diverse corpus produces a coarse split, and nearest-centroid assignment
amplifies it (no density notion → no abstention). Finer structure — the kind
Tier-1's *leaf*-mode secondary run surfaced (7/9 clusters >96% single-region) —
would come from PCA-to-32/64-d before clustering or a leaf-mode sample run;
flagged as follow-up in PHASE7B.md and not substituted in here.

### Agreement with the Tier-1 full clustering

On the 12,418 tiles common to both (all of Tier-1's tiles are still in the
index):

| comparison | n | ARI | AMI | homogeneity | completeness | V-measure |
|---|--:|--:|--:|--:|--:|--:|
| **non-noise in both** | 9,080 | **0.959** | 0.905 | 0.925 | 0.886 | 0.905 |
| all tiles (incl. Tier-1 noise) | 12,418 | 0.412 | 0.463 | 0.376 | 0.602 | 0.463 |

Non-noise contingency (Tier-1 cluster → at-scale cluster):

| Tier-1 | → at-scale | count |
|---|---|--:|
| c1 (largest) | **c4** | 7,705 / 7,754 |
| c0 | **c0** | 955 / 983 |
| c2 | **c3** | 311 / 343 |

**The sample-based method recovers the Tier-1 structure**: the 3 Tier-1
clusters map almost 1:1 onto 3 of the 6 at-scale clusters (ARI 0.96 on the
tiles Tier-1 is confident about). The all-tiles ARI is lower (0.41) for one
honest reason: hard nearest-centroid **labels every tile**, so it necessarily
disagrees with Tier-1 on exactly the 26.9% of tiles Tier-1 calls noise. An
abstention sweep (relabel low-confidence assignments as noise) confirms this —
all-tiles agreement peaks at ARI ≈ 0.53 around an abstain-cosine of 0.82
(≈21% noise) and at Tier-1's own 26.9% noise fraction sits at ARI 0.51:

| abstain cosine ≥ | noise | ARI (all tiles) |
|--:|--:|--:|
| 0.70 | 0.9% | 0.428 |
| 0.80 | 13.6% | 0.517 |
| 0.82 | 20.9% | **0.527** |
| 0.8325 (= Tier-1 noise frac) | 26.9% | 0.508 |
| 0.85 | 37.1% | 0.434 |

So: the *cluster geometry* transfers (non-noise ARI 0.96); centroid assignment
just draws the borderline/noise set differently than density-based HDBSCAN, and
no single abstention threshold fully reconciles the two. Reported plainly — this
is the known trade of the sample+assign approach.

---

## Tests / seams

`pytest -q`: **228 passed, 3 skipped** (the 3 skips are the pre-existing
`test_search_parity.py` guards). +20 new in `tests/test_phase7b_followup.py`:
`reconstruct_all` parity vs per-vector reconstruct + reopen; `region_key`
cases; `apply_score_threshold` prefix-cut; `filter_by_region` order; greedy
`suppress_near_duplicates`; `best_threshold_by_f1` picks the separating cut;
`stratified_sample_indices` floor/proportional/cap/determinism;
`cluster_sample_and_assign` recovers synthetic blobs, labels everything, unit-norm
centroids, abstention relabels.

`grep -rn "import sqlite3\|import faiss" src/`: unchanged — `sqlite3` only in
`catalog/{migrate,sqlite_repository}.py`, `faiss` only in
`vectorindex/faiss_flat.py`. New `scripts/*.py` import neither; all vector
access goes through `FaissFlatIPIndex` / `SearchEngine`. New code:
- `VectorIndex.reconstruct_all()` (default loop; `FaissFlatIPIndex` overrides
  with `reconstruct_n`) — additive, every existing caller unaffected.
- `src/geoseek/search/rerank.py` — new, pure, no I/O.
- `cluster.cluster_sample_and_assign` / `stratified_sample_indices` — new;
  `cluster_embeddings` and its callers/tests untouched (default path exercised
  by `test_phase5.py`).

## Run it

```
python scripts/eval_precision_at_scale.py
python scripts/diagnose_judge_transfer.py --top-n 12      # then eyeball data/eval_retrieval/judge_diagnosis/*.png
python scripts/cluster_at_scale.py --sample-size 20000 --n-jobs -1
pytest -q
```

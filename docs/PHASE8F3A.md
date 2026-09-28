# Phase 8F-3a — diagnosing the DOTA-to-Maxar domain gap

Status: **Step 1 built, waiting on human labelling.** Steps 3a/3b/3c/3e (the parts that don't need
Maxar ground truth) and Step 4 (training anomaly) are measured below. Step 2 (baseline) and the
GT-dependent halves of Step 3, and the final ranked Step 5 verdict, are **not done** — they need the
labelled test set this document sets up, and per the task's own instruction nothing was retrained,
fine-tuned, or invented as a placeholder for missing labels.

Builds on [[geoseek-phase8F2-detector]]. Read that first: the number this whole step exists to
explain is "0.854 AP50 on DOTA val, but a manual audit of 10 seeded Van Nuys crops found only ~2 of
~47 visible cars."

## Step 1 — labelled Maxar test set (built, NOT YET LABELLED)

`data/detect_eval/maxar_handlabeled_v1/` — 8 images from the Van Nuys quadkey
(`103001010C12B000_031311102120`), oriented boxes, DOTA text format
(`x1 y1 x2 y2 x3 y3 x4 y4 category difficult`), classes `small-vehicle` / `large-vehicle` only.
**Not authoritative ground truth** — one labeller, no adjudication pass; registered as such in the
provenance manifest (`maxar_vannuys_handlabeled_v1` analysis section) and in the set's own
`manifest.json` (`"authoritative": false`).

Selection (`manifest.json` has the exact rationale + tile/crop offset per image) was deliberately
**not** based on "where the detector already found something," to avoid building a GT set that only
confirms the model's existing blind spots:

| image | source | why |
|---|---|---|
| P9001 | tile r007_c005, crop 384px | dense parking lot + street, mixed use |
| P9002 | tile r014_c010, crop 384px | dense urban block, lot + street parking |
| P9003 | tile r011_c013, crop 384px | **the detector found ZERO objects on this full tile**, despite it visibly containing a dense lot (Sobel edge-density picked this window as texture-rich) |
| P9004 | tile r016_c010, crop 384px | same story — zero detections, visibly a dealership-style lot full of cars |
| P9005 | tile r010_c006, crop 384px | vehicles on a road beside a tall hangar (strong shadow) |
| P9006 | tile r009_c006, crop 384px | second hangar-shadow road, different sun geometry |
| P9007 | tile r008_c008, full 1024px | airport apron: aircraft rows + light ground traffic |
| P9008 | tile r006_c003, full 1024px | residential, sparse — negative-control density |

P9003/P9004 (picked by a Sobel edge-density scan among tiles with literally 0 detector output, not
by eye) are the most important images in the set: they were chosen specifically to catch whether the
model has a real blind spot, independent of confirming what it already half-sees.

### How to label (needs a human — I stopped here)

```
cd geoseek
python scripts/serve_label_tool.py
```
Opens `http://127.0.0.1:8765/` in the browser (offline, localhost-only, no external network/CDN).
Per image: **drag** on empty space to draw a box, **drag a corner** to resize, **drag the small
handle above the box** to rotate it to match the vehicle's heading, **1**/**2** to set
small-vehicle/large-vehicle, **Delete** to remove, **D** to flag an ambiguous/occluded instance as
`difficult` (excluded from scoring, same as DOTA), mouse wheel to zoom, right-drag to pan,
**Ctrl+S** or the Save button to write the image's `labelTxt/<stem>.txt` straight to disk (no
export/import step — it POSTs to the local server). **N** jumps to the next unlabelled image.
Autosaves a draft to the browser's local storage every few seconds as a crash safety net.

Label **every vehicle you can see** in each image, not just the ones the model already draws boxes
for (there are no model predictions shown in this tool at all — it never saw them, by design, so
there's no anchoring). Expect a few hundred instances total across the 8 images (dense lots have
tens to ~100 apiece); that's the intended scale, not a sign something's wrong. Once all 8 show
"labeled" in the left sidebar, tell me and I'll run Step 2.

`scripts/eval_maxar_handlabeled.py` (Step 2, ready to run, not yet run) scores the current detector
against whatever's labelled, reusing the existing evaluator (`geoseek.detect.evaluate`) unchanged —
confirmed it runs cleanly against the current empty label set (all-NaN, as expected; that dry run's
output was deleted, not kept as a "result").

## Step 3 — hypotheses tested without needing Maxar ground truth

### 3a. Pixel statistics — measured, one recovery test tried and it came back negative

250 random DOTA train chips vs 340 random tiles across all 4 staged Maxar quadkeys
(`data/detect_eval/phase8f3a_pixel_scale_stats.json`):

| | DOTA train chips | Maxar tiles |
|---|---|---|
| mean brightness (0-255) | 89.0 | **138.3** (1.55x brighter) |
| contrast (std) | 33.5 | 39.8 |
| sharpness (Laplacian variance) | 543 | **223** (less than half) |
| dark-pixel fraction (<40) | 0.127 | 0.132 (not different) |
| clipped-pixel fraction (<=1 or >=254) | 0.026 | 0.021 (not different) |

Maxar is meaningfully brighter and softer than DOTA; shadow/clipping stats are not distinguishable.
The softness matches the known ~1.6-1.9x upsampling from native WV02/GE01 resolution
([[geoseek-phase8F2-detector]]).

**Recovery test** (`phase8f3a_histmatch_test.json`): linearly matched 9 Maxar tiles' per-channel
mean/std to DOTA's measured mean/std, re-ran the detector at a low confidence floor (0.05, a
sensitivity probe, not the deployment operating point) to see whether new detections appear.
Result: **total small-vehicle detections went down (1295 -> 1181, -9%), not up.** A DOTA-chip
control run through the identical transform barely moved (+0.5 mean detections), confirming the
transform itself isn't a generic booster. **Global brightness/contrast matching alone does not
recover the missing vehicles** — ruled out as a sufficient standalone explanation, though it may
still be a contributing factor alongside others.

### 3b. Object scale — real but moderate, not dominant

DOTA v1.5 train (400 sampled label files, ~32k small-vehicle + ~5.9k large-vehicle instances):
small-vehicle long-side px **p10=10.0, p50=19.2, p90=39.1**. A real car at Maxar's 0.305 m/px GSD is
expected at **~14.8 px** long, **~5.9 px** wide (physical car dimensions, not measured on Maxar
directly — that needs the labelled set). Only **39%** of DOTA's small-vehicle training instances are
as small as a Maxar car; the median training example is ~30% bigger. Maxar cars sit in the lower
half of what the model trained on, not fully outside it (p10 already covers 10 px).

This lines up with the **already-run** Phase 8F-2 controlled-blur proxy
(`data/detect_eval/domain_shift_proxy.json`, k=1.8 area-downsample + cubic-upsample degradation of
DOTA monitor chips — the exact factor by which Maxar's delivered 0.305 m tiles are upsampled from
native WV02/GE01 resolution): vehicle AP50 0.875 -> 0.853, recall 0.910 -> 0.870. **A ~2-4 point
AP50 / ~4 point recall cost, not the ~20x recall collapse the audit saw.** Running the degraded
chips at `imgsz=1536` instead of 1024 barely helps (recall 0.870 -> 0.891) — resolution/scale is a
real, measured, secondary factor, not sufficient by itself.

### 3c. Viewing geometry and shadow — real and measured, impact on the detector not yet isolated

Live STAC fetch for the Van Nuys acquisition (`103001010C12B000`, quadkey `031311102120`):
**off-nadir 29.7°, incidence angle 56.2°, sun elevation only 33.0°** (WV02). DOTA's own paper
describes its source imagery as predominantly near-nadir Google Earth mosaics — no equivalent
per-image angle metadata ships with DOTA to compare against directly, so this is a real, quantified
fact about our tile, not a head-to-head number.

Visually this shows up exactly where you'd expect: P9005/P9006 (this doc's own labelling picks) sit
right next to a tall hangar with a pronounced dark lean/shadow band; a 1.5 m-tall vehicle under a
33° sun casts a shadow ~1.5x its own length. But the **aggregate** dark-pixel-fraction across ALL
staged Maxar tiles (0.132) was *not* distinguishably different from DOTA's (0.127) in 3a's numbers —
so this specific tile's shadow severity is not obviously a dataset-wide effect. **Open**: whether
recall is measurably worse specifically in the shadowed sub-regions needs the labelled set,
stratified by a shadow/lean measure, which isn't built yet.

### 3d. Preprocessing (true-colour rendering) — partially covered, not run as its own test

3a's clip-fraction numbers (Maxar 0.021 vs DOTA 0.026, not meaningfully different) argue against the
percentile-stretch/8-bit conversion being unusually aggressive or lossy compared to DOTA's own
source imagery. A dedicated "re-render the same tile three ways and diff the detections" test was
not built this pass — lower priority once 3a's brightness/sharpness numbers and the histogram-match
negative result were in hand. Flagged as an open item, not run.

### 3e. Inference chip size — ruled out by design, not just by assumption

Checked in code, not assumed: training used `imgsz=1024` (`geoseek/detect/train.py`), chips are
1024x1024 (`geoseek/detect/chipping.py: CHIP_SIZE = 1024`, chosen explicitly to match Maxar's own
1024 px tile size), and deployment (`YoloObbDetectionModel`, `scripts/detect_maxar.py`) also runs at
`imgsz=1024` by default. Train/deploy chip size already match — this was never a mismatch. The
existing blur-proxy's `imgsz=1536` variant (3b, above) tests *raising* inference resolution past
training resolution on already-degraded imagery and finds only a small recall change (0.870 ->
0.891) — consistent with chip size not being a lever here.

## Step 4 — the training anomaly (`best.pt` = epoch 1): investigated, not retrained

Pulled the actual run's `args.yaml` and `results.csv`
(`data/runs/detector/geoseek_obb_v15_yolo26s/`) rather than re-describing the phase report from
memory:

- **LR schedule is unremarkable**: AdamW, `lr0=1.5e-4 -> lrf ratio 0.1` (cosine), 2 warm-up epochs,
  weight decay 5e-4, `patience=100` (no early stopping triggered) — a standard fine-tuning schedule,
  not a misconfiguration.
- **Train loss keeps falling for all 20 epochs** (`train/box_loss` 1.132 -> 0.996, `train/cls_loss`
  0.825 -> 0.530 monotonically) — the optimizer is doing real work throughout, it isn't stuck.
- **Validation does not follow it**: `metrics/mAP50(B)` 0.899 (epoch 1) -> 0.891 (epoch 20), and
  `val/cls_loss` actually **rises** slightly (0.708 -> 0.747, noisily) over the same span — the
  textbook signature of continuing to fit the training set with no corresponding generalisation
  gain, not a stalled optimizer.

**Verdict**: this is not a bug to fix by retraining differently. The DOTA-pretrained checkpoint
already scored 0.821 monitor mAP50 with *zero* fine-tuning (only the class head transplanted); the
one thing epoch 1 changed was correcting the small-vehicle class boundary after the v1.0->v1.5
label-completeness fix ([[geoseek-phase8F2-detector]]). There was very little left to learn from more
epochs of the *same* data through the *same* head, so there wasn't any — consistent with, not
contradicting, the rest of this document's finding that the real gap is a **domain** gap (Maxar
looks different from DOTA), not a **training** gap.

## Step 5 — preliminary ranking (NOT the final verdict — Step 2 baseline is still missing)

| hypothesis | status | size of effect (measured) |
|---|---|---|
| training anomaly (Step 4) | ruled out as a contributing cause | n/a — expected transfer-learning saturation |
| inference chip size (3e) | ruled out by design | already matched; imgsz 1536 barely moves degraded recall |
| resolution / object scale (3b) | real, secondary | ~2-4 pt AP50, ~4 pt recall from a matched blur proxy |
| global brightness/contrast (3a) | real gap, but recovery test failed | matching stats made detections go DOWN 9%, not up |
| viewing geometry / shadow (3c) | real and quantified (29.7° off-nadir, 33° sun) | detector impact not yet isolated — no GT to stratify by |
| preprocessing / rendering (3d) | not run as its own test | clip-fraction argues against gross rendering pathology |

**Honest statement, per the task's own instruction to say so if no single cause explains the gap:**
none of the label-independent measurements above accounts for a drop from ~91% recall (DOTA,
matched-resolution proxy) to ~4% (the Van Nuys manual audit). Resolution/scale is real but small.
Brightness matching, tested directly, does not help. Viewing geometry is real but unquantified
against detector output. **The most likely reading right now is that no single factor explains it,
and the dominant cause is either (a) something not on this list, or (b) the *combination* of these
factors compounding past some threshold that no individual controlled test captures** — which is
exactly why Step 2's real baseline and Step 3's GT-dependent recovery tests (histogram-matching
*scored against real recall*, per-instance size/shadow-stratified AP) are necessary before naming a
primary cause. **The labelled set may also simply be too small (8 images) to distinguish between
several plausible causes with confidence** — that will become clear once Step 2 runs; if per-class
confidence intervals are too wide to separate hypotheses, that itself is the answer to give, not a
reason to force a verdict.

### What a fix would require (not implemented, per the task's instructions)

- If 3b (scale) turns out to dominate once measured on GT: fine-tune with heavier
  small-object augmentation (mosaic + explicit small-scale copy-paste of vehicle crops) and/or a
  P2 detection head for finer strides. **Medium effort** (a few days: augmentation pipeline change +
  one more training run, infrastructure already exists).
- If 3c (geometry/shadow) dominates: needs actual Maxar-domain training data (this labelled set is
  order-of-magnitude too small to fine-tune on; would need the 458-image DOTA-scale effort repeated
  for Maxar, or synthetic shadow augmentation on DOTA chips as a cheaper proxy). **Large effort**
  (weeks: a real Maxar-labelled training set, not just this eval set).
- If it's the brightness/radiometric axis after all (3a's linear match was too naive — e.g. per-tile
  histogram equalization or a learned colour-correction head might behave differently from simple
  moment matching): **small-to-medium effort** to try harder normalisation schemes, but 3a's
  negative result on the simplest version makes this the lower-priority guess.

# Phase 8F-2 — Oriented object detector: build, train, evaluate, wire in

Everything below was **measured on this machine on 2026-09-20** (RTX 4060 Laptop, 8.59 GB, ~7.2 GB usable; 15.3 GB RAM),
and every number is reproducible from a script named next to it. Where a number is unflattering it is stated as such.

> **Status of this document.** Complete: sections 0–4 (findings, detector choice, data, imbalance, pre-flight), 5 (training),
> 6 (held-out evaluation), 7 (Maxar inference), 8 (limitations, next steps) and 9 (acceptance checklist). Every number was copied
> from the run's own JSON artifacts (`data/detect_eval/phase8f2_tables.md` is the generated dump).

---

## 0. Two premises in the brief that turned out to be wrong (and what was done)

**0.1 "DOTA v1.5 annotations are oriented" — true of the dataset, false of the copy on disk.**
The pre-flight check the brief asked for ("confirm boxes sit on objects and rotation is correct") caught it: on a chip of
trucks parked at 45°, every ground-truth box was an axis-aligned square. Measured over *every* label file:
`labelTxt-v1.5/` was **100.00 % axis-aligned** (210,631 of 210,631 train instances, 69,565 of 69,565 val).

Root cause (found, not guessed): `DOTA-v1.5_<split>.zip` (oriented) and `DOTA-v1.5_<split>_hbb.zip` (horizontal) have flat,
identically named members. The Phase 8F-1 extractor unzipped **both into the same folder**, the HBB archive last, so every
oriented file was silently overwritten. The oriented labels were still in `_archives/`: re-extracted, they are 4.69 % / 6.29 %
axis-aligned, byte-identical to the archive. `labelTxt-v1.0/` was never affected (byte-identical to its OBB archive).
Fixed in `geoseek.staging.download_dota` (colliding archives get their own folder; `--repair-v15-labels` re-extracts and
verifies). A previous, uncommitted "Stage A" of this phase had trained its whole pipeline on the corrupted labels; its 15 GB
conversion was deleted and everything below was rebuilt.

**0.2 "`ObjectDetectionModel` exists as an ABC" — it was only a sketch.**
`docs/ARCHITECTURE.md` FW-5 described it; `models/base.py` had no such class. It is now implemented as designed
(`ObjectDetectionModel`, `Detection`, `TileGeoRef`), with one concrete class.

**0.3 v1.0 vs v1.5 — which labels to train on (decided on evidence).**
v1.5 has 126,501 train small-vehicles vs v1.0's 26,126. I expected the extras to be barely visible specks; they are not: the
v1.5-only small-vehicles have a **median long side of 17 px** (only 17.6 % under 10 px) — a real car at our Maxar 0.305 m
scale is ~14.8 px. 78 % of v1.5's small-vehicles sit in images where v1.0 labeled fewer than half of them; image `P1414` has
4,745 small-vehicles in v1.5 and **none** in v1.0 (`data/detect_preflight/label_source_compare_P1414_3200_640.png`: 358 cars
in one 384 px window, all correctly boxed in v1.5, none in v1.0). A detector trained on v1.0 is taught that such parking lots
are background, and v1.0 scoring counts correct detections there as false positives. **Decision: train and headline-evaluate
on v1.5 OBB; also score against v1.0 (the label set every published number and the pretrained weights use) and report both.**
The pretrained model's monitor result makes the same point from the other side (section 5): before fine-tuning its
small-vehicle AP50 against v1.5 is 0.36 while every non-vehicle class is 0.75–0.99.

## 1. Step 1 — detector choice and staging

| | YOLO26s-OBB (**chosen**) | YOLOv8n-OBB (prior pick) | Oriented R-CNN (MMRotate) |
|---|---|---|---|
| Availability | `ultralytics` 8.4.156 on PyPI (8.4.152 installed); weights in `ultralytics/assets` v8.4.0 | same package, v8.3.0 | `mmrotate` last tag v0.3.4 = **2023-02-01**; default branch last pushed 2024-09-28 |
| Licence | **AGPL-3.0** (LICENSE file is GNU AGPL v3 incl. §13 network clause) | same | Apache-2.0 |
| Superseded? | Current release (Jan 2026); docs list *only* YOLO26-OBB; YOLO27 "coming soon" | **Yes, by YOLO11 (Sep 2024) and YOLO26 (Jan 2026)** | 1.x line only ever pre-release |
| Self-reported val (15 cls, chip level, from the `.pt`) | **mAP50 0.789 / mAP50-95 0.645** | 0.765 / 0.601 | — |
| Published DOTA-v1.0 test mAP50 | 80.9 (docs, multi-scale, DOTA server) | not on the current docs pages (only YOLO26 is listed) — no figure quoted | 75.63 (fp16) / 75.69 (fp32) in MMRotate's README; 75.87 in the paper (R50-FPN) |
| Training memory | **measured 6.4–6.6 GB @ batch 8 / 1024²** | ~5.6 GB @ 16 (n) | **7.37 GB (fp16) / 8.46 GB @ batch 2 / 1024²** (MMRotate's own table) |
| Setup cost | 1 pip package + 22 MB `.pt` | same | mmcv/mmdet/mmengine lattice built for torch 1.x/2.0 vs our torch 2.2.2 |

Why **YOLO26s** against *our* constraints — measured with `scripts/detect_bench.py` on the real augmenting dataloader at
worst-case label density:

* **8 GB VRAM.** `s` at batch 16 does **not** fit; batch 8 does (20.5 samples/s, 6.4 GB). `n` at batch 16 runs 24.2 samples/s —
  only 1.35× faster (24.2 vs 17.7 samples/s, identical settings) for 2.3 mAP50 / 3.2 mAP50-95 less, so the small model saves little wall clock and gives up real accuracy
  where it matters (small vehicles). Oriented R-CNN's *documented* 7.4–8.5 GB at batch 2 leaves no headroom on a card with
  ~7.2 GB usable.
* **Offline after staging.** One package, one file.
* **Small objects.** YOLO26's small-target-aware label assignment targets exactly the ~15 px vehicles this track is for.

**⚠ Licence — flagged, and it matters beyond this prototype.** `ultralytics` and its pretrained weights are **AGPL-3.0**;
geoseek's `pyproject.toml` declares **Apache-2.0**. A distributed or network-served build that links the two must offer the
combined work's source (AGPL §13) or use an Ultralytics Enterprise licence. For this non-commercial, open SIH prototype it is
not a blocker. Mitigations implemented: `ultralytics` is an **optional extra** (`pip install geoseek[detect]`), imported only
inside `geoseek.models.yolo_obb` behind the dependency-free `ObjectDetectionModel` ABC, and a test asserts that importing
geoseek does not import it — swapping the detector is a new subclass. The fine-tuned weights are additionally
**non-commercial** (DOTA is academic-use-only; Google Earth imagery is under Google's terms).

Staged (SHA-256 + source + licence in `provenance_manifest.json` → `yolo_obb_detector`): `yolo26s-obb.pt`, `yolo26n-obb.pt`,
`yolo11n-obb.pt`, `yolov8n-obb.pt`, and `yolo26n.pt` (*not* a geoseek model — Ultralytics' internal AMP self-test fixture).
Corrections to the earlier note: MMRotate's last tag is dated 2023-02-01 (it had been read as 2023-01-02).

## 2. Step 2 — data conversion (`scripts/prepare_detect_data.py`, 154 s, parallel, resumable)

**Chip size 1024, overlap 200 px (stride 824).** 1024 is the Maxar tile size (`MAXAR_TILE_SIZE`, ~312 m at 0.305 m/px), so the
detector consumes tiles shaped exactly like what the ingest already produces, and it is the size the DOTAv1-pretrained weights
were trained and benchmarked at. DOTA's own GSD is 0.1–4.5 m (median 0.26 m, p10 0.12, p90 1.0): a 1024 chip spans ~266 m at
the median vs Maxar's 312 m; ~70 % of DOTA train images are ≤ 0.5 m/px, so Maxar's 0.305 m sits inside the bulk of the training
distribution. The 200 px overlap (DOTA_devkit's default) covers any vehicle (10–60 px), ship or plane completely in at least one
neighbouring chip; only long harbors/bridges can exceed it.

**Boundary handling.** Each box is clipped against the chip with shapely. Fully inside → original 4 points kept; visible
fraction ≥ 0.6 → the visible region's minimum-area rotated rectangle becomes the new box (re-fit, not a per-vertex clamp, which
would shear a rotated box); below 0.6 → dropped *from this chip only*. A test asserts a 45° vehicle clipped at the edge keeps
its heading (the "90° swap" failure mode).

**Train and val converted separately; three splits, disjoint by source image.**

| split | source images | chips | of which empty | kept-class instances | retained | clipped at a chip edge |
|---|---:|---:|---:|---:|---:|---:|
| `train` (DOTA train − monitor) | 1,285 (with chips) of 1,301 planned | 9,724 | 1,042 (521 hard negatives) | 183,810 | **183,725 (99.954 %)**, 85 never ≥ 60 % visible in any chip | 11,738 (3.76 %) |
| `monitor` (holdout from DOTA **train**) | 110 (7.8 % of train images) | 1,476 | 442 | 20,026 | 20,022 (99.98 %) | 1,569 (4.04 %) |
| `val` (official, natural distribution) | 458 | 5,297 | 2,282 | 67,377 | **67,374 (99.996 %)** | 4,385 (3.63 %) |

(1.70–1.94 label lines per retained instance: the overlap puts an object in more than one chip. 35,325 of val's 67,377
instances are flagged `difficult` — see section 6.) Verification re-reads what was written: 471,593 label lines valid, **zero
source images shared** between any two splits, 57 % of train+monitor boxes are >5° off-axis. The prior session's val kept only
positive chips (3,006 of 5,297), which silently removed all background false-positive pressure from the "held-out" metric; the
new val keeps every window.

**Why a `monitor` split.** The brief: val gets no training signal, no threshold selection, no derived statistics — yet asks
for per-epoch val loss and mAP. Per-epoch curves, checkpoint choice and the confidence threshold are *model selection*; doing
them on the official val would make the reported val number optimistic. So a class-stratified 8 % of the TRAIN source images
(whole images, never chips: overlapping chips of one photograph share pixels) is the training run's only validation set, and
the official val is opened exactly once, after training. (A first version of the selection swallowed **49 % of all train
helicopters**; a hard per-class cap fixed it — helicopter share is now 9.6 %, all classes 8–16 %.)

## 3. Step 3 — class subset and imbalance

**Classes: 8 of 16.** Kept: `small-vehicle`, `large-vehicle` (the core target), `ship`, `plane`, `helicopter` (vehicle-scale
mobile assets; Van Nuys has ~25 jets), `storage-tank`, `harbor`, `bridge` (infrastructure; Chungthang has a bridge). Dropped:
`baseball-diamond`, `tennis-court`, `basketball-court`, `ground-track-field`, `soccer-ball-field`, `swimming-pool`, `roundabout`
(large recreational / road-layout land use, out of scope) and `container-crane` (142 train / 14 val instances — unlearnable at
that support). Dropped-class windows are not wasted: chips containing them but no kept class are the natural **hard
negatives** (a round pool or roundabout is what a storage-tank detector hallucinates on), and they are sampled first as
background chips.

**The imbalance, as it really is.** On v1.5 train: small-vehicle 126,501 vs container-crane 142 (**~890:1**, as in the brief);
among the 8 kept classes small-vehicle:helicopter is **199:1**. Levers considered:

| lever | decision | why |
|---|---|---|
| class subsetting | **used** | removes the 890:1 outlier and the out-of-scope classes |
| chip-level repeat-factor sampling (LVIS, t = 0.25) | **used** | established long-tail recipe, no loss surgery; effect *measured*: effective instance max:min **176.5:1 → 48.9:1**, helicopter effective instances 1,007 → 5,400 (share 0.3 % → 1.3 %), list expansion 1.21× (11,809 entries/epoch). The ratio plateaus for t ≥ 0.2 because small-vehicle's dominance is a *crowding* effect (a few parking-lot chips hold thousands of instances), not a scarcity of vehicle chips |
| hard-negative background chips (12 % of positives, hard first) | **used** | false-positive suppression on known distractors |
| mosaic augmentation | **used** | co-occurrence mixing on top of the sampler |
| focal / easy-negative-down-weighting loss | **rejected** | Ultralytics' task-aligned assigner already gives IoU-aware soft targets and normalises BCE by the summed target score (what focal loss approximates); adding it means patching the OBB loss class across a 8.4.x release cadence, and would stack a second easy-example suppression on top of TAL. Revisit only if a rare class collapses in the per-class AP |
| per-class BCE weights | **rejected** | same patch-the-loss cost; not needed at this ratio with a DOTA-pretrained init |
| density cap | **used, train only** | Ultralytics pads every image's targets in a batch to the batch maximum, so one 2,400-label chip inflates the assigner's (batch × max_boxes × anchors) tensors for all images. Chips > 800 labels (22 chips = 0.23 % of chips, 7.3 % of label lines, mostly one 0.5 m car-storage scene) are excluded from *train*; val and monitor are never capped |

## 4. Step 4 — pre-flight checks (all passed after the fixes below)

**4.1 Five training chips with ground-truth oriented boxes, saved** (`scripts/render_detect_sanity_chips.py`; chosen by a fixed
seed from chips whose boxes are clearly oblique — median off-axis ≥ 15° — never by looking at anything but the labels; each
also has a 3× zoom):

```
data/detect_preflight/gt_chips/gt_P0685__0_824.png      (small vehicles, 30–60° tilt)
data/detect_preflight/gt_chips/gt_P0580__0_0.png        (school-bus lot, large vehicles 25–75°)
data/detect_preflight/gt_chips/gt_P0828__0_1169.png
data/detect_preflight/gt_chips/gt_P1173__2457_2784.png
data/detect_preflight/gt_chips/gt_P1059__2496_824.png
data/detect_preflight/augmented_batch_rotation_on.png   (a batch from Ultralytics' TRAINING loader, mosaic + rotation)
```
Boxes sit on the objects and are not off by 90° — the white line through each box is its long axis. Two independent numeric checks
back the pictures: **loader round-trip** (labels as Ultralytics' own dataset class loads them, converted back to corners):
*0 of 15,067 elongated boxes swapped by 90°*, corner error vs `minAreaRect(quad)` 0.0001 px median; and **rotation
augmentation is applied to the labels**: 91.4 % of elongated boxes are >5° off-axis with rotation on vs 75.3 % with it off.

*A finding the pre-flight surfaced:* DOTA's "OBB" labels are general **quadrilaterals** (hand-clicked corners), and the loader
fits the minimum-area *rectangle* to each. The median IoU between a quad and its own best rectangle is **0.925 for
small-vehicle** (0.94–0.96 for the other classes) — a real accuracy ceiling for mAP@0.5:0.95 on tiny objects, independent of
the model. (Section 6 scores against the original quads, not the rectangles.)

**4.2 One-batch forward/backward on the 4060 — measured over 62 real iterations at the intended batch and image size**
(`scripts/detect_bench.py`; real augmenting dataloader, worst-case-dense chips forced into the epoch; results in
`data/detect_preflight/bench_*.json`):

| run | batch | samples/s | peak VRAM (torch reserved / nvidia-smi) | result |
|---|---:|---:|---:|---|
| yolo26s, no cap | 16 | — | 13.1 / 7.8 GB | **crashed** (CUDA error at iteration 0) |
| yolo26s, no cap | 8 | 3.2 | 11.4 / 7.8 GB; RAM 0.14 GB free | ran, but 62-min epochs |
| yolo26s, VRAM cap 0.80 | 8 | 17.7 | 6.87 / 6.56 GB | good |
| yolo26s, VRAM cap + non-deterministic | **8** | **20.5** | 6.87 / 6.42 GB | **chosen** |
| yolo26n, VRAM cap | 16 | 24.2 | 6.87 / 6.46 GB; RAM 1.2 GB free | alternative |

**Why the cap:** on Windows the NVIDIA driver's system-memory fallback lets CUDA spill past VRAM into RAM instead of raising
OOM. That turned 17.7 samples/s into 3.2 *and* stopped Ultralytics' own per-image assigner OOM-retry from ever firing (it only
fires on a real OOM). `torch.cuda.set_per_process_memory_fraction(0.80)` restores real OOMs; dense batches then hit that retry.
**Batch 16 for `s` was adjusted to 8** (effective batch 64 via gradient accumulation). Further problems found by smoke tests and
fixed before the long run: 20 dataloader worker processes with system RAM at 96 % used (validation loader capped from 12 to 4
workers); the end-of-run final evaluation OOM-ing under the cap (now tolerated); a supervisor that tried to resume a *finished*
run (resume now skips finished / stripped / corrupt checkpoints); two supervisors fighting over one run directory (now a lock).
A kill-the-whole-process-tree-mid-epoch test confirmed the run resumes from `last.pt`.

**4.3 Planned configuration** (`python scripts/train_detector.py --print-config`; full JSON in the manifest,
`detector_training.config`): YOLO26s-OBB, DOTA-pretrained, 8-class head **transplanted** from the pretrained head by class name
(verified bit-exact: 48 rows, 0.0 max difference on class logits and every other head output); imgsz 1024; batch 8 × accumulation
to 64; **20 epochs**; **AdamW**, lr0 1.5e-4 → 1.5e-5 (cosine), 2 warm-up epochs, weight-decay 5e-4; **AMP** (fp16 autocast);
augmentation: **rotation ±180°**, flipud/fliplr 0.5, mosaic (off for the last 4 epochs), scale 0.4, translate 0.1, HSV defaults;
`max_det` 1000; checkpoint every epoch (`last.pt`) and every 2nd (`epochN.pt`); auto-resume supervisor; hard VRAM cap 0.80;
sleep prevention. lr was lowered from a first guess of 3e-4 after a smoke test where monitor mAP50 fell from 0.875 to 0.832
once the (few) full-LR optimizer steps began — consistent with AdamW moving every weight by ≈ lr per step.

## 5. Step 5 — training (`scripts/train_detector.py`; run dir `data/runs/detector/geoseek_obb_v15_yolo26s/`)

The configuration of §4.3 ran unchanged, launched from commit `636f89e`, **all 20 epochs completed**, and the supervisor never had
to restart the process (`resumes_after_interruption: 0`).

| | |
|---|---|
| model | YOLO26s-OBB, **10,526,414 parameters**; 8-class head transplanted from the DOTA-pretrained checkpoint (§4.3) |
| epochs · batch · precision | 20 · 8 (accumulated to 64) · AMP fp16 autocast; measured 20.5 samples/s in the pre-flight |
| time per epoch | mean **9.3 min** (8.7–10.5), *including* the per-epoch validation on the monitor split |
| wall clock | **4.7 h** (13:18 → 18:00 on 2026-09-20). About **1.6 h of that is the machine being suspended** during epoch 12 (that epoch's clock reads 105.6 min; it was also the one epoch run on battery) → **≈ 3.1 h of actual compute** |
| peak VRAM | torch-reserved **6.87 GB = the hard cap I set (0.80 × 8.59 GB)**, so the allocator ran right up to its limit and stayed there without an OOM; nvidia-smi read ≈ 6.4–6.6 GB during the benchmark |
| checkpoints | `last.pt` every epoch, `epochN.pt` every 2nd, `best.pt`; the weights used everywhere below are `best.pt` (SHA-256 `67f61717…e31e113`, in the manifest and the model card) |

**Curves** (PNG, from `results.csv`; the official val is *not* evaluated per epoch — the per-epoch "val" is the monitor holdout of §2):
`data/detect_eval/figures/detector_loss_curves.png` (train vs monitor box / cls / L1 / angle loss) and
`data/detect_eval/figures/detector_map_curves.png` (monitor mAP50 and mAP50-95 per epoch, with the pretrained starting point as a dashed line).
Per-epoch numbers (monitor split, chip-level protocol; train losses are on augmented data, monitor losses on plain chips):

| epoch | train box | train cls | monitor P | monitor R | mAP50 | mAP50-95 | monitor cls loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (pretrained, no fine-tuning) | — | — | 0.917 | 0.767 | 0.8209 | 0.6708 | — |
| **1** | 1.132 | 0.825 | 0.886 | 0.841 | **0.8991** | **0.7076** | 0.708 |
| 2 | 1.124 | 0.719 | 0.887 | 0.821 | 0.8914 | 0.6953 | 0.793 |
| 4 | 1.117 | 0.696 | 0.873 | 0.831 | 0.8966 | 0.7037 | 0.847 |
| 6 | 1.103 | 0.672 | 0.870 | 0.815 | 0.8858 | 0.6882 | 0.890 |
| 8 | 1.091 | 0.643 | 0.876 | 0.827 | 0.8943 | 0.6979 | 0.808 |
| 10 | 1.072 | 0.624 | 0.869 | 0.821 | 0.8863 | 0.6904 | 0.873 |
| 12 (suspended) | 1.068 | 0.610 | 0.884 | 0.827 | 0.8948 | 0.7006 | 0.802 |
| 14 | 1.059 | 0.601 | 0.871 | 0.841 | 0.8985 | 0.7046 | 0.751 |
| 16 | 1.040 | 0.577 | 0.872 | 0.833 | 0.8927 | 0.6996 | 0.749 |
| 17 (mosaic off) | 1.003 | 0.537 | 0.879 | 0.823 | 0.8872 | 0.6962 | 0.756 |
| 18 | 0.998 | 0.532 | 0.877 | 0.836 | 0.8881 | 0.7002 | 0.730 |
| 20 | 0.996 | 0.530 | 0.872 | 0.841 | 0.8911 | 0.7017 | 0.747 |

(All 20 rows are in `data/detect_eval/phase8f2_tables.md`, T1.)

**What the curves say, stated plainly.**

* **Essentially the whole fine-tuning gain arrived in the first epoch** (monitor mAP50 0.821 → 0.899, mAP50-95 0.671 → 0.708; recall
  0.767 → 0.841 — the small vehicles the v1.0-trained weights had learned to ignore). For the next 19 epochs the monitor metrics
  wander inside a ±0.01 band (mAP50 0.885–0.899, mAP50-95 0.687–0.705) with no trend, while the *train* losses keep falling
  (box 1.13 → 1.00, cls 0.83 → 0.53) and the monitor *box* loss stays flat (0.87 → 0.87). That is a fit that saturated early, not one that
  overfit: the monitor loss did not rise. The step at epoch 17 is mosaic switching off (`close_mosaic=4`) — un-mosaicked images are easier, so
  the train loss drops by construction.
* **`best.pt` is therefore the epoch-1 checkpoint, not a 20-epoch model.** It was chosen by Ultralytics' fitness (0.1·mAP50 + 0.9·mAP50-95)
  on the *monitor* split before the official val was opened; the model card records `weights_epoch: 1 of 20`. Epoch 1's fitness is 0.7267; epoch 14 is
  0.7240 and epoch 20 is 0.7206 — the ranking among epochs is noise-level. §6.6 scores the final epoch on the official val too:
  it is not worse, and is slightly better at strict localisation; I kept `best.pt` because switching to `last.pt` *after seeing val* would
  be selecting on val.
* The lesson for the next run is not "train longer": more epochs of the same data on the same head bought nothing measurable. What
  moved the metric was correcting the *labels* (v1.5 instead of v1.0); what will move it next is different *data* (§8).

## 6. Step 6 — honest evaluation on the held-out official val (`scripts/eval_detector.py`)

### 6.1 Was val ever used for anything? — explicit confirmation

**No.** Each item below is a check, not an assertion (`data/detect_eval/eval_results.json` → `integrity`):

1. **Disjoint source images:** 0 shared between train ↔ val, monitor ↔ val, and train ↔ monitor (set intersection of the converted lists).
2. **Training never looked at it:** the training `dataset.yaml` has `val: monitor/monitor.txt` and never mentions the official val; the run's
   `args.yaml` points at that file; `patience: 100`, so no early-stopping decision was taken either.
3. **No Ultralytics process ever opened the val labels:** `val/labels.cache` (written on the first load of a labelled split) **did not exist
   before the evaluation script ran**.
4. **Nothing was selected with it.** The checkpoint (`best.pt`) was picked on the monitor split; the operating confidence (**0.525**, the
   macro-F1 maximum over a grid) was chosen on the monitor split and only *applied* to val; the hyper-parameters were fixed before
   training (the single change — lr 3e-4 → 1.5e-4 — was made after a smoke test scored on the monitor split, §4.3).
5. Everything on val that follows the headline — last-epoch vs selected, the per-checkpoint curve, the pretrained comparison — is
   post-hoc reporting; none of it fed back.

One dependency I could not verify myself: the *pretrained* base weights were trained by Ultralytics, not by me. Their published
dataset definition (`ultralytics/cfg/datasets/DOTAv1.yaml`, read from the installed package) is `train: images/train` (1,411 images),
`val: images/val` (458 images) — the same partition used here — so the official val is outside the base weights' training set *as
documented*.

### 6.2 What is measured, and against what

Two protocols on the same predictions, because they answer different questions: **full-image, DOTA-devkit protocol** (headline; `geoseek.detect.evaluate`):
detections from all overlapping chips of a photograph are merged across chips, IoU is the *true polygon IoU against the original
hand-clicked quadrilateral*, objects flagged `difficult` are ignored (46 % of v1.5's kept-class instances are), AP50 is all-point
(VOC07 11-point shown alongside); and **chip level** (Ultralytics' own validator: `difficult` counted, per-chip, ProbIoU) — what the checkpoints' self-reported
numbers use. The evaluator was validated before it was trusted (oracle detector on the real val geometry: macro AP50 0.9997, precision =
recall = 1.000; the same oracle scored against v1.0 labels: precision 0.38 — the v1.0 incompleteness, quantified; jittered / degraded detectors fall
in the expected order; `data/detect_eval/evaluator_selftest.json`). Confidence intervals are 95 % percentile bootstrap over **images**
(100 resamples of the 458; chips of one photograph share pixels), so they are the honest width, including for the rare classes.

### 6.3 Headline — per class, official val, v1.5 ground truth (full-image protocol; n = non-`difficult` GT)

| class | n GT | AP50 [95 % CI] | AP50 (VOC07) | AP50:95 [95 % CI] | AP50 pretrained | AP50:95 pretrained | ΔAP50 from fine-tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| **small-vehicle** | 10,290 | 0.845 [0.738–0.905] | 0.811 | 0.413 [0.338–0.471] | 0.606 | 0.330 | **+0.239** |
| **large-vehicle** | 4,612 | 0.862 [0.823–0.896] | 0.838 | 0.529 [0.491–0.563] | 0.825 | 0.552 | +0.036 |
| ship | 10,206 | 0.938 [0.905–0.961] | 0.890 | 0.588 [0.550–0.628] | 0.915 | 0.608 | +0.024 |
| plane | 2,469 | 0.984 [0.968–0.993] | 0.908 | 0.743 [0.685–0.783] | 0.985 | 0.768 | −0.001 |
| helicopter | 77 | 0.656 [0.305–0.815] | 0.646 | 0.340 [0.162–0.436] | 0.736 | 0.366 | −0.080 |
| storage-tank | 1,888 | 0.919 [0.871–0.953] | 0.883 | 0.610 [0.566–0.655] | 0.918 | 0.627 | +0.001 |
| harbor | 2,083 | 0.782 [0.718–0.826] | 0.742 | 0.415 [0.379–0.456] | 0.795 | 0.455 | −0.013 |
| bridge | 427 | 0.553 [0.437–0.645] | 0.548 | 0.217 [0.172–0.263] | 0.575 | 0.235 | −0.023 |

("pretrained" = the DOTA-pretrained weights with only the class head transplanted, i.e. **no fine-tuning**, on identical chips.)

**By reporting group — vehicles are not averaged with infrastructure:**

| group | n GT | macro AP50 [95 % CI] | macro AP50:95 [95 % CI] | pretrained AP50 / AP50:95 |
|---|---:|---:|---:|---:|
| **ground vehicles** (small + large) | 14,902 | **0.854** [0.786–0.891] | **0.471** [0.424–0.508] | 0.716 / 0.441 |
| ships + aircraft | 12,752 | 0.859 [0.735–0.911] | 0.557 [0.477–0.598] | 0.879 / 0.581 |
| **infrastructure** (tanks, harbors, bridges) | 4,398 | **0.751** [0.704–0.784] | **0.414** [0.390–0.441] | 0.763 / 0.439 |
| all 8 classes | 32,052 | 0.817 [0.764–0.845] | 0.482 [0.447–0.498] | 0.794 / 0.493 |

### 6.4 "I expect vehicles to score materially lower" — what the data says

**At IoU 0.5 on DOTA val, they do not.** Ground vehicles score macro AP50 **0.854**, *above* infrastructure (0.751; the intervals almost
touch, 0.786 vs 0.784) and level with ships + aircraft. Reporting the expectation as met would be wrong. What *is* materially worse for
vehicles shows up along four other axes, and it is where this project's use-case lives:

1. **Strict localisation.** Small-vehicle AP50 0.845 falls to AP50:95 **0.413** — its AP50:95 is 49 % of its AP50, versus 76 % for
   planes, 63 % for ships, 66 % for tanks (only bridges, 39 %, are worse). Part of that is not the model's fault: a 15 px car is ±1 px of box error away from a
   0.1 IoU swing, and DOTA's quadrilateral labels are not rectangles (median IoU between a car's quad and its own best rectangle: 0.925, §4.1) — the
   evaluator's *oracle* detector, which outputs exactly the rectangle refit of every ground-truth quad, tops out at small-vehicle AP50:95
   **0.775**. So the model reaches 53 % of what was achievable, not 53 % of 1.0.
2. **Precision.** At the operating confidence (0.525) small-vehicle has the largest pile of false positives of any class: precision **0.597** at recall
   0.907 (9,338 TP, **6,307 FP**, 952 FN), vs ships 0.894 / 0.928 and planes 0.916 / 0.971. (Some of those "false" positives are real cars that even
   v1.5 did not label or flagged `difficult`; I cannot split that without re-annotating, so treat 0.597 as a lower bound on true precision.)
3. **Object size.** Below ~16 px the vehicle numbers collapse — see the size table.
4. **The Maxar domain (§7)** — by far the largest effect, and not measurable on DOTA at all.

**Vehicles by object size** (long side in px; a real car on the Maxar 0.305 m grid is ~15 px; GT outside a bucket is ignored; **recall / precision are at the
operating confidence 0.525**, `scripts/eval_size_operating_point.py`):

| class | long side px | n GT | AP50 | AP50:95 | recall @0.525 | precision @0.525 | recall at *any* confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| small-vehicle | 10–16 | 321 | 0.419 | 0.155 | 0.748 | **0.124** | 0.947 |
| small-vehicle | 16–32 | 6,269 | 0.778 | 0.320 | 0.898 | 0.611 | 0.982 |
| small-vehicle | 32–64 | 3,690 | 0.943 | 0.540 | 0.939 | 0.843 | 0.969 |
| large-vehicle | 10–16 | 14 | 0.006 | 0.002 | 0.000 | — | 0.357 |
| large-vehicle | 16–32 | 294 | 0.116 | 0.048 | 0.177 | 0.214 | 0.653 |
| large-vehicle | 32–64 | 1,937 | 0.839 | 0.512 | 0.868 | 0.759 | 0.952 |
| large-vehicle | 64+ | 2,367 | 0.944 | 0.593 | 0.914 | 0.905 | 0.975 |

Small vehicles of Maxar's size (10–16 px) *are found* on DOTA imagery (recall 0.75) but almost every 10–16 px detection is wrong (precision 0.12): AP50 0.42 there,
against 0.94 for cars of 32–64 px. And a large vehicle under 32 px is essentially a small vehicle — the class boundary is a size convention
(qualitative example 2 below is exactly this). Only 321 of the 10,290 small-vehicle GT are in the 10–16 px bucket, i.e. **DOTA is not a test of the
size regime that matters here**; the Maxar audit in §7 is.

### 6.5 Operating point (confidence 0.525, chosen on the monitor split; official val, v1.5 GT, IoU 0.5)

| class / group | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| small-vehicle | 0.597 | 0.907 | 0.720 | 9,338 | 6,307 | 952 |
| large-vehicle | 0.803 | 0.845 | 0.823 | 3,898 | 959 | 714 |
| ship | 0.894 | 0.928 | 0.910 | 9,467 | 1,128 | 739 |
| plane | 0.916 | 0.971 | 0.943 | 2,397 | 219 | 72 |
| helicopter | 0.699 | 0.662 | 0.680 | 51 | 22 | 26 |
| storage-tank | 0.872 | 0.868 | 0.870 | 1,639 | 241 | 249 |
| harbor | 0.776 | 0.821 | 0.798 | 1,711 | 495 | 372 |
| bridge | 0.646 | 0.518 | 0.575 | 221 | 121 | 206 |
| **ground vehicles** | 0.646 | 0.888 | 0.748 | 13,236 | 7,266 | 1,666 |
| **ships + aircraft** | 0.897 | 0.934 | 0.915 | 11,915 | 1,369 | 837 |
| **infrastructure** | 0.806 | 0.812 | 0.809 | 3,571 | 857 | 827 |
| **all 8 classes** | 0.752 | 0.896 | 0.818 | 28,722 | 9,492 | 3,330 |

The single threshold is the macro-F1 optimum on the monitor split (0.866 there); per-class optima on the monitor split range from 0.46 (helicopter) to 0.79 (plane)
and are not used. Scored against the *v1.0* labels the same detections give small-vehicle precision **0.168** (24,062 "false" positives) — the label-incompleteness
effect of §0.3, made concrete.

### 6.6 Selected checkpoint vs final epoch (official val, v1.5, full-image — reporting only)

| | AP50 selected (epoch 1) | AP50 final epoch (20) | AP50:95 selected | AP50:95 final |
|---|---:|---:|---:|---:|
| ground vehicles (macro) | 0.854 | 0.853 | 0.471 | 0.495 |
| ships + aircraft | 0.859 | 0.839 | 0.557 | 0.543 |
| infrastructure | 0.751 | 0.765 | 0.414 | 0.422 |
| **all 8 classes** | **0.817** | **0.815** | **0.482** | **0.486** |

Twenty epochs of training do not beat one epoch on the official val either (all-class AP50 0.817 vs 0.815; AP50:95 0.482 vs 0.486, both far inside the CI width);
the final epoch is a little sharper on vehicle box quality (+0.024 AP50:95, large-vehicle +0.037) and a little worse on helicopters (n = 77). Per-class rows:
`data/detect_eval/phase8f2_tables.md`, T2.

The same question over the whole run — official-val mAP of every saved checkpoint I scored, **chip-level protocol** (Ultralytics' validator; post-hoc, reporting only):

| after epoch | checkpoint | mAP50 | mAP50-95 |
|---:|---|---:|---:|
| 0 | pretrained, no fine-tuning | 0.787 | 0.612 |
| **1** | **`best.pt` (selected on the monitor split)** | **0.841** | **0.631** |
| 3 | `epoch2.pt` | 0.840 | 0.624 |
| 7 | `epoch6.pt` | 0.804 | 0.603 |
| 11 | `epoch10.pt` | 0.838 | 0.634 |
| 15 | `epoch14.pt` | 0.842 | 0.637 |
| 19 | `epoch18.pt` | 0.839 | 0.635 |
| 20 | `last.pt` (final epoch) | 0.837 | 0.633 |

The official-val curve tells the same story as the monitor curve: one epoch buys everything (+0.054 mAP50 over the pretrained weights), then it is flat within ±0.005 —
apart from a dip at epoch 7 (0.804) that the monitor split shows too (mAP50-95 0.688 at epochs 6–7), i.e. a real mid-training wobble at the peak learning rate with mosaic on, not evaluation noise.
The best checkpoint *on val* would have been epoch 15 (0.842 / 0.637); picking it after the fact would have "improved" mAP50-95 by 0.006 — inside the noise, and I did not.

### 6.7 Against published DOTA numbers — and the leakage check the brief asked for

The published figures are DOTA-v1.0 **test** (server-scored, VOC07 11-point AP50) for **Oriented R-CNN R50-FPN** (Xie et al., ICCV 2021, table read from the paper PDF);
mine are the official **val** split against the **v1.0** labels (the set every published number uses), so the comparison is indicative, not exact (val ≠ test;
YOLO26s is also a newer architecture than the 2021 baseline: Ultralytics' page lists 80.9 mAP50 for YOLO26s-OBB on the test set vs the paper's 75.87).

| class | n GT (v1.0) | fine-tuned (VOC07, v1.0 GT) | pretrained (no fine-tuning) | published Oriented R-CNN R50 (test) | fine-tuned − published |
|---|---:|---:|---:|---:|---:|
| small-vehicle | 5,090 | 0.620 | 0.765 | 0.789 | −0.169 (label effect, below) |
| large-vehicle | 4,293 | 0.844 | 0.848 | 0.830 | +0.014 |
| ship | 8,861 | 0.862 | 0.887 | 0.882 | −0.020 |
| plane | 2,450 | 0.906 | 0.905 | 0.895 | +0.011 |
| helicopter | 72 | 0.650 | 0.659 | 0.523 | **+0.127** |
| storage-tank | 1,869 | 0.883 | 0.889 | 0.847 | +0.036 |
| harbor | 2,065 | 0.723 | 0.793 | 0.749 | −0.026 |
| bridge | 424 | 0.545 | 0.565 | 0.548 | −0.003 |

* **No class is far above the published band except helicopter**, and helicopter does not indicate leakage: (a) 72 val instances — the bootstrap CI on the v1.5 helicopter
  AP50 is 0.31–0.82, which contains the published 0.52; (b) the *un-fine-tuned* pretrained weights show the same excess (0.659), so it is present before any of my training
  touched anything; (c) it *decreased* with fine-tuning on the monitor-selected checkpoint.
* **Small-vehicle is far *below* the published number, for the reason that matters:** after fine-tuning on v1.5 the detector marks cars the v1.0 labels never annotated
  (§0.3), so scored against v1.0 its precision collapses (0.168 at the operating point). The pretrained weights (trained on v1.0, which ignores those cars) are the ones
  that score 0.765 there. Same detections, different verdict from the label set — that is why v1.5 is the headline.
* **The behavioural signature of leakage is absent.** If val had leaked into training, the non-vehicle classes (whose labels did not change between v1.0 and v1.5)
  would *jump* relative to the pretrained baseline. They do not: ΔAP50 between −0.08 (helicopter, noise) and +0.036, all within the bootstrap intervals; the only large gain
  (+0.239) is on the one class whose labels were corrected. Together with §6.1's structural checks, I found no sign of leakage.

### 6.8 Chip-level protocol, and why my evaluator disagrees with Ultralytics' (measured, not assumed)

Ultralytics' own validator on the same val chips (every instance counts, per chip, ProbIoU, 101-point AP):

| | mAP50 | mAP50-95 |
|---|---:|---:|
| fine-tuned (`best.pt`) | 0.841 | 0.631 |
| pretrained, class-head-transplanted (no fine-tuning) | 0.787 | 0.612 |

Per class (fine-tuned): small-vehicle 0.746 / 0.473, large-vehicle 0.877 / 0.684, ship 0.965 / 0.773, plane 0.986 / 0.878, helicopter 0.741 / 0.539, storage-tank 0.872 / 0.706,
harbor 0.900 / 0.608, bridge 0.638 / 0.389. These are higher than the full-image headline (0.817 / 0.482) and I checked *why* instead of just picking the flattering one:

| evaluator (same predictions, chip-level protocol) | mean / max \|ΔAP50\| vs the validator | mean / max \|ΔAP50:95\| |
|---|---:|---:|
| **mine, true polygon IoU** (as used for the headline) | 0.069 / 0.117 | 0.193 / 0.218 |
| mine, **only the overlap measure swapped to ProbIoU** | 0.028 / 0.047 | 0.034 / 0.055 |
| Ultralytics' own overlap + matching + AP code, run on *my* saved predictions | 0.028 / 0.044 | 0.019 / 0.037 |

* **The overlap measure explains most of it.** ProbIoU is not an area ratio; it is a distance between Gaussian approximations of the boxes and it is looser at every threshold: for two
  same-shape boxes offset by 20 % of their width the true IoU is 0.67 and ProbIoU 0.76, and a ProbIoU of 0.5 is a true IoU of about 0.3. Swapping only that changes my AP50:95 by ≈ +0.16 and brings the gap from 0.19 to 0.03.
  (`probiou_polys` reproduces `ultralytics.utils.metrics.batch_probiou` to 4·10⁻¹¹ on random rectangles.) The headline numbers use the true polygon IoU against the original
  quadrilateral because that is what the DOTA devkit does and what published numbers mean.
* **The remaining ~0.03 is in the predictions, not the metric.** Ultralytics' code applied to my stored predictions scores *below* the validator's own numbers in every class (by 0.02–0.05 AP50): my stored predictions
  use a 0.005 confidence floor and single-label NMS, the validator uses 0.001 and `multi_label=True` (`ultralytics/models/yolo/detect/val.py`; the predictor's default is single-label; both take the
  same default external-NMS head path). I have not separated those two causes; the direction is one-sided, so my full-image numbers, if biased at all, are biased *low*.
* On top of the overlap measure, the full-image protocol ignores `difficult` objects and de-duplicates across overlapping chips — both of which also move numbers, in different directions.

### 6.9 Five qualitative results (val chips; predictions at confidence ≥ 0.525; TP / FP / FN at IoU 0.5 against the chip's own labels)

The chips were drawn by a fixed seed from the **chip index** (labels only) in five content strata, *before* any prediction was looked at. Left = ground truth, right = predictions,
both coloured by class (`data/detect_eval/qualitative/`).

| # | stratum | chip | GT | pred | TP | FP | FN | what the picture shows |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | dense small vehicles | `P2271__1648_824` | 38 | 5 | 3 | 2 | **35** | **A failure.** It finds the three isolated cars on the road and one at the lot; it misses the tight rows of parked cars beside the hangar (≈ 8–12 px each) and the two helicopters on the pad |
| 2 | large + small vehicles | `P1860__824_0` | 15 | 17 | 12 | 5 | 3 | Boxes sit exactly on the vehicles; the errors are the **small ↔ large class boundary** (four white vans labelled large: two predicted small; one small predicted large) and one missed van |
| 3 | ships / harbor | `P0706__87_158` | 612 | 632 | 583 | 49 | 29 | A marina packed with boats: 95 % recall, 92 % precision, tilted boxes on every hull. Some false positives are large "harbor" boxes over the piers, which this chip's labels do not have |
| 4 | aircraft | `P1397__4944_824` | 20 | 23 | 20 | 3 | 0 | Aircraft boneyard: every plane found, boxes aligned with the fuselage; the 3 FPs are extra boxes on partly visible planes at the chip edge |
| 5 | infrastructure | `P2766__2472_2472` | 10 | 6 | 5 | 1 | **5** | Hazy industrial scene: the five clear storage tanks are found, five small/faint ones are missed, plus one small-vehicle FP on a road |

Two of five are clear failures and are shown as such; none was chosen for looking good or bad.

## 7. Step 7 — wired in, and run on the staged Maxar tiles

### 7.1 The seam

`geoseek.models.base.ObjectDetectionModel` (abstract; `class_names`, `detect(tile_rgb_uint8, *, classes, min_score, geo)`, `detect_batch`, `load`, `info`) with the value objects `Detection`
(class, score, oriented box in tile pixels, 4-corner polygon, heading, WKT footprint in EPSG:4326) and `TileGeoRef`. The concrete
`geoseek.models.yolo_obb.YoloObbDetectionModel` is the **only** module that imports `ultralytics` (lazily, in `load()`; a test runs
`import geoseek.models, geoseek.detect` in a subprocess and asserts `ultralytics` is not in `sys.modules`). It takes the same
`HxWx3 uint8 RGB` tile the embedding pipeline produces, flips to BGR exactly once (ultralytics' ndarray convention), windows tiles larger than
`imgsz` with the training overlap and de-duplicates across windows, reads its default confidence (0.525) from `<weights>.card.json`, and has an
opt-in `upscale` factor (§7.5). **Checked on real data, not only with the fake predictor:** on 9 monitor chips, `detect(rgb_array)` and the file-path route
used for every DOTA number return the same 660 detections — 0 class mismatches, max score difference 0.002, max centre difference 0.7 px (fp16 numerics) — so
nothing below is a channel-order or scaling artefact. Detections are registered as `DerivedProduct(kind="detection")` (`derived_id = detection:<observation_id>`, GeoJSON of oriented polygons with class,
score, long side and heading), which inherits the provenance chain like every other derived layer.

### 7.2 Detections per tile (`scripts/detect_maxar.py`; 0.07–0.09 s per 1024² tile on the 4060, 21–25 s per 289-tile observation; confidence 0.525)

| observation (collection `maxar-opendata`) | role | tiles | with ≥ 1 detection | detections | by class | per tile (mean / max) |
|---|---|---:|---:|---:|---|---|
| Chungthang town + Teesta-III dam, WV02 | single date | 289 | 4 | **5** | storage-tank 2, small-vehicle 1, bridge 1, large-vehicle 1 | 0.02 / 2 |
| lake quadkey 120220030330, pre-event, WV02 | negative control (no man-made objects) | 289 | 1 | **1** | helicopter 1 | 0.00 / 1 |
| lake quadkey 120220030330, post-event, GE01 | negative control | 289 | 1 | **1** | helicopter 1 | 0.00 / 1 |
| Van Nuys airport, Los Angeles, WV02 | dense-vehicle / aircraft demo | 289 | 190 | **867** | small-vehicle 550, plane 223, ship 70, large-vehicle 12, storage-tank 11, helicopter 1 | 3.0 / 46 |

Per-tile counts by class: `data/detections/<observation_id>/summary.json`; every detection as an oriented polygon: `detections.geojson` beside it; annotated tiles (the three highest-count tiles + two
seeded-random ones per observation): `data/detections/samples/`; roll-up and model info: `data/detections/maxar_inference_report.json` (also the manifest section `detector_maxar_inference`).
Van Nuys detection scores: 10th / 50th / 90th percentile 0.536 / 0.609 / 0.893 — most accepted detections sit just above the threshold.

### 7.3 What the annotated samples show

* **Large, isolated, high-contrast objects are found and the boxes are good.** Van Nuys tile r010_c006: 31 aircraft parked in rows on the apron, each with a tight oriented box (one doubtful box
  on a hangar edge); Chungthang tile r003_c008: the bridge over the river, boxed along its deck; tile r015_c010: two round white rooftop tanks tagged `storage-tank`.
* **Cars are mostly not found at the operating confidence.** In the same Van Nuys tile the street parking, the lot to the left and the row of cars along the road are essentially all unboxed; the tile has 15
  `small-vehicle` detections against well over a hundred visible cars (visual estimate). In tile r007_c005 (25 detections) the parking lot at the top right has dozens of cars and a handful of boxes.
* **Chungthang has few vehicles, and the detector finds fewer still**: 5 detections in 289 tiles; the parking area at r002_c007 (≈ 10 visible vehicles) gets 1 box.
* **The negative control is clean on natural terrain, but it is a weak control.** 2 detections in 578 tiles of glacier / snow / rock, both `helicopter` (scores 0.66 / 0.63) on ice and crevasse features. Natural terrain
  is not where this detector's false positives are: Van Nuys shows they are man-made **rooftops** (70 `ship` detections at an airport).

### 7.4 A small manual audit, because there is no ground truth here (`scripts/audit_maxar_crops.py`)

**This is a subjective tally on a handful of crops, not a metric.** Ten 256 × 256 px windows (78 m), 2 per tile from 5 tiles drawn by a fixed seed from *all 289* Van Nuys tiles (not from tiles the detector fired on),
viewed at 3× against the raw imagery (`data/detections/audit/*__pair.png`). Counts of visible cars are mine, approximate (±20 %):

| | visible cars | correct car detections | wrong detections | other |
|---|---:|---:|---:|---|
| 10 random Van Nuys crops, native scale | ≈ 47 | **2** (≈ 4 %) | 2 (`ship` boxes on a flat rooftop) | 1 of 1 aircraft found |
| Chungthang parking area (r002_c007, a tile chosen because vehicles are known to be there) | ≈ 10 | 1 | 0 | — |

Five of the ten random crops contained cars and got **zero** boxes. What stands out in the missed cars: dark bodies with a bright glint on the roof (off-nadir sun), rows of cars in shade, cars under trees. The detected ones
are mostly white / tan / isolated.

### 7.5 DOTA → Maxar: what differs, what I measured, what I did not

| | DOTA (train / val) | Maxar tiles here |
|---|---|---|
| GSD | 0.1–4.5 m (median 0.26 m); mostly aerial / Google Earth | 0.305 m grid, but **resampled from a 0.46–0.58 m sensor** (WV02 ≈ 0.58 m native as staged, GE01 ≈ 0.49 m), pan-sharpened, JPEG-like artefacts |
| a 4.5 m car | 15–45 px, crisp | ≈ 15 px on the grid, ≈ 8–9 px of real detail, soft |
| geometry / lighting | mostly near-nadir | off-nadir, low sun, strong shadows |

Three experiments, all with the *same weights*:

1. **Resolution alone is not the explanation** (`scripts/eval_domain_shift_proxy.py`, monitor split — train-domain images, **not** the official val; degrade = area-downsample 1.8× then cubic-upsample, labels unchanged):

   | monitor chips | vehicles AP50 | ground-vehicle recall | small-vehicle AP50 | macro AP50 / AP50:95 |
   |---|---:|---:|---:|---:|
   | original, imgsz 1024 | 0.875 | 0.910 | 0.941 | 0.898 / 0.548 |
   | degraded k = 1.8, imgsz 1024 | 0.853 | 0.870 | 0.908 | 0.888 / 0.518 |
   | degraded k = 1.8, imgsz 1536 | 0.867 | 0.891 | 0.915 | 0.886 / 0.528 |
   | original, imgsz 1536 | 0.895 | 0.930 | 0.939 | 0.906 / 0.558 |

   A Maxar-like softening costs ≈ 4 points of vehicle recall on DOTA imagery and test-time upscaling wins back about half. Yet on the Maxar audit the recall is ≈ 4 %, not ≈ 87 %. **So most of the gap is something
   this proxy does not model** — sensor radiometry and pan-sharpening artefacts, off-nadir / sun geometry, the scene mix. I did not isolate which; those are hypotheses, not findings.
2. **Object size alone is not the explanation either.** On DOTA val the detector finds 75 % of the labelled 10–16 px cars and 90 % of the 16–32 px ones at the same operating confidence (§6.4) — sharp imagery, same pixel size, recall 0.75–0.90.
3. **Test-time upscaling helps, and is not a fix** (`scripts/maxar_scale_probe.py`, exploratory; `YoloObbDetectionModel(upscale=…)` resamples the tile, detects, and maps boxes back to original pixels). Van Nuys, whole observation, same confidence:

   | scale | detections | small-vehicle | ship | plane | tiles with ≥ 1 |
   |---|---:|---:|---:|---:|---:|
   | 1.0 (default) | 867 | 550 | 70 | 223 | 190 / 289 |
   | 1.5 | 3,338 | 2,680 | 389 | 231 | 268 / 289 |
   | 2.0 | 6,958 | 6,038 | 636 | 248 | 285 / 289 |

   On the same ten audit crops, correctly-classed car detections go from ≈ 2 to ≈ 14 of ≈ 47 visible (≈ 30 %), and a few more real vehicles are found but labelled `ship`; the parking-lot crop that had none at native scale gets 4 of ≈ 9. **But** `ship` detections grow 9× (70 → 636; rooftop
   structures and white vans), the dark cars on the road crops are still missed at 2×, and Chungthang goes from 5 to 33 detections (25 `small-vehicle`, plus a `plane`, a `harbor` and 3 `ship` in a mountain town — not credible). Nothing here is validated against ground truth, so the default stays `upscale = 1.0`.

**Bottom line for this track:** as trained, the detector is a **reliable finder of aircraft and other large, isolated objects on this Maxar product, and not a usable car counter** — DOTA-val numbers for vehicles (§6) do not transfer to this imagery.
The Maxar-side evidence is qualitative (a few dozen boxes inspected by eye), and I say so; the DOTA-side numbers are quantitative and say nothing about this sensor.

## 8. Limitations, licence, and what I would do next

**Limitations of the result**

1. **Vehicles on Maxar (§7)** — the dominant one. There is no ground truth on the staged tiles, so I can quantify neither recall nor precision there; the audit puts recall at a few percent at the DOTA-calibrated operating point.
2. **The training gain is one epoch of label correction, not 20 epochs of learning** (§5–6). Longer training on the same data will not help; different *data* will.
3. **Single training run, single seed.** Run-to-run variance is unmeasured; checkpoint-to-checkpoint noise is ±0.005 mAP50 (chip level), ±0.01 on the monitor split, and one mid-run dip of 0.037.
4. **Wide intervals for rare classes.** Helicopter has 77 val instances (AP50 CI 0.31–0.82) and bridge 427; per-class conclusions there are weak. The bootstrap resamples images (100 replicates), not training runs.
5. **Label noise bounds AP50:95 for small objects** (quadrilateral vs rectangle IoU 0.925; oracle ceiling 0.775 for small vehicles), and 46 % of v1.5 kept instances are `difficult` and ignored by the headline protocol.
6. **My full-image numbers are slightly conservative** relative to Ultralytics' validator (confidence floor / single-label prediction sets, §6.8) — direction known, size not separated.
7. **Quality of the DOTA labels themselves:** v1.5 labels the tiny vehicles v1.0 missed, but it is still hand-annotated; "false positives" against it can be real cars.
8. **8 of 16 classes.** The other seven DOTA classes and container-crane are out of scope by design (§3).

**Licence.** `ultralytics` and the pretrained weights: **AGPL-3.0** (§13 network clause); geoseek: Apache-2.0. Isolated as an optional extra behind an ABC, with a test that guards the isolation; a distributed or network-served build that includes the detector must offer
the combined source or take an Ultralytics enterprise licence. The fine-tuned weights are additionally **non-commercial / academic-use** (DOTA terms, Google Earth imagery).

**Next steps, ordered by expected payoff**

1. **Get a Maxar-domain test set before tuning anything else:** hand-label vehicles in a few dozen Maxar tiles (or obtain xView — 0.3 m WorldView imagery — / DOTA-v2.0, both account-gated, verified live in Phase 8F-1). Without it every Maxar statement stays anecdotal.
2. **Train on the domain, not longer:** sensor-matched degradation in the augmentation (blur k ≈ 1.5–3, JPEG, pan-sharpening halo), scale jitter biased towards 8–16 px vehicles, brightness / glint augmentation; then re-evaluate the size-stratified numbers of §6.4 on the Maxar labels.
3. **Re-calibrate the confidence on Maxar** labels (per class: 0.46 (helicopter) to 0.79 (plane) on the monitor split shows one threshold is a compromise) and test `upscale = 1.5–2` there with ground truth.
4. Merge the small ↔ large vehicle boundary (a size convention, §6.4) or train a size-conditioned head.
5. Only if the AGPL is a problem for deployment: re-implement the head behind the same ABC on an Apache-2.0 detector (MMRotate Oriented R-CNN needs ≈ 7.4–8.5 GB at batch 2 — it does not fit this card comfortably).

## 9. Acceptance checklist (Phase 8F-2 brief)

| requirement | where |
|---|---|
| detector staged with source, licence, SHA-256 in the manifest | §1; `provenance_manifest.json` → `yolo_obb_detector`, `detector_model_card` |
| conversion counts (chips / split, instances retained, boxes clipped) | §2 table; `data/datasets/dota_obb/conversion_report.json` |
| imbalance strategy | §3 |
| 5 annotated GT chips, paths printed | §4.1 |
| one-batch fwd/bwd on the 4060 with peak VRAM; full planned config | §4.2–4.3 |
| training to completion, curves as PNG, wall clock, peak VRAM, checkpoints, model card | §5; `data/detect_eval/figures/detector_loss_curves.png`, `detector_map_curves.png`; `data/models/detector/geoseek_obb_v15_yolo26s.card.json` |
| per-class mAP@0.5 and @0.5:0.95 on held-out val, vehicles broken out, published-baseline comparison / leakage check | §6.3–6.7 |
| 5 qualitative predictions-vs-GT images | §6.9; `data/detect_eval/qualitative/` |
| explicit confirmation that val was never used in training or threshold selection | §6.1 |
| concrete `ObjectDetectionModel` behind the ABC; detections per Maxar tile; annotated samples; DOTA↔Maxar difference and what was observed | §7 |
| pytest green, the original 275 tests unchanged | **351 passed, 3 skipped** in 103 s (275 pre-existing + 76 new: 35 data prep, 8 training run-control, 17 evaluator, 4 eval I/O, 12 detection model); `data/detect_eval/pytest_full.log` |

*Generated tables for §5–§7 (all numbers above are copied from these):* `data/detect_eval/phase8f2_tables.md`, from `data/detect_eval/eval_results.json`, `evaluator_selftest.json`, `domain_shift_proxy.json`,
`size_operating_point.json`, `data/detections/scale_probe/`, `data/detections/maxar_inference_report.json`.

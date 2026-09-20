# Phase 8F-2 — Oriented object detector: build, train, evaluate, wire in

Everything below was **measured on this machine on 2026-09-20** (RTX 4060 Laptop, 8.59 GB, ~7.2 GB usable; 15.3 GB RAM),
and every number is reproducible from a script named next to it. Where a number is unflattering it is stated as such.

> **Status of this document.** Sections 0–4 (findings, detector choice, data, imbalance, pre-flight) are final.
> Sections 5–8 (training, evaluation, Maxar inference, limitations) are filled in from the run's own artifacts — see the
> bottom of each section for the file it was generated from.

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

<!-- SECTIONS 5-8 ARE APPENDED BELOW AFTER TRAINING / EVALUATION / MAXAR INFERENCE -->

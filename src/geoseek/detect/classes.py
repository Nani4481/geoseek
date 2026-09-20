"""Class subset, label-source and grouping decisions for the Phase 8F-2 detector.

Made explicit here (not buried in a training script) so the conversion,
sanity-check, training and evaluation code all read from one source of truth.

Label source: DOTA **v1.5 OBB** (``labelTxt-v1.5``), evaluated against v1.0 as a secondary view
------------------------------------------------------------------------------------------------
Two facts, both measured (2026-09-20), decide this:

1. The staged ``labelTxt-v1.5/`` directory was CORRUPTED by an extraction collision: the OBB zip and
   the HBB zip have flat, identically named members and were unzipped into the same folder, so the HBB
   files overwrote every OBB file (210,631 of 210,631 train instances axis-aligned). The oriented v1.5
   labels were recovered from the retained archives (``download_dota.repair_v15_label_dirs``; byte-
   identical to ``DOTA-v1.5_<split>.zip``, 4.69% / 6.29% axis-aligned). ``labelTxt-v1.0/`` was never
   affected (byte-identical to ``labelTxt.zip``).

2. v1.0's vehicle labels are INCOMPLETE, v1.5's are not. v1.5 has 126,501 train small-vehicles vs
   v1.0's 26,126, and the extras are not specks: their median long side is 17 px (only 17.6% are under
   10 px) - the size of a real car at our 0.305 m Maxar scale (~15 px). 78% of v1.5's small-vehicles sit
   in images where v1.0 labeled fewer than half of them; e.g. one image has 4,745 v1.5 small-vehicles
   and none in v1.0 (a parking lot of hundreds of clearly visible, well-fit cars, see
   data/detect_preflight/label_source_compare_P1414_3200_640.png). A detector trained on v1.0 is taught
   that such lots are background, and v1.0 scoring counts correct detections there as false
   positives. For a vehicle detector meant for dense Maxar scenes v1.5 is the right target.

So the detector TRAINS and is primarily EVALUATED on v1.5 OBB. The same model is also scored against
v1.0 GT as the published-baseline-comparable view (the Ultralytics DOTAv1 weights and every published
DOTA number are v1.0), reported alongside with the incompleteness caveat - never instead.

The imbalance quoted in the brief is real on v1.5: small-vehicle 126,501 vs container-crane 142 (~890:1).
container-crane is dropped (below); among the 8 kept classes small-vehicle:helicopter is 199:1.

Class subset decision
----------------------
DOTA v1.5 ships 16 classes. Our operational targets are vehicle-scale objects and infrastructure
(this track exists to detect things analysts care about in Maxar VHR imagery: vehicles, aircraft,
vessels, storage/port infrastructure, bridges - see geoseek.staging.download_maxar and the
Chungthang / Van Nuys quadkeys already staged, which are dense in exactly these classes).

KEPT (8):
  small-vehicle, large-vehicle  - the core target, the reason this track exists
  ship                          - vehicle-scale mobile asset (harbour / reservoir monitoring)
  plane, helicopter             - vehicle-scale aviation assets (matches our Van Nuys Airport quadkey)
  storage-tank                  - infrastructure (industrial / fuel-site monitoring)
  harbor, bridge                - infrastructure (bridge matches our Chungthang quadkey directly)

DROPPED (8 of 16):
  baseball-diamond, tennis-court, basketball-court, ground-track-field,
  soccer-ball-field, swimming-pool, roundabout
      - large-footprint recreational / road-layout land-use classes, outside this track's
        operational scope ("find tennis courts after a flood" is not a query anyone asks). They
        are NOT discarded from the data: windows that contain them but none of the kept classes
        are the natural HARD NEGATIVES for the 8-class model (a round roundabout or swimming
        pool is exactly what a storage-tank detector will hallucinate on), and
        geoseek.detect.convert samples them preferentially as background chips.
  container-crane
      - the ~890:1 extreme minority (142 train / 14 val instances): statistically unlearnable at
        that support whatever the scope, so dropping it also removes the worst of the imbalance.

Imbalance-handling strategy (within the 8 kept classes)
----------------------------------------------------------
Measured v1.5 train instance counts: small-vehicle 126,501, ship 32,973, large-vehicle 22,218,
plane 8,072, harbor 6,016, storage-tank 5,346, bridge 2,075, helicopter 635 (max:min = 199:1).
Levers, and what we decided:

  1. Class subsetting (above)        - removes the outlier and the out-of-scope classes.  USED.
  2. Chip-level repeat-factor sampling (RFS, Gupta et al. 2019 / LVIS):
     f_c = fraction of train chips that contain class c;
     r_c = max(1, sqrt(t / f_c)); a chip repeats max_c r_c times.  USED - the
     established long-tail recipe, needs no loss surgery, and the effective
     per-class instance counts before/after are MEASURED and reported
     (geoseek.detect.sampling).
  3. Mosaic augmentation (Ultralytics default)  - co-occurrence mixing.  USED.
  4. Hard-negative background chips  - windows with no kept class, biased to
     those holding a dropped-class distractor.  USED.
  5. Focal / down-weight-easy-negatives loss  - REJECTED, with reasons:
     Ultralytics' assigner (TAL) already gives IoU-aware soft classification
     targets and normalises the BCE by the summed target score, which is what
     focal loss approximates; the v8 head dropped focal loss; adding it means
     patching the OBB loss class (fragile across the frequent 8.4.x releases)
     and would stack a second easy-example suppression on top of TAL. We
     measure per-class AP first; if a rare class collapses, that is the
     evidence to revisit it.
  6. Per-class BCE weights  - REJECTED for the same patch-the-loss reason, and
     because chip-level RFS + a DOTA-pretrained init handle the long tail
     (small-vehicle's dominance is a crowding effect - a few parking-lot
     chips hold thousands of instances - not a scarcity of vehicle chips).
"""

from __future__ import annotations

# Order defines the YOLO class index - never re-order without re-running the conversion.
KEPT_CLASSES = (
    "small-vehicle",
    "large-vehicle",
    "ship",
    "plane",
    "helicopter",
    "storage-tank",
    "harbor",
    "bridge",
)

# The other 8 of DOTA v1.5's 16 classes (see docstring). container-crane exists only in v1.5.
DROPPED_CLASSES = (
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "soccer-ball-field",
    "swimming-pool",
    "roundabout",
    "container-crane",
)

V15_ONLY_CLASSES = ("container-crane",)

CLASS_NAMES = KEPT_CLASSES
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Reporting groups (Step 6: vehicles must be visible on their own, not averaged into the rest).
CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "ground_vehicles": ("small-vehicle", "large-vehicle"),
    "ships_and_aircraft": ("ship", "plane", "helicopter"),
    "infrastructure": ("storage-tank", "harbor", "bridge"),
}

# Oriented annotation directories under each staged DOTA split. PRIMARY = what we train and headline-evaluate on.
LABEL_SETS = {"v1.5": "labelTxt-v1.5", "v1.0": "labelTxt-v1.0"}
PRIMARY_LABEL_SET = "v1.5"
OBB_LABEL_DIR = LABEL_SETS[PRIMARY_LABEL_SET]

# Ultralytics' DOTAv1 pretrained head uses these 15 names (with spaces). Map to ours for head transplant / baselines.
DOTA15_NAMES = (
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool",
)


def dota15_index_to_ours() -> dict[int, int]:
    """Pretrained-15-class index -> our 8-class index, for the classes we keep."""
    out: dict[int, int] = {}
    for i, name in enumerate(DOTA15_NAMES):
        ours = name.replace(" ", "-")
        if ours in CLASS_TO_INDEX:
            out[i] = CLASS_TO_INDEX[ours]
    return out


def group_of(class_name: str) -> str:
    for g, members in CLASS_GROUPS.items():
        if class_name in members:
            return g
    raise KeyError(class_name)

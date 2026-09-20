"""Class-imbalance handling for the detector's TRAIN list: repeat-factor sampling.

Repeat Factor Sampling (Gupta, Dollar & Girshick, "LVIS", CVPR 2019), applied at
the chip level:

    f_c   = fraction of train chips that contain at least one instance of class c
    r_c   = max(1, sqrt(t / f_c))              (classes rarer than t get boosted)
    r_chip = max over the classes present in the chip of r_c
    a chip appears round(r_chip) times in the train list (stochastic rounding
    with a fixed seed, so E[repeats] == r_chip and the list is reproducible)

It needs no change to the loss, so it survives Ultralytics' fast release
cadence, and every quantity is computed from the TRAIN split only. The effect
is *measured* - :func:`build_repeat_list` returns the effective per-class
instance counts before and after, and the max:min ratio, so "did this actually
rebalance anything?" has a number as its answer.

Repeats are realised by repeating the chip's path in ``train.txt`` (Ultralytics'
image-list datasets keep duplicates), never by copying files.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from geoseek.detect.classes import KEPT_CLASSES

# Chosen from the measured sweep in scripts/prepare_detect_data.py (see the report's
# rfs.sensitivity table): the smallest t that pulls the effective instance
# max:min ratio under ~15:1 without more than ~1.5x list expansion.
RFS_THRESHOLD = 0.25
RFS_SEED = 8241


@dataclass(frozen=True)
class ChipClasses:
    """The only thing sampling needs to know about a chip."""
    chip_id: str
    class_counts: dict[str, int]      # kept-class name -> #label lines in the chip


def presence_fractions(chips: list[ChipClasses]) -> dict[str, float]:
    n = max(len(chips), 1)
    present = Counter()
    for c in chips:
        for cls, k in c.class_counts.items():
            if k > 0:
                present[cls] += 1
    return {cls: present[cls] / n for cls in KEPT_CLASSES}


def class_repeat_factors(freq: dict[str, float], t: float) -> dict[str, float]:
    """r_c = max(1, sqrt(t / f_c)); a class absent from the split gets r_c = 1 (nothing to repeat)."""
    return {cls: (max(1.0, math.sqrt(t / f)) if f > 0 else 1.0) for cls, f in freq.items()}


def chip_repeat_factor(class_counts: dict[str, int], r_c: dict[str, float]) -> float:
    present = [r_c.get(cls, 1.0) for cls, k in class_counts.items() if k > 0]
    return max(present) if present else 1.0


def stochastic_round(x: float, rng: random.Random) -> int:
    base = math.floor(x)
    return base + (1 if rng.random() < (x - base) else 0)


def effective_instance_counts(chips: list[ChipClasses], repeats: dict[str, int] | None = None) -> dict[str, int]:
    out = Counter()
    for c in chips:
        r = 1 if repeats is None else repeats.get(c.chip_id, 1)
        for cls, k in c.class_counts.items():
            out[cls] += k * r
    return {cls: int(out[cls]) for cls in KEPT_CLASSES}


def _ratio(counts: dict[str, int]) -> float | None:
    vals = [v for v in counts.values() if v > 0]
    return round(max(vals) / min(vals), 2) if vals else None


def build_repeat_list(
    chips: list[ChipClasses], *, t: float = RFS_THRESHOLD, seed: int = RFS_SEED,
    sweep: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
) -> tuple[dict[str, int], dict]:
    """(chip_id -> repeat count, report). ``chips`` are the TRAIN chips only."""
    freq = presence_fractions(chips)
    r_c = class_repeat_factors(freq, t)
    rng = random.Random(seed)
    repeats = {c.chip_id: max(1, stochastic_round(chip_repeat_factor(c.class_counts, r_c), rng)) for c in chips}

    before = effective_instance_counts(chips)
    after = effective_instance_counts(chips, repeats)
    n_list = sum(repeats.values())

    sensitivity = {}
    for tt in sweep:
        rc = class_repeat_factors(freq, tt)
        expected = {c.chip_id: chip_repeat_factor(c.class_counts, rc) for c in chips}   # expectation, no rounding noise
        eff = Counter()
        for c in chips:
            for cls, k in c.class_counts.items():
                eff[cls] += k * expected[c.chip_id]
        eff = {cls: int(eff[cls]) for cls in KEPT_CLASSES}
        sensitivity[str(tt)] = {
            "list_expansion": round(sum(expected.values()) / max(len(chips), 1), 3),
            "effective_instance_ratio_max_over_min": _ratio(eff),
        }

    report = {
        "method": "repeat-factor sampling (LVIS), chip level, stochastic rounding, seeded",
        "threshold_t": t,
        "seed": seed,
        "n_unique_train_chips": len(chips),
        "n_list_entries": n_list,
        "list_expansion": round(n_list / max(len(chips), 1), 3),
        "chip_presence_fraction_by_class": {k: round(v, 4) for k, v in freq.items()},
        "class_repeat_factor": {k: round(v, 3) for k, v in r_c.items()},
        "instances_per_class_unique": before,
        "instances_per_class_effective": after,
        "instance_ratio_max_over_min_unique": _ratio(before),
        "instance_ratio_max_over_min_effective": _ratio(after),
        "share_of_instances_unique": {k: round(v / max(sum(before.values()), 1), 4) for k, v in before.items()},
        "share_of_instances_effective": {k: round(v / max(sum(after.values()), 1), 4) for k, v in after.items()},
        "sensitivity": sensitivity,
    }
    return repeats, report

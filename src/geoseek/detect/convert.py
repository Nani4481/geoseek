"""DOTA OBB (v1.5 by default) -> YOLO-OBB chip conversion, in two passes, three disjoint splits.

Splits (all disjoint at the SOURCE-IMAGE level, so overlapping chips of one
photograph can never straddle a boundary):

  train    DOTA train minus the monitor images. Positives + a bounded number of
           background chips (hard negatives first) -> repeat-factor-sampled
           ``train.txt``.
  monitor  ~8% of the DOTA *train* source images (class-stratified). Natural
           distribution (every window, empty ones included). The training run's
           ONLY validation set: per-epoch loss / mAP curves, checkpoint choice
           and the confidence-threshold choice all use it.
  val      the official DOTA val split, natural distribution (every window,
           empty ones included). Never referenced by the training yaml; only
           ``dataset_official_val.yaml`` (evaluation scripts) points at it.

Why a monitor split at all: the user's rule is that val gives no training
signal, no threshold selection and no derived statistics. Per-epoch curves and
"which epoch / which conf threshold" are model selection; doing that on the
official val would make the reported val number optimistic. So selection runs
on ``monitor`` and ``val`` is touched once, at the end.

Pass 1 (PLAN) is pure geometry - label files + image *headers* only - so the
train negatives / monitor / repeat-factor decisions are made globally and the
whole thing is unit-testable without pixels. Pass 2 (WRITE) decodes each source
image once (in a worker pool) and writes its chips atomically.

Chip ids encode the window origin (``P0001__824_0``) so detections can be mapped
back to full-image coordinates for the full-image DOTA evaluation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from geoseek.detect.chipping import (
    CHIP_SIZE,
    OVERLAP,
    STRIDE,
    VISIBILITY_THRESHOLD,
    ImagePlan,
    axis_deviation_deg,
    crop_chip,
    format_yolo_obb_line,
    normalize_points,
    parse_dota_label_file,
    parse_gsd,
    plan_windows,
)
from geoseek.detect.classes import (
    CLASS_TO_INDEX,
    DROPPED_CLASSES,
    KEPT_CLASSES,
    OBB_LABEL_DIR,
)
from geoseek.detect.sampling import RFS_SEED, RFS_THRESHOLD, ChipClasses, build_repeat_list

Image.MAX_IMAGE_PIXELS = None  # DOTA has 60 MP scenes; these are trusted local files

MONITOR_FRACTION = 0.08        # target share of each kept class's TRAIN instances held out for monitoring
MONITOR_CAP_FACTOR = 2.0       # ... and no class may exceed this multiple of the target in the holdout
TRAIN_MAX_LABELS_PER_CHIP = 800  # TRAIN only: Ultralytics pads every image's targets in a batch to the batch maximum, so
                                 # one 2,400-label chip inflates the assigner's (batch x max_boxes x anchors) tensors for
                                 # all 16 images (OOM -> slow CPU fallback). val / monitor are NEVER capped.
NEGATIVE_FRACTION = 0.12       # train background chips = this share of the positive-window count
HARD_NEGATIVE_SHARE = 0.5      # of that budget, up to this share from windows holding dropped-class distractors
SPLIT_SEED = 8241
PNG_COMPRESSION = 1            # fast to write; decode speed is unaffected (files are a bit larger)
OFF_AXIS_DEG = 5.0             # "oriented" = long edge > this many degrees off horizontal/vertical


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class SourcePlan:
    stem: str
    image_path: Path
    width: int
    height: int
    gsd: float | None
    plan: ImagePlan
    class_orig: Counter                       # kept-class instance counts in the source labels
    n_difficult_kept: int
    n_off_axis: int                           # kept instances whose long edge is > OFF_AXIS_DEG off-axis
    n_dropped_class: int


@dataclass
class ChipRecord:
    chip_id: str
    split: str
    source: str
    x0: int
    y0: int
    valid_w: int
    valid_h: int
    kind: str                                 # positive | negative_hard | negative_random | negative
    lines: list[str] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    n_difficult: int = 0
    n_clipped: int = 0
    gsd: float | None = None


# --------------------------------------------------------------------------
# Pass 1: PLAN
# --------------------------------------------------------------------------


def find_images_dir(dota_split_dir: Path) -> Path:
    """Each staged ``images/partN.zip`` has its own top folder called ``images/``, so the PNGs live in ``images/images/``."""
    base = dota_split_dir / "images"
    nested = base / "images"
    if nested.is_dir() and any(nested.glob("*.png")):
        return nested
    return base


def plan_source(image_path: Path, label_path: Path) -> SourcePlan:
    with Image.open(image_path) as im:            # header only - no pixel decode
        width, height = im.size
    text = label_path.read_text(encoding="utf-8", errors="replace")
    instances = parse_dota_label_file(text)
    plan = plan_windows(width, height, instances)

    kept = [i for i in instances if i.category in CLASS_TO_INDEX]
    return SourcePlan(
        stem=image_path.stem, image_path=image_path, width=width, height=height, gsd=parse_gsd(text), plan=plan,
        class_orig=Counter(i.category for i in kept),
        n_difficult_kept=sum(1 for i in kept if i.difficult),
        n_off_axis=sum(1 for i in kept if axis_deviation_deg(i.points) > OFF_AXIS_DEG),
        n_dropped_class=len(instances) - len(kept),
    )


def plan_split(dota_split_dir: Path, *, label_dir: str = OBB_LABEL_DIR) -> tuple[list[SourcePlan], list[str]]:
    images_dir = find_images_dir(dota_split_dir)
    labels_dir = dota_split_dir / label_dir
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"{labels_dir} not found")
    plans: list[SourcePlan] = []
    skipped: list[str] = []
    for label_path in sorted(labels_dir.glob("P*.txt")):
        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.is_file():
            skipped.append(label_path.stem)
            continue
        plans.append(plan_source(image_path, label_path))
    return plans, skipped


def choose_monitor_sources(
    sources: list[SourcePlan], *, fraction: float = MONITOR_FRACTION, seed: int = SPLIT_SEED,
    cap_factor: float = MONITOR_CAP_FACTOR,
) -> set[str]:
    """Class-stratified holdout of whole TRAIN source images, rarest class first.

    For each class (rarest first) add random source images that contain it until the holdout holds
    ``fraction`` of that class's instances. A HARD CAP keeps any class from being swallowed: an image is
    skipped if adding it would push ANY class's holdout share above ``cap_factor * fraction``. (Without
    the cap, a rare class whose instances sit in a few big images - helicopters on an air base - ends up
    mostly in the holdout: the first version of this function held out 49% of all train helicopters.)
    If the cap makes the target unreachable the class simply ends up below target; the achieved shares
    are reported, never assumed.
    """
    rng = random.Random(seed)
    total = Counter()
    for s in sources:
        total.update(s.class_orig)
    cap = {c: math.ceil(cap_factor * fraction * total[c]) for c in KEPT_CLASSES}
    have: Counter = Counter()
    chosen: set[str] = set()
    for cls in sorted(KEPT_CLASSES, key=lambda c: total[c]):
        if total[cls] == 0:
            continue
        target = math.ceil(fraction * total[cls])
        cands = [s for s in sources if s.class_orig[cls] > 0 and s.stem not in chosen]
        rng.shuffle(cands)
        for s in cands:
            if have[cls] >= target:
                break
            if any(have[c] + s.class_orig[c] > cap[c] for c in KEPT_CLASSES if s.class_orig[c]):
                continue
            chosen.add(s.stem)
            have.update(s.class_orig)
    return chosen


def _records_for_source(src: SourcePlan, split: str, *, only: set[tuple[int, int]] | None = None,
                        kinds: dict[tuple[int, int], str] | None = None) -> list[ChipRecord]:
    out: list[ChipRecord] = []
    for w in src.plan.windows:
        key = (w.x0, w.y0)
        if only is not None and key not in only:
            continue
        rec = ChipRecord(
            chip_id=f"{src.stem}__{w.x0}_{w.y0}", split=split, source=src.stem, x0=w.x0, y0=w.y0,
            valid_w=min(CHIP_SIZE, src.width - w.x0), valid_h=min(CHIP_SIZE, src.height - w.y0),
            kind=(kinds or {}).get(key, "positive" if w.labels else "negative"), gsd=src.gsd,
        )
        for lab in w.labels:
            rec.lines.append(format_yolo_obb_line(lab.class_index, normalize_points(lab.points, CHIP_SIZE)))
            rec.class_counts[KEPT_CLASSES[lab.class_index]] += 1
            rec.n_difficult += int(lab.difficult)
            rec.n_clipped += int(lab.visible_fraction < 0.999)
        out.append(rec)
    return out


def select_train_windows(
    sources: list[SourcePlan], *, neg_fraction: float = NEGATIVE_FRACTION,
    hard_share: float = HARD_NEGATIVE_SHARE, seed: int = SPLIT_SEED,
    max_labels: int | None = TRAIN_MAX_LABELS_PER_CHIP, excluded: list | None = None,
) -> list[ChipRecord]:
    """All positive windows (except over-dense ones, see TRAIN_MAX_LABELS_PER_CHIP) + a bounded background sample
    (hard negatives first). Over-dense chips are appended to ``excluded`` (if given) so the report can list them."""
    positives: list[ChipRecord] = []
    hard: list[tuple[SourcePlan, tuple[int, int]]] = []
    easy: list[tuple[SourcePlan, tuple[int, int]]] = []
    for s in sources:
        pos_keys = {(w.x0, w.y0) for w in s.plan.windows if w.labels}
        if max_labels is not None:
            too_dense = {(w.x0, w.y0) for w in s.plan.windows if len(w.labels) > max_labels}
            if too_dense and excluded is not None:
                excluded += _records_for_source(s, "train", only=too_dense)
            pos_keys -= too_dense
        positives += _records_for_source(s, "train", only=pos_keys)
        for w in s.plan.windows:
            if not w.labels:
                (hard if w.n_distractors > 0 else easy).append((s, (w.x0, w.y0)))

    budget = round(neg_fraction * len(positives))
    rng = random.Random(seed)
    rng.shuffle(hard)
    rng.shuffle(easy)
    n_hard = min(len(hard), round(hard_share * budget))
    chosen = [(x, "negative_hard") for x in hard[:n_hard]]
    chosen += [(x, "negative_random") for x in easy[: max(0, budget - n_hard)]]
    # if the easy pool ran short, top up from the remaining hard ones
    if len(chosen) < budget:
        chosen += [(x, "negative_hard") for x in hard[n_hard: n_hard + (budget - len(chosen))]]

    by_src: dict[str, tuple[SourcePlan, dict[tuple[int, int], str]]] = {}
    for (s, key), kind in chosen:
        by_src.setdefault(s.stem, (s, {}))[1][key] = kind
    negatives: list[ChipRecord] = []
    for s, kinds in by_src.values():
        negatives += _records_for_source(s, "train", only=set(kinds), kinds=kinds)
    return positives + negatives


# --------------------------------------------------------------------------
# Pass 2: WRITE
# --------------------------------------------------------------------------


def _atomic_write_bytes_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_source_job(job: dict) -> dict:
    """Worker: decode one source image once, write all its chips + labels. Idempotent / resumable."""
    import cv2  # local import: keeps worker start-up lean and the module importable without a display stack

    cv2.setNumThreads(1)
    image = None
    n_img = n_skipped = 0
    out_dir = Path(job["out_dir"])
    for win in job["windows"]:
        img_path = out_dir / "images" / f"{win['chip_id']}.png"
        lbl_path = out_dir / "labels" / f"{win['chip_id']}.txt"
        _atomic_write_bytes_text(lbl_path, "\n".join(win["lines"]) + ("\n" if win["lines"] else ""))
        if img_path.is_file() and img_path.stat().st_size > 0:
            n_skipped += 1
            continue
        if image is None:
            image = cv2.imread(job["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cv2 failed to read {job['image_path']}")
        chip = crop_chip(image, win["x0"], win["y0"], CHIP_SIZE)
        tmp = img_path.with_name(img_path.stem + ".tmp.png")
        if not cv2.imwrite(str(tmp), chip, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise RuntimeError(f"cv2 failed to write {tmp}")
        os.replace(tmp, img_path)
        n_img += 1
    return {"source": job["source"], "written": n_img, "skipped_existing": n_skipped}


def write_chips(records: list[ChipRecord], sources: dict[str, SourcePlan], out_root: Path, *, workers: int,
                progress_every: int = 50) -> dict:
    by_src: dict[tuple[str, str], list[ChipRecord]] = {}
    for r in records:
        by_src.setdefault((r.split, r.source), []).append(r)
    jobs = []
    for (split, stem), recs in by_src.items():
        d = out_root / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        jobs.append({
            "source": stem, "image_path": str(sources[stem].image_path), "out_dir": str(d),
            "windows": [{"chip_id": r.chip_id, "x0": r.x0, "y0": r.y0, "lines": r.lines} for r in recs],
            "_px": sources[stem].width * sources[stem].height,
        })
    jobs.sort(key=lambda j: -j["_px"])                     # biggest first: keeps the pool balanced to the tail

    t0 = time.time()
    written = skipped = done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_write_source_job, {k: v for k, v in j.items() if k != "_px"}) for j in jobs]
        for f in as_completed(futs):
            res = f.result()                                # re-raises worker errors loudly
            written += res["written"]
            skipped += res["skipped_existing"]
            done += 1
            if done % progress_every == 0 or done == len(futs):
                print(f"[convert]   {done}/{len(futs)} source images  ({written} chips written, "
                      f"{skipped} already present)  {time.time() - t0:.0f}s", flush=True)
    return {"chips_written": written, "chips_already_present": skipped, "elapsed_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------
# Lists, yamls, index
# --------------------------------------------------------------------------


def _rel(chip_id: str) -> str:
    return f"./images/{chip_id}.png"


def write_lists(out_root: Path, train: list[ChipRecord], monitor: list[ChipRecord], val: list[ChipRecord],
                repeats: dict[str, int], *, seed: int = SPLIT_SEED) -> dict:
    rng = random.Random(seed)
    train_lines = [_rel(c.chip_id) for c in train for _ in range(repeats.get(c.chip_id, 1))]
    rng.shuffle(train_lines)
    (out_root / "train" / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_root / "monitor" / "monitor.txt").write_text("\n".join(_rel(c.chip_id) for c in monitor) + "\n", encoding="utf-8")
    (out_root / "val" / "val.txt").write_text("\n".join(_rel(c.chip_id) for c in val) + "\n", encoding="utf-8")
    return {"train_list_entries": len(train_lines), "monitor_list_entries": len(monitor), "val_list_entries": len(val)}


def _names_block() -> str:
    return "\n".join(f"  {i}: {name}" for i, name in enumerate(KEPT_CLASSES))


def write_dataset_yamls(out_root: Path) -> dict[str, Path]:
    """Two yamls. The TRAINING one's ``val:`` is the monitor split - the training process cannot even
    open the official val images. The official-val yaml is for the evaluation scripts only."""
    root = out_root.resolve().as_posix()
    train_yaml = out_root / "dataset.yaml"
    train_yaml.write_text(
        f"# TRAINING dataset. val: is the *monitor* holdout carved from DOTA train, NOT the official val.\n"
        f"path: {root}\ntrain: train/train.txt\nval: monitor/monitor.txt\nnames:\n{_names_block()}\n",
        encoding="utf-8",
    )
    val_yaml = out_root / "dataset_official_val.yaml"
    val_yaml.write_text(
        f"# EVALUATION ONLY: the official DOTA val split (natural distribution). Never used for training,\n"
        f"# early stopping, checkpoint choice or threshold selection. (train: is a required key; it is unused.)\n"
        f"path: {root}\ntrain: monitor/monitor.txt\nval: val/val.txt\nnames:\n{_names_block()}\n",
        encoding="utf-8",
    )
    monitor_yaml = out_root / "dataset_monitor.yaml"
    monitor_yaml.write_text(
        f"# Monitor holdout (from DOTA train) as the eval split - used for threshold / checkpoint selection.\n"
        f"path: {root}\ntrain: monitor/monitor.txt\nval: monitor/monitor.txt\nnames:\n{_names_block()}\n",
        encoding="utf-8",
    )
    return {"train": train_yaml, "official_val": val_yaml, "monitor": monitor_yaml}


def write_chip_index(out_root: Path, records: list[ChipRecord]) -> Path:
    path = out_root / "chip_index.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "chip_id", "source", "x0", "y0", "valid_w", "valid_h", "kind", "n_labels",
                    "n_difficult", "n_clipped", "gsd"] + list(KEPT_CLASSES))
        for r in records:
            w.writerow([r.split, r.chip_id, r.source, r.x0, r.y0, r.valid_w, r.valid_h, r.kind, len(r.lines),
                        r.n_difficult, r.n_clipped, "" if r.gsd is None else r.gsd] +
                       [r.class_counts.get(c, 0) for c in KEPT_CLASSES])
    return path


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _gsd_stats(sources: list[SourcePlan]) -> dict:
    g = np.array([s.gsd for s in sources if s.gsd is not None], dtype=float)
    if not len(g):
        return {"n_with_gsd": 0}
    return {
        "n_with_gsd": int(len(g)), "n_null": len(sources) - int(len(g)),
        "min_m": round(float(g.min()), 3), "p10_m": round(float(np.percentile(g, 10)), 3),
        "median_m": round(float(np.median(g)), 3), "p90_m": round(float(np.percentile(g, 90)), 3),
        "max_m": round(float(g.max()), 3), "share_le_0.5m": round(float((g <= 0.5).mean()), 3),
        "share_le_0.35m": round(float((g <= 0.35).mean()), 3),
    }


def _density(chips: list[ChipRecord]) -> dict:
    n = np.array([len(c.lines) for c in chips], dtype=float)
    if not len(n):
        return {}
    return {
        "mean": round(float(n.mean()), 1), "median": float(np.median(n)), "p90": float(np.percentile(n, 90)),
        "p99": float(np.percentile(n, 99)), "max": int(n.max()),
        "n_chips_gt_300": int((n > 300).sum()), "n_chips_gt_1000": int((n > 1000).sum()),
    }


def split_report(name: str, sources: list[SourcePlan], chips: list[ChipRecord], *, skipped_no_image: int = 0) -> dict:
    orig = Counter()
    for s in sources:
        orig.update(s.class_orig)
    n_orig = sum(orig.values())
    n_retained = sum(s.plan.n_retained for s in sources)
    lines = sum(len(c.lines) for c in chips)
    occ = Counter()
    for c in chips:
        occ.update(c.class_counts)
    return {
        "split": name,
        "n_source_images": len(sources),
        "n_label_files_without_image": skipped_no_image,
        "n_candidate_windows": sum(len(s.plan.windows) for s in sources),
        "n_chips_written": len(chips),
        "n_positive_chips": sum(1 for c in chips if c.lines),
        "n_negative_chips": sum(1 for c in chips if not c.lines),
        "n_negative_hard_chips": sum(1 for c in chips if c.kind == "negative_hard"),
        "n_original_instances_kept_classes": n_orig,
        "n_instances_retained": n_retained,
        "n_instances_dropped_never_visible_enough": n_orig - n_retained,
        "n_zero_area_instances_skipped": sum(s.plan.n_invalid_instances for s in sources),
        "retained_pct": round(100.0 * n_retained / max(n_orig, 1), 3),
        "n_chip_label_lines": lines,
        "chip_lines_per_retained_instance": round(lines / max(n_retained, 1), 3),
        "n_boxes_clipped_and_refit_at_chip_edge": sum(c.n_clipped for c in chips),
        "pct_chip_boxes_clipped": round(100.0 * sum(c.n_clipped for c in chips) / max(lines, 1), 2),
        "original_instances_by_class": {k: orig[k] for k in KEPT_CLASSES},
        "chip_occurrences_by_class": {k: occ[k] for k in KEPT_CLASSES},
        "n_difficult_kept_instances": sum(s.n_difficult_kept for s in sources),
        "labels_per_chip": _density(chips),
        "orientation": {
            "n_instances_off_axis_gt_5deg": sum(s.n_off_axis for s in sources),
            "pct_off_axis": round(100.0 * sum(s.n_off_axis for s in sources) / max(n_orig, 1), 2),
        },
        "n_dropped_class_instances_in_source_labels": sum(s.n_dropped_class for s in sources),
        "gsd_m": _gsd_stats(sources),
    }


def verify_conversion(out_root: Path) -> dict:
    """Re-read what was written: every image has a label file, every label line is well-formed, the three
    splits share no source image, and the training yaml never mentions the official val."""
    problems: list[str] = []
    stems: dict[str, set[str]] = {}
    n_lines = 0
    n_off = 0
    for split in ("train", "monitor", "val"):
        imgs = {p.stem for p in (out_root / split / "images").glob("*.png")}
        lbls = {p.stem for p in (out_root / split / "labels").glob("*.txt")}
        if imgs != lbls:
            problems.append(f"{split}: {len(imgs ^ lbls)} chips without a matching label/image file")
        stems[split] = {i.split("__")[0] for i in imgs}
        for p in (out_root / split / "labels").glob("*.txt"):
            for ln in p.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 9:
                    problems.append(f"{p.name}: malformed line {ln!r}")
                    continue
                if not (0 <= int(parts[0]) < len(KEPT_CLASSES)) or any(not (0.0 <= float(v) <= 1.0) for v in parts[1:]):
                    problems.append(f"{p.name}: out-of-range value in {ln!r}")
                n_lines += 1
                if split != "val":
                    pts = [(float(parts[i]), float(parts[i + 1])) for i in range(1, 9, 2)]
                    n_off += int(axis_deviation_deg(pts) > OFF_AXIS_DEG)
    overlap = {
        "train&monitor": stems["train"] & stems["monitor"],
        "train&val": stems["train"] & stems["val"],
        "monitor&val": stems["monitor"] & stems["val"],
    }
    for k, v in overlap.items():
        if v:
            problems.append(f"source images shared between {k}: {sorted(v)[:5]}")
    yaml_txt = (out_root / "dataset.yaml").read_text(encoding="utf-8")
    if "val/val.txt" in yaml_txt.replace("monitor/monitor.txt", ""):
        problems.append("training dataset.yaml references the official val split")
    return {
        "ok": not problems, "problems": problems[:20], "n_label_lines_checked": n_lines,
        "source_images": {k: len(v) for k, v in stems.items()},
        "source_image_overlap": {k: len(v) for k, v in overlap.items()},
        "pct_off_axis_train_and_monitor_label_lines": round(100.0 * n_off / max(n_lines, 1), 2),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def convert_dota(
    dota_root: Path, out_root: Path, *, workers: int = 12, monitor_fraction: float = MONITOR_FRACTION,
    neg_fraction: float = NEGATIVE_FRACTION, rfs_t: float = RFS_THRESHOLD, do_write: bool = True,
    label_dir: str = OBB_LABEL_DIR,
) -> dict:
    t_all = time.time()
    out_root.mkdir(parents=True, exist_ok=True)

    print("[convert] PASS 1 - planning windows (labels + image headers only) ...", flush=True)
    val_sources, val_skipped = plan_split(dota_root / "val", label_dir=label_dir)
    train_sources_all, train_skipped = plan_split(dota_root / "train", label_dir=label_dir)
    print(f"[convert]   val: {len(val_sources)} images, train: {len(train_sources_all)} images "
          f"({time.time() - t_all:.0f}s)", flush=True)

    monitor_stems = choose_monitor_sources(train_sources_all, fraction=monitor_fraction)
    monitor_sources = [s for s in train_sources_all if s.stem in monitor_stems]
    train_sources = [s for s in train_sources_all if s.stem not in monitor_stems]

    dense_excluded: list[ChipRecord] = []
    train_chips = select_train_windows(train_sources, neg_fraction=neg_fraction, excluded=dense_excluded)
    monitor_chips = [r for s in monitor_sources for r in _records_for_source(s, "monitor")]
    val_chips = [r for s in val_sources for r in _records_for_source(s, "val")]

    repeats, rfs_report = build_repeat_list(
        [ChipClasses(c.chip_id, dict(c.class_counts)) for c in train_chips], t=rfs_t, seed=RFS_SEED
    )

    src_index = {s.stem: s for s in train_sources_all + val_sources}
    all_chips = train_chips + monitor_chips + val_chips
    print(f"[convert] planned chips: train={len(train_chips)}  monitor={len(monitor_chips)}  val={len(val_chips)}  "
          f"(RFS list entries: {rfs_report['n_list_entries']})", flush=True)

    write_report: dict = {}
    if do_write:
        print(f"[convert] PASS 2 - writing {len(all_chips)} chips with {workers} workers ...", flush=True)
        write_report = write_chips(all_chips, src_index, out_root, workers=workers)
        write_lists(out_root, train_chips, monitor_chips, val_chips, repeats)
        write_dataset_yamls(out_root)
        write_chip_index(out_root, all_chips)

    monitor_orig = Counter()
    for s in monitor_sources:
        monitor_orig.update(s.class_orig)
    train_all_orig = Counter()
    for s in train_sources_all:
        train_all_orig.update(s.class_orig)

    report = {
        "purpose": "Phase 8F-2: DOTA OBB -> YOLO-OBB chips (train / monitor / official val)",
        "label_source": label_dir,
        "chip_size": CHIP_SIZE, "overlap_px": OVERLAP, "stride_px": STRIDE, "visibility_threshold": VISIBILITY_THRESHOLD,
        "png_compression": PNG_COMPRESSION,
        "classes": list(KEPT_CLASSES), "dropped_classes": list(DROPPED_CLASSES),
        "monitor_fraction_target": monitor_fraction, "negative_fraction": neg_fraction,
        "splits": {
            "train": split_report("train", train_sources, train_chips, skipped_no_image=len(train_skipped)),
            "monitor": split_report("monitor", monitor_sources, monitor_chips),
            "val": split_report("val", val_sources, val_chips, skipped_no_image=len(val_skipped)),
        },
        "monitor_holdout": {
            "n_source_images": len(monitor_sources),
            "share_of_train_source_images": round(len(monitor_sources) / max(len(train_sources_all), 1), 4),
            "share_of_train_instances_by_class": {
                k: round(monitor_orig[k] / max(train_all_orig[k], 1), 4) for k in KEPT_CLASSES},
        },
        "train_density_cap": {
            "max_labels_per_chip": TRAIN_MAX_LABELS_PER_CHIP,
            "n_chips_excluded_from_train": len(dense_excluded),
            "label_lines_excluded": sum(len(c.lines) for c in dense_excluded),
            "share_of_train_label_lines_excluded": round(
                sum(len(c.lines) for c in dense_excluded)
                / max(sum(len(c.lines) for c in dense_excluded + train_chips), 1), 4),
            "excluded_chips": sorted(((c.chip_id, len(c.lines), c.gsd) for c in dense_excluded), key=lambda t: -t[1]),
            "note": "val and monitor are never capped (natural distribution).",
        },
        "rfs": rfs_report,
        "write": write_report,
        "total_elapsed_s": round(time.time() - t_all, 1),
    }
    if do_write:
        report["verification"] = verify_conversion(out_root)
        (out_root / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

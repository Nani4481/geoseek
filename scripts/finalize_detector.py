"""Phase 8F-2 Step 5 (tail): freeze the trained detector and write its model card into the provenance manifest.

  * copies the monitor-selected ``best.pt`` to data/models/detector/geoseek_obb_v15_yolo26s.pt (+ records SHA256),
  * writes the model card next to it (``<weights>.card.json``, which YoloObbDetectionModel reads for classes and the
    operating confidence) and into the manifest (``detector_model_card``, ``detector_training``),
  * saves the loss / mAP curves as PNGs.

Re-runnable: run it once right after training, and again after ``scripts/eval_detector.py`` so the card also carries the
evaluation summary and the operating point (chosen on the monitor split).

    python scripts/finalize_detector.py [--run geoseek_obb_v15_yolo26s]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_GROUPS, KEPT_CLASSES, LABEL_SETS, PRIMARY_LABEL_SET  # noqa: E402
from geoseek.detect.plots import plot_training_curves  # noqa: E402
from geoseek.staging.manifest import record_analysis_section, record_artifact  # noqa: E402

WEIGHTS_NAME = "geoseek_obb_v15_yolo26s.pt"
LICENSE = ("AGPL-3.0 (derivative of Ultralytics' AGPL-3.0 YOLO26s-OBB weights) AND academic-use-only / non-commercial "
           "(fine-tuned on DOTA v1.5; Google Earth imagery under Google's terms). Not for commercial use.")

VEHICLE_CLASSES = CLASS_GROUPS["ground_vehicles"]  # ("small-vehicle", "large-vehicle") - tuned on xView, not DOTA monitor


def build_operating_point(dota_eval: dict, xview_eval: dict | None) -> dict:
    """PER-CLASS confidence thresholds (replaces the old single global cutoff - see the docstring below for why).

    Vehicles (small-vehicle, large-vehicle) are tuned on xView: an independently-labelled dataset the detector was
    NEVER trained on, giving a real held-out signal specifically for the classes a pipeline-parity test showed the
    old global threshold most undertuned. Every other class is tuned on the DOTA MONITOR split (never the official
    val, never Van Nuys - Van Nuys stays untouched for evaluation only). Each entry's own best-F1 threshold over its
    full PR curve is used, not one value compromising across all 8 classes together.
    """
    per_class: dict[str, dict] = {}
    for cls in VEHICLE_CLASSES:
        if xview_eval is None:
            raise RuntimeError(f"no xView eval results - cannot set a per-class threshold for {cls!r}")
        op = xview_eval["hbb_fair"]["operating_point"][cls]
        per_class[cls] = {
            "conf": op["best_f1_threshold"], "f1": op["f1"], "precision": op["precision"], "recall": op["recall"],
            "chosen_on": "xView (846 images, 214,612+ GT instances) - independent test set, never trained on",
            "protocol": ("full-image evaluation, prediction's own enclosing axis-aligned box vs xView's axis-aligned "
                        "GT (hbb_fair - removes the tight-OBB-vs-loose-HBB IoU bias), best-F1 threshold over the PR curve"),
            "source": "data/detect_eval/xview/eval_results.json",
        }
    for cls, entry in dota_eval["operating_point"]["per_class_best"].items():
        if cls in VEHICLE_CLASSES:
            continue
        per_class[cls] = {
            "conf": entry["conf"], "f1": entry["f1"],
            "chosen_on": "DOTA monitor split (class-stratified holdout of TRAIN source images)",
            "protocol": "full-image DOTA-protocol evaluation (geoseek.detect.evaluate), best-F1 threshold over the PR curve",
            "source": "data/detect_eval/eval_results.json",
        }
    missing = set(KEPT_CLASSES) - set(per_class)
    if missing:
        raise RuntimeError(f"no per-class threshold available for {sorted(missing)}")
    return {
        "schema": "per_class",
        "note": ("Each class has its OWN confidence threshold - see per_class[<class>].chosen_on / .protocol for "
                "how each was picked. Supersedes deprecated_global_operating_point below: a single threshold "
                "chosen for macro-F1 across all 8 classes together was confirmed too conservative for vehicles "
                "specifically - on a fixed 10-crop Van Nuys audit, correcting small-vehicle's threshold alone "
                "(0.525 -> its own best-F1 value) recovered 4x more detections on the identical imagery, though "
                "that alone did not close the full gap to xView's recall (see docs/PHASE8F3A.md)."),
        "per_class": per_class,
        "deprecated_global_operating_point": {
            "conf": dota_eval["operating_point"]["conf"], "chosen_on": "monitor (macro-F1 across all 8 classes together)",
            "macro_f1_on_monitor": dota_eval["operating_point"]["macro_f1_on_monitor"],
            "superseded_reason": ("A single cutoff tuned across all 8 classes together undertunes vehicles "
                                  "specifically - see the 'note' above and docs/PHASE8F3A.md."),
        },
    }


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=30,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return ""


def read_progress(run_dir: Path) -> list[dict]:
    p = run_dir / "progress.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.is_file() else []


def parse_args_yaml(run_dir: Path) -> dict:
    out = {}
    for ln in (run_dir / "args.yaml").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", ln)
        if m:
            v = m.group(2).strip()
            out[m.group(1)] = (float(v) if re.fullmatch(r"-?\d+\.\d+(e-?\d+)?", v) else int(v) if re.fullmatch(r"-?\d+", v)
                               else {"true": True, "false": False}.get(v.lower(), v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="geoseek_obb_v15_yolo26s")
    args = ap.parse_args()

    settings = get_settings()
    run_dir = settings.data_dir / "runs" / "detector" / args.run
    weights_src = run_dir / "weights" / "best.pt"
    if not weights_src.is_file():
        raise SystemExit(f"{weights_src} not found - has training finished?")
    dest_dir = settings.models_dir / "detector"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / WEIGHTS_NAME
    shutil.copy2(weights_src, dest)
    w_sha = sha256(dest)

    import torch

    ck = torch.load(str(dest), map_location="cpu", weights_only=False)
    model = ck["model"]
    n_params = sum(p.numel() for p in model.parameters())
    progress = [r for r in read_progress(run_dir) if r["epoch"] <= r["epochs"]]
    train_args = parse_args_yaml(run_dir)
    init = settings.models_dir / "yolo_obb" / "geoseek_init_yolo26s_8cls.pt"
    pre = settings.models_dir / "yolo_obb" / "yolo26s-obb.pt"
    conv = json.loads((settings.datasets_dir / "dota_obb" / "conversion_report.json").read_text(encoding="utf-8"))
    launch_commit = (run_dir.parent / "git_commit_at_launch.txt").read_text().strip() if (run_dir.parent / "git_commit_at_launch.txt").is_file() else None
    log_path = run_dir.parent / f"train_{args.run}.log"
    log_txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    n_resumes = len(re.findall(r"\[train\] RESUMING", log_txt))
    epoch_secs = [r["epoch_seconds"] for r in progress]
    unplugged = [r["epoch"] for r in progress if r.get("ac_power") is False]

    # WHICH epoch are these weights from? best.pt is chosen on the MONITOR split by Ultralytics' fitness (0.1*mAP50 + 0.9*mAP50-95);
    # it need not be the last epoch, and the card must not imply "20 epochs of training" when it is an early one.
    import csv
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    col = {h: [float(r[i]) for r in rows[1:]] for i, h in enumerate(head)}
    fit = [0.1 * a + 0.9 * b for a, b in zip(col["metrics/mAP50(B)"], col["metrics/mAP50-95(B)"])]
    best_i = max(range(len(fit)), key=fit.__getitem__)
    weights_epoch = int(col["epoch"][best_i])
    fitness_by_epoch = {int(e): round(f, 4) for e, f in zip(col["epoch"], fit)}

    eval_path = settings.data_dir / "detect_eval" / "eval_results.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else None
    xview_eval_path = settings.data_dir / "detect_eval" / "xview" / "eval_results.json"
    xview_ev = json.loads(xview_eval_path.read_text(encoding="utf-8")) if xview_eval_path.is_file() else None

    card = {
        "name": "geoseek-obb-v15-yolo26s",
        "task": "oriented object detection (OBB) in nadir aerial / satellite RGB",
        "architecture": {
            "family": "Ultralytics YOLO26s-OBB", "head": type(model.model[-1]).__name__, "n_params": int(n_params),
            "input": "RGB, 1024x1024 (larger inputs are windowed with 200 px overlap)",
            "dual_head": "one-to-many (NMS) + one-to-one (NMS-free); inference here uses the NMS path (better on dense scenes: "
                         "monitor mAP50 0.821 vs 0.810 for the NMS-free head at epoch 0)",
            "ultralytics_version_trained_with": __import__("ultralytics").__version__,
            "torch": torch.__version__,
        },
        "classes": list(KEPT_CLASSES), "class_groups": {k: list(v) for k, v in CLASS_GROUPS.items()},
        "initialisation": {
            "pretrained": "yolo26s-obb.pt (DOTAv1, 15 classes)", "pretrained_sha256": sha256(pre) if pre.is_file() else None,
            "class_head": "8 of the 15 pretrained class rows transplanted by NAME into both heads at every FPN level "
                          "(geoseek.detect.surgery); verified bit-exact (0.0 max diff)",
            "init_sha256": sha256(init) if init.is_file() else None,
        },
        "dataset": {
            "name": "DOTA v1.5 OBB (oriented) - repaired from the retained archives", "label_set": PRIMARY_LABEL_SET,
            "label_dir": LABEL_SETS[PRIMARY_LABEL_SET], "chip_size": conv["chip_size"], "overlap_px": conv["overlap_px"],
            "license": "academic use only; commercial use prohibited",
            "splits": {
                "train": {k: conv["splits"]["train"][k] for k in ("n_source_images", "n_chips_written", "n_positive_chips",
                                                                    "n_negative_chips", "n_original_instances_kept_classes")},
                "monitor": {k: conv["splits"]["monitor"][k] for k in ("n_source_images", "n_chips_written", "n_original_instances_kept_classes")},
                "official_val": {k: conv["splits"]["val"][k] for k in ("n_source_images", "n_chips_written", "n_original_instances_kept_classes")},
            },
            "split_policy": "monitor = class-stratified holdout of TRAIN source images (per-epoch curves, checkpoint and "
                            "threshold selection); official val touched only by scripts/eval_detector.py, after training",
            "train_list_entries_per_epoch": conv["rfs"]["n_list_entries"],
            "imbalance": {"method": "chip-level repeat-factor sampling (LVIS) t=%s + hard-negative background chips + mosaic" % conv["rfs"]["threshold_t"],
                          "instance_ratio_max_over_min": {"before": conv["rfs"]["instance_ratio_max_over_min_unique"],
                                                          "after": conv["rfs"]["instance_ratio_max_over_min_effective"]}},
            "train_density_cap": conv["train_density_cap"]["max_labels_per_chip"],
        },
        "hyperparameters": train_args,
        "weights_epoch": {
            "epoch": weights_epoch, "of": int(max(col["epoch"])),
            "selected_by": "Ultralytics fitness (0.1*mAP50 + 0.9*mAP50-95) on the MONITOR split - never on the official val",
            "monitor_fitness_by_epoch": fitness_by_epoch,
            "note": ("these weights are the checkpoint after epoch %d of %d, not the final epoch" % (weights_epoch, int(max(col["epoch"]))))
                    if weights_epoch != int(max(col["epoch"])) else "these weights are the final epoch",
        },
        "training": {
            "epochs_completed": len(progress), "epochs_planned": train_args.get("epochs"),
            "wall_clock_train_and_val_hours": round(sum(epoch_secs) / 3600, 2) if epoch_secs else None,
            "mean_epoch_seconds": round(sum(epoch_secs) / max(len(epoch_secs), 1), 1),
            "peak_vram_gb_torch_reserved": max((r["peak_vram_reserved_gb"] for r in progress), default=None),
            "vram_cap_fraction": 0.80, "resumes_after_interruption": n_resumes, "epochs_run_on_battery": unplugged,
            "started_at": progress[0]["wall_time"] if progress else None, "finished_at": progress[-1]["wall_time"] if progress else None,
            "gpu": "NVIDIA GeForce RTX 4060 Laptop (8.59 GB)", "mixed_precision": "AMP (fp16 autocast)",
        },
        "provenance": {
            "git_commit_at_training_launch": launch_commit, "git_commit_at_finalize": git("rev-parse", "HEAD"),
            "working_tree_dirty_at_finalize": bool(git("status", "--porcelain")),
            "weights_sha256": w_sha, "weights_source_run": str(run_dir), "finalized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "weights_sha256": w_sha,
        "license": LICENSE,
    }
    if ev:
        card["operating_point"] = build_operating_point(ev, xview_ev)
        card["evaluation_summary"] = {
            "official_val_full_image_v15": {g: {k: v for k, v in s.items() if k.startswith("macro_")} for g, s in ev["val_full_image_v15"]["groups"].items()},
            "per_class_AP50_v15": {n: ev["val_full_image_v15"]["per_class"][n]["AP50"] for n in KEPT_CLASSES},
            "see": "data/detect_eval/eval_results.json and docs/PHASE8F2.md",
        }

    card_path = dest.with_suffix(".card.json")
    card_path.write_text(json.dumps(card, indent=1), encoding="utf-8")
    rec = record_artifact(name="geoseek-obb-v15-yolo26s", source_url=f"local: trained by scripts/train_detector.py at git {launch_commit}",
                          local_path=dest, license=LICENSE)
    record_analysis_section("detector_model_card", card)
    record_analysis_section("detector_training", {"config": train_args, "progress_per_epoch": progress, "log": str(log_path),
                                                   "resumes_after_interruption": n_resumes})

    figs = settings.data_dir / "detect_eval" / "figures"
    baseline = None
    bj = settings.data_dir / "detect_preflight" / "epoch0_monitor_baseline.json"
    if bj.is_file():
        b = json.loads(bj.read_text(encoding="utf-8"))["nms_one2many"]
        baseline = {"monitor": {"mAP50": b["mAP50"], "mAP50_95": b["mAP50-95"]}}
    official = ev.get("official_val_by_checkpoint", []) if ev else []
    written = plot_training_curves(run_dir, figs, baseline=baseline, official_points=official)

    print(f"[finalize] weights: {dest}  sha256={w_sha}  ({rec.byte_size} B)")
    print(f"[finalize] model card: {card_path}")
    for w in written:
        print(f"[finalize] curve: {w}")
    print(f"[finalize] epochs completed: {len(progress)}; mean epoch {card['training']['mean_epoch_seconds']} s; "
          f"resumes: {n_resumes}; peak VRAM {card['training']['peak_vram_gb_torch_reserved']} GB")


if __name__ == "__main__":
    main()

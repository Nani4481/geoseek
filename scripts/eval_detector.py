"""Phase 8F-2 Step 6: HONEST evaluation of the trained detector on the held-out official DOTA val split.

Everything is measured, nothing is tuned to look better:

  * two protocols on the same predictions -
      chip level   : Ultralytics' own validator (all instances incl. `difficult` count, per-chip, ProbIoU) - comparable with
                     the pretrained checkpoints' self-reported numbers;
      full image   : DOTA-devkit protocol (geoseek.detect.evaluate: cross-chip merge, polygon IoU vs the original quad GT,
                     difficult ignored) - comparable with published DOTA results;
  * two label sets - v1.5 OBB (what the detector was trained for; PRIMARY) and v1.0 OBB (what every published number and the
    pretrained weights use; secondary, and v1.0's vehicle labels are incomplete - see geoseek.detect.classes);
  * two models - the fine-tuned detector and its own starting point ("pretrained-restricted": the DOTA-pretrained weights
    with the class head transplanted, i.e. NO fine-tuning), on identical chips;
  * per class, per reporting group (ground vehicles / ships+aircraft / infrastructure, never averaged together), and
    size-stratified for the vehicle classes;
  * the operating confidence is chosen on the MONITOR holdout only and merely APPLIED to val.

Val is opened here for the first time: the training side never touched it (checked below: disjoint source images, the
training yaml's val: is the monitor list, and no ``val/labels.cache`` existed before this script).

    python scripts/eval_detector.py                       # everything (GPU, ~40 min)
    python scripts/eval_detector.py --stages monitor,val  # just the predictions + protocol scores
    python scripts/eval_detector.py --stages qual,curves  # figures from cached predictions
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_GROUPS, KEPT_CLASSES, LABEL_SETS  # noqa: E402
from geoseek.detect.eval_io import assemble_full_image_dets, load_dota_gt, parse_chip_id  # noqa: E402
from geoseek.detect.evaluate import (  # noqa: E402
    ImageDets, ImageGT, best_f1_threshold, evaluate_dataset, group_summary, match_class, merge_cross_chip,
    operating_point, pr_curve, xywhr_to_polys)
from geoseek.detect.infer import load_predictions, predict_chips, save_predictions  # noqa: E402
from geoseek.detect.plots import plot_pr, plot_per_class, plot_training_curves, read_results_csv  # noqa: E402

CONF_FLOOR_FULL = 0.005        # candidate floor for the full-image protocol (the chip-level validator uses 0.001)
NMS_IOU = 0.7                  # the detector's own per-chip NMS
MERGE_IOU = 0.5                # cross-chip de-duplication
MAX_DET = 2500                 # per chip (the densest val chip holds 2,314 GT)
SEED = 20260920
CLASS_BGR = {"small-vehicle": (0, 255, 0), "large-vehicle": (0, 200, 255), "ship": (255, 200, 0), "plane": (255, 0, 255),
             "helicopter": (0, 0, 255), "storage-tank": (255, 255, 0), "harbor": (255, 128, 0), "bridge": (128, 0, 255)}


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------


def sanitize(o):
    """numpy -> plain Python, NaN/inf -> None (strictly valid JSON)."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    return o


def chip_paths(root: Path, split: str) -> list[Path]:
    lines = (root / split / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    seen, out = set(), []
    for ln in lines:
        if ln and ln not in seen:
            seen.add(ln)
            out.append(root / split / ln[2:] if ln.startswith("./") else Path(ln))
    return out


def get_pred(tag: str, weights: Path, split: str, root: Path, out_dir: Path, *, force=False, **kw) -> dict:
    f = out_dir / f"pred_{tag}_{split}.npz"
    if f.is_file() and not force:
        print(f"[eval] cached predictions: {f.name}", flush=True)
        return load_predictions(f)
    print(f"[eval] predicting {tag} on {split} ({len(chip_paths(root, split))} chips) ...", flush=True)
    pred = predict_chips(weights, chip_paths(root, split), conf=CONF_FLOOR_FULL, iou=NMS_IOU, max_det=MAX_DET, **kw)
    save_predictions(f, pred)
    print(f"[eval]   {len(pred['conf'])} raw detections in {pred['seconds']:.0f}s", flush=True)
    return pred


def full_image_dets(pred: dict) -> dict[str, ImageDets]:
    per = assemble_full_image_dets(pred["chip_ids"], pred["det_chip_idx"], pred["xywhr"], pred["conf"], pred["cls"])
    return {stem: merge_cross_chip(d, MERGE_IOU) for stem, d in per.items()}


def score_full(pred: dict, dota_root: Path, split: str, label_set: str, *, size_buckets: bool):
    stems = {parse_chip_id(c)[0] for c in pred["chip_ids"]}
    gt = load_dota_gt(dota_root / split / LABEL_SETS[label_set], stems)
    dets = full_image_dets(pred)
    res = evaluate_dataset(gt, dets, KEPT_CLASSES, size_buckets=size_buckets)
    matches = res.pop("_matches")
    res["groups"] = group_summary(res["per_class"], CLASS_GROUPS)
    res["n_images"] = len(gt)
    res["n_gt_total_incl_difficult"] = int(sum(len(g.classes) for g in gt.values()))
    res["n_gt_difficult"] = int(sum(g.difficult.sum() for g in gt.values()))
    res["n_detections_after_merge"] = int(sum(len(d.scores) for d in dets.values()))
    return res, matches


def operating_metrics(matches: dict, conf: float) -> dict:
    per = {n: operating_point(m, conf) for n, m in matches.items()}
    groups = {}
    for g, names in list(CLASS_GROUPS.items()) + [("all_kept_classes", tuple(KEPT_CLASSES))]:
        tp = sum(per[n]["TP"] for n in names)
        fp = sum(per[n]["FP"] for n in names)
        fn = sum(per[n]["FN"] for n in names)
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        groups[g] = {"TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else float("nan")}
    return {"conf": conf, "per_class": per, "groups": groups}


def choose_operating_conf(matches: dict) -> dict:
    """The single confidence maximising macro-F1 over the 8 classes on the MONITOR split (grid), plus per-class optima."""
    grid = np.round(np.arange(0.05, 0.951, 0.025), 3)
    macro = []
    for c in grid:
        f1s = [operating_point(matches[n], float(c))["f1"] for n in KEPT_CLASSES]
        macro.append(float(np.nanmean(np.nan_to_num(f1s, nan=0.0))))
    i = int(np.argmax(macro))
    per_class = {n: dict(zip(("conf", "f1"), best_f1_threshold(matches[n]))) for n in KEPT_CLASSES}
    return {"conf": float(grid[i]), "macro_f1_on_monitor": macro[i], "chosen_on": "monitor (NOT val)",
            "grid": {str(g): round(m, 4) for g, m in zip(grid, macro)}, "per_class_best": per_class}


def chip_level_val(weights: Path, yaml_path: Path, split_key: str = "val") -> dict:
    """Ultralytics' validator on a chip list (all instances count; ProbIoU; 101-point AP)."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    r = m.val(data=str(yaml_path), split=split_key, imgsz=1024, batch=8, conf=0.001, iou=NMS_IOU, max_det=MAX_DET, half=True,
              plots=False, workers=6, device=0, verbose=False, end2end=False,
              project=str(Path(os.environ.get("TEMP", ".")) / "geoseek_eval_val"), name="tmp", exist_ok=True)
    b = r.box
    idx = [int(i) for i in b.ap_class_index]
    return {
        "mAP50": float(b.map50), "mAP50_95": float(b.map), "precision": float(b.mp), "recall": float(b.mr),
        "per_class_AP50": {KEPT_CLASSES[c]: float(v) for c, v in zip(idx, b.ap50)},
        "per_class_AP50_95": {KEPT_CLASSES[c]: float(v) for c, v in zip(idx, b.ap)},
    }


# --------------------------------------------------------------------------
# qualitative examples (selected from the chip index + a fixed seed: never from the predictions)
# --------------------------------------------------------------------------

STRATA = [
    ("dense small vehicles", lambda r: r["small-vehicle"] >= 30 and r["small-vehicle"] >= 0.6 * r["n"]),
    ("large + small vehicles", lambda r: r["large-vehicle"] >= 6 and r["large-vehicle"] + r["small-vehicle"] >= 0.7 * r["n"]),
    ("ships / harbor", lambda r: r["ship"] >= 8 and r["ship"] >= 0.5 * r["n"]),
    ("aircraft", lambda r: r["plane"] + r["helicopter"] >= 4 and r["plane"] + r["helicopter"] >= 0.5 * r["n"]),
    ("infrastructure", lambda r: r["storage-tank"] + r["bridge"] + r["harbor"] >= 3 and
     r["storage-tank"] + r["bridge"] + r["harbor"] >= 0.6 * r["n"]),
]


def pick_qualitative(root: Path) -> list[tuple[str, str]]:
    rows = []
    with open(root / "chip_index.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] != "val" or int(r["n_labels"]) == 0:
                continue
            d = {k: int(r[k]) for k in KEPT_CLASSES}
            d["n"] = int(r["n_labels"])
            d["chip_id"] = r["chip_id"]
            rows.append(d)
    rng = random.Random(SEED)
    chosen, used = [], set()
    for name, pred in STRATA:
        pool = [r for r in rows if pred(r) and r["chip_id"] not in used]
        if pool:
            r = rng.choice(pool)
            chosen.append((name, r["chip_id"]))
            used.add(r["chip_id"])
    return chosen


def render_qualitative(root: Path, pred: dict, conf: float, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_of = {c: i for i, c in enumerate(pred["chip_ids"])}
    written = []
    for name, chip_id in pick_qualitative(root):
        img = cv2.imread(str(root / "val" / "images" / f"{chip_id}.png"))
        left, right = img.copy(), img.copy()
        gt_lines = [ln.split() for ln in (root / "val" / "labels" / f"{chip_id}.txt").read_text().splitlines() if ln.strip()]
        gt_cls = np.array([int(p[0]) for p in gt_lines], dtype=int)
        gt_polys = np.array([[[float(p[i]) * 1024, float(p[i + 1]) * 1024] for i in range(1, 9, 2)] for p in gt_lines])
        for c, poly in zip(gt_cls, gt_polys):
            cv2.polylines(left, [poly.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
        m = (pred["det_chip_idx"] == idx_of[chip_id]) & (pred["conf"] >= conf)
        px, pc, pk = pred["xywhr"][m], pred["conf"][m], pred["cls"][m]
        ppoly = xywhr_to_polys(px) if len(pc) else np.zeros((0, 4, 2))
        for poly, k in zip(ppoly, pk):
            cv2.polylines(right, [poly.astype(np.int32).reshape(-1, 1, 2)], True, CLASS_BGR[KEPT_CLASSES[k]], 2, cv2.LINE_AA)
        # per-chip TP / FP / FN at IoU 0.5 vs the chip's own (rectangle) labels, difficult unknown at chip level -> all count
        stem = "c"
        g = ImageGT(stem, gt_cls, gt_polys, np.zeros(len(gt_cls), bool))
        d = ImageDets(stem, pk.astype(int), pc, ppoly, np.zeros(len(pc), int))
        ops = [operating_point(match_class({stem: g}, {stem: d}, c), conf) for c in range(len(KEPT_CLASSES))]
        tp, fp, fn = sum(o["TP"] for o in ops), sum(o["FP"] for o in ops), sum(o["FN"] for o in ops)
        cap_l = f"GROUND TRUTH ({len(gt_cls)} objects) - {name} - {chip_id}"
        cap_r = f"PREDICTIONS conf>={conf:.2f} ({len(pc)}) | IoU0.5: TP {tp} FP {fp} FN {fn}"
        for im, cap in ((left, cap_l), (right, cap_r)):
            cv2.rectangle(im, (0, 0), (1024, 26), (0, 0, 0), -1)
            cv2.putText(im, cap, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        path = out_dir / f"qualitative_{len(written) + 1}_{chip_id}.png"
        cv2.imwrite(str(path), np.hstack([left, right]))
        written.append({"stratum": name, "chip_id": chip_id, "path": str(path), "n_gt": int(len(gt_cls)),
                        "n_pred": int(len(pc)), "TP": tp, "FP": fp, "FN": fn})
        print(f"[eval] qualitative: {path.name}  (TP {tp} FP {fp} FN {fn})", flush=True)
    return written


# --------------------------------------------------------------------------
# integrity checks (Step 6: "confirm explicitly that val was never used ...")
# --------------------------------------------------------------------------


def integrity_checks(root: Path, run_dir: Path, weights: Path) -> dict:
    stems = {}
    for split in ("train", "monitor", "val"):
        stems[split] = {parse_chip_id(Path(p).stem)[0] for p in chip_paths(root, split)}
    yaml_txt = (root / "dataset.yaml").read_text(encoding="utf-8")
    args_txt = (run_dir / "args.yaml").read_text(encoding="utf-8") if (run_dir / "args.yaml").is_file() else ""
    val_cache = root / "val" / "labels.cache"
    return {
        "source_images_shared_train_val": len(stems["train"] & stems["val"]),
        "source_images_shared_monitor_val": len(stems["monitor"] & stems["val"]),
        "source_images_shared_train_monitor": len(stems["train"] & stems["monitor"]),
        "training_yaml_val_entry": next((ln for ln in yaml_txt.splitlines() if ln.startswith("val:")), None),
        "training_yaml_mentions_official_val": "val/val.txt" in yaml_txt.replace("monitor/monitor.txt", ""),
        "run_args_data_yaml": next((ln.strip() for ln in args_txt.splitlines() if ln.startswith("data:")), None),
        "run_args_early_stopping_patience": next((ln.strip() for ln in args_txt.splitlines() if ln.startswith("patience:")), None),
        "official_val_labels_cache_existed_before_this_evaluation": val_cache.is_file(),
        "checkpoint_selected_on": "monitor holdout (best.pt = Ultralytics fitness on the monitor split)",
        "operating_confidence_selected_on": "monitor holdout",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="geoseek_obb_v15_yolo26s")
    ap.add_argument("--weights", type=Path, default=None, help="default: <run>/weights/best.pt")
    ap.add_argument("--stages", default="monitor,val,baseline,chip,checkpoints,qual,curves")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--checkpoint-epochs", default="1,3,7,11,15,19")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    dota_root = settings.datasets_dir / "dota"
    run_dir = settings.data_dir / "runs" / "detector" / args.run
    weights = args.weights or run_dir / "weights" / "best.pt"
    init = settings.models_dir / "yolo_obb" / "geoseek_init_yolo26s_8cls.pt"
    out_dir = args.out or settings.data_dir / "detect_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}

    def save():
        results_path.write_text(json.dumps(sanitize(results), indent=1), encoding="utf-8")

    results["meta"] = {"weights": str(weights), "baseline_weights": str(init), "conf_floor_full_image": CONF_FLOOR_FULL,
                       "nms_iou": NMS_IOU, "cross_chip_merge_iou": MERGE_IOU, "max_det_per_chip": MAX_DET,
                       "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    results["integrity"] = integrity_checks(root, run_dir, weights)
    save()

    # ---- MONITOR: choose the operating confidence (never on val) ----------------------------------------------------
    if "monitor" in stages:
        pm = get_pred("finetuned", weights, "monitor", root, out_dir, force=args.force)
        res, mm = score_full(pm, dota_root, "train", "v1.5", size_buckets=False)
        results["monitor_v15_full_image"] = {"per_class": res["per_class"], "groups": res["groups"]}
        results["operating_point"] = choose_operating_conf(mm)
        pb = get_pred("pretrained_restricted", init, "monitor", root, out_dir, force=args.force)
        rb, _ = score_full(pb, dota_root, "train", "v1.5", size_buckets=False)
        results["monitor_v15_full_image_pretrained_restricted"] = {"per_class": rb["per_class"], "groups": rb["groups"]}
        print(f"[eval] operating confidence chosen on MONITOR: {results['operating_point']['conf']} "
              f"(macro-F1 {results['operating_point']['macro_f1_on_monitor']:.3f})", flush=True)
        save()
    conf = results["operating_point"]["conf"]

    # ---- OFFICIAL VAL, fine-tuned -----------------------------------------------------------------------------------
    pv = None
    if "val" in stages:
        pv = get_pred("finetuned", weights, "val", root, out_dir, force=args.force)
        r15, m15 = score_full(pv, dota_root, "val", "v1.5", size_buckets=True)
        r10, m10 = score_full(pv, dota_root, "val", "v1.0", size_buckets=False)
        results["val_full_image_v15"] = {"per_class": r15["per_class"], "groups": r15["groups"],
                                         "counts": {k: r15[k] for k in ("n_images", "n_gt_total_incl_difficult", "n_gt_difficult", "n_detections_after_merge")}}
        results["val_full_image_v10"] = {"per_class": r10["per_class"], "groups": r10["groups"],
                                         "counts": {k: r10[k] for k in ("n_images", "n_gt_total_incl_difficult", "n_gt_difficult")}}
        results["val_operating_point_v15"] = operating_metrics(m15, conf)
        results["val_operating_point_v10"] = operating_metrics(m10, conf)
        save()
        _MATCHES["ft15"] = m15

    # ---- OFFICIAL VAL, pretrained-restricted baseline ---------------------------------------------------------------
    if "baseline" in stages:
        pbv = get_pred("pretrained_restricted", init, "val", root, out_dir, force=args.force)
        b15, mb15 = score_full(pbv, dota_root, "val", "v1.5", size_buckets=True)
        b10, mb10 = score_full(pbv, dota_root, "val", "v1.0", size_buckets=False)
        results["val_full_image_v15_pretrained_restricted"] = {"per_class": b15["per_class"], "groups": b15["groups"]}
        results["val_full_image_v10_pretrained_restricted"] = {"per_class": b10["per_class"], "groups": b10["groups"]}
        results["val_operating_point_v15_pretrained_restricted"] = operating_metrics(mb15, conf)
        _MATCHES["base15"] = mb15
        save()

    # ---- chip-level (Ultralytics) protocol --------------------------------------------------------------------------
    if "chip" in stages:
        print("[eval] Ultralytics chip-level validation (fine-tuned) ...", flush=True)
        results["val_chip_level_ultralytics"] = {"finetuned": chip_level_val(weights, root / "dataset_official_val.yaml")}
        print("[eval] Ultralytics chip-level validation (pretrained-restricted) ...", flush=True)
        results["val_chip_level_ultralytics"]["pretrained_restricted"] = chip_level_val(init, root / "dataset_official_val.yaml")
        results["monitor_chip_level_ultralytics"] = {
            "finetuned": chip_level_val(weights, root / "dataset_monitor.yaml", "val"),
            "pretrained_restricted": chip_level_val(init, root / "dataset_monitor.yaml", "val")}
        save()

    # ---- post-hoc official-val curve over saved checkpoints (reporting only) ------------------------------------------
    if "checkpoints" in stages:
        pts = [{"epoch": 0, **results["val_chip_level_ultralytics"]["pretrained_restricted"]}] if "val_chip_level_ultralytics" in results else []
        for e in [int(x) for x in args.checkpoint_epochs.split(",")]:
            ck = run_dir / "weights" / f"epoch{e - 1}.pt"
            if ck.is_file():
                print(f"[eval] official-val chip-level, checkpoint after epoch {e} ...", flush=True)
                v = chip_level_val(ck, root / "dataset_official_val.yaml")
                pts.append({"epoch": e, "mAP50": v["mAP50"], "mAP50_95": v["mAP50_95"]})
        if "val_chip_level_ultralytics" in results:
            f = results["val_chip_level_ultralytics"]["finetuned"]
            pts.append({"epoch": int(read_results_csv(run_dir)["epoch"].max()), "mAP50": f["mAP50"], "mAP50_95": f["mAP50_95"]})
        results["official_val_by_checkpoint"] = [{k: p[k] for k in ("epoch", "mAP50", "mAP50_95")} for p in pts]
        save()

    # ---- qualitative -----------------------------------------------------------------------------------------------
    if "qual" in stages:
        pv = pv or get_pred("finetuned", weights, "val", root, out_dir)
        results["qualitative"] = render_qualitative(root, pv, conf, out_dir / "qualitative")
        save()

    # ---- figures ---------------------------------------------------------------------------------------------------
    if "curves" in stages:
        figs = out_dir / "figures"
        figs.mkdir(exist_ok=True)
        base = None
        if "monitor_chip_level_ultralytics" in results:
            base = {"monitor": results["monitor_chip_level_ultralytics"]["pretrained_restricted"]}
        written = [str(p) for p in plot_training_curves(run_dir, figs, baseline=base,
                                                        official_points=results.get("official_val_by_checkpoint", []))]
        if "val_full_image_v15" in results and "val_full_image_v15_pretrained_restricted" in results:
            written.append(str(plot_per_class(results["val_full_image_v15"], results["val_full_image_v15_pretrained_restricted"], figs, "v1.5")))
            written.append(str(plot_per_class(results["val_full_image_v10"], results["val_full_image_v10_pretrained_restricted"], figs, "v1.0")))
        if "ft15" in _MATCHES and "base15" in _MATCHES:
            written.append(str(plot_pr(_MATCHES["ft15"], _MATCHES["base15"], figs)))
        results["figures"] = written
        for w in written:
            print(f"[eval] figure: {w}", flush=True)
        save()

    save()
    print(f"[eval] wrote {results_path}", flush=True)


_MATCHES: dict = {}

if __name__ == "__main__":
    main()

"""Evaluate the trained Phase 8F-2 detector (geoseek_obb_v15_yolo26s) against xView as an
independent, never-trained-on TEST set for the small-vehicle / large-vehicle classes.

Two IoU protocols on the SAME predictions, reported side by side (see
geoseek.detect.evaluate.obb_to_enclosing_hbb docstring for why the second one is the headline
number): xView ships axis-aligned ground truth only, and true polygon IoU between a TIGHT oriented
prediction and a LOOSE axis-aligned box is geometrically exact but not a fair comparison - a
perfect 45-degree detection can score well under the 0.5 threshold purely from the shape mismatch.

  raw_obb   : the detector's actual oriented output, scored as-is against xView's HBB ground truth
              (biased LOW for any genuinely rotated vehicle - reported for transparency, not headline)
  hbb_fair  : each prediction replaced by its own enclosing axis-aligned box before scoring against
              the same HBB ground truth (both sides "loose" the same way) - THE HEADLINE NUMBER

    python scripts/eval_xview.py                  # chip + infer + score (GPU, ~20-40 min first run)
    python scripts/eval_xview.py --stages score    # re-score cached predictions only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("GDAL_DATA", r"C:\AnacondaPython\anaconda3\envs\geoseek\Library\share\gdal")
os.environ.setdefault("PROJ_LIB", r"C:\AnacondaPython\anaconda3\envs\geoseek\Library\share\proj")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.chipping import CHIP_SIZE, chip_origins, crop_chip  # noqa: E402
from geoseek.detect.classes import CLASS_TO_INDEX  # noqa: E402
from geoseek.detect.eval_io import assemble_full_image_dets, parse_chip_id  # noqa: E402
from geoseek.detect.evaluate import (  # noqa: E402
    ImageDets, best_f1_threshold, dets_to_enclosing_hbb, evaluate_dataset, merge_cross_chip, operating_point)
from geoseek.detect.infer import load_predictions, predict_chips, save_predictions  # noqa: E402
from geoseek.staging.download_xview import images_dir, labels_zip_path, load_xview_gt  # noqa: E402
from geoseek.staging.manifest import record_analysis_section  # noqa: E402

XVIEW_CLASSES = ("small-vehicle", "large-vehicle")
MERGE_IOU = 0.3          # cross-chip de-duplication, same convention as scripts/eval_detector.py
NMS_IOU = 0.7
CONF_FLOOR = 0.005
MAX_DET = 1500


def sanitize(o):
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


def build_chips(chips_dir: Path, force: bool = False) -> list[Path]:
    """Chip every xView image at CHIP_SIZE/STRIDE (same convention as training), writing BGR PNGs
    so ultralytics' own cv2.imread round-trips identically to how the DOTA training chips were
    made. Idempotent: skips an image whose full expected chip set is already on disk."""
    chips_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(images_dir().glob("*.tif"))
    print(f"[eval_xview] {len(paths)} source images", flush=True)
    out: list[Path] = []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        stem = p.stem
        with rasterio.open(p) as ds:
            w, h = ds.width, ds.height
            origins = chip_origins(w, h, CHIP_SIZE)
            expected = [chips_dir / f"{stem}__{x0}_{y0}.png" for x0, y0 in origins]
            if not force and all(e.is_file() for e in expected):
                out.extend(expected)
                continue
            rgb = ds.read().transpose(1, 2, 0)  # (band,H,W) R,G,B -> (H,W,3)
            bgr = rgb[:, :, ::-1]
            for (x0, y0), out_path in zip(origins, expected):
                chip = crop_chip(bgr, x0, y0, CHIP_SIZE)
                cv2.imwrite(str(out_path), chip)
                out.append(out_path)
        if i % 100 == 0:
            print(f"[eval_xview]   chipped {i}/{len(paths)} images, {len(out)} chips so far, {time.time() - t0:.0f}s", flush=True)
    print(f"[eval_xview] {len(out)} chips total, {time.time() - t0:.0f}s", flush=True)
    return out


def get_pred(weights: Path, chip_paths: list[Path], out_dir: Path, *, force=False) -> dict:
    f = out_dir / "pred_xview.npz"
    if f.is_file() and not force:
        print(f"[eval_xview] cached predictions: {f.name}", flush=True)
        return load_predictions(f)
    print(f"[eval_xview] predicting on {len(chip_paths)} chips ...", flush=True)
    pred = predict_chips(weights, chip_paths, conf=CONF_FLOOR, iou=NMS_IOU, max_det=MAX_DET, imgsz=1024)
    save_predictions(f, pred)
    print(f"[eval_xview]   {len(pred['conf'])} raw detections in {pred['seconds']:.0f}s", flush=True)
    return pred


def full_image_dets(pred: dict) -> dict[str, ImageDets]:
    per = assemble_full_image_dets(pred["chip_ids"], pred["det_chip_idx"], pred["xywhr"], pred["conf"], pred["cls"])
    return {stem: merge_cross_chip(d, MERGE_IOU) for stem, d in per.items()}


def score_one(gt: dict, dets: dict, *, size_buckets: bool) -> tuple[dict, dict]:
    res = evaluate_dataset(gt, dets, XVIEW_CLASSES, size_buckets=size_buckets)
    matches = res.pop("_matches")
    return res, matches


def operating_metrics(matches: dict) -> dict:
    out = {}
    for name, m in matches.items():
        thr, f1 = best_f1_threshold(m)
        out[name] = {"best_f1_threshold": thr, **operating_point(m, thr)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="chip,infer,score")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    settings = get_settings()
    weights = args.weights or settings.models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt"
    out_dir = args.out or settings.data_dir / "detect_eval" / "xview"
    chips_dir = out_dir / "chips"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}

    def save():
        results_path.write_text(json.dumps(sanitize(results), indent=1), encoding="utf-8")

    results["meta"] = {"weights": str(weights), "conf_floor": CONF_FLOOR, "nms_iou": NMS_IOU,
                       "cross_chip_merge_iou": MERGE_IOU, "max_det_per_chip": MAX_DET,
                       "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save()

    chip_paths: list[Path] = []
    if "chip" in stages:
        chip_paths = build_chips(chips_dir, force=args.force)
        results["n_chips"] = len(chip_paths)
        save()
    if not chip_paths:
        chip_paths = sorted(chips_dir.glob("*.png"))

    pred = None
    if "infer" in stages:
        pred = get_pred(weights, chip_paths, out_dir, force=args.force)

    if "score" in stages:
        pred = pred or load_predictions(out_dir / "pred_xview.npz")
        stems = {parse_chip_id(c)[0] for c in pred["chip_ids"]}
        print(f"[eval_xview] loading xView ground truth for {len(stems)} images ...", flush=True)
        gt, gt_stats = load_xview_gt(labels_zip_path(), stems)
        results["gt_stats"] = gt_stats
        results["n_images"] = len(gt)
        results["n_gt_total"] = int(sum(len(g.classes) for g in gt.values()))

        dets_raw = full_image_dets(pred)
        dets_fair = {stem: dets_to_enclosing_hbb(d) for stem, d in dets_raw.items()}

        print("[eval_xview] scoring raw_obb (biased low - reported for transparency) ...", flush=True)
        raw_res, raw_matches = score_one(gt, dets_raw, size_buckets=True)
        print("[eval_xview] scoring hbb_fair (headline) ...", flush=True)
        fair_res, fair_matches = score_one(gt, dets_fair, size_buckets=True)

        results["raw_obb"] = {"per_class": raw_res["per_class"], "operating_point": operating_metrics(raw_matches)}
        results["hbb_fair"] = {"per_class": fair_res["per_class"], "operating_point": operating_metrics(fair_matches)}
        results["n_detections_after_merge"] = int(sum(len(d.scores) for d in dets_raw.values()))
        save()

        for tag, res in (("raw_obb", results["raw_obb"]), ("hbb_fair", results["hbb_fair"])):
            print(f"\n[eval_xview] === {tag} ===", flush=True)
            for cls in XVIEW_CLASSES:
                pc = res["per_class"][cls]
                op = res["operating_point"][cls]
                print(f"[eval_xview]   {cls}: AP50={pc['AP50']:.4f} AP50:95={pc['AP50_95']:.4f} "
                      f"n_gt={pc['n_gt']}  P={op['precision']:.3f} R={op['recall']:.3f} F1={op['f1']:.3f} "
                      f"@thr={op['score_thr']:.3f}", flush=True)

        record_analysis_section("xview_detector_eval", {
            "purpose": "TEST-ONLY: the DOTA-trained detector scored against xView's independent GT (never trained on).",
            "protocols": {
                "raw_obb": "detector's true oriented output vs xView HBB GT (geometrically exact IoU, biased low for rotated objects)",
                "hbb_fair": "prediction's own enclosing axis-aligned box vs xView HBB GT (headline - removes the tight-vs-loose bias)",
            },
            "meta": results["meta"], "n_images": results["n_images"], "n_gt_total": results["n_gt_total"],
            "gt_stats": gt_stats, "n_chips": results.get("n_chips"),
            "n_detections_after_merge": results["n_detections_after_merge"],
            "raw_obb": results["raw_obb"], "hbb_fair": results["hbb_fair"],
        })
        save()

    print(f"\n[eval_xview] wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()

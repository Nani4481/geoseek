"""Phase 8F-3a Step 2: score the trained detector against the hand-labelled Maxar test set.

This is the number to beat - the first real (non-anecdotal) recall/precision/AP figure for the
detector on Maxar imagery, using the held-out set built by scripts/build_label_set + labelled via
scripts/serve_label_tool.py. NOT run until the labels exist (see docs/PHASE8F3A.md); running it
against an empty labelTxt/ directory produces n_gt=0 / NaN APs, which is expected and meaningless.

    python scripts/eval_maxar_handlabeled.py [--weights ...] [--min-score 0.001] [--set data/detect_eval/maxar_handlabeled_v1]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_NAMES, CLASS_TO_INDEX, CLASS_GROUPS  # noqa: E402
from geoseek.detect.eval_io import load_dota_gt  # noqa: E402
from geoseek.detect.evaluate import ImageDets, class_ap, match_class, operating_point, best_f1_threshold  # noqa: E402
from geoseek.models import YoloObbDetectionModel  # noqa: E402

VEHICLE_CLASSES = CLASS_GROUPS["ground_vehicles"]  # ("small-vehicle", "large-vehicle") - the only classes this set labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--set", default="data/detect_eval/maxar_handlabeled_v1")
    ap.add_argument("--min-score", type=float, default=0.001, help="low floor for a full PR sweep, not the deployment operating point")
    args = ap.parse_args()

    settings = get_settings()
    dataset_dir = Path(args.set).resolve()
    images_dir, labels_dir = dataset_dir / "images", dataset_dir / "labelTxt"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    gt = load_dota_gt(labels_dir)
    n_labelled = sum(1 for g in gt.values() if len(g.classes))
    print(f"[eval] loaded GT for {len(gt)} images, {n_labelled} with >=1 vehicle instance "
          f"({sum(len(g.classes) for g in gt.values())} instances total)")
    if n_labelled == 0:
        print("[eval] WARNING: no labelled instances found - labelling is not done yet. Numbers below are meaningless.")

    weights = args.weights or settings.models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt"
    model = YoloObbDetectionModel(weights, min_score=args.min_score)
    model.load()
    print(f"[eval] model: {json.dumps(model.info)}")

    dets: dict[str, ImageDets] = {}
    for img_meta in manifest["images"]:
        stem = img_meta["stem"]
        bgr = cv2.imread(str(images_dir / f"{stem}.png"))
        if bgr is None:
            raise FileNotFoundError(images_dir / f"{stem}.png")
        rgb = bgr[..., ::-1]
        detections = model.detect(rgb, min_score=args.min_score)
        vehicle_dets = [d for d in detections if d.class_name in VEHICLE_CLASSES]
        if vehicle_dets:
            dets[stem] = ImageDets(
                stem, np.array([CLASS_TO_INDEX[d.class_name] for d in vehicle_dets], dtype=int),
                np.array([d.score for d in vehicle_dets], dtype=float),
                np.array([d.polygon_px for d in vehicle_dets], dtype=float).reshape(-1, 4, 2))
        else:
            dets[stem] = ImageDets(stem, np.zeros(0, int), np.zeros(0), np.zeros((0, 4, 2)))

    report = {"model": model.info, "min_score_floor": args.min_score, "n_images": len(manifest["images"]),
              "n_gt_instances": {}, "per_class": {}}
    macro_ap50, macro_ap5095 = [], []
    for cls_name in VEHICLE_CLASSES:
        c = CLASS_TO_INDEX[cls_name]
        m = match_class(gt, dets, c)
        result = class_ap(m)
        thr, f1 = best_f1_threshold(m)
        op_default = operating_point(m, model.min_score)
        op_best_f1 = operating_point(m, thr)
        report["n_gt_instances"][cls_name] = result["n_gt"]
        report["per_class"][cls_name] = {
            "AP50": result["AP50"], "AP50_voc07": result["AP50_voc07"], "AP50_95": result["AP50_95"],
            "operating_point_default_conf": op_default, "best_f1_threshold": thr, "best_f1": f1,
            "operating_point_best_f1": op_best_f1,
        }
        if result["n_gt"] > 0:
            macro_ap50.append(result["AP50"]); macro_ap5095.append(result["AP50_95"])
        print(f"[eval] {cls_name}: n_gt={result['n_gt']} AP50={result['AP50']:.3f} AP50:95={result['AP50_95']:.3f} "
              f"P@default={op_default['precision']:.3f} R@default={op_default['recall']:.3f} "
              f"P@bestF1={op_best_f1['precision']:.3f} R@bestF1={op_best_f1['recall']:.3f} (thr={thr:.3f})")

    report["macro_AP50"] = float(np.mean(macro_ap50)) if macro_ap50 else float("nan")
    report["macro_AP50_95"] = float(np.mean(macro_ap5095)) if macro_ap5095 else float("nan")
    print(f"[eval] macro (ground_vehicles): AP50={report['macro_AP50']:.3f} AP50:95={report['macro_AP50_95']:.3f}")

    out_path = dataset_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()

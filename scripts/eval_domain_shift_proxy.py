"""Phase 8F-2 Step 7 companion: a CONTROLLED proxy for one component of the DOTA -> Maxar domain gap (effective resolution).

There is no ground truth on the staged Maxar tiles, so no accuracy can be measured there. What the tiles do show (a look at
the Van Nuys sample) is soft, pansharpened imagery: the ARD grid is 0.305 m but the sensors are WV02 (~0.58 m native) and GE01
(~0.49 m native), i.e. the tiles are UPSAMPLED by ~1.6-1.9x. DOTA chips are much crisper at the same pixel size.

This script isolates that one factor with ground truth: take DOTA **monitor** chips (NOT the official val: the result informs an
inference-time choice, and val must never inform choices), degrade them the way the Maxar product is degraded - downsample by
``k`` (area) and upsample back (cubic), so pixel size and labels are unchanged but effective resolution is k times coarser -
and score the detector on them with the same full-image protocol. It then re-runs the degraded chips with test-time upscaling
(``imgsz`` 1536) to see whether that recovers the loss. What it can NOT model: sensor radiometry, off-nadir geometry, colour
processing, scene mix - those remain unmeasured.

    python scripts/eval_domain_shift_proxy.py [--k 1.8]
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_GROUPS, KEPT_CLASSES, LABEL_SETS  # noqa: E402
from geoseek.detect.eval_io import assemble_full_image_dets, load_dota_gt, parse_chip_id  # noqa: E402
from geoseek.detect.evaluate import evaluate_dataset, group_summary, merge_cross_chip, operating_point  # noqa: E402
from geoseek.detect.infer import predict_chips  # noqa: E402

CONF_FLOOR = 0.005
MERGE_IOU = 0.3


def degrade(src: Path, dst: Path, k: float) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def score(pred: dict, gt: dict, conf: float) -> dict:
    per = assemble_full_image_dets(pred["chip_ids"], pred["det_chip_idx"], pred["xywhr"], pred["conf"], pred["cls"])
    dets = {s: merge_cross_chip(d, MERGE_IOU) for s, d in per.items()}
    res = evaluate_dataset(gt, dets, KEPT_CLASSES)
    matches = res.pop("_matches")
    g = group_summary(res["per_class"], CLASS_GROUPS)
    op = {n: operating_point(matches[n], conf) for n in KEPT_CLASSES}

    def grp(names):
        tp, fp, fn = (sum(op[n][k] for n in names) for k in ("TP", "FP", "FN"))
        return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "TP": tp, "FP": fp, "FN": fn}

    return {"per_class_AP50": {n: res["per_class"][n]["AP50"] for n in KEPT_CLASSES},
            "per_class_AP50_95": {n: res["per_class"][n]["AP50_95"] for n in KEPT_CLASSES},
            "macro_AP50": g["all_kept_classes"]["macro_AP50"], "macro_AP50_95": g["all_kept_classes"]["macro_AP50_95"],
            "ground_vehicles_macro_AP50": g["ground_vehicles"]["macro_AP50"],
            "operating_point": {"ground_vehicles": grp(CLASS_GROUPS["ground_vehicles"]), "all": grp(KEPT_CLASSES)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=1.8)
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()

    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    dota = settings.datasets_dir / "dota"
    weights = args.weights or settings.data_dir / "runs" / "detector" / "geoseek_obb_v15_yolo26s" / "weights" / "best.pt"
    out_dir = settings.data_dir / "detect_eval"
    ev = json.loads((out_dir / "eval_results.json").read_text(encoding="utf-8"))
    conf = ev["operating_point"]["conf"]                      # chosen on the monitor split

    chips = [root / "monitor" / ln[2:] for ln in (root / "monitor" / "monitor.txt").read_text().splitlines() if ln]
    deg_dir = out_dir / f"monitor_degraded_k{args.k}"
    deg_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deg = []
    for c in chips:
        d = deg_dir / c.name
        if not d.is_file():
            degrade(c, d, args.k)
        deg.append(d)
    print(f"[proxy] {len(deg)} degraded chips (k={args.k}) ready in {time.time() - t0:.0f}s", flush=True)

    stems = {parse_chip_id(c.stem)[0] for c in chips}
    gt = load_dota_gt(dota / "train" / LABEL_SETS["v1.5"], stems)
    out = {"k": args.k, "conf": conf, "weights": str(weights), "split": "monitor (train-domain images; NOT the official val)", "runs": {}}

    runs = (("original_imgsz1024", chips, 1024), (f"degraded_k{args.k}_imgsz1024", deg, 1024),
            (f"degraded_k{args.k}_imgsz1536", deg, 1536), ("original_imgsz1536", chips, 1536))
    for name, paths, imgsz in runs:
        pred = predict_chips(weights, paths, conf=CONF_FLOOR, iou=0.7, max_det=2500, imgsz=imgsz, batch=4 if imgsz > 1024 else 8,
                             log_every=0)
        out["runs"][name] = score(pred, gt, conf)
        r = out["runs"][name]
        print(f"[proxy] {name:28s} macro AP50 {r['macro_AP50']:.3f}  AP50-95 {r['macro_AP50_95']:.3f} | vehicles AP50 "
              f"{r['ground_vehicles_macro_AP50']:.3f}  P {r['operating_point']['ground_vehicles']['precision']:.3f} "
              f"R {r['operating_point']['ground_vehicles']['recall']:.3f} | SV AP50 {r['per_class_AP50']['small-vehicle']:.3f}", flush=True)

    (out_dir / "domain_shift_proxy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[proxy] wrote {out_dir / 'domain_shift_proxy.json'}")


if __name__ == "__main__":
    main()

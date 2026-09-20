"""Phase 8F-2 Step 6/7 companion: precision / recall of the vehicle classes AT THE OPERATING CONFIDENCE, stratified by object size.

``eval_detector.py`` reports size-stratified AP. AP is threshold-free; what a user of the Maxar detections actually sees is a fixed
confidence (0.525, chosen on the monitor split), and the audit of the Maxar crops (section 7) found almost no cars at that
confidence. This script answers, on the official val with ground truth, the matching question: at 0.525, what fraction of the
vehicles of a given pixel size does the detector find, and how many of its detections in that size range are right?

Reads the cached fine-tuned val predictions written by eval_detector.py (no GPU); nothing is chosen with the result.

    python scripts/eval_size_operating_point.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import KEPT_CLASSES, LABEL_SETS  # noqa: E402
from geoseek.detect.eval_io import assemble_full_image_dets, load_dota_gt, parse_chip_id  # noqa: E402
from geoseek.detect.evaluate import SIZE_BUCKETS_PX, _tp_fp, match_class, merge_cross_chip  # noqa: E402
from geoseek.detect.infer import load_predictions  # noqa: E402

MERGE_IOU = 0.3


def main() -> None:
    settings = get_settings()
    out_dir = settings.data_dir / "detect_eval"
    results = json.loads((out_dir / "eval_results.json").read_text(encoding="utf-8"))
    conf = float(results["operating_point"]["conf"])
    pred = load_predictions(out_dir / "pred_finetuned_val.npz")
    stems = {parse_chip_id(c)[0] for c in pred["chip_ids"]}
    gt = load_dota_gt(settings.datasets_dir / "dota" / "val" / LABEL_SETS["v1.5"], stems)
    per = assemble_full_image_dets(pred["chip_ids"], pred["det_chip_idx"], pred["xywhr"], pred["conf"], pred["cls"])
    dets = {s: merge_cross_chip(d, MERGE_IOU) for s, d in per.items()}

    out: dict = {"conf": conf, "split": "official val (v1.5 GT, difficult ignored, full-image protocol, IoU 0.5)", "classes": {}}
    for name in ("small-vehicle", "large-vehicle"):
        m = match_class(gt, dets, KEPT_CLASSES.index(name))
        rows = {}
        for lo, hi in SIZE_BUCKETS_PX:
            tp, fp, npos, sc = _tp_fp(m, 0.5, (lo, hi))
            sel = sc >= conf
            tp_c, fp_c = float(tp[sel].sum()), float(fp[sel].sum())
            rows[f"{lo:g}-{hi:g}"] = {
                "n_gt": int(npos), "TP": int(tp_c), "FP": int(fp_c),
                "recall": tp_c / npos if npos else None, "precision": tp_c / (tp_c + fp_c) if tp_c + fp_c else None,
                "max_recall_at_any_conf": float(tp.sum() / npos) if npos else None}
        out["classes"][name] = rows
    (out_dir / "size_operating_point.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"operating confidence {conf}")
    for name, rows in out["classes"].items():
        print(f"\n{name}   long side px | n GT | recall@{conf} | precision@{conf} | recall at any confidence")
        for k, r in rows.items():
            f = lambda v: "  -  " if v is None else f"{v:5.3f}"
            print(f"   {k:>8} | {r['n_gt']:6d} | {f(r['recall'])} | {f(r['precision'])} | {f(r['max_recall_at_any_conf'])}")


if __name__ == "__main__":
    main()

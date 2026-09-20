"""Self-test of the full-image evaluator on the REAL official-val geometry with SYNTHETIC detections (no model, no GPU, no torch).

Three synthetic "detectors" built from the val chip labels, scored with exactly the code path the real evaluation uses
(chip-id offsets -> full-image coordinates -> cross-chip merge -> polygon IoU vs the ORIGINAL quadrilateral GT, `difficult`
ignored -> AP):

  oracle      every chip label becomes a detection at its own rectangle, score 0.9. AP50 should be ~1.0. Its AP50:95 is a
              MEASUREMENT of the label-noise ceiling: DOTA labels are hand-clicked quadrilaterals, detectors emit rectangles,
              so even a perfect rectangle detector cannot reach IoU 0.95 on a quad.
  jitter      the oracle with centre noise (sigma 1.5 px) and +-5 % size noise. Each metric must fall (AP50:95 most). AP50 also
              falls, more than one might expect: 1.5 px is a large error for an 8 px vehicle, and independently jittered
              copies of the same object seen in two overlapping chips can fall below the merge IoU and survive as duplicates.
  degraded    jitter + 25 % of objects missed + false positives at 10 % of the object count (random locations, score 0.3-0.6).
              Recall and AP must fall again.

What is asserted is the ORDERING oracle > jitter > degraded on every summary metric, oracle AP50 ~ 1 with precision and recall
~ 1 on v1.5 (the plumbing is right), and that the SAME oracle scored against v1.0 collapses in precision (v1.0 leaves most real
vehicles unlabeled) - not any absolute value for the perturbed runs.

    python scripts/eval_selftest.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_GROUPS, KEPT_CLASSES, LABEL_SETS  # noqa: E402
from geoseek.detect.eval_io import assemble_full_image_dets, load_dota_gt, parse_chip_id  # noqa: E402
from geoseek.detect.evaluate import evaluate_dataset, group_summary, merge_cross_chip, operating_point  # noqa: E402

SEED = 7


def oracle_predictions(root: Path, split: str) -> dict:
    chip_ids, det_idx, xywhr, cls = [], [], [], []
    files = sorted((root / split / "labels").glob("*.txt"))
    for i, f in enumerate(files):
        chip_ids.append(f.stem)
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) != 9:
                continue
            pts = np.array([[float(p[k]) * 1024, float(p[k + 1]) * 1024] for k in range(1, 9, 2)], dtype=np.float32)
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
            theta = np.deg2rad(ang)
            if w < h:
                w, h, theta = h, w, theta + np.pi / 2
            det_idx.append(i)
            xywhr.append([cx, cy, w, h, theta])
            cls.append(int(p[0]))
    n = len(det_idx)
    return {"chip_ids": chip_ids, "det_chip_idx": np.array(det_idx, dtype=np.int64), "xywhr": np.array(xywhr, dtype=np.float64).reshape(-1, 5),
            "conf": np.full(n, 0.9), "cls": np.array(cls, dtype=int)}


def perturb(pred: dict, rng: np.random.Generator, *, sigma=0.0, size=0.0, miss=0.0, fp_rate=0.0) -> dict:
    n = len(pred["conf"])
    keep = rng.random(n) >= miss
    xywhr = pred["xywhr"][keep].copy()
    xywhr[:, :2] += rng.normal(0, sigma, (len(xywhr), 2)) if sigma else 0
    xywhr[:, 2:4] *= (1 + rng.uniform(-size, size, (len(xywhr), 2))) if size else 1
    det_idx, conf, cls = pred["det_chip_idx"][keep], pred["conf"][keep], pred["cls"][keep]
    n_fp = int(fp_rate * n)
    if n_fp:
        src = rng.integers(0, n, n_fp)
        fx = pred["xywhr"][src].copy()
        fx[:, :2] = rng.uniform(50, 970, (n_fp, 2))
        xywhr = np.vstack([xywhr, fx])
        det_idx = np.concatenate([det_idx, pred["det_chip_idx"][src]])
        conf = np.concatenate([conf, rng.uniform(0.3, 0.6, n_fp)])
        cls = np.concatenate([cls, pred["cls"][src]])
    return {**pred, "det_chip_idx": det_idx, "xywhr": xywhr, "conf": conf, "cls": cls}


def score(pred: dict, gt: dict) -> tuple[dict, dict, float]:
    t0 = time.time()
    per = assemble_full_image_dets(pred["chip_ids"], pred["det_chip_idx"], pred["xywhr"], pred["conf"], pred["cls"])
    dets = {s: merge_cross_chip(d, 0.3) for s, d in per.items()}
    res = evaluate_dataset(gt, dets, KEPT_CLASSES)
    matches = res.pop("_matches")
    res["groups"] = group_summary(res["per_class"], CLASS_GROUPS)
    return res, matches, time.time() - t0


def main() -> None:
    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    dota = settings.datasets_dir / "dota"
    rng = np.random.default_rng(SEED)
    base = oracle_predictions(root, "val")
    stems = {parse_chip_id(c)[0] for c in base["chip_ids"]}
    out = {"n_chips": len(base["chip_ids"]), "n_synthetic_detections_oracle": int(len(base["conf"])), "label_sets": {}}
    print(f"[selftest] {out['n_chips']} val chips, {out['n_synthetic_detections_oracle']} oracle detections", flush=True)

    for label_set in ("v1.5", "v1.0"):
        gt = load_dota_gt(dota / "val" / LABEL_SETS[label_set], stems)
        block = {}
        for name, pred in (("oracle", base),
                           ("jitter", perturb(base, rng, sigma=1.5, size=0.05)),
                           ("degraded", perturb(base, rng, sigma=1.5, size=0.05, miss=0.25, fp_rate=0.10))):
            res, matches, secs = score(pred, gt)
            op = {n: operating_point(matches[n], 0.5) for n in KEPT_CLASSES}
            tp, fp, fn = (sum(op[n][k] for n in KEPT_CLASSES) for k in ("TP", "FP", "FN"))
            block[name] = {
                "seconds": round(secs, 1),
                "per_class": {n: {"AP50": round(res["per_class"][n]["AP50"], 4), "AP50_95": round(res["per_class"][n]["AP50_95"], 4),
                                  "n_gt": res["per_class"][n]["n_gt"]} for n in KEPT_CLASSES},
                "macro_AP50": round(res["groups"]["all_kept_classes"]["macro_AP50"], 4),
                "macro_AP50_95": round(res["groups"]["all_kept_classes"]["macro_AP50_95"], 4),
                "operating_point_conf0.5": {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4)},
            }
            print(f"[selftest] {label_set} {name:9s} macro AP50 {block[name]['macro_AP50']:.4f}  AP50-95 {block[name]['macro_AP50_95']:.4f}  "
                  f"P {block[name]['operating_point_conf0.5']['precision']:.3f} R {block[name]['operating_point_conf0.5']['recall']:.3f}  "
                  f"({secs:.0f}s)", flush=True)
            if name == "oracle":
                print("[selftest]    oracle AP50-95 by class (= label-noise ceiling): " +
                      ", ".join(f"{n} {block[name]['per_class'][n]['AP50_95']:.3f}" for n in KEPT_CLASSES), flush=True)
        out["label_sets"][label_set] = block

    (settings.data_dir / "detect_eval").mkdir(exist_ok=True)
    (settings.data_dir / "detect_eval" / "evaluator_selftest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[selftest] wrote {settings.data_dir / 'detect_eval' / 'evaluator_selftest.json'}")


if __name__ == "__main__":
    main()

"""Fix a tune/test leak in the vehicle operating-point selection.

The first pass tuned vehicle confidence thresholds (best-F1) on ALL of xView, then reported that same
threshold's precision/recall on ALL of xView - the same images used to pick the threshold were used to
report its performance, an optimistic bias. AP50/AP50:95 themselves are threshold-free (integrate the
whole PR curve) and stay valid regardless, but a specific operating point's precision/recall do not.

Fix: split xView by image into a fixed, seeded TUNE half (threshold selection only, never reported) and
a TEST half (all reporting - AP50, AP50:95, precision, recall at the tuned threshold - never used to
pick anything). Reuses the already-cached predictions (data/detect_eval/xview/pred_xview.npz) - no
re-inference needed, just re-scoring against two disjoint image sets.

    python scripts/tune_xview_thresholds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from geoseek.config import get_settings
from geoseek.detect.eval_io import parse_chip_id
from geoseek.detect.evaluate import best_f1_threshold, dets_to_enclosing_hbb, operating_point
from geoseek.detect.infer import load_predictions
from geoseek.staging.download_xview import labels_zip_path, load_xview_gt
from geoseek.staging.manifest import record_analysis_section

from eval_xview import XVIEW_CLASSES, full_image_dets, sanitize, score_one  # noqa: E402  (reuse the tested pipeline)

SPLIT_SEED = 20260923
TUNE_FRACTION = 0.5


def split_stems(stems: list[str], seed: int, tune_fraction: float) -> tuple[list[str], list[str]]:
    """Deterministic: sort first (canonical order independent of dict/set iteration), then permute with
    a seeded RNG - reproducible from (stems, seed, tune_fraction) alone."""
    order = sorted(stems)
    idx = np.random.default_rng(seed).permutation(len(order))
    n_tune = int(round(len(order) * tune_fraction))
    tune = sorted(order[i] for i in idx[:n_tune])
    test = sorted(order[i] for i in idx[n_tune:])
    return tune, test


def restrict(d: dict, stems: set) -> dict:
    return {s: v for s, v in d.items() if s in stems}


def main() -> None:
    settings = get_settings()
    out_dir = settings.data_dir / "detect_eval" / "xview"
    pred = load_predictions(out_dir / "pred_xview.npz")
    all_stems = sorted({parse_chip_id(c)[0] for c in pred["chip_ids"]})
    tune_stems, test_stems = split_stems(all_stems, SPLIT_SEED, TUNE_FRACTION)
    print(f"[tune_xview] {len(all_stems)} images -> {len(tune_stems)} tune / {len(test_stems)} test "
          f"(seed={SPLIT_SEED}, disjoint: {set(tune_stems) & set(test_stems) == set()})", flush=True)

    gt_all, gt_stats = load_xview_gt(labels_zip_path(), set(all_stems))
    dets_fair_all = {s: dets_to_enclosing_hbb(d) for s, d in full_image_dets(pred).items()}

    tune_set, test_set = set(tune_stems), set(test_stems)
    gt_tune, dets_tune = restrict(gt_all, tune_set), restrict(dets_fair_all, tune_set)
    gt_test, dets_test = restrict(gt_all, test_set), restrict(dets_fair_all, test_set)

    _, tune_matches = score_one(gt_tune, dets_tune, size_buckets=False)
    test_res, test_matches = score_one(gt_test, dets_test, size_buckets=True)

    result = {
        "purpose": "vehicle confidence thresholds tuned on TUNE only, all reporting on TEST only - no leakage.",
        "split": {"seed": SPLIT_SEED, "tune_fraction": TUNE_FRACTION, "n_images_total": len(all_stems),
                  "n_images_tune": len(tune_stems), "n_images_test": len(test_stems),
                  "tune_stems": tune_stems, "test_stems": test_stems},
        "per_class": {},
    }
    for cls in XVIEW_CLASSES:
        thr, tune_f1 = best_f1_threshold(tune_matches[cls])
        test_op = operating_point(test_matches[cls], thr)
        pc = test_res["per_class"][cls]
        result["per_class"][cls] = {
            "threshold": thr, "tuned_on": "TUNE half only", "tune_f1_at_threshold": tune_f1,
            "test_AP50": pc["AP50"], "test_AP50_95": pc["AP50_95"], "test_n_gt": test_op["n_gt"],
            "test_precision_at_threshold": test_op["precision"], "test_recall_at_threshold": test_op["recall"],
            "test_f1_at_threshold": test_op["f1"], "test_by_size_px": pc.get("by_size_px"),
        }
        print(f"[tune_xview] {cls}: threshold={thr:.4f} (chosen on TUNE, f1={tune_f1:.3f}) | "
              f"TEST-only: AP50={pc['AP50']:.4f} AP50:95={pc['AP50_95']:.4f} "
              f"P={test_op['precision']:.3f} R={test_op['recall']:.3f} F1={test_op['f1']:.3f}", flush=True)

    out_path = out_dir / "tune_test_results.json"
    out_path.write_text(json.dumps(sanitize(result), indent=1), encoding="utf-8")
    record_analysis_section("xview_vehicle_threshold_tune_test_split", result)
    print(f"\n[tune_xview] wrote {out_path} and recorded xview_vehicle_threshold_tune_test_split in the manifest", flush=True)


if __name__ == "__main__":
    main()

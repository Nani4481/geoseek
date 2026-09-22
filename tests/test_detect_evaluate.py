"""Phase 8F-2: the full-image DOTA-protocol evaluator (synthetic ground truth with known answers; numpy/scipy/cv2 only)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from geoseek.detect.evaluate import (
    ImageDets,
    bootstrap_ap,
    ImageGT,
    _convex_iou,
    best_f1_threshold,
    dets_to_enclosing_hbb,
    evaluate_dataset,
    group_summary,
    long_side,
    match_class,
    merge_cross_chip,
    obb_to_enclosing_hbb,
    operating_point,
    xywhr_to_polys,
)

NAMES = ("small-vehicle", "large-vehicle")


def rect(cx, cy, w, h, deg=0.0):
    return xywhr_to_polys(np.array([[cx, cy, w, h, math.radians(deg)]]))[0]


def gt_of(stem, items):
    """items: [(class, poly, difficult)]"""
    if not items:
        return ImageGT(stem, np.zeros(0, int), np.zeros((0, 4, 2)), np.zeros(0, bool))
    return ImageGT(stem, np.array([c for c, _, _ in items]), np.stack([p for _, p, _ in items]),
                   np.array([d for _, _, d in items], dtype=bool))


def dets_of(stem, items):
    """items: [(class, score, poly, chip_id)]"""
    if not items:
        return ImageDets(stem, np.zeros(0, int), np.zeros(0), np.zeros((0, 4, 2)), np.zeros(0, int))
    return ImageDets(stem, np.array([c for c, *_ in items]), np.array([s for _, s, *_ in items]),
                     np.stack([p for _, _, p, _ in items]), np.array([k for *_, k in items]))


def ap(res, cls="small-vehicle", key="AP50"):
    return res["per_class"][cls][key]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_xywhr_to_polys_matches_opencv_boxpoints_up_to_vertex_order():
    for deg in (0, 17, 45, 90, 133):
        p = xywhr_to_polys(np.array([[100.0, 50.0, 40.0, 12.0, math.radians(deg)]]))[0]
        q = cv2.boxPoints(((100.0, 50.0), (40.0, 12.0), deg)).astype(np.float64)
        d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=2)
        assert d.min(axis=1).max() < 1e-4 and d.min(axis=0).max() < 1e-4


def test_convex_iou_known_values():
    a = rect(50, 50, 20, 10)
    assert _convex_iou(a, a) == pytest.approx(1.0, abs=1e-6)
    assert _convex_iou(a, rect(50 + 10, 50, 20, 10)) == pytest.approx(1 / 3, abs=1e-4)      # half overlap of two equal boxes
    assert _convex_iou(a, rect(500, 500, 20, 10)) == 0.0
    assert _convex_iou(rect(50, 50, 30, 10, 0), rect(50, 50, 30, 10, 90)) == pytest.approx(100 / (300 + 300 - 100), abs=1e-3)
    assert long_side(a[None])[0] == pytest.approx(20.0)


def test_obb_to_enclosing_hbb_is_identity_on_axis_aligned_boxes():
    a = rect(50, 50, 20, 10, 0)
    np.testing.assert_allclose(obb_to_enclosing_hbb(a[None])[0], a, atol=1e-9)


def test_obb_to_enclosing_hbb_area_matches_known_geometry_for_a_rotated_square():
    # a side-s square rotated 45 deg has an axis-aligned enclosing box of side s*sqrt(2) -> area 2x
    s = 10.0
    sq = rect(0, 0, s, s, 45)
    hbb = obb_to_enclosing_hbb(sq[None])[0]
    enclosing_area = (hbb[:, 0].max() - hbb[:, 0].min()) * (hbb[:, 1].max() - hbb[:, 1].min())
    assert enclosing_area == pytest.approx(2 * s * s, rel=1e-3)


def test_enclosing_hbb_fixes_the_tight_obb_vs_loose_hbb_iou_penalty():
    """The scenario the fix exists for: a PERFECT rotated detection of a car scored against its
    own axis-aligned (xView-style) ground-truth box. Raw OBB IoU falls noticeably below 1.0 purely
    from the shape mismatch (not a detection error); the enclosing-box conversion recovers ~1.0."""
    obb_pred = rect(50, 50, 8, 4, 45)                                    # a tight oriented "car"
    hbb_gt = obb_to_enclosing_hbb(obb_pred[None])[0]                     # xView-style: the axis-aligned box AROUND it

    raw_iou = _convex_iou(obb_pred, hbb_gt)
    assert raw_iou < 0.6                                                 # would be discarded at the standard 0.5 threshold

    fixed_iou = _convex_iou(obb_to_enclosing_hbb(obb_pred[None])[0], hbb_gt)
    assert fixed_iou == pytest.approx(1.0, abs=1e-6)


def test_dets_to_enclosing_hbb_only_changes_polys():
    d = dets_of("a", [(0, 0.9, rect(50, 50, 20, 8, 30), 2), (1, 0.7, rect(150, 80, 20, 8, 75), 5)])
    out = dets_to_enclosing_hbb(d)
    assert out.stem == d.stem
    np.testing.assert_array_equal(out.classes, d.classes)
    np.testing.assert_array_equal(out.scores, d.scores)
    np.testing.assert_array_equal(out.chip_ids, d.chip_ids)
    np.testing.assert_allclose(out.polys, obb_to_enclosing_hbb(d.polys))
    assert not np.allclose(out.polys[0], d.polys[0])                     # actually changed for the rotated one


# --------------------------------------------------------------------------
# AP semantics
# --------------------------------------------------------------------------


def test_perfect_detections_give_ap_one_at_every_iou_threshold():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 8, 30), False), (0, rect(150, 80, 20, 8, 75), False)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 8, 30), 0), (0, 0.8, rect(150, 80, 20, 8, 75), 0)])}
    r = evaluate_dataset(g, d, NAMES)
    assert ap(r) == pytest.approx(1.0) and ap(r, key="AP50_95") == pytest.approx(1.0) and ap(r, key="AP50_voc07") == pytest.approx(1.0)


def test_ap_at_high_iou_drops_when_boxes_are_offset():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False)])}
    off = rect(53, 50, 20, 10)                                                    # IoU = 17/23 ~ 0.74
    d = {"a": dets_of("a", [(0, 0.9, off, 0)])}
    r = evaluate_dataset(g, d, NAMES)
    assert ap(r) == pytest.approx(1.0)                                            # passes 0.5 ... 0.70
    by = r["per_class"]["small-vehicle"]["AP_by_iou"]
    assert by["0.7"] == pytest.approx(1.0) and by["0.75"] == 0.0 and by["0.95"] == 0.0
    assert ap(r, key="AP50_95") == pytest.approx(5 / 10)                          # thresholds 0.50..0.70 pass


def test_false_positive_scored_above_the_true_positive_lowers_ap():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False)])}
    hi_fp = (0, 0.95, rect(300, 300, 20, 10), 0)
    tp = (0, 0.5, rect(50, 50, 20, 10), 0)
    assert ap(evaluate_dataset(g, {"a": dets_of("a", [hi_fp, tp])}, NAMES)) == pytest.approx(0.5)   # precision 1/2 at recall 1
    assert ap(evaluate_dataset(g, {"a": dets_of("a", [tp])}, NAMES)) == pytest.approx(1.0)


def test_duplicate_detection_of_one_object_is_a_false_positive_not_a_second_hit():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.8, rect(51, 50, 20, 10), 1)])}
    m = match_class(g, d, 0)
    op = operating_point(m, 0.0)
    assert (op["TP"], op["FP"], op["FN"]) == (1, 1, 0)
    assert ap(evaluate_dataset(g, d, NAMES)) == pytest.approx(1.0)                # the FP arrives after full recall


def test_difficult_gt_is_ignored_both_as_a_target_and_as_a_reason_to_penalise():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False), (0, rect(150, 50, 20, 10), True)])}
    only_easy = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0)])}
    with_diff = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.85, rect(150, 50, 20, 10), 0)])}
    for dets in (only_easy, with_diff):                                           # detecting the difficult one costs nothing
        r = evaluate_dataset(g, dets, NAMES)
        assert r["per_class"]["small-vehicle"]["n_gt"] == 1 and ap(r) == pytest.approx(1.0)
    op = operating_point(match_class(g, with_diff, 0), 0.0)
    assert (op["TP"], op["FP"], op["FN"]) == (1, 0, 0)
    # ... but if the difficult objects were NOT ignored they would be 2 GT: the chip-level protocol's different answer
    assert evaluate_dataset(g, only_easy, NAMES)["per_class"]["small-vehicle"]["n_gt"] == 1


def test_classes_are_scored_independently():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False), (1, rect(150, 50, 30, 12), False)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.8, rect(150, 50, 30, 12), 0)])}   # 2nd: wrong class
    r = evaluate_dataset(g, d, NAMES)
    assert ap(r, "small-vehicle") == pytest.approx(1.0)                           # TP first, the wrong-class hit is a later FP
    assert ap(r, "large-vehicle") == 0.0                                          # the large vehicle was never detected as such


def test_voc07_eleven_point_differs_from_all_point_on_a_stepped_curve():
    g = {"a": gt_of("a", [(0, rect(50 + 100 * i, 50, 20, 10), False) for i in range(4)])}
    dets = [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.8, rect(900, 900, 20, 10), 0),
            (0, 0.7, rect(150, 50, 20, 10), 0), (0, 0.6, rect(250, 50, 20, 10), 0), (0, 0.5, rect(350, 50, 20, 10), 0)]
    r = evaluate_dataset(g, {"a": dets_of("a", dets)}, NAMES)
    assert 0.0 < ap(r, key="AP50") <= 1.0 and 0.0 < ap(r, key="AP50_voc07") <= 1.0
    assert ap(r, key="AP50") != pytest.approx(ap(r, key="AP50_voc07"), abs=1e-6)


# --------------------------------------------------------------------------
# cross-chip merge
# --------------------------------------------------------------------------


def test_cross_chip_merge_removes_the_same_object_seen_in_two_overlapping_chips_only():
    same_object_a = rect(500, 500, 20, 10)
    same_object_b = rect(501, 500, 20, 10)                                        # same car, second chip, IoU ~0.9
    neighbour = rect(500, 512, 20, 10)                                            # different car right next to it (touching)
    d = dets_of("a", [(0, 0.9, same_object_a, 0), (0, 0.8, same_object_b, 1), (0, 0.7, neighbour, 0)])
    m = merge_cross_chip(d, 0.5)
    assert len(m.scores) == 2 and set(np.round(m.scores, 1)) == {0.9, 0.7}
    # same-chip near-duplicates are NOT touched (the detector's own NMS owns that decision)
    d2 = dets_of("a", [(0, 0.9, same_object_a, 0), (0, 0.8, same_object_b, 0)])
    assert len(merge_cross_chip(d2, 0.5).scores) == 2
    # different classes never suppress each other
    d3 = dets_of("a", [(0, 0.9, same_object_a, 0), (1, 0.8, same_object_b, 1)])
    assert len(merge_cross_chip(d3, 0.5).scores) == 2


def test_merging_removes_the_false_positive_a_chip_overlap_would_have_created():
    g = {"a": gt_of("a", [(0, rect(500, 500, 20, 10), False)])}
    raw = {"a": dets_of("a", [(0, 0.9, rect(500, 500, 20, 10), 0), (0, 0.8, rect(501, 500, 20, 10), 1)])}
    assert operating_point(match_class(g, raw, 0), 0.0)["FP"] == 1
    merged = {"a": merge_cross_chip(raw["a"], 0.5)}
    assert operating_point(match_class(g, merged, 0), 0.0)["FP"] == 0


# --------------------------------------------------------------------------
# operating point, size buckets, groups
# --------------------------------------------------------------------------


def test_operating_point_and_best_f1_threshold():
    g = {"a": gt_of("a", [(0, rect(50 + 100 * i, 50, 20, 10), False) for i in range(3)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.8, rect(150, 50, 20, 10), 0),
                            (0, 0.4, rect(700, 700, 20, 10), 0), (0, 0.3, rect(250, 50, 20, 10), 0)])}
    m = match_class(g, d, 0)
    hi = operating_point(m, 0.5)
    assert (hi["TP"], hi["FP"], hi["FN"]) == (2, 0, 1) and hi["precision"] == 1.0 and hi["recall"] == pytest.approx(2 / 3)
    lo = operating_point(m, 0.0)
    assert (lo["TP"], lo["FP"], lo["FN"]) == (3, 1, 0)
    thr, f1 = best_f1_threshold(m)
    assert thr in (0.3, 0.8) and f1 == pytest.approx(max(2 * 1 * (2 / 3) / (1 + 2 / 3), 2 * 0.75 * 1 / 1.75), abs=1e-6)


def test_size_buckets_score_small_and_large_objects_separately():
    small_gt = rect(50, 50, 8, 4)                                                 # long side 8 px -> bucket 0-10
    big_gt = rect(300, 50, 40, 16)                                                # 40 px -> bucket 32-64
    g = {"a": gt_of("a", [(0, small_gt, False), (0, big_gt, False)])}
    d = {"a": dets_of("a", [(0, 0.9, big_gt, 0)])}                                # only the big one is found
    r = evaluate_dataset(g, d, NAMES, size_buckets=True)
    by = r["per_class"]["small-vehicle"]["by_size_px"]
    assert by["0-10"]["n_gt"] == 1 and by["0-10"]["AP50"] == 0.0
    assert by["32-64"]["n_gt"] == 1 and by["32-64"]["AP50"] == pytest.approx(1.0)
    assert by["10-16"]["n_gt"] == 0 and math.isnan(by["10-16"]["AP50"])


def test_group_summary_reports_groups_separately_with_macro_and_weighted_means():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False)] + [(1, rect(50 + 100 * i, 200, 30, 12), False) for i in range(3)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (1, 0.9, rect(50, 200, 30, 12), 0)])}
    r = evaluate_dataset(g, d, NAMES)
    s = group_summary(r["per_class"], {"vehicles": NAMES, "small_only": ("small-vehicle",)})
    assert s["small_only"]["macro_AP50"] == pytest.approx(1.0)
    assert s["vehicles"]["n_gt"] == 4
    assert s["vehicles"]["macro_AP50"] == pytest.approx((1.0 + r["per_class"]["large-vehicle"]["AP50"]) / 2)
    w = (1 * 1.0 + 3 * r["per_class"]["large-vehicle"]["AP50"]) / 4
    assert s["vehicles"]["weighted_AP50"] == pytest.approx(w)


# --------------------------------------------------------------------------
# bootstrap confidence intervals (over images)
# --------------------------------------------------------------------------


def _many_images(n_img=40, hit_rate=0.7, seed=0):
    rng = np.random.default_rng(seed)
    gt, dets = {}, {}
    for i in range(n_img):
        stem = f"P{i:04d}"
        polys = [rect(50 + 60 * k, 100, 20, 8) for k in range(5)]
        gt[stem] = gt_of(stem, [(0, p, False) for p in polys])
        found = [p for p in polys if rng.random() < hit_rate]
        dets[stem] = dets_of(stem, [(0, float(rng.uniform(0.5, 1.0)), p, 0) for p in found])
    return gt, dets


def test_bootstrap_ci_contains_the_point_estimate_and_is_reproducible():
    gt, dets = _many_images()
    one = ("small-vehicle",)                                                    # a class with no GT would have NaN CIs (nan != nan)
    point = evaluate_dataset(gt, dets, one)["per_class"]["small-vehicle"]["AP50"]
    a = bootstrap_ap(gt, dets, one, {"vehicles": one}, n_boot=60, seed=3)
    b = bootstrap_ap(gt, dets, one, {"vehicles": one}, n_boot=60, seed=3)
    lo, hi = a["per_class"]["small-vehicle"]["AP50_ci"]
    assert lo <= point <= hi and hi - lo > 0
    assert a == b
    assert a["n_images"] == 40 and a["level"] == 0.95 and "vehicles" in a["groups"] and "all_kept_classes" in a["groups"]
    empty = bootstrap_ap(gt, dets, NAMES, {"v": NAMES}, n_boot=5, seed=0)["per_class"]["large-vehicle"]["AP50_ci"]
    assert all(np.isnan(empty))                                                 # no GT for that class -> NaN, silently


def test_bootstrap_ci_is_narrower_with_more_images():
    small = bootstrap_ap(*_many_images(n_img=10, seed=1), NAMES, {"v": NAMES}, n_boot=80, seed=1)["per_class"]["small-vehicle"]["AP50_ci"]
    big = bootstrap_ap(*_many_images(n_img=120, seed=1), NAMES, {"v": NAMES}, n_boot=80, seed=1)["per_class"]["small-vehicle"]["AP50_ci"]
    assert (big[1] - big[0]) < (small[1] - small[0])


def test_match_class_refactor_is_unchanged_on_a_mixed_case():
    g = {"a": gt_of("a", [(0, rect(50, 50, 20, 10), False), (0, rect(150, 50, 20, 10), True)]),
         "b": gt_of("b", [(0, rect(80, 80, 20, 10), False)])}
    d = {"a": dets_of("a", [(0, 0.9, rect(50, 50, 20, 10), 0), (0, 0.8, rect(51, 50, 20, 10), 1), (0, 0.7, rect(150, 50, 20, 10), 0)]),
         "b": dets_of("b", [(0, 0.6, rect(300, 300, 20, 10), 0)])}
    op = operating_point(match_class(g, d, 0), 0.0)
    assert (op["TP"], op["FP"], op["FN"]) == (1, 2, 1)             # 1 hit; duplicate + a miss in b are FP; b's GT is missed; difficult ignored

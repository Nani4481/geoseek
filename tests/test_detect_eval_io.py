"""Phase 8F-2: chip-id parsing, original-quad GT loading and chip -> full-image assembly (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.detect.classes import CLASS_TO_INDEX
from geoseek.detect.eval_io import assemble_full_image_dets, load_dota_gt, parse_chip_id
from geoseek.detect.evaluate import merge_cross_chip, xywhr_to_polys


def test_parse_chip_id():
    assert parse_chip_id("P0001__824_0") == ("P0001", 824, 0)
    assert parse_chip_id("P2809__0_1648") == ("P2809", 0, 1648)


def test_load_dota_gt_keeps_original_quads_kept_classes_and_difficult_flags(tmp_path):
    (tmp_path / "P0001.txt").write_text(
        "imagesource:GoogleEarth\ngsd:0.3\n"
        "742 384 728 401 672 359 687 337 ship 0\n"              # a genuine quadrilateral, not a rectangle
        "10 10 20 10 20 20 10 20 small-vehicle 1\n"
        "30 30 90 30 90 90 30 90 tennis-court 0\n", encoding="utf-8")   # dropped class
    (tmp_path / "P0002.txt").write_text("imagesource:x\ngsd:null\n", encoding="utf-8")         # image with no kept objects
    gt = load_dota_gt(tmp_path)
    assert set(gt) == {"P0001", "P0002"}
    g = gt["P0001"]
    assert list(g.classes) == [CLASS_TO_INDEX["ship"], CLASS_TO_INDEX["small-vehicle"]]
    assert list(g.difficult) == [False, True]
    np.testing.assert_allclose(g.quads[0], [[742, 384], [728, 401], [672, 359], [687, 337]])     # NOT refit to a rectangle
    assert len(gt["P0002"].classes) == 0
    assert set(load_dota_gt(tmp_path, {"P0002"})) == {"P0002"}


def test_assemble_moves_chip_local_detections_to_full_image_coordinates():
    chips = ["P0001__0_0", "P0001__824_0", "P0002__0_824"]
    xywhr = np.array([[100.0, 100.0, 20.0, 8.0, 0.0],        # chip 0 -> stays
                      [50.0, 60.0, 20.0, 8.0, 0.3],          # chip 1 -> +824 in x
                      [10.0, 20.0, 30.0, 9.0, 1.0]])         # chip 2 (other image) -> +824 in y
    out = assemble_full_image_dets(chips, np.array([0, 1, 2]), xywhr, np.array([0.9, 0.8, 0.7]), np.array([0, 2, 1]))
    assert set(out) == {"P0001", "P0002"}
    a = out["P0001"]
    assert len(a.scores) == 2 and set(a.chip_ids) == {0, 1}
    centres = a.polys.mean(axis=1)
    assert any(np.allclose(c, [100.0, 100.0], atol=1e-6) for c in centres)
    assert any(np.allclose(c, [50.0 + 824, 60.0], atol=1e-6) for c in centres)
    assert np.allclose(out["P0002"].polys.mean(axis=1)[0], [10.0, 20.0 + 824], atol=1e-6)
    assert assemble_full_image_dets(chips, np.zeros(0, int), np.zeros((0, 5)), np.zeros(0), np.zeros(0, int)) == {}


def test_same_object_seen_in_two_overlapping_chips_is_merged_after_assembly():
    # chip 0 covers x 0..1023, chip 1 covers x 824..1847: a car at full-image x=900 appears in both (local 900 and 76)
    chips = ["P0001__0_0", "P0001__824_0"]
    xywhr = np.array([[900.0, 300.0, 24.0, 10.0, 0.4], [76.0, 300.0, 24.0, 10.0, 0.4]])
    per = assemble_full_image_dets(chips, np.array([0, 1]), xywhr, np.array([0.9, 0.8]), np.array([0, 0]))
    assert len(per["P0001"].scores) == 2
    merged = merge_cross_chip(per["P0001"], 0.5)
    assert len(merged.scores) == 1 and merged.scores[0] == pytest.approx(0.9)

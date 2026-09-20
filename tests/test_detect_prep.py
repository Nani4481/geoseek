"""Phase 8F-2: DOTA OBB -> YOLO-OBB conversion tests (classes, chipping, sampling, conversion).

Hermetic - synthetic images/labels only, no dependency on the staged DOTA archive
(which may not be present in a CI / fresh clone).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from geoseek.detect.chipping import (
    CHIP_SIZE,
    STRIDE,
    DotaInstance,
    axis_deviation_deg,
    chip_origins,
    clip_instance_to_chip,
    crop_chip,
    format_yolo_obb_line,
    is_axis_aligned,
    long_edge_angle_deg,
    normalize_points,
    parse_dota_label_file,
    parse_gsd,
    plan_windows,
)
from geoseek.detect.classes import (
    CLASS_GROUPS,
    CLASS_TO_INDEX,
    DOTA15_NAMES,
    DROPPED_CLASSES,
    KEPT_CLASSES,
    LABEL_SETS,
    OBB_LABEL_DIR,
    PRIMARY_LABEL_SET,
    V15_ONLY_CLASSES,
    dota15_index_to_ours,
)
from geoseek.detect.convert import (
    SourcePlan,
    choose_monitor_sources,
    find_images_dir,
    plan_source,
    select_train_windows,
    verify_conversion,
    write_dataset_yamls,
)
from geoseek.detect.sampling import (
    ChipClasses,
    build_repeat_list,
    chip_repeat_factor,
    class_repeat_factors,
    presence_fractions,
)


def rot_rect(cx, cy, length, width, deg):
    """4 corner points of a length x width rectangle rotated by ``deg`` (image coords), clockwise from one corner."""
    a = math.radians(deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cy])
    hl, hw = length / 2, width / 2
    return tuple(tuple(map(float, p)) for p in (c - hl * u - hw * v, c + hl * u - hw * v, c + hl * u + hw * v, c - hl * u + hw * v))


# --------------------------------------------------------------------------
# classes.py
# --------------------------------------------------------------------------


def test_kept_and_dropped_partition_the_16_dota_v15_classes():
    all_v15 = {n.replace(" ", "-") for n in DOTA15_NAMES} | set(V15_ONLY_CLASSES)
    assert len(all_v15) == 16
    assert set(KEPT_CLASSES) | set(DROPPED_CLASSES) == all_v15
    assert set(KEPT_CLASSES).isdisjoint(DROPPED_CLASSES)


def test_container_crane_is_the_v15_only_class_and_is_dropped():
    # ~890:1 against small-vehicle in train (142 vs 126,501); unlearnable at that support
    assert V15_ONLY_CLASSES == ("container-crane",)
    assert "container-crane" in DROPPED_CLASSES and "container-crane" not in KEPT_CLASSES


def test_primary_label_set_is_the_oriented_v15_and_v10_is_the_secondary_view():
    assert PRIMARY_LABEL_SET == "v1.5" and OBB_LABEL_DIR == "labelTxt-v1.5"
    assert LABEL_SETS["v1.0"] == "labelTxt-v1.0"


def test_class_to_index_is_contiguous_and_matches_kept_order():
    assert CLASS_TO_INDEX == {name: i for i, name in enumerate(KEPT_CLASSES)}


def test_groups_partition_kept_classes_and_vehicles_are_their_own_group():
    members = [c for g in CLASS_GROUPS.values() for c in g]
    assert sorted(members) == sorted(KEPT_CLASSES)
    assert CLASS_GROUPS["ground_vehicles"] == ("small-vehicle", "large-vehicle")


def test_dota15_index_mapping_covers_exactly_the_kept_classes():
    m = dota15_index_to_ours()
    assert len(m) == len(KEPT_CLASSES)
    assert {DOTA15_NAMES[i].replace(" ", "-") for i in m} == set(KEPT_CLASSES)
    assert all(KEPT_CLASSES[j] == DOTA15_NAMES[i].replace(" ", "-") for i, j in m.items())


# --------------------------------------------------------------------------
# chipping.py: parsing
# --------------------------------------------------------------------------


def test_parse_dota_label_file_skips_header_lines_and_reads_difficult():
    text = "imagesource:GoogleEarth\ngsd:0.146343590398\n10 10 20 10 20 20 10 20 small-vehicle 1\n"
    (inst,) = parse_dota_label_file(text)
    assert inst.category == "small-vehicle" and inst.difficult == 1
    assert inst.points == ((10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0))


def test_parse_dota_label_file_handles_empty_text():
    assert parse_dota_label_file("") == []


def test_parse_gsd():
    assert parse_gsd("imagesource:GoogleEarth\ngsd:0.25\n") == 0.25
    assert parse_gsd("imagesource:GoogleEarth\ngsd:null\n") is None
    assert parse_gsd("") is None


# --------------------------------------------------------------------------
# chipping.py: windows / crop
# --------------------------------------------------------------------------


def test_chip_origins_small_image_is_single_origin():
    assert chip_origins(500, 500) == [(0, 0)]
    assert chip_origins(CHIP_SIZE, CHIP_SIZE) == [(0, 0)]


def test_chip_origins_last_chip_snapped_to_edge_never_overflowing():
    w = h = CHIP_SIZE + STRIDE + 100
    for x, y in chip_origins(w, h):
        assert x + CHIP_SIZE <= w and y + CHIP_SIZE <= h
    assert max(x for x, _ in chip_origins(w, h)) == w - CHIP_SIZE


def test_chip_origins_cover_the_image_with_no_gap():
    xs = sorted({x for x, _ in chip_origins(3000, 3000)})
    assert all(b - a <= STRIDE for a, b in zip(xs, xs[1:]))
    assert xs[-1] + CHIP_SIZE == 3000


def test_crop_chip_pads_past_the_edge_and_is_exact_inside():
    image = np.full((600, 600, 3), 255, dtype=np.uint8)
    chip = crop_chip(image, 0, 0, CHIP_SIZE)
    assert chip.shape == (CHIP_SIZE, CHIP_SIZE, 3) and chip[:600, :600].min() == 255 and chip[600:].max() == 0
    big = np.arange(2048 * 2048 * 3, dtype=np.uint8).reshape(2048, 2048, 3)
    np.testing.assert_array_equal(crop_chip(big, 100, 200), big[200:200 + CHIP_SIZE, 100:100 + CHIP_SIZE])


# --------------------------------------------------------------------------
# chipping.py: clipping (the boundary-handling core)
# --------------------------------------------------------------------------


def test_fully_inside_instance_kept_verbatim_and_translated():
    inst = DotaInstance(points=((1124, 1124), (1174, 1124), (1174, 1174), (1124, 1174)), category="ship", difficult=0)
    assert clip_instance_to_chip(inst, x0=1000, y0=1000) == ((124, 124), (174, 124), (174, 174), (124, 174))


def test_instance_outside_chip_is_none_and_mostly_outside_is_dropped():
    far = DotaInstance(points=((5000, 5000), (5050, 5000), (5050, 5050), (5000, 5050)), category="ship", difficult=0)
    assert clip_instance_to_chip(far, 0, 0) is None
    sliver = DotaInstance(points=((980, 500), (1080, 500), (1080, 600), (980, 600)), category="ship", difficult=0)
    assert clip_instance_to_chip(sliver, 0, 0) is None            # ~40% visible < 0.6


def test_mostly_inside_instance_is_refit_within_the_chip():
    inst = DotaInstance(points=((934, 500), (1034, 500), (1034, 600), (934, 600)), category="ship", difficult=0)
    pts = clip_instance_to_chip(inst, 0, 0)
    assert pts is not None
    assert max(p[0] for p in pts) <= CHIP_SIZE + 0.01 and 500 - 0.01 <= min(p[1] for p in pts)


def test_rotated_instance_keeps_its_heading_when_clipped_at_the_edge():
    # a 100 x 20 vehicle at 45 deg whose far end pokes past the chip edge (~85% visible): the re-fit must
    # stay long-and-thin along the SAME diagonal, not flip to the perpendicular one (the "90 degrees off" bug)
    pts = rot_rect(CHIP_SIZE - 25, 512, 100, 20, 45)
    inst = DotaInstance(points=pts, category="large-vehicle", difficult=0)
    out = clip_instance_to_chip(inst, 0, 0)
    assert out is not None
    assert abs(long_edge_angle_deg(out) - 45.0) < 6.0


def test_orientation_helpers():
    assert is_axis_aligned(((0, 0), (10, 0), (10, 5), (0, 5)))
    assert not is_axis_aligned(rot_rect(50, 50, 30, 10, 30))
    assert axis_deviation_deg(((0, 0), (10, 0), (10, 5), (0, 5))) == pytest.approx(0.0, abs=1e-6)
    assert axis_deviation_deg(rot_rect(50, 50, 30, 10, 45)) == pytest.approx(45.0, abs=1e-6)
    assert long_edge_angle_deg(rot_rect(50, 50, 30, 10, 30)) == pytest.approx(30.0, abs=1e-6)
    assert long_edge_angle_deg(rot_rect(50, 50, 30, 10, 120)) == pytest.approx(120.0, abs=1e-6)


def test_normalize_and_format():
    pts = ((0, 0), (CHIP_SIZE, 0), (CHIP_SIZE, CHIP_SIZE), (0, CHIP_SIZE))
    assert normalize_points(pts) == (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    assert max(normalize_points(((-10, -10), (CHIP_SIZE + 10, CHIP_SIZE + 10), (0, 0), (0, 0)))) <= 1.0
    parts = format_yolo_obb_line(3, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)).split()
    assert parts[0] == "3" and len(parts) == 9


# --------------------------------------------------------------------------
# chipping.py: plan_windows
# --------------------------------------------------------------------------


def test_plan_windows_filters_dropped_classes_but_counts_them_as_distractors():
    insts = [
        DotaInstance(rot_rect(300, 300, 60, 20, 30), "tennis-court", 0),          # dropped class
        DotaInstance(rot_rect(600, 600, 60, 20, 30), "large-vehicle", 0),         # kept
    ]
    plan = plan_windows(900, 900, insts)
    assert plan.n_kept_instances == 1 and len(plan.windows) == 1
    (w,) = plan.windows
    assert len(w.labels) == 1 and w.labels[0].class_index == CLASS_TO_INDEX["large-vehicle"]
    assert w.n_distractors == 1


def test_plan_windows_boundary_instance_is_kept_in_the_window_that_sees_it_whole():
    w = h = CHIP_SIZE + STRIDE                      # two origins per axis: 0 and w-CHIP_SIZE
    x_second = w - CHIP_SIZE
    inst = DotaInstance(rot_rect(x_second + 40, 500, 60, 20, 10), "large-vehicle", 0)
    plan = plan_windows(w, h, [inst])
    assert plan.n_retained == 1 and plan.n_dropped == 0
    assert sum(len(x.labels) for x in plan.windows) >= 1


def test_plan_windows_counts_never_visible_enough_instances_as_dropped():
    # To be < 60% visible in EVERY 1024 window an object must be longer than 1024 / 0.6 ~ 1707 px. A 1900 px
    # bridge is at best 1024/1900 = 54% visible (window x0=824); a 1400 px one would be 73% visible there.
    inst = DotaInstance(rot_rect(1500, 500, 1900, 30, 0), "bridge", 0)
    plan = plan_windows(3000, 1200, [inst])
    assert plan.n_retained == 0 and plan.n_dropped == 1
    assert plan.best_visibility[0] == pytest.approx(1024 / 1900, abs=0.01)
    ok = DotaInstance(rot_rect(1500, 500, 1400, 30, 0), "bridge", 0)         # 73% visible in one window -> retained
    assert plan_windows(3000, 1200, [ok]).n_retained == 1


def test_plan_windows_preserves_orientation_through_the_label_text():
    inst = DotaInstance(rot_rect(400, 400, 80, 24, 37), "small-vehicle", 0)
    plan = plan_windows(800, 800, [inst])
    (lab,) = plan.windows[0].labels
    assert abs(long_edge_angle_deg(lab.points) - 37.0) < 1e-6              # untouched, fully inside
    line = format_yolo_obb_line(lab.class_index, normalize_points(lab.points))
    back = [(float(line.split()[i]) * CHIP_SIZE, float(line.split()[i + 1]) * CHIP_SIZE) for i in range(1, 9, 2)]
    assert abs(long_edge_angle_deg(back) - 37.0) < 0.05                    # ... and through 6-decimal serialisation


# --------------------------------------------------------------------------
# convert.py: plan_source / monitor / negatives / verification (synthetic DOTA on disk)
# --------------------------------------------------------------------------


def _write_dota_source(root, stem, w, h, instances, gsd="0.3"):
    images = root / "images" / "images"
    labels = root / OBB_LABEL_DIR
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)).save(images / f"{stem}.png")
    lines = ["imagesource:synthetic", f"gsd:{gsd}"]
    for pts, cat in instances:
        lines.append(" ".join(f"{v:.1f}" for p in pts for v in p) + f" {cat} 0")
    (labels / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def test_find_images_dir_handles_the_double_nested_layout(tmp_path):
    _write_dota_source(tmp_path, "P0001", 600, 600, [])
    assert find_images_dir(tmp_path) == tmp_path / "images" / "images"


def test_plan_source_reads_header_labels_and_gsd_without_decoding_pixels(tmp_path):
    _write_dota_source(tmp_path, "P0001", 700, 500, [(rot_rect(200, 200, 60, 20, 30), "ship"),
                                                       (rot_rect(400, 300, 50, 50, 0), "roundabout")], gsd="0.42")
    src = plan_source(tmp_path / "images" / "images" / "P0001.png", tmp_path / OBB_LABEL_DIR / "P0001.txt")
    assert (src.width, src.height, src.gsd) == (700, 500, 0.42)
    assert src.class_orig == Counter({"ship": 1}) and src.n_dropped_class == 1
    assert src.n_off_axis == 1                                            # the 30-degree ship


def _fake_source(stem, counts):
    from geoseek.detect.chipping import ImagePlan, WindowPlan
    plan = ImagePlan(width=1024, height=1024, windows=[WindowPlan(0, 0)], n_kept_instances=sum(counts.values()),
                     n_invalid_instances=0, best_visibility=np.ones(sum(counts.values())))
    return SourcePlan(stem=stem, image_path=None, width=1024, height=1024, gsd=None, plan=plan,
                      class_orig=Counter(counts), n_difficult_kept=0, n_off_axis=0, n_dropped_class=0)


def test_monitor_selection_is_deterministic_and_covers_rare_classes():
    sources = [_fake_source(f"P{i:04d}", {"small-vehicle": 50}) for i in range(60)]
    sources += [_fake_source(f"H{i:04d}", {"helicopter": 5, "small-vehicle": 5}) for i in range(20)]
    a = choose_monitor_sources(sources, fraction=0.1, seed=1)
    b = choose_monitor_sources(sources, fraction=0.1, seed=1)
    assert a == b and a != choose_monitor_sources(sources, fraction=0.1, seed=2)
    heli = sum(s.class_orig["helicopter"] for s in sources if s.stem in a)
    assert heli >= math.ceil(0.1 * 100)                                    # rarest class reached its target


def test_monitor_selection_does_not_let_one_image_swallow_a_rare_class():
    sources = [_fake_source("BASE", {"helicopter": 90})] + [_fake_source(f"H{i}", {"helicopter": 2}) for i in range(5)]
    chosen = choose_monitor_sources(sources, fraction=0.1, seed=3)
    assert "BASE" not in chosen                                            # 90 >> the ~10 still needed, alternatives exist


def test_monitor_cap_holds_even_when_the_only_holders_are_big_images():
    # the regression that held out 49% of all train helicopters: every helicopter is in a few big images
    sources = [_fake_source(f"B{i}", {"helicopter": 60}) for i in range(4)] + \
              [_fake_source(f"P{i}", {"small-vehicle": 100}) for i in range(50)]
    chosen = choose_monitor_sources(sources, fraction=0.08, seed=5)
    held = sum(s.class_orig["helicopter"] for s in sources if s.stem in chosen)
    assert held <= math.ceil(2.0 * 0.08 * 240)                             # never above cap_factor * fraction of the class


def test_train_negatives_are_bounded_and_prefer_windows_with_dropped_class_distractors():
    from geoseek.detect.chipping import ImagePlan, ChipLabel, WindowPlan

    def src(stem, wins):
        plan = ImagePlan(width=4096, height=1024, windows=wins, n_kept_instances=1, n_invalid_instances=0,
                         best_visibility=np.ones(1))
        return SourcePlan(stem=stem, image_path=None, width=4096, height=1024, gsd=None, plan=plan,
                          class_orig=Counter({"ship": 1}), n_difficult_kept=0, n_off_axis=0, n_dropped_class=0)

    lab = ChipLabel(class_index=2, points=((10, 10), (30, 10), (30, 20), (10, 20)), visible_fraction=1.0,
                    difficult=0, source_index=0)
    wins = [WindowPlan(0, 0, [lab])] + [WindowPlan(1000 * i, 0, [], n_distractors=(1 if i % 2 else 0)) for i in range(1, 4)]
    sources = [src(f"S{k}", list(wins)) for k in range(10)]               # 10 positives, 30 empty windows (15 hard)
    chips = select_train_windows(sources, neg_fraction=0.5, hard_share=1.0, seed=0)
    pos = [c for c in chips if c.lines]
    neg = [c for c in chips if not c.lines]
    assert len(pos) == 10 and len(neg) == 5                                # budget = 50% of positives
    assert all(c.kind == "negative_hard" for c in neg)                     # hard_share=1.0 -> hard first


def test_write_dataset_yamls_keeps_official_val_out_of_the_training_yaml(tmp_path):
    paths = write_dataset_yamls(tmp_path)
    train_txt = paths["train"].read_text(encoding="utf-8")
    assert "val: monitor/monitor.txt" in train_txt
    assert "val/val.txt" not in train_txt.replace("monitor/monitor.txt", "")
    assert "val: val/val.txt" in paths["official_val"].read_text(encoding="utf-8")
    for i, name in enumerate(KEPT_CLASSES):
        assert f"  {i}: {name}" in train_txt


def test_verify_conversion_flags_a_leaked_source_image_and_a_bad_label(tmp_path):
    for split in ("train", "monitor", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    write_dataset_yamls(tmp_path)

    def chip(split, stem, line="0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"):
        (tmp_path / split / "images" / f"{stem}__0_0.png").write_bytes(b"x")
        (tmp_path / split / "labels" / f"{stem}__0_0.txt").write_text(line + "\n", encoding="utf-8")

    chip("train", "P0001"); chip("monitor", "P0002"); chip("val", "P0003")
    assert verify_conversion(tmp_path)["ok"]

    chip("val", "P0001")                                                   # same source image now in train AND val
    bad = verify_conversion(tmp_path)
    assert not bad["ok"] and any("train&val" in p for p in bad["problems"])

    (tmp_path / "val" / "images" / "P0001__0_0.png").unlink()
    (tmp_path / "val" / "labels" / "P0001__0_0.txt").unlink()
    chip("val", "P0004", line="9 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2")         # class 9 does not exist
    assert not verify_conversion(tmp_path)["ok"]


# --------------------------------------------------------------------------
# sampling.py
# --------------------------------------------------------------------------


def _chips():
    return ([ChipClasses(f"a{i}", {"small-vehicle": 20}) for i in range(90)] +
            [ChipClasses(f"h{i}", {"helicopter": 2, "small-vehicle": 3}) for i in range(10)])


def test_presence_fractions_and_repeat_factors():
    f = presence_fractions(_chips())
    assert f["small-vehicle"] == 1.0 and f["helicopter"] == pytest.approx(0.1)
    r = class_repeat_factors(f, t=0.4)
    assert r["small-vehicle"] == 1.0                                       # frequent class: never boosted
    assert r["helicopter"] == pytest.approx(2.0)                           # sqrt(0.4 / 0.1)
    assert r["bridge"] == 1.0                                              # absent class: nothing to repeat
    assert chip_repeat_factor({"helicopter": 2, "small-vehicle": 3}, r) == pytest.approx(2.0)   # max over classes
    assert chip_repeat_factor({}, r) == 1.0                                # background chip


def test_build_repeat_list_boosts_rare_class_and_reports_the_measured_effect():
    repeats, rep = build_repeat_list(_chips(), t=0.4, seed=7)
    assert all(repeats[f"a{i}"] == 1 for i in range(90))
    assert all(repeats[f"h{i}"] == 2 for i in range(10))                   # integer repeat factor -> exact
    assert rep["n_list_entries"] == 110 and rep["list_expansion"] == pytest.approx(1.1)
    assert rep["instances_per_class_effective"]["helicopter"] == 2 * rep["instances_per_class_unique"]["helicopter"]
    assert rep["instance_ratio_max_over_min_effective"] < rep["instance_ratio_max_over_min_unique"]


def test_build_repeat_list_is_reproducible_and_stochastic_rounding_is_unbiased():
    chips = [ChipClasses(f"h{i}", {"helicopter": 1}) for i in range(2000)] + \
            [ChipClasses(f"a{i}", {"small-vehicle": 1}) for i in range(8000)]
    a, _ = build_repeat_list(chips, t=0.1 * 2.25, seed=3)                  # r_heli = sqrt(0.225/0.2) = 1.06 (fractional)
    b, _ = build_repeat_list(chips, t=0.1 * 2.25, seed=3)
    assert a == b
    mean_r = np.mean([a[f"h{i}"] for i in range(2000)])
    assert mean_r == pytest.approx(math.sqrt(0.225 / 0.2), abs=0.03)


def test_train_density_cap_excludes_over_dense_chips_but_reports_them():
    from geoseek.detect.chipping import ChipLabel, ImagePlan, WindowPlan

    def lab():
        return ChipLabel(class_index=0, points=((10, 10), (30, 10), (30, 20), (10, 20)), visible_fraction=1.0,
                         difficult=0, source_index=0)

    wins = [WindowPlan(0, 0, [lab() for _ in range(5)]), WindowPlan(1000, 0, [lab() for _ in range(50)])]
    plan = ImagePlan(width=2048, height=1024, windows=wins, n_kept_instances=55, n_invalid_instances=0,
                     best_visibility=np.ones(55))
    src = SourcePlan(stem="S", image_path=None, width=2048, height=1024, gsd=None, plan=plan,
                     class_orig=Counter({"small-vehicle": 55}), n_difficult_kept=0, n_off_axis=0, n_dropped_class=0)
    dropped: list = []
    chips = select_train_windows([src], neg_fraction=0.0, max_labels=20, excluded=dropped)
    assert [c.chip_id for c in chips] == ["S__0_0"]
    assert [c.chip_id for c in dropped] == ["S__1000_0"] and len(dropped[0].lines) == 50
    assert len(select_train_windows([src], neg_fraction=0.0, max_labels=None)) == 2      # cap disabled -> both kept

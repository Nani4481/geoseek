"""Phase 8F-2 pre-flight (Step 4): prove the training labels are correct BEFORE any long run.

Three independent checks, all reading the converted TRAIN chips:

  1. GT chips      - 5 training chips with their ground-truth oriented boxes drawn on, chosen (seeded, not
                     cherry-picked) among chips whose boxes are clearly OBLIQUE, so a wrong rotation or a
                     90-degree swap would be visible. Saved as PNGs; paths printed.
  2. Loader trip   - the labels as Ultralytics' OWN dataset class loads them (polygon -> xywhr via
                     minAreaRect) are converted back to corners and compared with the label-file polygon.
                     Any 90-degree / vertex-order / normalisation bug shows up as a corner error.
  3. Augmented     - a batch drawn from Ultralytics' TRAINING dataset with the real augmentation config
                     (mosaic + rotation), boxes drawn from the loader's tensors. Also measures how the
                     off-axis fraction changes with vs without rotation, i.e. that rotation augmentation
                     is present AND actually applied to the labels.

    python scripts/render_detect_sanity_chips.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.chipping import axis_deviation_deg, long_edge_angle_deg  # noqa: E402
from geoseek.detect.classes import CLASS_TO_INDEX, KEPT_CLASSES  # noqa: E402

SEED = 20260920
N_CHIPS = 5
MIN_LABELS = 8
MIN_MEDIAN_DEVIATION_DEG = 15.0

# BGR, one colour per class (fixed so every render in this project reads the same)
CLASS_BGR = {
    "small-vehicle": (0, 255, 0), "large-vehicle": (0, 200, 255), "ship": (255, 200, 0), "plane": (255, 0, 255),
    "helicopter": (0, 0, 255), "storage-tank": (255, 255, 0), "harbor": (255, 128, 0), "bridge": (128, 0, 255),
}
ELONGATED = {"small-vehicle", "large-vehicle", "ship", "plane"}


def read_label_polys(label_path: Path, size: int = 1024) -> list[tuple[int, np.ndarray]]:
    out = []
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 9:
            continue
        pts = np.array([[float(p[i]) * size, float(p[i + 1]) * size] for i in range(1, 9, 2)], dtype=np.float64)
        out.append((int(p[0]), pts))
    return out


def draw_obbs(img: np.ndarray, items: list[tuple[int, np.ndarray]], *, thickness: int = 2, axis: bool = True) -> np.ndarray:
    for cls, pts in items:
        name = KEPT_CLASSES[cls]
        col = CLASS_BGR[name]
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, col, thickness, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in pts[0]), 3, (0, 0, 255), -1)          # red dot = first vertex
        if axis and name in ELONGATED:                                              # thin white line along the long axis
            edges = np.roll(pts, -1, axis=0) - pts
            k = int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))
            d = edges[k] / (np.linalg.norm(edges[k]) + 1e-9)
            c = pts.mean(axis=0)
            half = 0.5 * np.linalg.norm(edges[k])
            cv2.line(img, tuple(int(v) for v in c - d * half), tuple(int(v) for v in c + d * half), (255, 255, 255), 1, cv2.LINE_AA)
    return img


def legend(img: np.ndarray, counts: dict[str, int], title: str) -> np.ndarray:
    pad = 26 + 18 * (len(counts) + 1)
    cv2.rectangle(img, (0, 0), (330, pad), (0, 0, 0), -1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (k, v) in enumerate(counts.items()):
        cv2.putText(img, f"{k}: {v}", (6, 40 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_BGR[k], 1, cv2.LINE_AA)
    return img


def chip_stats(label_path: Path) -> dict | None:
    polys = read_label_polys(label_path)
    if len(polys) < MIN_LABELS:
        return None
    devs = [axis_deviation_deg(p) for _, p in polys]
    counts: dict[str, int] = {}
    for c, _ in polys:
        counts[KEPT_CLASSES[c]] = counts.get(KEPT_CLASSES[c], 0) + 1
    return {"n": len(polys), "median_dev": float(np.median(devs)), "counts": counts,
            "dominant": max(counts, key=counts.get)}


def pick_gt_chips(labels_dir: Path, rng: random.Random) -> list[Path]:
    """One chip per bucket (small-vehicle / large-vehicle / ship / plane / infrastructure) among chips whose boxes
    are clearly oblique. Seeded draw within each bucket - selection never looks at anything but the labels."""
    files = sorted(labels_dir.glob("*.txt"))
    rng.shuffle(files)
    buckets = {"small-vehicle": [], "large-vehicle": [], "ship": [], "plane": [], "infrastructure": []}
    for f in files:
        if all(len(v) >= 3 for v in buckets.values()):
            break
        s = chip_stats(f)
        if s is None or s["median_dev"] < MIN_MEDIAN_DEVIATION_DEG:
            continue
        dom = s["dominant"]
        key = dom if dom in buckets else ("infrastructure" if dom in ("storage-tank", "harbor", "bridge") else None)
        if key and len(buckets[key]) < 3:
            buckets[key].append(f)
    return [v[0] for v in buckets.values() if v][:N_CHIPS]


def render_gt_chips(train_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    paths = []
    for lp in pick_gt_chips(train_dir / "labels", rng):
        img = cv2.imread(str(train_dir / "images" / f"{lp.stem}.png"))
        items = read_label_polys(lp)
        s = chip_stats(lp)
        img = draw_obbs(img, items)
        img = legend(img, s["counts"], f"{lp.stem}  n={s['n']}  median off-axis={s['median_dev']:.0f} deg")
        out = out_dir / f"gt_{lp.stem}.png"
        cv2.imwrite(str(out), img)
        paths.append(out)
        # 2x zoom of the densest 256x256 window - orientation of individual vehicles is legible here
        pts = np.array([p.mean(axis=0) for _, p in items])
        best, best_n = (0, 0), -1
        for cy in range(0, 1024 - 255, 64):
            for cx in range(0, 1024 - 255, 64):
                n = int(np.sum((pts[:, 0] >= cx) & (pts[:, 0] < cx + 256) & (pts[:, 1] >= cy) & (pts[:, 1] < cy + 256)))
                if n > best_n:
                    best, best_n = (cx, cy), n
        raw = cv2.imread(str(train_dir / "images" / f"{lp.stem}.png"))
        crop = raw[best[1]:best[1] + 256, best[0]:best[0] + 256]
        crop = cv2.resize(crop, (768, 768), interpolation=cv2.INTER_CUBIC)
        shifted = [(c, (p - np.array(best)) * 3.0) for c, p in items]
        zoom = draw_obbs(crop, shifted, thickness=2)
        zpath = out_dir / f"gt_{lp.stem}_zoom3x.png"
        cv2.imwrite(str(zpath), zoom)
    return paths


# --------------------------------------------------------------------------
# Ultralytics loader checks
# --------------------------------------------------------------------------


def _ul_dataset(yaml_path: Path, mode: str, overrides: dict, batch: int = 16):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg(overrides={"task": "obb", "imgsz": 1024, "data": str(yaml_path), "model": "yolo26s-obb.pt", **overrides})
    data = check_det_dataset(str(yaml_path))
    img_path = data["train"] if mode == "train" else data["val"]
    return build_yolo_dataset(cfg, img_path, batch, data, mode=mode, stride=32), cfg


def _xywhr_to_poly(b: np.ndarray, size: int) -> np.ndarray:
    cx, cy, w, h, r = (float(v) for v in b)
    cx, cy, w, h = cx * size, cy * size, w * size, h * size
    c, s = math.cos(r), math.sin(r)
    u, v = np.array([c, s]), np.array([-s, c])
    ctr = np.array([cx, cy])
    return np.array([ctr - u * w / 2 - v * h / 2, ctr + u * w / 2 - v * h / 2,
                     ctr + u * w / 2 + v * h / 2, ctr - u * w / 2 + v * h / 2])


def _corner_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max over corners of the distance to the nearest corner of the other quad (order/rotation-invariant)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _aspect(poly: np.ndarray) -> float:
    e = np.roll(poly, -1, axis=0) - poly
    l = np.hypot(e[:, 0], e[:, 1])
    return float(max(l[0], l[1]) / max(min(l[0], l[1]), 1e-6))


def _min_area_rect_poly(quad: np.ndarray) -> np.ndarray:
    """The rectangle Ultralytics fits to a DOTA quad (cv2.minAreaRect) - what the model is actually trained on."""
    return cv2.boxPoints(cv2.minAreaRect(quad.astype(np.float32))).astype(np.float64)


def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    from shapely.geometry import Polygon
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    u = pa.union(pb).area
    return float(pa.intersection(pb).area / u) if u > 0 else 0.0


def loader_roundtrip(yaml_path: Path, n_chips: int = 400) -> dict:
    """Ultralytics-loaded (xywhr) boxes -> corners, compared with the label-file polygons.

    DOTA 'OBB' labels are general QUADRILATERALS (4 clicked corners, not exact rectangles); Ultralytics fits the
    minimum-area enclosing rectangle to each. So two different comparisons are made:

      * vs cv2.minAreaRect(quad)  - the PASS/FAIL check: the loader must reproduce that rectangle to sub-pixel
                                    precision, and never swap width/height (a '90 degree' bug).
      * vs the raw quad           - informational: how much the quad -> rectangle approximation itself moves the
                                    corners / costs in IoU. This is label noise inherent to OBB training on DOTA,
                                    largest for small objects, and is reported per class.

    Each truth box is matched to the loaded box OF THE SAME CLASS with the smallest corner error (nearest-centroid
    matching mispairs a harbor with the ships nested inside it). Angles are compared only for clearly elongated
    boxes (aspect >= 1.5) - a near-square box has no defined long axis."""
    ds, _ = _ul_dataset(yaml_path, "val", {})
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(ds)), min(len(ds), n_chips * 2))
    err_rect, err_quad, ang_err, iou_by_cls = [], [], [], {}
    n_boxes, n_chips_used, n_count_mismatch = 0, 0, 0
    for i in idxs:
        item = ds[i]
        truth = read_label_polys(Path(ds.label_files[i]))
        if not truth:
            continue
        n_chips_used += 1
        b = item["bboxes"].numpy()
        cls_loaded = item["cls"].numpy().reshape(-1).astype(int)
        if len(b) != len(truth):
            n_count_mismatch += 1
            continue
        loaded = [(int(c), _xywhr_to_poly(row, 1024)) for c, row in zip(cls_loaded, b)]
        for cls, quad in truth:
            rect = _min_area_rect_poly(quad)
            cands = [p for c, p in loaded if c == cls]
            best = min(cands, key=lambda p: _corner_error(p, rect))
            err_rect.append(_corner_error(best, rect))
            err_quad.append(_corner_error(best, quad))
            if _aspect(rect) >= 1.5:
                ang_err.append(abs(((long_edge_angle_deg(best) - long_edge_angle_deg(rect) + 90) % 180) - 90))
            iou_by_cls.setdefault(KEPT_CLASSES[cls], []).append(_poly_iou(quad, rect))
            n_boxes += 1
        if n_chips_used >= n_chips:
            break
    err_rect, err_quad, ang_err = np.array(err_rect), np.array(err_quad), np.array(ang_err)
    q = lambda a, p: round(float(np.percentile(a, p)), 4)
    return {
        "chips_compared": n_chips_used, "boxes_compared": n_boxes, "chips_with_box_count_mismatch": n_count_mismatch,
        "PASS_loader_vs_minAreaRect_corner_error_px": {"median": q(err_rect, 50), "p99": q(err_rect, 99), "max": round(float(err_rect.max()), 4)},
        "elongated_boxes_compared": int(len(ang_err)),
        "PASS_long_axis_angle_error_deg_vs_minAreaRect": {"median": q(ang_err, 50), "p99": q(ang_err, 99), "max": round(float(ang_err.max()), 4)},
        "PASS_elongated_boxes_with_angle_error_gt_45deg (== a 90-degree swap)": int(np.sum(ang_err > 45)),
        "INFO_loader_vs_raw_quad_corner_error_px": {"median": q(err_quad, 50), "p99": q(err_quad, 99), "max": round(float(err_quad.max()), 2)},
        "INFO_quad_vs_its_min_area_rectangle_IoU_by_class": {
            k: {"n": len(v), "median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3)}
            for k, v in sorted(iou_by_cls.items())},
    }


def augmented_render(yaml_path: Path, out_dir: Path, aug: dict, n: int = 8) -> dict:
    """Draw boxes from the TRAINING loader's tensors (augmentation on) and compare the off-axis fraction with
    rotation on vs off (same seed -> same source chips)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for tag, over in (("rotation_on", aug), ("rotation_off", {**aug, "degrees": 0.0, "flipud": 0.0, "fliplr": 0.0})):
        ds, cfg = _ul_dataset(yaml_path, "train", over)
        random.seed(SEED); np.random.seed(SEED)
        devs, panels = [], []
        for i in range(48):
            item = ds[i * 37 % len(ds)]
            b = item["bboxes"].numpy()
            img = item["img"].numpy().transpose(1, 2, 0)[:, :, ::-1].copy()          # CHW RGB -> HWC BGR
            polys = [(int(c), _xywhr_to_poly(row, 1024)) for c, row in zip(item["cls"].numpy().reshape(-1), b)]
            devs += [axis_deviation_deg(p) for _, p in polys if _aspect(p) >= 1.5 and max(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])) > 6]
            if len(panels) < n and len(polys) >= 5:
                panels.append(draw_obbs(img, polys, thickness=2))
        stats[tag] = {"n_elongated_boxes": len(devs), "pct_off_axis_gt_5deg": round(100 * float(np.mean(np.array(devs) > 5.0)), 1),
                      "median_axis_deviation_deg": round(float(np.median(devs)), 1)}
        if tag == "rotation_on" and panels:
            small = [cv2.resize(p, (512, 512)) for p in panels[:8]]
            while len(small) < 8:
                small.append(np.zeros_like(small[0]))
            sheet = np.vstack([np.hstack(small[:4]), np.hstack(small[4:8])])
            cv2.imwrite(str(out_dir / "augmented_batch_rotation_on.png"), sheet)
    return stats


def main() -> None:
    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    out_dir = settings.data_dir / "detect_preflight"
    print(f"[sanity] GT chips -> {out_dir / 'gt_chips'}")
    paths = render_gt_chips(root / "train", out_dir / "gt_chips")
    for p in paths:
        print(f"[sanity]   {p}")

    print("[sanity] Ultralytics loader round-trip (labels -> xywhr -> corners vs label file) ...")
    rt = loader_roundtrip(root / "dataset_monitor.yaml")
    print(json.dumps(rt, indent=1))

    print("[sanity] augmented training batch (mosaic + rotation) ...")
    aug = {"mosaic": 1.0, "degrees": 180.0, "flipud": 0.5, "fliplr": 0.5, "scale": 0.5, "translate": 0.1}
    st = augmented_render(root / "dataset.yaml", out_dir, aug)
    print(json.dumps(st, indent=1))
    (out_dir / "preflight_labels.json").write_text(
        json.dumps({"gt_chips": [str(p) for p in paths], "loader_roundtrip": rt, "augmentation": st}, indent=2), encoding="utf-8")
    print(f"[sanity] wrote {out_dir / 'preflight_labels.json'}  and {out_dir / 'augmented_batch_rotation_on.png'}")


if __name__ == "__main__":
    main()

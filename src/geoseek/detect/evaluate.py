"""Full-image DOTA-protocol evaluation for oriented detections (pure numpy / scipy / OpenCV; no ultralytics import).

Why not just use the trainer's validator? Its chip-level metric (a) counts every ``difficult`` instance as ordinary ground
truth - 46% of DOTA-v1.5's kept-class instances are flagged difficult, the official devkit IGNORES them - (b) scores the same
object separately in every overlapping chip (1.8 label lines per instance here), and (c) matches on a Gaussian-overlap
approximation (ProbIoU) against *rectangles fitted to* the hand-clicked quadrilateral labels. This module follows the DOTA
devkit's protocol instead, so the numbers are comparable with published DOTA results:

  * detections from all chips of a source image are moved to full-image coordinates and de-duplicated ACROSS chips
    (per-class rotated NMS, only between detections that came from different chips - same-chip duplicates were already
    removed by the detector's own NMS, and adjacent parked cars must not suppress each other);
  * IoU is the true polygon IoU between the predicted rectangle and the ORIGINAL ground-truth quadrilateral;
  * matching is the VOC / DOTA-devkit rule: each detection is compared with its best-IoU ground truth (over ALL the class's
    GT in the image); IoU >= t and GT not ``difficult`` and not yet taken -> TP; taken -> FP; the GT is ``difficult`` -> the
    detection is IGNORED (neither TP nor FP); IoU < t -> FP;
  * AP is reported at IoU 0.5 both as VOC07 11-point (what DOTA's leaderboard uses) and all-point, and as the mean over
    IoU 0.50:0.05:0.95 (COCO-style, all-point).

Because the matching rule depends only on the score-sorted order, one pass yields the whole precision-recall curve, the AP at
every IoU threshold, precision / recall / F1 at any operating threshold, and size-stratified AP (GT outside a size bucket
are ignored like ``difficult``; unmatched detections outside the bucket are dropped, as in COCO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.951, 0.05), 2))
SIZE_BUCKETS_PX = ((0.0, 10.0), (10.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, float("inf")))   # by long side


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def xywhr_to_polys(xywhr: np.ndarray) -> np.ndarray:
    """(N,5) [cx, cy, w, h, angle_rad] -> (N,4,2) corners (Ultralytics' convention: angle rotates the w axis)."""
    xywhr = np.asarray(xywhr, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, r = xywhr.T
    c, s = np.cos(r), np.sin(r)
    ux, uy = c * w / 2, s * w / 2
    vx, vy = -s * h / 2, c * h / 2
    pts = np.stack([
        np.stack([cx - ux - vx, cy - uy - vy], -1), np.stack([cx + ux - vx, cy + uy - vy], -1),
        np.stack([cx + ux + vx, cy + uy + vy], -1), np.stack([cx - ux + vx, cy - uy + vy], -1)], axis=1)
    return pts


def long_side(polys: np.ndarray) -> np.ndarray:
    e = np.roll(polys, -1, axis=1) - polys
    l = np.hypot(e[..., 0], e[..., 1])
    return np.maximum(l[:, 0], l[:, 1])


def _convex_iou(a: np.ndarray, b: np.ndarray) -> float:
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    if inter <= 0:
        return 0.0
    aa, ab = abs(cv2.contourArea(a32)), abs(cv2.contourArea(b32))
    u = aa + ab - inter
    return float(inter / u) if u > 0 else 0.0


def _neighbours(a_polys: np.ndarray, b_polys: np.ndarray) -> list[np.ndarray]:
    """For each polygon in a, the indices of polygons in b whose centre is within the sum of half-diagonals."""
    if len(a_polys) == 0 or len(b_polys) == 0:
        return [np.zeros(0, dtype=np.int64) for _ in range(len(a_polys))]
    ca, cb = a_polys.mean(axis=1), b_polys.mean(axis=1)
    ra = np.linalg.norm(a_polys - ca[:, None, :], axis=2).max(axis=1)
    rb = np.linalg.norm(b_polys - cb[:, None, :], axis=2).max(axis=1)
    tree = cKDTree(cb)
    rmax = float(rb.max())
    out = []
    for i in range(len(a_polys)):
        idx = np.asarray(tree.query_ball_point(ca[i], ra[i] + rmax), dtype=np.int64)
        if len(idx):
            idx = idx[np.linalg.norm(cb[idx] - ca[i], axis=1) <= ra[i] + rb[idx]]
        out.append(idx)
    return out


# --------------------------------------------------------------------------
# data holders
# --------------------------------------------------------------------------


@dataclass
class ImageGT:
    stem: str
    classes: np.ndarray            # (M,) int
    quads: np.ndarray              # (M,4,2) original ground-truth quadrilaterals
    difficult: np.ndarray          # (M,) bool


@dataclass
class ImageDets:
    stem: str
    classes: np.ndarray            # (N,) int
    scores: np.ndarray             # (N,) float
    polys: np.ndarray              # (N,4,2) full-image coordinates
    chip_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # (N,) source chip, for cross-chip NMS

    def __post_init__(self):
        if len(self.chip_ids) != len(self.scores):
            self.chip_ids = np.full(len(self.scores), -1, dtype=np.int64)


def merge_cross_chip(dets: ImageDets, iou_thr: float = 0.5) -> ImageDets:
    """Per-class rotated NMS between detections from DIFFERENT chips (score order; higher score wins)."""
    n = len(dets.scores)
    if n == 0:
        return dets
    keep = np.ones(n, dtype=bool)
    for c in np.unique(dets.classes):
        idx = np.flatnonzero(dets.classes == c)
        idx = idx[np.argsort(-dets.scores[idx], kind="stable")]
        polys = dets.polys[idx]
        chips = dets.chip_ids[idx]
        neigh = _neighbours(polys, polys)
        alive = np.ones(len(idx), dtype=bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            for j in neigh[i]:
                if j <= i or not alive[j] or chips[j] == chips[i] or chips[i] < 0:
                    continue
                if _convex_iou(polys[i], polys[j]) > iou_thr:
                    alive[j] = False
        keep[idx[~alive]] = False
    return ImageDets(dets.stem, dets.classes[keep], dets.scores[keep], dets.polys[keep], dets.chip_ids[keep])


# --------------------------------------------------------------------------
# AP
# --------------------------------------------------------------------------


def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, npos: int, method: str) -> float:
    if npos == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / npos
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    if method == "voc07":                                   # 11-point (DOTA leaderboard)
        return float(np.mean([prec[rec >= t].max() if np.any(rec >= t) else 0.0 for t in np.arange(0.0, 1.1, 0.1)]))
    mrec = np.concatenate(([0.0], rec, [1.0]))              # all-point (VOC12 / area under the monotone PR curve)
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@dataclass
class ClassMatches:
    """Per-class, per-detection best-GT bookkeeping, independent of the IoU threshold."""
    scores: np.ndarray             # (D,)
    best_iou: np.ndarray           # (D,)
    gt_key: np.ndarray             # (D,) global id of the best GT (-1: none overlapping)
    gt_difficult: np.ndarray       # (D,) bool  best GT is difficult
    gt_size: np.ndarray            # (D,) long side of the best GT (nan if none)
    det_size: np.ndarray           # (D,) long side of the detection
    n_gt: int                      # non-difficult GT
    gt_sizes_all: np.ndarray       # (n_gt_total,) long side of every GT
    gt_difficult_all: np.ndarray   # (n_gt_total,) bool


def _image_matches(g: ImageGT | None, d: ImageDets | None, cls: int) -> dict:
    """Best-GT bookkeeping for ONE image and class. ``gt_local`` indexes this image's class-``cls`` GT (-1: none)."""
    gi = np.flatnonzero(g.classes == cls) if g is not None else np.zeros(0, dtype=np.int64)
    gq = g.quads[gi] if g is not None else np.zeros((0, 4, 2))
    gd = g.difficult[gi] if g is not None else np.zeros(0, dtype=bool)
    gs = long_side(gq) if len(gq) else np.zeros(0)
    di = np.flatnonzero(d.classes == cls) if d is not None else np.zeros(0, dtype=np.int64)
    n = len(di)
    out = {"scores": np.zeros(n), "best_iou": np.zeros(n), "gt_local": np.full(n, -1, dtype=np.int64),
           "gt_diff": np.zeros(n, dtype=bool), "gt_size": np.full(n, np.nan), "det_size": np.zeros(n),
           "gt_sizes": gs, "gt_diffs": gd}
    if n:
        dp = d.polys[di]
        neigh = _neighbours(dp, gq)
        out["det_size"] = long_side(dp)
        out["scores"] = d.scores[di].astype(float)
        for k in range(n):
            bi, bj = 0.0, -1
            for j in neigh[k]:
                v = _convex_iou(dp[k], gq[j])
                if v > bi:
                    bi, bj = v, int(j)
            out["best_iou"][k] = bi
            out["gt_local"][k] = bj
            if bj >= 0:
                out["gt_diff"][k] = bool(gd[bj])
                out["gt_size"][k] = float(gs[bj])
    return out


def _combine(parts: list[dict]) -> ClassMatches:
    """Concatenate per-image pieces; every copy of a piece gets its own GT-key range (so bootstrap duplicates stay distinct)."""
    keys, offset = [], 0
    for q in parts:
        keys.append(np.where(q["gt_local"] >= 0, q["gt_local"] + offset, -1))
        offset += len(q["gt_sizes"])
    cat = lambda k, dt=float: np.concatenate([q[k] for q in parts]).astype(dt) if parts else np.zeros(0, dtype=dt)
    gd_all = cat("gt_diffs", bool)
    return ClassMatches(
        scores=cat("scores"), best_iou=cat("best_iou"), gt_key=np.concatenate(keys).astype(np.int64) if keys else np.zeros(0, dtype=np.int64),
        gt_difficult=cat("gt_diff", bool), gt_size=cat("gt_size"), det_size=cat("det_size"),
        n_gt=int((~gd_all).sum()), gt_sizes_all=cat("gt_sizes"), gt_difficult_all=gd_all,
    )


def match_class_by_image(gt: dict[str, ImageGT], dets: dict[str, ImageDets], cls: int) -> list[dict]:
    return [_image_matches(gt.get(s), dets.get(s), cls) for s in sorted(set(gt) | set(dets))]


def match_class(gt: dict[str, ImageGT], dets: dict[str, ImageDets], cls: int) -> ClassMatches:
    return _combine(match_class_by_image(gt, dets, cls))


def bootstrap_ap(
    gt: dict[str, ImageGT], dets: dict[str, ImageDets], class_names: tuple[str, ...], groups: dict[str, tuple[str, ...]],
    *, n_boot: int = 100, seed: int = 0, level: float = 0.95,
) -> dict:
    """Percentile confidence intervals over IMAGES (the independent unit; chips of one photograph share pixels) for the
    per-class and per-group AP50 / AP50:95. The same resampled image set is used for every class in a replicate."""
    per_class_parts = [match_class_by_image(gt, dets, c) for c in range(len(class_names))]
    n_img = len(per_class_parts[0])
    rng = np.random.default_rng(seed)
    ap50 = np.full((n_boot, len(class_names)), np.nan)
    ap5095 = np.full((n_boot, len(class_names)), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, n_img, n_img)
        for c in range(len(class_names)):
            m = _combine([per_class_parts[c][i] for i in pick])
            r = class_ap(m)
            ap50[b, c], ap5095[b, c] = r["AP50"], r["AP50_95"]
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    ci = lambda a: [float("nan")] * 2 if np.all(np.isnan(a)) else [float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))]
    out = {"n_boot": n_boot, "n_images": n_img, "level": level, "per_class": {}, "groups": {}}
    for c, name in enumerate(class_names):
        out["per_class"][name] = {"AP50_ci": ci(ap50[:, c]), "AP50_95_ci": ci(ap5095[:, c])}
    for g, names in list(groups.items()) + [("all_kept_classes", tuple(class_names))]:
        idx = [class_names.index(n) for n in names]
        out["groups"][g] = {"macro_AP50_ci": ci(np.nanmean(ap50[:, idx], axis=1)), "macro_AP50_95_ci": ci(np.nanmean(ap5095[:, idx], axis=1))}
    return out


def _tp_fp(m: ClassMatches, thr: float, size_range: tuple[float, float] | None = None):
    """(tp, fp, npos, scores_sorted) for one IoU threshold, VOC/DOTA rule, optional size bucket."""
    order = np.argsort(-m.scores, kind="stable")
    sc, iou, key = m.scores[order], m.best_iou[order], m.gt_key[order]
    diff, gsz, dsz = m.gt_difficult[order], m.gt_size[order], m.det_size[order]

    hit = (iou >= thr) & (key >= 0)
    ignore = hit & diff                                        # matched a difficult GT -> ignored
    if size_range is not None:
        lo, hi = size_range
        in_g = (gsz >= lo) & (gsz < hi)
        in_d = (dsz >= lo) & (dsz < hi)
        ignore |= hit & ~in_g                                  # matched a GT of another size -> ignored
        ignore |= ~hit & ~in_d                                 # unmatched and itself outside the bucket -> dropped
        npos = int(np.sum(~m.gt_difficult_all & (m.gt_sizes_all >= lo) & (m.gt_sizes_all < hi)))
    else:
        npos = m.n_gt

    tp = np.zeros(len(sc), dtype=bool)
    cand = np.flatnonzero(hit & ~ignore)
    if len(cand):
        _, first = np.unique(key[cand], return_index=True)     # cand is score-ordered: first occurrence = highest score
        tp[cand[first]] = True
    fp = ~tp & ~ignore
    keep = ~ignore
    return tp[keep].astype(np.float64), fp[keep].astype(np.float64), npos, sc[keep]


def class_ap(m: ClassMatches, *, size_range=None) -> dict:
    out = {"n_gt": None}
    tp, fp, npos, _ = _tp_fp(m, 0.5, size_range)
    out["n_gt"] = npos
    out["AP50_voc07"] = _ap_from_pr(tp, fp, npos, "voc07")
    out["AP50"] = _ap_from_pr(tp, fp, npos, "all")
    aps = []
    for t in IOU_THRESHOLDS:
        tp, fp, npos, _ = _tp_fp(m, float(t), size_range)
        aps.append(_ap_from_pr(tp, fp, npos, "all"))
    out["AP50_95"] = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan")
    out["AP_by_iou"] = {str(t): a for t, a in zip(IOU_THRESHOLDS, aps)}
    return out


def operating_point(m: ClassMatches, score_thr: float, iou_thr: float = 0.5) -> dict:
    """Precision / recall / F1 counting only detections with score >= score_thr (difficult GT ignored)."""
    tp, fp, npos, sc = _tp_fp(m, iou_thr)
    sel = sc >= score_thr
    TP, FP = float(tp[sel].sum()), float(fp[sel].sum())
    P = TP / (TP + FP) if TP + FP > 0 else float("nan")
    R = TP / npos if npos > 0 else float("nan")
    F1 = 2 * P * R / (P + R) if P == P and R == R and P + R > 0 else float("nan")
    return {"score_thr": score_thr, "TP": int(TP), "FP": int(FP), "FN": int(npos - TP), "n_gt": npos,
            "precision": P, "recall": R, "f1": F1}


def pr_curve(m: ClassMatches, iou_thr: float = 0.5):
    """(recall, precision, scores) along the score-sorted detections (difficult GT ignored) - for plotting."""
    tp, fp, npos, sc = _tp_fp(m, iou_thr)
    if npos == 0 or len(sc) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    return ctp / npos, ctp / np.maximum(ctp + cfp, 1e-12), sc


def best_f1_threshold(m: ClassMatches, iou_thr: float = 0.5) -> tuple[float, float]:
    """(score threshold, F1) maximising F1 for one class."""
    tp, fp, npos, sc = _tp_fp(m, iou_thr)
    if npos == 0 or len(sc) == 0:
        return 0.25, float("nan")
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    p = ctp / np.maximum(ctp + cfp, 1e-12)
    r = ctp / npos
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1))
    return float(sc[i]), float(f1[i])


def evaluate_dataset(
    gt: dict[str, ImageGT], dets: dict[str, ImageDets], class_names: tuple[str, ...], *, size_buckets: bool = False,
) -> dict:
    """AP per class (+ optional size buckets). ``dets`` must already be cross-chip merged."""
    per_class: dict[str, dict] = {}
    matches: dict[str, ClassMatches] = {}
    for c, name in enumerate(class_names):
        m = match_class(gt, dets, c)
        matches[name] = m
        per_class[name] = class_ap(m)
        if size_buckets:
            per_class[name]["by_size_px"] = {
                f"{lo:g}-{hi:g}": class_ap(m, size_range=(lo, hi)) for lo, hi in SIZE_BUCKETS_PX}
    return {"per_class": per_class, "_matches": matches}


def group_summary(per_class: dict[str, dict], groups: dict[str, tuple[str, ...]]) -> dict:
    """Macro-average (over classes) and instance-weighted AP per reporting group + overall."""
    def agg(names):
        rows = [per_class[n] for n in names if per_class[n]["n_gt"]]
        if not rows:
            return {}
        w = np.array([r["n_gt"] for r in rows], dtype=float)
        out = {"classes": list(names), "n_gt": int(w.sum())}
        for k in ("AP50", "AP50_voc07", "AP50_95"):
            v = np.array([r[k] for r in rows], dtype=float)
            out[f"macro_{k}"] = float(np.nanmean(v))
            out[f"weighted_{k}"] = float(np.nansum(v * w) / w[~np.isnan(v)].sum())
        return out

    res = {g: agg(names) for g, names in groups.items()}
    res["all_kept_classes"] = agg(tuple(per_class))
    return res

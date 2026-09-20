"""Pure chip/box-clipping geometry for DOTA -> YOLO-OBB (no file I/O; unit-tested directly).

Chip size and overlap
-----------------------
CHIP_SIZE = 1024, matching geoseek.staging.download_maxar's own
MAXAR_TILE_SIZE (1024 px @ ~0.305 m/px, ~312 m per side). A detector trained on
these chips consumes tiles shaped identically to what the Maxar ingest already
produces - no resize / re-chip step at inference. 1024 is also the size the
Ultralytics DOTAv1 pretrained OBB weights were trained and benchmarked at, so
our numbers stay comparable to the published ones. (DOTA's own GSD varies
0.1-4.5 m, so 1024 px covers ~100 m to ~4.6 km of ground; the Maxar chip is
312 m - inside that range. The relevant scale match is object size in pixels,
not chip footprint, and that is a domain-shift risk we measure in Step 7 rather
than assume away.)

OVERLAP = 200 px (STRIDE = 824), the DOTA_devkit ``ImgSplit`` default and the
Ultralytics ``split_trainval`` default: an object cut by one chip boundary has
a real chance of being whole in the neighbouring chip, without ballooning the
image area we store and train on. A 200 px overlap covers a typical vehicle
(10-60 px), ship and plane (30-200 px) completely; only harbors and long bridges
can exceed it, and those are handled by the clipping rule below.

Boundary handling
-------------------
A box is clipped against the chip rectangle with shapely. Let ``frac`` be the
visible fraction of its original area:

  * frac == 1 (AABB entirely inside the chip)   -> original 4 points kept verbatim
  * VISIBILITY_THRESHOLD <= frac < 1            -> the visible region's minimum-area
                                                   rotated rectangle (cv2.minAreaRect)
                                                   becomes the new box. This is what
                                                   "re-projecting" a clipped OBB means:
                                                   re-fit to the visible pixels, NOT a
                                                   per-vertex clamp (which would shear a
                                                   rotated box).
  * frac <  VISIBILITY_THRESHOLD                -> dropped FROM THIS CHIP only.

Nothing is lost project-wide by the last rule as long as the 200 px overlap
puts the object >= 60% visible in some neighbour; the conversion counts the
instances for which that never happens (``n_instances_dropped``) and reports
them instead of hiding them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

from geoseek.detect.classes import CLASS_TO_INDEX

CHIP_SIZE = 1024
OVERLAP = 200
STRIDE = CHIP_SIZE - OVERLAP  # 824
VISIBILITY_THRESHOLD = 0.6
FULLY_VISIBLE_THRESHOLD = 0.999  # above this, keep the original points verbatim (no minAreaRect re-fit)
DISTRACTOR_OVERLAP = 0.5         # a dropped-class instance "is in" a window if >= this much of its AABB is


@dataclass(frozen=True)
class DotaInstance:
    points: tuple[tuple[float, float], ...]  # 4 (x, y) absolute-pixel points, as given in the label file
    category: str
    difficult: int


@dataclass(frozen=True)
class ChipLabel:
    class_index: int
    points: tuple[tuple[float, float], ...]  # chip-local pixel coordinates (4 points)
    visible_fraction: float
    difficult: int
    source_index: int                        # index into the image's KEPT-class instance list


@dataclass
class WindowPlan:
    x0: int
    y0: int
    labels: list[ChipLabel] = field(default_factory=list)
    n_distractors: int = 0                   # dropped-class instances (>= DISTRACTOR_OVERLAP inside) - hard-negative signal

    @property
    def is_positive(self) -> bool:
        return bool(self.labels)


@dataclass
class ImagePlan:
    width: int
    height: int
    windows: list[WindowPlan]
    n_kept_instances: int
    n_invalid_instances: int                 # zero-area polygons skipped
    best_visibility: np.ndarray              # per kept instance: max visible fraction over all windows

    @property
    def n_retained(self) -> int:
        return int(np.count_nonzero(self.best_visibility >= VISIBILITY_THRESHOLD))

    @property
    def n_dropped(self) -> int:
        return self.n_kept_instances - self.n_retained


# --------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------


def parse_dota_label_file(text: str) -> list[DotaInstance]:
    """Parse a DOTA labelTxt file's contents (v1.0 OBB format).

    Each instance line: 'x1 y1 x2 y2 x3 y3 x4 y4 category difficult'. The header
    lines (``imagesource:...``, ``gsd:...``) are skipped (they have < 10 fields).
    """
    instances: list[DotaInstance] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        coords = [float(v) for v in parts[:8]]
        points = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
        category = parts[8]
        difficult = int(parts[9]) if parts[9].isdigit() else 0
        instances.append(DotaInstance(points=points, category=category, difficult=difficult))
    return instances


def parse_gsd(text: str) -> float | None:
    """The ``gsd:`` header value in metres/pixel, or None when absent / 'null'."""
    for line in text.splitlines()[:4]:
        if line.startswith("gsd:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------
# Windows and pixels
# --------------------------------------------------------------------------


def chip_origins(img_w: int, img_h: int, chip_size: int = CHIP_SIZE, stride: int = STRIDE) -> list[tuple[int, int]]:
    """(x0, y0) top-left origins covering the image with the given chip size/stride.

    The final chip in each row/column is snapped back so it still ends exactly
    at the image edge (never runs past it), matching DOTA_devkit ImgSplit.
    Images smaller than one chip get a single origin at (0, 0) - the caller pads.
    """

    def axis_origins(dim: int) -> list[int]:
        if dim <= chip_size:
            return [0]
        origins = list(range(0, dim - chip_size + 1, stride))
        last = dim - chip_size
        if origins[-1] != last:
            origins.append(last)
        return origins

    xs = axis_origins(img_w)
    ys = axis_origins(img_h)
    return [(x, y) for y in ys for x in xs]


def crop_chip(image: np.ndarray, x0: int, y0: int, chip_size: int = CHIP_SIZE) -> np.ndarray:
    """Crop a chip_size x chip_size window from ``image`` at (x0, y0), zero-padding past the edge."""
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else None
    shape = (chip_size, chip_size, channels) if channels else (chip_size, chip_size)
    chip = np.zeros(shape, dtype=image.dtype)

    x1, y1 = min(x0 + chip_size, w), min(y0 + chip_size, h)
    src = image[y0:y1, x0:x1]
    chip[: src.shape[0], : src.shape[1]] = src
    return chip


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------


def safe_polygon(points) -> Polygon:
    """A valid shapely Polygon for a DOTA quad. A quad with swapped vertices
    self-intersects; its convex hull is the rectangle that was meant."""
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.convex_hull
    return poly


def visibility_and_clip(
    poly: Polygon,
    instance_points: tuple[tuple[float, float], ...],
    x0: int,
    y0: int,
    chip_size: int = CHIP_SIZE,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
) -> tuple[float, tuple[tuple[float, float], ...] | None]:
    """One shapely intersection -> (visible_fraction, chip-local clipped points or None).

    ``poly`` is the instance's pre-built polygon (build once per instance, not
    once per instance-chip pair). Returns (0.0, None) for a degenerate or
    non-overlapping polygon; (frac, None) when frac is below the threshold
    (visible somewhere, just not enough to keep in THIS chip); (frac, points)
    when the instance is kept.
    """
    if not poly.is_valid or poly.area <= 0:
        return 0.0, None

    intersection = poly.intersection(shapely_box(x0, y0, x0 + chip_size, y0 + chip_size))
    if intersection.is_empty or intersection.area <= 0:
        return 0.0, None

    frac = intersection.area / poly.area
    if frac < visibility_threshold:
        return frac, None

    if frac >= FULLY_VISIBLE_THRESHOLD:
        pts = instance_points
    else:
        piece = intersection
        if piece.geom_type != "Polygon":
            # a clipped rectangle can degenerate to a MultiPolygon / collection; keep the largest polygon
            polys = [g for g in getattr(piece, "geoms", []) if g.geom_type == "Polygon"]
            if not polys:
                return frac, None
            piece = max(polys, key=lambda g: g.area)
        coords = np.array(piece.exterior.coords[:-1], dtype=np.float32)
        rect = cv2.minAreaRect(coords)
        pts = tuple(map(tuple, cv2.boxPoints(rect)))

    return frac, tuple((float(px - x0), float(py - y0)) for px, py in pts)


def clip_instance_to_chip(
    instance: DotaInstance,
    x0: int,
    y0: int,
    chip_size: int = CHIP_SIZE,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
) -> tuple[tuple[float, float], ...] | None:
    """Convenience wrapper over :func:`visibility_and_clip` for one-off / test use."""
    _frac, pts = visibility_and_clip(
        safe_polygon(instance.points), instance.points, x0, y0, chip_size, visibility_threshold
    )
    return pts


def _shoelace_area(pts: np.ndarray) -> np.ndarray:
    """Polygon areas for an (N, 4, 2) array."""
    x, y = pts[:, :, 0], pts[:, :, 1]
    return 0.5 * np.abs((x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1))


def plan_windows(
    width: int,
    height: int,
    instances: list[DotaInstance],
    *,
    class_index: dict[str, int] = CLASS_TO_INDEX,
    chip_size: int = CHIP_SIZE,
    stride: int = STRIDE,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
) -> ImagePlan:
    """Plan every window of one image: which kept-class boxes land in it (clipped,
    chip-local) and how many dropped-class distractors it holds. Pure geometry -
    the image is never read."""
    kept = [inst for inst in instances if inst.category in class_index]
    dropped = [inst for inst in instances if inst.category not in class_index]

    K = len(kept)
    if K:
        kp = np.asarray([inst.points for inst in kept], dtype=np.float64).reshape(K, 4, 2)
        kminx, kmaxx = kp[:, :, 0].min(axis=1), kp[:, :, 0].max(axis=1)
        kminy, kmaxy = kp[:, :, 1].min(axis=1), kp[:, :, 1].max(axis=1)
        valid = _shoelace_area(kp) > 1e-6
    else:
        kminx = kmaxx = kminy = kmaxy = np.zeros(0)
        valid = np.zeros(0, dtype=bool)

    if dropped:
        dp = np.asarray([inst.points for inst in dropped], dtype=np.float64).reshape(len(dropped), 4, 2)
        dminx, dmaxx = dp[:, :, 0].min(axis=1), dp[:, :, 0].max(axis=1)
        dminy, dmaxy = dp[:, :, 1].min(axis=1), dp[:, :, 1].max(axis=1)
        darea = np.maximum((dmaxx - dminx) * (dmaxy - dminy), 1e-6)
    else:
        dminx = dmaxx = dminy = dmaxy = darea = np.zeros(0)

    polys: dict[int, Polygon] = {}
    best_vis = np.zeros(K, dtype=np.float64)
    windows: list[WindowPlan] = []

    for (x0, y0) in chip_origins(width, height, chip_size, stride):
        x1, y1 = x0 + chip_size, y0 + chip_size
        plan = WindowPlan(x0=x0, y0=y0)

        if K:
            cand = np.flatnonzero(valid & (kmaxx > x0) & (kminx < x1) & (kmaxy > y0) & (kminy < y1))
            for k in cand:
                inst = kept[k]
                inside = kminx[k] >= x0 and kmaxx[k] <= x1 and kminy[k] >= y0 and kmaxy[k] <= y1
                if inside:
                    frac = 1.0
                    local = tuple((px - x0, py - y0) for px, py in inst.points)
                else:
                    poly = polys.get(int(k))
                    if poly is None:
                        poly = polys[int(k)] = safe_polygon(inst.points)
                    frac, local = visibility_and_clip(poly, inst.points, x0, y0, chip_size, visibility_threshold)
                best_vis[k] = max(best_vis[k], frac)
                if local is not None:
                    plan.labels.append(ChipLabel(
                        class_index=class_index[inst.category], points=local,
                        visible_fraction=float(frac), difficult=inst.difficult, source_index=int(k),
                    ))

        if len(darea):
            ix = np.clip(np.minimum(dmaxx, x1) - np.maximum(dminx, x0), 0, None)
            iy = np.clip(np.minimum(dmaxy, y1) - np.maximum(dminy, y0), 0, None)
            plan.n_distractors = int(np.count_nonzero(ix * iy / darea >= DISTRACTOR_OVERLAP))

        windows.append(plan)

    return ImagePlan(
        width=width, height=height, windows=windows, n_kept_instances=K,
        n_invalid_instances=int(K - np.count_nonzero(valid)) if K else 0, best_visibility=best_vis,
    )


# --------------------------------------------------------------------------
# Normalisation / label text
# --------------------------------------------------------------------------


def normalize_points(points: tuple[tuple[float, float], ...], chip_size: int = CHIP_SIZE) -> tuple[float, ...]:
    """Chip-local absolute-pixel points -> flat (x1,y1,...,x4,y4) normalised to [0, 1], clamped."""
    flat: list[float] = []
    for px, py in points:
        flat.append(min(max(px / chip_size, 0.0), 1.0))
        flat.append(min(max(py / chip_size, 0.0), 1.0))
    return tuple(flat)


def format_yolo_obb_line(class_index: int, norm_points: tuple[float, ...]) -> str:
    coords = " ".join(f"{v:.6f}" for v in norm_points)
    return f"{class_index} {coords}"


# --------------------------------------------------------------------------
# Orientation statistics - the "are these boxes really rotated?" check
# --------------------------------------------------------------------------


def long_edge_angle_deg(points) -> float:
    """Direction of the polygon's longest edge, in [0, 180) degrees (image coords, y down)."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    dx, dy = edges[int(np.argmax(lengths))]
    return math.degrees(math.atan2(dy, dx)) % 180.0


def axis_deviation_deg(points) -> float:
    """How far the box is from axis-aligned: 0 for a horizontal/vertical rectangle, up to 45."""
    a = long_edge_angle_deg(points) % 90.0
    return min(a, 90.0 - a)


def is_axis_aligned(points, tol_px: float = 1e-3) -> bool:
    """True iff the 4 points form an axis-parallel rectangle (2 distinct x and 2 distinct y values)."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    return len(set(np.round(pts[:, 0] / max(tol_px, 1e-9)).astype(np.int64))) == 2 and \
        len(set(np.round(pts[:, 1] / max(tol_px, 1e-9)).astype(np.int64))) == 2

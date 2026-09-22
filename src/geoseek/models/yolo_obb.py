"""YOLO-OBB implementation of :class:`geoseek.models.base.ObjectDetectionModel` (Phase 8F-2).

This is the ONLY module that touches ``ultralytics`` (AGPL-3.0, network-use clause included), and only lazily inside
:meth:`YoloObbDetectionModel.load` - importing geoseek, or this module, never imports it. The weights it runs are the
fine-tuned checkpoint produced by ``scripts/train_detector.py`` (DOTAv1-pretrained YOLO26s-OBB, 8 classes); they inherit
the AGPL-3.0 licence of the pretrained weights and DOTA's academic-use-only terms - see ``docs/PHASE8F2.md``.

Behaviour worth knowing:

  * RGB in, BGR to ultralytics. ultralytics treats ndarray inputs as OpenCV-style BGR and flips them to RGB itself; our
    tiles are RGB, so they are flipped once here. (Feeding RGB straight through would silently swap the R and B channels.)
  * Tiles larger than ``imgsz`` are run as overlapping windows (same 200 px overlap as training) and de-duplicated across
    windows with a per-class rotated NMS - the same merge the full-image evaluation uses.
  * The default score threshold is PER-CLASS, read from the model card (``<weights>.card.json`` ->
    ``operating_point.per_class.<class>.conf``): vehicles (small-vehicle, large-vehicle) are tuned on xView (an
    independent, never-trained-on test set); the other 6 classes are tuned on the *monitor* split (never the official
    val, never Van Nuys). A single global threshold (the old ``operating_point.conf``) was confirmed too conservative
    for vehicles specifically - see ``operating_point.deprecated_global_operating_point`` in the card for why. A class
    with no entry falls back to ``DEFAULT_MIN_SCORE``. Internally this means inference runs at the LOWEST per-class
    threshold in effect (so no class's candidates are prematurely dropped before NMS), then each surviving detection
    is filtered by its OWN class's threshold - the same two-stage pattern the evaluation scripts already use.
  * ``upscale`` (default 1.0 = off) resamples the tile by that factor (cubic) before inference and reports every box back in
    ORIGINAL tile pixels (and, with a TileGeoRef, at the same place on the ground). It exists because the detector is
    much weaker on ~15 px objects than on 25-30 px ones (docs/PHASE8F2.md sec. 6-7); it is off by default because its effect
    on the Maxar tiles has only been checked by eye, not against ground truth.
  * Runs offline: ``YOLO_OFFLINE=1`` is set before ultralytics is imported.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from geoseek.models.base import Detection, ObjectDetectionModel, TileGeoRef

DEFAULT_MIN_SCORE = 0.25
OVERLAP_PX = 200
CROSS_WINDOW_MERGE_IOU = 0.3      # same de-duplication threshold as the full-image evaluation (DOTA devkit standard)


def _polys_from_xywhr(xywhr: np.ndarray) -> np.ndarray:
    from geoseek.detect.evaluate import xywhr_to_polys

    return xywhr_to_polys(xywhr)


class YoloObbDetectionModel(ObjectDetectionModel):
    def __init__(
        self,
        weights_path: str | Path,
        *,
        min_score: float | dict[str, float] | None = None,
        iou: float = 0.7,
        max_det: int = 1000,
        imgsz: int = 1024,
        device: str | None = None,
        half: bool | None = None,
        batch_size: int = 4,
        upscale: float = 1.0,
    ):
        if not upscale > 0:
            raise ValueError(f"upscale must be > 0, got {upscale}")
        self.upscale = float(upscale)
        self.weights_path = Path(weights_path)
        self.card = self._read_card(self.weights_path)
        self._min_score_override = min_score
        self.iou, self.max_det, self.imgsz, self.batch_size = iou, max_det, imgsz, batch_size
        self._device, self._half = device, half
        self._model = None
        self._names: tuple[str, ...] | None = None

    # -- interface -----------------------------------------------------------------------------------------

    @property
    def class_names(self) -> tuple[str, ...]:
        if self._names is None:
            if self.card and self.card.get("classes"):
                self._names = tuple(self.card["classes"])
            else:
                self.load()
        return self._names

    @property
    def min_score(self) -> dict[str, float]:
        """Per-class confidence thresholds, resolved lazily (never forces a model load just to read this):

        - an explicit dict at construction is used as-is;
        - an explicit float applies uniformly to every class already known (the card's ``classes`` list, or
          ``class_names`` if the underlying model happens to be loaded already) - {} if none are known yet;
        - otherwise the card's ``operating_point.per_class`` (current schema: each class tuned independently -
          vehicles on xView, everything else on the DOTA monitor split - see the module docstring) or
          ``operating_point.conf`` (legacy single value, applied uniformly);
        - a class with no entry either way falls back to :data:`DEFAULT_MIN_SCORE` at lookup time, not here.
        """
        override, card = self._min_score_override, self.card
        known_classes = list((card or {}).get("classes") or (self._names or []))
        if isinstance(override, dict):
            return {k: float(v) for k, v in override.items()}
        if isinstance(override, (int, float)):
            return {c: float(override) for c in known_classes}
        op = (card or {}).get("operating_point") or {}
        per_class = op.get("per_class")
        if per_class:
            return {c: float(v["conf"]) for c, v in per_class.items()}
        if "conf" in op:
            return {c: float(op["conf"]) for c in known_classes}
        return {}

    @property
    def info(self) -> dict:
        return {
            "model": "YoloObbDetectionModel", "weights": str(self.weights_path),
            "weights_sha256": (self.card or {}).get("weights_sha256"), "architecture": (self.card or {}).get("architecture"),
            "classes": list(self.class_names), "min_score_default": dict(self.min_score), "nms_iou": self.iou,
            "max_det": self.max_det, "imgsz": self.imgsz, "upscale": self.upscale,
        }

    def _effective_min_score_map(self, override: float | dict[str, float] | None) -> dict[str, float]:
        """A complete {class_name: threshold} map for one call, filling any gap with DEFAULT_MIN_SCORE."""
        names = self.class_names
        if isinstance(override, dict):
            base = override
        elif isinstance(override, (int, float)):
            return {c: float(override) for c in names}
        else:
            base = self.min_score
        return {c: float(base.get(c, DEFAULT_MIN_SCORE)) for c in names}

    def load(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("YOLO_OFFLINE", "1")                  # never touch the network at inference time
        from ultralytics import YOLO                                  # AGPL-3.0: imported here and nowhere else

        self._model = YOLO(str(self.weights_path))
        names = self._model.names
        self._names = tuple(names[i] for i in sorted(names))
        if self._device is None:
            import torch

            self._device = 0 if torch.cuda.is_available() else "cpu"
        if self._half is None:
            self._half = self._device != "cpu"

    def detect(
        self,
        tile_rgb_uint8: np.ndarray,
        *,
        classes: Sequence[str] | None = None,
        min_score: float | dict[str, float] | None = None,
        geo: TileGeoRef | None = None,
    ) -> list[Detection]:
        return self.detect_batch([tile_rgb_uint8], classes=classes, min_score=min_score, geo=geo)[0]

    def detect_batch(self, tiles, *, classes=None, min_score=None, geo=None, geos=None) -> list[list[Detection]]:
        """True batching for tiles that fit in one window; larger tiles fall back to windowed inference.
        ``geos`` (one TileGeoRef per tile) takes precedence over ``geo`` for batches of different tiles.
        ``min_score`` overrides the model's default PER-CLASS thresholds for this call: a float applies uniformly,
        a dict is used as a per-class override for this call only (any class it omits falls back to
        :data:`DEFAULT_MIN_SCORE`, not the model's own stored default - an override replaces the whole map, it
        doesn't patch it)."""
        self.load()
        eff_map = self._effective_min_score_map(min_score)
        floor = min(eff_map.values()) if eff_map else DEFAULT_MIN_SCORE
        wanted = None if classes is None else {c for c in classes}
        if wanted is not None:
            unknown = wanted - set(self.class_names)
            if unknown:
                raise ValueError(f"unknown class(es) {sorted(unknown)}; this model emits {list(self.class_names)}")

        if self.upscale != 1.0:
            return self._detect_batch_upscaled(tiles, floor, eff_map, wanted, geo, geos)
        return self._detect_batch_native(tiles, floor, eff_map, wanted, geo, geos)

    # -- internals -----------------------------------------------------------------------------------------

    def _detect_batch_upscaled(self, tiles, floor, eff_map, wanted, geo, geos) -> list[list[Detection]]:
        import cv2

        s = self.upscale
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        big = [cv2.resize(t, None, fx=s, fy=s, interpolation=interp) for t in tiles]
        geos_s = [_scale_geo(geos[i] if geos else geo, s) for i in range(len(tiles))]
        inner = self._detect_batch_native(big, floor, eff_map, wanted, None, geos_s)
        return [[_rescale_detection(d, s) for d in dets] for dets in inner]

    def _detect_batch_native(self, tiles, floor, eff_map, wanted, geo, geos) -> list[list[Detection]]:
        out: list[list[Detection] | None] = [None] * len(tiles)
        small = [i for i, t in enumerate(tiles) if max(t.shape[:2]) <= self.imgsz]
        for start in range(0, len(small), self.batch_size):
            idxs = small[start:start + self.batch_size]
            raw = self._predict([tiles[i] for i in idxs], floor)
            for i, (xywhr, conf, cls) in zip(idxs, raw):
                out[i] = self._to_detections(xywhr, conf, cls, wanted, (geos[i] if geos else geo), eff_map)
        for i, t in enumerate(tiles):
            if out[i] is None:
                out[i] = self._detect_large(t, floor, eff_map, wanted, (geos[i] if geos else geo))
        return out  # type: ignore[return-value]

    @staticmethod
    def _read_card(weights_path: Path) -> dict | None:
        card = weights_path.with_suffix(".card.json")
        if card.is_file():
            try:
                return json.loads(card.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _predict(self, tiles_rgb: list[np.ndarray], thr: float):
        """-> per tile (xywhr (N,5) float64, conf (N,), cls (N,) int), tile pixel coordinates."""
        bgr = [np.ascontiguousarray(t[..., ::-1]) for t in tiles_rgb]            # RGB -> BGR (ultralytics ndarray convention)
        extra = {"quantize": 16} if self._half else {}                            # fp16 ('half=' is deprecated in ultralytics 8.4.x)
        results = self._model.predict(
            bgr, imgsz=self.imgsz, conf=thr, iou=self.iou, max_det=self.max_det, device=self._device,
            verbose=False, augment=False, **extra,
        )
        out = []
        for r in results:
            if r.obb is None or len(r.obb) == 0:
                out.append((np.zeros((0, 5)), np.zeros(0), np.zeros(0, dtype=int)))
                continue
            out.append((r.obb.xywhr.cpu().numpy().astype(np.float64), r.obb.conf.cpu().numpy().astype(np.float64),
                        r.obb.cls.cpu().numpy().astype(int)))
        return out

    def _detect_large(self, tile: np.ndarray, floor: float, eff_map: dict[str, float], wanted, geo) -> list[Detection]:
        from geoseek.detect.chipping import chip_origins
        from geoseek.detect.evaluate import ImageDets, merge_cross_chip

        h, w = tile.shape[:2]
        origins = chip_origins(w, h, self.imgsz, self.imgsz - OVERLAP_PX)
        polys_l, conf_l, cls_l, chip_l = [], [], [], []
        for k in range(0, len(origins), self.batch_size):
            group = origins[k:k + self.batch_size]
            crops = [tile[y0:y0 + self.imgsz, x0:x0 + self.imgsz] for x0, y0 in group]
            for (x0, y0), (xywhr, conf, cls) in zip(group, self._predict(crops, floor)):
                if len(conf):
                    xywhr = xywhr.copy()
                    xywhr[:, 0] += x0
                    xywhr[:, 1] += y0
                    polys_l.append(xywhr)
                    conf_l.append(conf)
                    cls_l.append(cls)
                    chip_l.append(np.full(len(conf), len(chip_l)))
        if not conf_l:
            return []
        xywhr, conf, cls, chip = (np.concatenate(a) for a in (polys_l, conf_l, cls_l, chip_l))
        # cross-window de-duplication happens on ALL floor-level candidates - the per-class threshold is
        # applied afterwards in _to_detections, same ordering as the full-image evaluation pipeline
        merged = merge_cross_chip(ImageDets("tile", cls, conf, _polys_from_xywhr(xywhr), chip), CROSS_WINDOW_MERGE_IOU)
        # re-derive xywhr for the survivors by matching polygons back to their source rows
        keep = np.array([int(np.argmin(np.abs(_polys_from_xywhr(xywhr) - p).sum(axis=(1, 2)))) for p in merged.polys])
        return self._to_detections(xywhr[keep], conf[keep], cls[keep], wanted, geo, eff_map)

    def _to_detections(self, xywhr, conf, cls, wanted, geo, eff_map: dict[str, float]) -> list[Detection]:
        if len(conf) == 0:
            return []
        names = self.class_names
        polys = _polys_from_xywhr(xywhr)
        wkts = _polys_to_wkt_4326(polys, geo) if geo is not None else [None] * len(conf)
        dets = []
        for i in range(len(conf)):
            name = names[int(cls[i])]
            if wanted is not None and name not in wanted:
                continue
            if float(conf[i]) < eff_map.get(name, DEFAULT_MIN_SCORE):        # per-class threshold, applied post-NMS
                continue
            cx, cy, w, h, r = (float(v) for v in xywhr[i])
            dets.append(Detection(
                class_name=name, class_id=int(cls[i]), score=float(conf[i]), obb_px=(cx, cy, w, h, r),
                polygon_px=tuple((float(x), float(y)) for x, y in polys[i]), geom_wkt_4326=wkts[i]))
        dets.sort(key=lambda d: -d.score)
        return dets


def _scale_geo(geo: TileGeoRef | None, s: float) -> TileGeoRef | None:
    """The affine of the tile after it was resampled by ``s`` (a pixel of the big tile is 1/s of an original pixel)."""
    if geo is None:
        return None
    a, b, c, d, e, f = geo.transform
    return TileGeoRef(transform=(a / s, b / s, c, d / s, e / s, f), crs=geo.crs)


def _rescale_detection(d: Detection, s: float) -> Detection:
    """A detection found on the ``s``-times resampled tile, expressed in ORIGINAL tile pixels (the footprint is unchanged)."""
    cx, cy, w, h, r = d.obb_px
    return Detection(class_name=d.class_name, class_id=d.class_id, score=d.score, obb_px=(cx / s, cy / s, w / s, h / s, r),
                     polygon_px=tuple((x / s, y / s) for x, y in d.polygon_px), geom_wkt_4326=d.geom_wkt_4326)


_TRANSFORMERS: dict[str, object] = {}


def _polys_to_wkt_4326(polys: np.ndarray, geo: TileGeoRef) -> list[str]:
    """Tile-pixel quads -> WKT POLYGON (lon lat) via the tile's affine + CRS."""
    import pyproj

    tr = _TRANSFORMERS.get(geo.crs)
    if tr is None:
        tr = _TRANSFORMERS[geo.crs] = pyproj.Transformer.from_crs(geo.crs, "EPSG:4326", always_xy=True)
    a, b, c, d, e, f = geo.transform
    out = []
    for poly in polys:
        x = a * poly[:, 0] + b * poly[:, 1] + c
        y = d * poly[:, 0] + e * poly[:, 1] + f
        lon, lat = tr.transform(x, y)
        ring = list(zip(lon, lat)) + [(lon[0], lat[0])]
        out.append("POLYGON ((" + ", ".join(f"{px:.7f} {py:.7f}" for px, py in ring) + "))")
    return out

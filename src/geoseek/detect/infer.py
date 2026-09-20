"""Chip-level inference for evaluation: run a YOLO-OBB checkpoint over chip files, return flat detection arrays.

Used by ``scripts/eval_detector.py`` (official-val / monitor evaluation) - NOT the deployment path (that is
:class:`geoseek.models.yolo_obb.YoloObbDetectionModel`). Images are read by ultralytics itself (OpenCV, BGR) exactly as in
training, so there is no channel-order ambiguity here.

``class_map`` remaps predicted class indices and DROPS everything unmapped: with it the 15-class DOTA-pretrained checkpoint
is scored on our 8 classes by class NAME (:func:`geoseek.detect.classes.dota15_index_to_ours`). Dropping classes after the
detector's own (class-aware) NMS is equivalent to never having predicted them.

ultralytics (AGPL-3.0) is imported lazily.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


def predict_chips(
    weights: Path,
    chip_paths: list[Path],
    *,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 2500,
    imgsz: int = 1024,
    batch: int = 8,
    device: int | str = 0,
    half: bool = True,
    class_map: dict[int, int] | None = None,
    nms_free: bool = False,
    log_every: int = 500,
) -> dict:
    """-> {"chip_ids": [...], "det_chip_idx": (N,), "xywhr": (N,5), "conf": (N,), "cls": (N,), "seconds": float}"""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False, augment=False)
    if half:
        kw["quantize"] = 16                                # fp16 ('half=' is deprecated in ultralytics 8.4.x)
    if nms_free:
        kw["nms"] = False                                # NMS-free one-to-one head; default (omitted) = classic head + NMS,
                                                         # which scored higher on dense scenes at epoch 0 (0.821 vs 0.810)
    chip_ids = [Path(p).stem for p in chip_paths]
    idx_l, xywhr_l, conf_l, cls_l = [], [], [], []
    t0 = time.time()
    # Batch MANUALLY. With a LIST source, ultralytics 8.4.152 ignores ``batch=`` and stacks the whole list into one tensor
    # (1,476 chips -> a 23 GiB conv allocation); it is invisible on short lists, so it must not be relied on.
    for start in range(0, len(chip_paths), batch):
        group = [str(p) for p in chip_paths[start:start + batch]]
        for j, r in enumerate(model.predict(group, **kw)):
            i = start + j
            if r.obb is not None and len(r.obb):
                xywhr = r.obb.xywhr.cpu().numpy().astype(np.float64)
                c = r.obb.conf.cpu().numpy().astype(np.float64)
                k = r.obb.cls.cpu().numpy().astype(int)
                if class_map is not None:
                    keep = np.array([kk in class_map for kk in k], dtype=bool)
                    xywhr, c, k = xywhr[keep], c[keep], np.array([class_map[kk] for kk in k[keep]], dtype=int)
                if len(c):
                    idx_l.append(np.full(len(c), i, dtype=np.int64))
                    xywhr_l.append(xywhr)
                    conf_l.append(c)
                    cls_l.append(k)
        if log_every and ((start + batch) // batch) % max(log_every // batch, 1) == 0:
            print(f"[infer]   {min(start + batch, len(chip_paths))}/{len(chip_paths)} chips  {time.time() - t0:.0f}s", flush=True)
    cat = lambda l, shape: np.concatenate(l) if l else np.zeros(shape)
    return {
        "chip_ids": chip_ids, "det_chip_idx": cat(idx_l, (0,)).astype(np.int64), "xywhr": cat(xywhr_l, (0, 5)),
        "conf": cat(conf_l, (0,)), "cls": cat(cls_l, (0,)).astype(int), "seconds": time.time() - t0,
    }


def save_predictions(path: Path, pred: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chip_ids=np.array(pred["chip_ids"]), det_chip_idx=pred["det_chip_idx"], xywhr=pred["xywhr"],
                        conf=pred["conf"], cls=pred["cls"], seconds=np.array(pred["seconds"]))


def load_predictions(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {"chip_ids": [str(s) for s in z["chip_ids"]], "det_chip_idx": z["det_chip_idx"], "xywhr": z["xywhr"],
            "conf": z["conf"], "cls": z["cls"], "seconds": float(z["seconds"])}

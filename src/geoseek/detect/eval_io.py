"""Glue between chip-level detections and the full-image evaluator (:mod:`geoseek.detect.evaluate`). numpy only.

  parse_chip_id            "P0001__824_0"           -> ("P0001", 824, 0)
  load_dota_gt             labelTxt dir             -> {stem: ImageGT}   original quadrilaterals + difficult flags
  assemble_full_image_dets raw per-chip arrays      -> {stem: ImageDets} full-image coordinates, chip ids kept

Ground truth is read from the ORIGINAL DOTA label files (not from the chip labels, which are rectangles fitted to the
quads and clipped at chip edges), for the KEPT classes only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geoseek.detect.chipping import parse_dota_label_file
from geoseek.detect.classes import CLASS_TO_INDEX
from geoseek.detect.evaluate import ImageDets, ImageGT, xywhr_to_polys


def parse_chip_id(chip_id: str) -> tuple[str, int, int]:
    stem, rest = chip_id.split("__")
    x0, y0 = rest.split("_")
    return stem, int(x0), int(y0)


def load_dota_gt(label_dir: Path, stems: set[str] | None = None) -> dict[str, ImageGT]:
    out: dict[str, ImageGT] = {}
    for f in sorted(Path(label_dir).glob("P*.txt")):
        if stems is not None and f.stem not in stems:
            continue
        inst = [i for i in parse_dota_label_file(f.read_text(encoding="utf-8", errors="replace"))
                if i.category in CLASS_TO_INDEX]
        if inst:
            out[f.stem] = ImageGT(
                f.stem, np.array([CLASS_TO_INDEX[i.category] for i in inst], dtype=int),
                np.array([i.points for i in inst], dtype=np.float64).reshape(len(inst), 4, 2),
                np.array([bool(i.difficult) for i in inst]))
        else:
            out[f.stem] = ImageGT(f.stem, np.zeros(0, int), np.zeros((0, 4, 2)), np.zeros(0, bool))
    return out


def assemble_full_image_dets(
    chip_ids: list[str], det_chip_idx: np.ndarray, xywhr: np.ndarray, conf: np.ndarray, cls: np.ndarray,
) -> dict[str, ImageDets]:
    """Move chip-local detections to full-image coordinates. ``det_chip_idx[i]`` indexes ``chip_ids``. No cross-chip
    de-duplication here - call :func:`geoseek.detect.evaluate.merge_cross_chip` on each result."""
    origins = np.array([parse_chip_id(c)[1:] for c in chip_ids], dtype=np.float64).reshape(-1, 2)
    stems = np.array([parse_chip_id(c)[0] for c in chip_ids])
    out: dict[str, ImageDets] = {}
    if len(conf) == 0:
        return out
    shifted = np.asarray(xywhr, dtype=np.float64).copy()
    shifted[:, 0] += origins[det_chip_idx, 0]
    shifted[:, 1] += origins[det_chip_idx, 1]
    polys = xywhr_to_polys(shifted)
    det_stems = stems[det_chip_idx]
    for stem in np.unique(det_stems):
        m = det_stems == stem
        out[str(stem)] = ImageDets(str(stem), np.asarray(cls)[m].astype(int), np.asarray(conf)[m].astype(float),
                                   polys[m], np.asarray(det_chip_idx)[m].astype(np.int64))
    return out

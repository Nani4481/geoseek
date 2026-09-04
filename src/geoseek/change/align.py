"""Pairwise alignment summary: co-registration + relative radiometric offset.

A light wrapper over :mod:`geoseek.change.coregister` and
:mod:`geoseek.change.normalize` that produces the ``coregistration`` /
``radiometric_normalization`` provenance blocks for ONE subject observation
against a reference observation - the same shapes ``geoseek.change.prep`` writes
for the original pair, but computed over a window (no acceptance gate, no
rasters). Used to attach alignment provenance to a newly-staged third date so
:class:`geoseek.temporal.matcher.TemporalObservationMatcher` has a real
co-registration status for the new pairs.

Offline: reads only from ``data/datasets/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from geoseek.change.coregister import estimate_pair_shift
from geoseek.change.normalize import (
    NORM_BANDS,
    RadiometricNormalization,
    iterate_pif_normalization,
    select_pif_mask,
)

_PIF_BANDS = ("B04", "B03", "B02", "B08")


def _read_window(scene_dir: Path, bands, win: Window) -> dict[str, np.ndarray]:
    out = {}
    for b in bands:
        with rasterio.open(scene_dir / f"{b}.tif") as ds:
            out[b] = ds.read(1, window=win)
    return out


def summarize_pair_alignment(
    subject_dir: Path | str,
    reference_dir: Path | str,
    *,
    reference_observation_id: str,
    subject_observation_id: str,
    subject_date: str,
    reference_date: str,
    window: tuple[int, int, int, int] = (1500, 1500, 5000, 5000),
) -> dict:
    """``{"coregistration": {...}, "radiometric_normalization": {...}}`` for subject vs reference.

    ``window`` is ``(col_off, row_off, width, height)`` on the 10 m grid - a large
    interior box gives stable PIF statistics without loading the full 82 km scene.
    """
    subject_dir, reference_dir = Path(subject_dir), Path(reference_dir)

    # -- co-registration (phase correlation on B08) -----------------------------
    res = estimate_pair_shift(reference_dir, subject_dir, band="B08", tile_px=1024, grid=3)
    coreg = res.to_manifest_dict()
    # link it to the reference OBSERVATION id so the matcher can resolve the pair
    coreg["reference_scene"] = reference_observation_id
    coreg["moving_scene"] = subject_observation_id
    coreg["subject_date"] = subject_date
    coreg["reference_date"] = reference_date

    # -- relative radiometric normalization (PIF additive, windowed) -----------
    win = Window(*window)
    all_bands = tuple(sorted(set(NORM_BANDS) | {"SCL"}))
    ref = _read_window(reference_dir, all_bands, win)
    subj = _read_window(subject_dir, all_bands, win)

    pif0 = select_pif_mask(
        {b: ref[b] for b in _PIF_BANDS}, {b: subj[b] for b in _PIF_BANDS},
        ref_scl=ref["SCL"], subj_scl=subj["SCL"],
    )
    per_band, mask = iterate_pif_normalization(
        {b: ref[b] for b in NORM_BANDS}, {b: subj[b] for b in NORM_BANDS},
        pif0, bands=NORM_BANDS, method="additive",
    )
    norm = RadiometricNormalization(
        reference_scene=reference_observation_id, subject_scene=subject_observation_id,
        reference_date=reference_date, subject_date=subject_date,
        per_band=per_band, n_pif=int(mask.sum()),
        pif_selection={
            "gate": "valid + SCL-good (both dates) + not water + |NDVI| < 0.25 (both dates)",
            "refinement": "gross cross-band outlier removal (iterate_pif_normalization)",
            "window_col_row_w_h": list(window),
            "model": "per-band scene-wide additive offset (reference_DN = subject_DN + offset_b)",
        },
    )
    norm_dict = norm.to_manifest_dict()
    norm_dict["note"] = ("windowed summary for a staged third date - not gated; the authoritative "
                         "acceptance check remains geoseek.change.prep for the original pair.")
    return {"coregistration": coreg, "radiometric_normalization": norm_dict}

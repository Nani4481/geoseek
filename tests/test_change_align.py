"""Tests for geoseek.change.align.summarize_pair_alignment (Phase 3.5 Step 4)."""

from __future__ import annotations

import pytest

from geoseek.change.align import summarize_pair_alignment
from geoseek.config import get_settings

SUBJECT = "S2A_44RPQ_20210304_1_L2A_scaled"      # the third date
REFERENCE = "S2A_44RPQ_20240308_0_L2A_scaled"    # the date the pair normalizes against


def _dir(name: str):
    p = get_settings().datasets_dir / name
    for b in ("B04", "B03", "B02", "B08", "B11", "SCL"):
        if not (p / f"{b}.tif").is_file():
            pytest.skip(f"{name} not fully staged - run `python -m geoseek.staging.download_datasets --third-date`")
    return p


def test_summary_shape_and_coregistration_is_subpixel():
    subj, ref = _dir(SUBJECT), _dir(REFERENCE)
    out = summarize_pair_alignment(
        subject_dir=subj, reference_dir=ref,
        reference_observation_id=REFERENCE, subject_observation_id=SUBJECT,
        subject_date="2021-03-04", reference_date="2024-03-08",
        window=(1500, 1500, 4000, 4000),
    )
    coreg = out["coregistration"]
    assert coreg["reference_scene"] == REFERENCE
    assert coreg["moving_scene"] == SUBJECT
    # same-MGRS-tile multitemporal registration is sub-pixel
    assert coreg["median_magnitude_px"] < 1.0
    assert coreg["subpixel_threshold_px"] == 0.5

    norm = out["radiometric_normalization"]
    assert norm["reference_date"] == "2024-03-08" and norm["subject_date"] == "2021-03-04"
    per_band = norm["per_band_dn"]
    assert set(per_band) == {"B04", "B03", "B02", "B08", "B11"}
    for b, c in per_band.items():
        assert c["gain"] == 1.0                       # additive model
        assert c["offset_dn"] == c["offset_dn"]       # finite (not NaN)
        assert c["n"] > 10000
    # blue carries the largest (haze) offset and the weakest inter-date correlation,
    # the same signature Phase 3a found for the 2019<->2024 pair
    assert abs(per_band["B02"]["offset_dn"]) > abs(per_band["B08"]["offset_dn"])
    assert per_band["B02"]["inter_date_corr"] < per_band["B08"]["inter_date_corr"]

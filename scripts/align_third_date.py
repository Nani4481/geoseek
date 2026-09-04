"""Phase 3.5 Step 4 (offline): attach alignment provenance to the third date.

Computes co-registration + a windowed PIF additive radiometric-normalization
summary for the 2021-03-04 observation against the SAME reference date the
original pair uses (2024-03-08), then writes it onto that observation's
``radiometry`` / ``coregistration`` in the catalog and adds a
``third_date_alignment`` manifest section. No change detection.
"""

from __future__ import annotations

import dataclasses
import json

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.change.align import summarize_pair_alignment
from geoseek.config import get_settings
from geoseek.staging.manifest import load_manifest, record_analysis_section

THIRD = "S2A_44RPQ_20210304_1_L2A_scaled"
REFERENCE = "S2A_44RPQ_20240308_0_L2A_scaled"


def main() -> None:
    settings = get_settings()
    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")
    try:
        third = repo.get_observation(THIRD)
        reference = repo.get_observation(REFERENCE)
        if third is None or reference is None:
            raise SystemExit(f"need both observations staged+ingested: {THIRD}, {REFERENCE}")

        summary = summarize_pair_alignment(
            subject_dir=settings.datasets_dir / (third.dataset_dir or THIRD),
            reference_dir=settings.datasets_dir / (reference.dataset_dir or REFERENCE),
            reference_observation_id=REFERENCE,
            subject_observation_id=THIRD,
            subject_date=third.acquired_at,
            reference_date=reference.acquired_at,
        )

        coreg = summary["coregistration"]
        norm = summary["radiometric_normalization"]
        print("co-registration (2021-03-04 vs 2024-03-08 reference, B08 phase correlation):")
        print(f"  median (dy,dx) px = {coreg['median_shift_px']}  |  magnitude = "
              f"{coreg['median_magnitude_px']} px  (threshold {coreg['subpixel_threshold_px']} px)")
        print(f"  correction_needed = {coreg['correction_needed']}   correction_applied = {coreg['correction_applied']}")
        print("\nrelative radiometric normalization (PIF additive, subject 2021 -> reference 2024):")
        for b, c in norm["per_band_dn"].items():
            print(f"  {b}: offset {c['offset_dn']:+.1f} DN   inter-date r = {c['inter_date_corr']}   "
                  f"rmse {c['rmse_dn']:.1f} DN   n = {c['n']:,}")

        fixed = load_manifest().get("radiometry", {})
        new_radiometry = {
            "fixed_true_color": fixed,
            "role": "subject (normalized onto 2024-03-08 reference)",
            "relative_normalization": norm,
        }
        updated = dataclasses.replace(third, radiometry=new_radiometry, coregistration=coreg)
        repo.register_observation(updated)
        print(f"\ncatalog: updated observation {THIRD} radiometry + coregistration")

        record_analysis_section("third_date_alignment", {
            "subject_observation": THIRD,
            "reference_observation": REFERENCE,
            "subject_date": third.acquired_at,
            "reference_date": reference.acquired_at,
            **summary,
        })
        print("manifest: wrote 'third_date_alignment' section")
    finally:
        repo.close()


if __name__ == "__main__":
    main()

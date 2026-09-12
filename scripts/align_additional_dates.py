"""Phase 8 Step A (offline): attach alignment provenance to the new 2025/2026 dates.

Generalizes ``scripts/align_third_date.py`` to any number of "additional"
dates staged by ``geoseek.staging.download_datasets.stage_phase8_dates``:
for each, computes co-registration + a windowed PIF additive
radiometric-normalization summary against the SAME reference the original
pair and the third date use (2024-03-08), writes it onto that observation's
``radiometry`` / ``coregistration`` in the catalog, and merges the result into
the manifest's ``additional_dates_alignment`` section (a dict keyed by
observation id, so running this for 2026 does not clobber 2025's entry).

No change detection here - see ``geoseek.change.analyze`` for that.

    python scripts/align_additional_dates.py [--only <observation_id> ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.change.align import summarize_pair_alignment
from geoseek.config import get_settings
from geoseek.staging.download_datasets import ADDITIONAL_DATES, LARGE_AOI_DIR_SUFFIX, REFERENCE_DATE_SCENE_ID
from geoseek.staging.manifest import load_manifest, record_analysis_section

REFERENCE = REFERENCE_DATE_SCENE_ID + LARGE_AOI_DIR_SUFFIX


def align_one(repo, observation_id: str) -> dict:
    settings = get_settings()
    subject = repo.get_observation(observation_id)
    reference = repo.get_observation(REFERENCE)
    if subject is None or reference is None:
        raise SystemExit(f"need both observations staged+ingested: {observation_id}, {REFERENCE}")

    summary = summarize_pair_alignment(
        subject_dir=settings.datasets_dir / (subject.dataset_dir or observation_id),
        reference_dir=settings.datasets_dir / (reference.dataset_dir or REFERENCE),
        reference_observation_id=REFERENCE,
        subject_observation_id=observation_id,
        subject_date=subject.acquired_at,
        reference_date=reference.acquired_at,
    )

    coreg = summary["coregistration"]
    norm = summary["radiometric_normalization"]
    print(f"\nco-registration ({subject.acquired_at} vs {reference.acquired_at} reference, B08 phase correlation):")
    print(f"  median (dy,dx) px = {coreg['median_shift_px']}  |  magnitude = "
          f"{coreg['median_magnitude_px']} px  (threshold {coreg['subpixel_threshold_px']} px)")
    print(f"  correction_needed = {coreg['correction_needed']}   correction_applied = {coreg['correction_applied']}")
    print(f"relative radiometric normalization (PIF additive, subject {subject.acquired_at} -> "
          f"reference {reference.acquired_at}):")
    for b, c in norm["per_band_dn"].items():
        print(f"  {b}: offset {c['offset_dn']:+.1f} DN   inter-date r = {c['inter_date_corr']}   "
              f"rmse {c['rmse_dn']:.1f} DN   n = {c['n']:,}")

    fixed = load_manifest().get("radiometry", {})
    new_radiometry = {
        "fixed_true_color": fixed,
        "role": "subject (normalized onto 2024-03-08 reference)",
        "relative_normalization": norm,
    }
    updated = dataclasses.replace(subject, radiometry=new_radiometry, coregistration=coreg)
    repo.register_observation(updated)
    print(f"catalog: updated observation {observation_id} radiometry + coregistration")

    return {
        "subject_observation": observation_id,
        "reference_observation": REFERENCE,
        "subject_date": subject.acquired_at,
        "reference_date": reference.acquired_at,
        **summary,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-align-additional-dates", description=__doc__)
    p.add_argument("--only", nargs="*", default=None,
                   help="observation id(s) to align (default: every ADDITIONAL_DATES entry)")
    args = p.parse_args(argv)

    targets = args.only or [e["scene_id"] + LARGE_AOI_DIR_SUFFIX for e in ADDITIONAL_DATES]

    settings = get_settings()
    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")
    try:
        section = dict(load_manifest().get("additional_dates_alignment", {}))
        for obs_id in targets:
            print(f"\n=== aligning {obs_id} vs {REFERENCE} ===")
            section[obs_id] = align_one(repo, obs_id)
        record_analysis_section("additional_dates_alignment", section)
        print(f"\nmanifest: wrote 'additional_dates_alignment' ({len(section)} observation(s))")
    finally:
        repo.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[align_additional_dates] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

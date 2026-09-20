"""Phase 8F-2: convert staged DOTA (v1.5 OBB labels by default) into YOLO-OBB chips.

  data/datasets/dota/{train,val}/  ->  data/datasets/dota_obb/{train,monitor,val}/

Reads only from the staged DOTA archive (no network). Train and val are converted
SEPARATELY; the ``monitor`` split is a class-stratified holdout of whole TRAIN
source images (the training run's only validation set - see geoseek.detect.convert).

  python scripts/prepare_detect_data.py --plan-only     # measure, write nothing
  python scripts/prepare_detect_data.py                 # convert (resumable: existing chips are skipped)

This SUPERSEDES the 2026-09-15 conversion, which read a labelTxt-v1.5 folder that an extraction collision had
overwritten with axis-aligned (HBB) labels (repaired since; see geoseek.staging.download_dota and
geoseek.detect.classes), so that output could not train an oriented detector.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings, print_startup_banner  # noqa: E402
from geoseek.detect.classes import LABEL_SETS, PRIMARY_LABEL_SET  # noqa: E402
from geoseek.detect.convert import MONITOR_FRACTION, NEGATIVE_FRACTION, convert_dota  # noqa: E402
from geoseek.detect.sampling import RFS_THRESHOLD  # noqa: E402
from geoseek.staging.manifest import record_analysis_section  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true", help="plan + report only; write no chips")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rfs-t", type=float, default=RFS_THRESHOLD)
    ap.add_argument("--monitor-fraction", type=float, default=MONITOR_FRACTION)
    ap.add_argument("--neg-fraction", type=float, default=NEGATIVE_FRACTION)
    ap.add_argument("--label-set", choices=sorted(LABEL_SETS), default=PRIMARY_LABEL_SET)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default data/datasets/dota_obb; a non-primary label set -> dota_obb_<set>)")
    args = ap.parse_args(argv)

    settings = get_settings()
    print_startup_banner(settings)
    dota_root = settings.datasets_dir / "dota"
    suffix = "" if args.label_set == PRIMARY_LABEL_SET else "_" + args.label_set.replace(".", "")
    out_root = args.out or settings.datasets_dir / f"dota_obb{suffix}"
    print(f"\n=== DOTA {args.label_set} OBB -> YOLO-OBB chips: {dota_root} -> {out_root} ===")

    t0 = time.time()
    report = convert_dota(
        dota_root, out_root, workers=args.workers, monitor_fraction=args.monitor_fraction,
        neg_fraction=args.neg_fraction, rfs_t=args.rfs_t, do_write=not args.plan_only,
        label_dir=LABEL_SETS[args.label_set],
    )
    print(f"\n[convert] done in {time.time() - t0:.0f}s")
    print(json.dumps(report, indent=1))

    if not args.plan_only:
        report["supersedes"] = (
            "2026-09-15 'Stage A' conversion (data/datasets/dota_yolo/): it read a labelTxt-v1.5 folder that an "
            "extraction collision had overwritten with axis-aligned (HBB) labels - 210,631/210,631 train and "
            "69,565/69,565 val instances - so it could not train an oriented detector. Its val also dropped every "
            "background chip (3,006 of 5,297 windows)."
        )
        record_analysis_section("detect_data_conversion" + suffix, report)
        print(f"[convert] Manifest: {settings.provenance_manifest_path}")
        v = report["verification"]
        if not v["ok"]:
            raise SystemExit(f"[convert] VERIFICATION FAILED: {v['problems']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[convert] FATAL: {e}", file=sys.stderr)
        raise

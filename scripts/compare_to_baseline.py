"""Diff a candidate snapshot against a baseline snapshot.

    python scripts/compare_to_baseline.py data/eval/baseline_v1.json data/eval/candidate.json
    python scripts/compare_to_baseline.py BASE CAND --only retrieval --significant-only --md diff.md --json diff.json
    python scripts/compare_to_baseline.py BASE CAND --fail-on-regression      # exit 1 if any metric regressed beyond noise

Prints a table of metric / baseline / candidate / delta / noise band / exceeds-noise / verdict. A delta
counts only when it exceeds the metric's recorded noise band, or - where both snapshots scored the same queries /
regions - when its paired bootstrap 95% CI excludes zero (verdict *_paired). Any phase that claims an improvement must
attach this diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.eval.compare import diff_snapshots, render_markdown, skipped_placeholders, summarize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--only", default=None, help="restrict to metric paths starting with this prefix (e.g. retrieval.ayodhya_3267)")
    ap.add_argument("--significant-only", action="store_true", help="hide within-noise / no-band rows")
    ap.add_argument("--md", type=Path, default=None, help="also write the markdown table here")
    ap.add_argument("--json", type=Path, default=None, help="also write the rows as JSON here")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 if any metric regressed (band or paired test)")
    ap.add_argument("--no-paired", action="store_true", help="skip the paired-bootstrap column (band rule only)")
    args = ap.parse_args(argv)

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand = json.loads(args.candidate.read_text(encoding="utf-8"))
    rows = diff_snapshots(base, cand, only_prefix=args.only, paired=not args.no_paired)
    skipped = skipped_placeholders(base, cand)
    md = render_markdown(rows, only_significant=args.significant_only, skipped=skipped)
    print(md)
    if args.md:
        args.md.write_text(md + "\n", encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps({"summary": summarize(rows), "skipped": skipped,
                                         "rows": [r.as_dict() for r in rows]}, indent=1), encoding="utf-8")
    if args.fail_on_regression and any(r.verdict in ("regressed", "regressed_paired") for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

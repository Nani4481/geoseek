"""Rebuild the Ayodhya tile index from the two scaled 2019/2024 scenes with the corrected (fixed-bounds,
harmonized) true-colour. The model is loaded once and reused for both scenes.

SAFE BY DEFAULT. Without ``--force`` this writes a NEW index (``tiles.faiss`` + ``tiles.sqlite``) into a new
directory and touches nothing else - not the production index, not the provenance manifest.

    python scripts/rebuild_index.py                    # -> data/index/rebuild_<UTC timestamp>/
    python scripts/rebuild_index.py --out-dir D        # -> D (must not exist yet)
    python scripts/rebuild_index.py --force            # DESTRUCTIVE: deletes the production index + catalog first

``--force`` deletes the production ``tiles.faiss`` and ``tiles.sqlite`` and rebuilds them from these two scenes
ONLY. The production index holds every other scene as well (100k+ vectors, the analyst decisions audit table,
watch areas); none of that is recreated by this script. It exists for the original two-scene demo setup. To
re-embed the whole catalog with a different model, use ``scripts/reembed.py``, which never writes to the
production index.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings  # noqa: E402
from geoseek.ingest.embed import load_model_once  # noqa: E402
from geoseek.ingest.pipeline import ingest_scene  # noqa: E402
from geoseek.ingest.store import DB_FILENAME, INDEX_FILENAME  # noqa: E402

SCALED_SCENES = ("S2B_44RPQ_20190330_1_L2A_scaled", "S2A_44RPQ_20240308_0_L2A_scaled")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="DESTRUCTIVE: delete the production index + catalog and rebuild them in place from the two scenes")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the new index when --force is not given (default data/index/rebuild_<timestamp>); "
                         "must not already exist")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    settings = get_settings()
    index_dir = settings.index_dir

    if args.force:
        if args.out_dir is not None:
            raise SystemExit("--force rebuilds the production index in place; it cannot be combined with --out-dir")
        target, record_manifest = None, True
        for fn in (INDEX_FILENAME, DB_FILENAME):
            p = index_dir / fn
            if p.is_file():
                p.unlink()
                print(f"[rebuild] --force: removed {p}")
    else:
        target = args.out_dir or index_dir / f"rebuild_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        if target.exists():
            raise SystemExit(f"{target} already exists; refusing to write into it. Choose a new --out-dir.")
        record_manifest = False        # a scratch rebuild must not rewrite the shared provenance manifest
        print(f"[rebuild] writing a NEW index to {target} (production index untouched; pass --force to replace it)")

    load_model_once()  # resident for both scenes

    t0 = time.time()
    total = 0
    for name in SCALED_SCENES:
        scene_dir = settings.datasets_dir / name
        report = ingest_scene(scene_dir, save_sample=False, record_manifest=record_manifest, index_dir=target)
        total = report["index_total_vectors"]
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"[rebuild] DONE: {total} vectors in {target or index_dir}, re-embed wall time {dt:.2f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

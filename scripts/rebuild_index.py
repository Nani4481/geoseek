"""Rebuild the production tile index over both scaled scenes with the corrected
(fixed-bounds, harmonized) true-color. Model is loaded once and reused for both
scenes. Deletes the old FAISS+SQLite first so this is a clean rebuild, not an
append.
"""

from __future__ import annotations

import time
from pathlib import Path

from geoseek.config import get_settings
from geoseek.ingest.embed import load_model_once
from geoseek.ingest.pipeline import ingest_scene
from geoseek.ingest.store import DB_FILENAME, INDEX_FILENAME

SCALED_SCENES = ("S2B_44RPQ_20190330_1_L2A_scaled", "S2A_44RPQ_20240308_0_L2A_scaled")


def main() -> None:
    settings = get_settings()
    index_dir = settings.index_dir
    for fn in (INDEX_FILENAME, DB_FILENAME):
        p = index_dir / fn
        if p.is_file():
            p.unlink()
            print(f"[rebuild] removed stale {p}")

    load_model_once()  # resident for both scenes

    t0 = time.time()
    total = 0
    for name in SCALED_SCENES:
        scene_dir = settings.datasets_dir / name
        report = ingest_scene(scene_dir, save_sample=False, record_manifest=True)
        total = report["index_total_vectors"]
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"[rebuild] DONE: {total} vectors in the index, re-embed wall time {dt:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

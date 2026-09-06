"""Phase 7b driver: stage + ingest diverse-AOI scenes, cycling through regions,
until the FAISS index reaches a target vector count. One embedding model
instance is loaded once and reused for every scene (matches how a real
ingestion service would run, and avoids re-paying model load per scene).

Samples peak process RSS (psutil) and peak CUDA memory (torch) across each
scene's ingest call and appends one ledger row per scene to
data/eval_retrieval/diverse_ingest_ledger.json - the source for the Phase 7b
measurement tables (scripts/measure_tier.py reads this ledger).

Usage:
    python scripts/run_diverse_ingest.py --target 10000 --regions dehradun jaisalmer sundarbans delhi_ncr kanha
    python scripts/run_diverse_ingest.py --target 50000
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import psutil
import torch

from geoseek.config import get_settings
from geoseek.models import RemoteCLIPEmbeddingModel
from geoseek.vectorindex import FaissFlatIPIndex
from ingest_diverse_scene import ingest_diverse_scene
from stage_diverse_aois import REGIONS, stage_one

LEDGER_PATH = get_settings().data_dir / "eval_retrieval" / "diverse_ingest_ledger.json"


class MemSampler:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.proc = psutil.Process()
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemSampler":
        self.peak_rss = self.proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def current_faiss_count() -> int:
    settings = get_settings()
    return FaissFlatIPIndex(settings.index_dir / "tiles.faiss").count()


def append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.is_file() else []
    data.append(entry)
    LEDGER_PATH.write_text(json.dumps(data, indent=2))


def run_until(target_count: int, region_cycle: list[str], max_scenes: int) -> None:
    model = RemoteCLIPEmbeddingModel()
    model.load()
    region_cycle = list(region_cycle)
    ri = 0
    n_scenes = 0

    while True:
        count = current_faiss_count()
        if count >= target_count:
            print(f"[driver] TARGET REACHED: index has {count} vectors (target {target_count})")
            break
        if n_scenes >= max_scenes:
            print(f"[driver] max_scenes ({max_scenes}) reached - stopping short at {count} vectors")
            break
        if not region_cycle:
            print(f"[driver] no regions left to try - stopping short at {count} vectors")
            break

        region = region_cycle[ri % len(region_cycle)]
        ri += 1
        print(f"\n[driver] --- scene {n_scenes + 1} (index at {count}/{target_count}) region={region} ---")
        try:
            staged = stage_one(region)
        except Exception as e:
            print(f"[driver] STAGE FAILED for '{region}': {e!r} - dropping this region for the rest of the run")
            region_cycle = [r for r in region_cycle if r != region]
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with MemSampler() as mem:
                report = ingest_diverse_scene(staged["scene_id"], embedding_model=model)
        except Exception as e:
            print(f"[driver] INGEST FAILED for '{staged['scene_id']}': {e!r} - skipping, region stays in rotation")
            continue
        wall_s = time.time() - t0
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None

        entry = {
            **report,
            "peak_rss_mb": round(mem.peak_rss / (1024 * 1024), 1),
            "peak_vram_mb": round(vram_mb, 1) if vram_mb is not None else None,
            "wall_s": round(wall_s, 2),
        }
        append_ledger(entry)
        n_scenes += 1
        print(f"[driver] #{n_scenes} {report['scene_id']} ({report['region']}/{report['category']}) "
              f"+{report['tiles_added']} tiles -> cumulative {report['index_total_vectors']} "
              f"| {report['tiles_per_sec']} tiles/s | RSS peak {entry['peak_rss_mb']:.0f} MB "
              f"| VRAM peak {entry['peak_vram_mb']} MB | wall {wall_s:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True, help="stop once the FAISS index reaches this many vectors")
    ap.add_argument("--regions", nargs="+", default=list(REGIONS), choices=sorted(REGIONS))
    ap.add_argument("--max-scenes", type=int, default=300, help="safety cap on scenes staged this run")
    args = ap.parse_args()
    run_until(args.target, args.regions, args.max_scenes)


if __name__ == "__main__":
    main()

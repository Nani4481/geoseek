"""Phase 8F-2 pre-flight (Step 4): can the intended training config run end-to-end on this 8 GB GPU, how fast, and at
WORST-CASE label density?

One model / batch size per process (clean CUDA state). It runs a real Ultralytics ``train()`` for one epoch over a
benchmark list drawn from the real train list - so the real augmenting dataloader (mosaic + rotation), the real OBB loss
and the real optimiser step are all exercised - and measures:

  * throughput (samples/s), steady state (first iterations excluded)
  * peak VRAM: torch max_memory_allocated / reserved AND nvidia-smi (which also sees the CUDA context)
  * GPU utilisation (is the dataloader starving the GPU?) and system RAM headroom
  * label density per batch (Ultralytics pads every image's targets to the batch maximum, so one dense chip inflates the
    task-aligned assigner's (batch x max_boxes x anchors) tensors for all images) and whether the assigner had to fall
    back to the CPU ("CUDA OutOfMemoryError in TaskAlignedAssigner")
  * that the loss stays finite

The benchmark list is a seeded random sample of the TRAIN list plus the densest training chips, so the worst case
appears early in the epoch instead of being missed.

    python scripts/detect_bench.py --model yolo26s-obb.pt --batch 16 --out data/detect_preflight/bench_yolo26s_b16.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
# fragmentation-tolerant allocator settings (varying batch shapes); must be set before torch initialises CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import KEPT_CLASSES  # noqa: E402


class _Poller(threading.Thread):
    """nvidia-smi + psutil sampler (0.5 s)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.vram_mib: list[int] = []
        self.gpu_util: list[int] = []
        self.ram_avail_gb: list[float] = []
        self.cpu_util: list[float] = []

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.vram_mib.append(int(out[0]))
                self.gpu_util.append(int(out[1]))
            except Exception:
                pass
            self.ram_avail_gb.append(psutil.virtual_memory().available / 1e9)
            self.cpu_util.append(psutil.cpu_percent(interval=None))
            time.sleep(0.5)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def build_bench_dataset(root: Path, tmp: Path, n_samples: int, n_dense: int, seed: int) -> Path:
    import csv

    train_lines = (root / "train" / "train.txt").read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    sample = rng.sample(train_lines, min(n_samples, len(train_lines)))

    dense = []
    with open(root / "chip_index.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    rows.sort(key=lambda r: -int(r["n_labels"]))
    dense = [f"./images/{r['chip_id']}.png" for r in rows[:n_dense]]

    lines = sample + dense * 2                       # the densest chips appear twice
    rng.shuffle(lines)
    bench_dir = tmp / "train"
    # mirror the dataset layout so Ultralytics finds labels via /images/ -> /labels/
    (bench_dir).mkdir(parents=True, exist_ok=True)
    abs_lines = [str((root / "train" / ln[2:]).resolve()) for ln in lines]
    lst = tmp / "bench_train.txt"
    lst.write_text("\n".join(abs_lines) + "\n", encoding="utf-8")
    mon = (root / "monitor" / "monitor.txt").read_text(encoding="utf-8").splitlines()[:16]
    mlst = tmp / "bench_val.txt"
    mlst.write_text("\n".join(str((root / "monitor" / ln[2:]).resolve()) for ln in mon) + "\n", encoding="utf-8")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(KEPT_CLASSES))
    y = tmp / "bench.yaml"
    y.write_text(f"path: {tmp.resolve().as_posix()}\ntrain: {lst.resolve().as_posix()}\nval: {mlst.resolve().as_posix()}\n"
                 f"names:\n{names}\n", encoding="utf-8")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--dense", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--cache", default="False", help="False | disk | ram")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="torch.cuda.set_per_process_memory_fraction. On Windows the driver's system-memory fallback "
                         "spills past VRAM into RAM instead of raising OOM, which (a) wrecks throughput and (b) stops "
                         "Ultralytics' per-image assigner retry from ever firing. A hard cap restores real OOMs.")
    ap.add_argument("--deterministic", default="True", help="Ultralytics deterministic flag (True|False)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    ckpt = settings.models_dir / "yolo_obb" / args.model
    tmp = Path(os.environ.get("TEMP", ".")) / f"geoseek_bench_{args.model}_{args.batch}_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    yaml_path = build_bench_dataset(root, tmp, args.samples, args.dense, args.seed)

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    cap = _Capture()
    LOGGER.addHandler(cap)
    poller = _Poller()
    stats = {"iter_s": [], "max_labels_per_image": [], "loss": []}
    state = {"t": None}

    def on_train_start(trainer):
        orig = trainer.preprocess_batch

        def wrapped(batch):
            bi = batch["batch_idx"].long()
            stats["max_labels_per_image"].append(int(torch.bincount(bi).max().item()) if len(bi) else 0)
            return orig(batch)

        trainer.preprocess_batch = wrapped
        torch.cuda.reset_peak_memory_stats()

    def on_batch_start(trainer):
        state["t"] = time.perf_counter()

    def on_batch_end(trainer):
        torch.cuda.synchronize()
        stats["iter_s"].append(time.perf_counter() - state["t"])
        try:
            stats["loss"].append([float(x) for x in trainer.tloss])
        except Exception:
            pass

    model = YOLO(str(ckpt))
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_batch_start)
    model.add_callback("on_train_batch_end", on_batch_end)

    batt0 = psutil.sensors_battery()
    poller.start()
    t0 = time.time()
    err = None
    try:
        model.train(
            data=str(yaml_path), imgsz=1024, batch=args.batch, epochs=1, device=0, workers=args.workers,
            optimizer="AdamW", lr0=0.0005, cos_lr=True, warmup_epochs=0.5, amp=True,
            mosaic=1.0, degrees=180.0, flipud=0.5, fliplr=0.5, scale=args.scale, translate=0.1, close_mosaic=0,
            val=False, plots=False, save=False, exist_ok=True, project=str(tmp / "runs"), name="bench",
            cache=(False if args.cache == "False" else args.cache), seed=args.seed, verbose=False,
            deterministic=(args.deterministic == "True"),
        )
    except Exception as e:  # report, don't crash the harness: an OOM IS the answer
        err = f"{type(e).__name__}: {e}"
    wall = time.time() - t0
    poller.stop.set()
    time.sleep(0.6)

    it = np.array(stats["iter_s"])
    steady = it[10:] if len(it) > 15 else it
    mx = np.array(stats["max_labels_per_image"]) if stats["max_labels_per_image"] else np.zeros(1)
    losses = np.array([l for l in stats["loss"] if len(l)], dtype=float) if stats["loss"] else np.zeros((0, 4))
    fallback_msgs = [m for m in cap.msgs if "out of memory" in m.lower() or "using cpu" in m.lower()]
    res = {
        "deterministic": args.deterministic, "model": args.model, "batch": args.batch, "workers": args.workers, "cache": args.cache, "imgsz": 1024,
        "amp": True, "error": err, "wall_s": round(wall, 1), "iterations": int(len(it)),
        "samples_per_s_steady": None if not len(steady) else round(args.batch / float(np.mean(steady)), 2),
        "iter_s_median": None if not len(steady) else round(float(np.median(steady)), 3),
        "iter_s_p95": None if not len(steady) else round(float(np.percentile(steady, 95)), 3),
        "iter_s_max": None if not len(it) else round(float(it.max()), 2),
        "n_iters_gt_3x_median": None if not len(steady) else int((steady > 3 * np.median(steady)).sum()),
        "peak_vram_torch_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_vram_torch_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "peak_vram_nvidia_smi_gb": round(max(poller.vram_mib or [0]) / 1024, 2),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        "gpu_util_mean_pct": round(float(np.mean(poller.gpu_util)), 1) if poller.gpu_util else None,
        "cpu_util_mean_pct": round(float(np.mean(poller.cpu_util)), 1) if poller.cpu_util else None,
        "ram_available_min_gb": round(min(poller.ram_avail_gb or [0]), 2),
        "max_labels_per_image_in_batch": {"median": float(np.median(mx)), "p90": float(np.percentile(mx, 90)),
                                          "max": int(mx.max())},
        "mem_fraction": args.mem_fraction,
        "assigner_oom_retry_messages": len(fallback_msgs),
        "assigner_oom_retry_text": (fallback_msgs[0][:200] if fallback_msgs else None),
        "loss_finite": bool(np.isfinite(losses).all()) if losses.size else None,
        "loss_first_last": None if not len(losses) else [[round(v, 3) for v in losses[0]], [round(v, 3) for v in losses[-1]]],
        "battery": {"on_ac_start": None if batt0 is None else bool(batt0.power_plugged),
                    "percent_start": None if batt0 is None else round(batt0.percent),
                    "on_ac_end": None if psutil.sensors_battery() is None else bool(psutil.sensors_battery().power_plugged)},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\nBENCH_RESULT " + json.dumps(res))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

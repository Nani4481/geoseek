"""Training configuration + run control for the Phase 8F-2 detector (fine-tuning YOLO26-OBB onto our 8 classes).

Everything that shapes the run lives in :func:`build_train_args` so it can be printed, unit-tested and recorded in the
model card. Run control (:func:`train`) adds what a long run on THIS machine needs:

  * a hard torch VRAM cap (``set_per_process_memory_fraction``). On Windows the NVIDIA driver's system-memory fallback
    lets CUDA spill past VRAM into RAM instead of raising OOM; measured on this machine that turned 17.7 samples/s into
    3.2 samples/s AND stopped Ultralytics' own per-image assigner retry from ever firing (it only fires on a real OOM).
    With the cap, dense batches hit that retry and training stays fast.
  * sleep prevention (``SetThreadExecutionState``) - the laptop must not suspend mid-run;
  * a per-epoch ``progress.jsonl`` (wall time, VRAM, AC power) so an interruption or a throttled run is visible;
  * :func:`find_resume_checkpoint` - resume from ``last.pt``, or the newest periodic checkpoint that still loads
    (a crash during ``torch.save`` can corrupt ``last.pt``).

ultralytics (AGPL-3.0) is imported lazily.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

VRAM_FRACTION = 0.80          # of total VRAM (8.59 GB card -> 6.87 GB); ~1.3 GB is held by the Windows desktop
VAL_WORKERS = 4               # Ultralytics builds the val loader with workers*2 (=12 here) and keeps every worker alive for
                              # the whole run: measured 20 worker processes x ~450 MB, system RAM 96.4% used / 0.60 GB free.
DEFAULT_EPOCHS = 20
RUN_NAME = "geoseek_obb_v15_yolo26s"


def build_train_args(
    data_yaml: Path, *, project: Path, name: str = RUN_NAME, epochs: int = DEFAULT_EPOCHS, batch: int = 8,
    workers: int = 6, device: int = 0, fraction: float = 1.0,
) -> dict:
    """The complete, explicit training configuration (nothing left to ``optimizer=auto``)."""
    return {
        # --- data / schedule -------------------------------------------------------------------------------
        "data": str(data_yaml),
        "imgsz": 1024,                 # = chip size = Maxar tile size; small vehicles need the resolution
        "epochs": epochs,
        "batch": batch,                # 8: measured max that fits under the VRAM cap with dense mosaic batches
        "nbs": 64,                     # gradient accumulation to an effective batch of 64
        "workers": workers,            # RAM-limited (15 GB): 6 workers left ~2.5 GB free in the benchmark
        "device": device,
        "fraction": fraction,          # 1.0 for the real run; <1 only for smoke tests
        "cache": False,                # measured: the decode cache did not speed anything up
        # --- optimiser (explicit: 'auto' would pick AdamW lr=0.000833 here and could switch to MuSGD >10k its) ---
        "optimizer": "AdamW",
        "lr0": 0.00015,                # gentle: AdamW moves every weight by ~lr per step whatever the gradient size
        "lrf": 0.1,                    # final lr = 1.5e-5
        "cos_lr": True,
        "momentum": 0.9,               # AdamW beta1
        "weight_decay": 0.0005,
        "warmup_epochs": 2.0,          # ~370 optimizer steps before full lr (a 3-epoch smoke at 3e-4/1 warmup lost ~4 mAP50 pts)
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,         # required for Adam-family optimisers (default 0.1 would explode the biases)
        # --- precision -------------------------------------------------------------------------------------
        "amp": True,                   # mixed precision (fp16 autocast + GradScaler)
        "deterministic": False,        # measured +13% throughput; seeds are still set
        "seed": 0,
        # --- augmentation (rotation is REQUIRED for oriented aerial imagery) --------------------------------
        "mosaic": 1.0,
        "close_mosaic": 4,             # last 4 epochs without mosaic: settle on natural-looking chips
        "degrees": 180.0,              # full rotation: nadir imagery has no canonical 'up'
        "flipud": 0.5,
        "fliplr": 0.5,
        "scale": 0.4,                  # 0.6x-1.4x; milder than the pretrained recipe's 0.9 (tiny cars vanish at 0.1x)
        "translate": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # --- validation / selection: on the MONITOR holdout (data yaml's val:), never the official val ---------
        "val": True,
        "max_det": 1000,               # dense parking lots exceed the default 300 per chip
        "patience": 100,               # early stopping effectively off (the schedule is fixed in advance)
        "save": True,
        "save_period": 2,              # keep epochN.pt too: post-hoc evaluation curves + resume fallback
        "plots": True,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "verbose": True,
    }


def prevent_sleep() -> None:
    """Ask Windows not to suspend the machine while this process runs (cleared automatically on exit)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def find_resume_checkpoint(run_dir: Path, epochs: int) -> Path | None:
    """last.pt if it is resumable, else the newest periodic epochN.pt that is; None if nothing usable.

    Resumable = loads, still carries optimizer state (a finished run's last.pt is stripped), and its epoch is not the
    final one (Ultralytics asserts ``0 < start_epoch < epochs``: resuming a completed run is an error, and an
    ``epochN.pt`` saved on the last epoch still has optimizer state, which is how the first supervisor got fooled)."""
    import torch

    weights = run_dir / "weights"
    if not weights.is_dir():
        return None
    cands = [weights / "last.pt"] + sorted(weights.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if p.is_file():
            try:
                ck = torch.load(str(p), map_location="cpu", weights_only=False)
                if ck.get("optimizer") is not None and 0 <= int(ck.get("epoch", -1)) + 1 < epochs:
                    return p
            except Exception:
                continue
    return None


def training_finished(run_dir: Path, epochs: int) -> bool:
    """True when the per-epoch log shows the final epoch completed AND a last.pt exists (the log row is written before
    the checkpoint, so a process killed in between must not count as finished)."""
    log = run_dir / "progress.jsonl"
    if not log.is_file() or not (run_dir / "weights" / "last.pt").is_file():
        return False
    last = 0
    for ln in log.read_text(encoding="utf-8").splitlines():
        try:
            last = max(last, int(json.loads(ln)["epoch"]))
        except Exception:
            continue
    return last >= epochs


class RunLock:
    """One supervisor per run directory. A second launch on the same run dir (which fights the first one for the GPU
    and RAM - it happened during the smoke tests) refuses to start; a stale lock from a dead process is taken over."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "supervisor.lock"

    def __enter__(self):
        import psutil

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            try:
                pid = int(self.path.read_text().strip())
                if psutil.pid_exists(pid) and pid != os.getpid():
                    raise SystemExit(f"[train] another supervisor (pid {pid}) already owns {self.path.parent}; refusing to start a second one")
            except ValueError:
                pass
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except OSError:
            pass
        return False


def _progress_callbacks(run_dir: Path):
    import psutil
    import torch

    log = run_dir / "progress.jsonl"

    def on_fit_epoch_end(trainer):
        batt = psutil.sensors_battery()
        row = {
            "epoch": int(trainer.epoch) + 1, "epochs": int(trainer.epochs), "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch_seconds": round(float(getattr(trainer, "epoch_time", 0.0) or 0.0), 1),
            "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "ac_power": None if batt is None else bool(batt.power_plugged),
            "battery_pct": None if batt is None else round(batt.percent),
            "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
            "metrics": {k: (round(float(v), 5) if isinstance(v, (int, float)) else v) for k, v in trainer.metrics.items()},
            "lr": {k: round(float(v), 8) for k, v in (getattr(trainer, "lr", {}) or {}).items()},
        }
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        gc.collect()
        torch.cuda.empty_cache()          # under the VRAM cap, cached-but-fragmented blocks otherwise starve the next val / final eval
        if batt is not None and not batt.power_plugged:
            print("[train] WARNING: AC power unplugged - throughput will drop sharply (see geoseek-phase7b-followup)", flush=True)

    return {"on_fit_epoch_end": on_fit_epoch_end}


def lean_obb_trainer():
    """OBBTrainer whose validation loader uses VAL_WORKERS workers instead of 2 x ``workers`` (RAM)."""
    from ultralytics.models.yolo.obb import OBBTrainer

    class LeanOBBTrainer(OBBTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            saved = self.args.workers
            self.args.workers = max(1, VAL_WORKERS // 2)             # the parent multiplies by 2 for non-train modes
            try:
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            finally:
                self.args.workers = saved

        def final_eval(self):
            """The stock end-of-run validation of best.pt died with CUDA OOM under the VRAM cap (allocator blocks from
            training still held) AFTER every epoch and checkpoint had been saved. Free them first, and never let this
            optional step fail an otherwise finished run: the official evaluation is a separate script."""
            import torch

            gc.collect()
            torch.cuda.empty_cache()
            try:
                super().final_eval()
            except RuntimeError as e:
                print(f"[train] final_eval skipped ({type(e).__name__}: {str(e).splitlines()[0][:120]}); "
                      f"weights are saved, run scripts/eval_detector.py", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

    return LeanOBBTrainer


def train(args: dict, *, init_weights: Path, resume: Path | None = None) -> Path:
    """Run (or resume) training in-process. Returns the run directory."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    import torch

    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    prevent_sleep()

    from ultralytics import YOLO

    run_dir = Path(args["project"]) / args["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        model = YOLO(str(resume))
        for ev, fn in _progress_callbacks(run_dir).items():
            model.add_callback(ev, fn)
        model.train(resume=True, trainer=lean_obb_trainer())
    else:
        model = YOLO(str(init_weights))
        for ev, fn in _progress_callbacks(run_dir).items():
            model.add_callback(ev, fn)
        model.train(trainer=lean_obb_trainer(), **args)
    return run_dir

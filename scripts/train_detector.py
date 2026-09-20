"""Phase 8F-2 Step 5: train the 8-class oriented detector (fine-tune YOLO26s-OBB, transplanted class head).

    python scripts/train_detector.py --print-config      # show the full planned config, run nothing
    python scripts/train_detector.py                       # supervised run: auto-resumes after a crash / interruption
    python scripts/train_detector.py --epochs 1 --name smoke    # smoke test

The run directory (default data/runs/detector/geoseek_obb_v15_yolo26s) holds weights/last.pt, best.pt (selected on the
MONITOR holdout, never the official val), epoch*.pt every 2 epochs, results.csv, progress.jsonl. Re-running the same
command resumes from the newest usable checkpoint. Only the supervisor loop lives here; the config is in
geoseek.detect.train.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.convert import write_dataset_yamls  # noqa: E402
from geoseek.detect.surgery import transplant_head, verify_transplant  # noqa: E402
from geoseek.detect.train import (  # noqa: E402
    DEFAULT_EPOCHS, RUN_NAME, RunLock, build_train_args, find_resume_checkpoint, train, training_finished)

MAX_RESTARTS = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default=RUN_NAME)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the list (smoke tests only)")
    ap.add_argument("--pretrained", default="yolo26s-obb.pt")
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--child", action="store_true", help="internal: run one (resumable) training attempt in-process")
    args = ap.parse_args()

    settings = get_settings()
    root = settings.datasets_dir / "dota_obb"
    yamls = write_dataset_yamls(root)                                    # refresh absolute paths for THIS checkout
    project = settings.data_dir / "runs" / "detector"
    run_dir = project / args.name
    pretrained = settings.models_dir / "yolo_obb" / args.pretrained
    init = settings.models_dir / "yolo_obb" / f"geoseek_init_{Path(args.pretrained).stem.replace('-obb', '')}_8cls.pt"
    cfg = build_train_args(yamls["train"], project=project, name=args.name, epochs=args.epochs, batch=args.batch,
                           workers=args.workers, fraction=args.fraction)

    if args.print_config:
        print(json.dumps({"init_weights": str(init), "transplanted_from": str(pretrained), **cfg}, indent=2))
        return

    if not init.is_file():
        print(f"[train] building the class-head-transplanted init weights -> {init}", flush=True)
        print(json.dumps(transplant_head(pretrained, init), indent=1), flush=True)
        v = verify_transplant(pretrained, init)
        print("[train] transplant verification:", json.dumps(v), flush=True)
        if not v["ok"]:
            raise SystemExit("[train] class-head transplant failed verification - refusing to train on it")

    if args.child:
        resume = find_resume_checkpoint(run_dir, args.epochs)
        print(f"[train] {'RESUMING from ' + str(resume) if resume else 'starting fresh from ' + str(init)}", flush=True)
        train(cfg, init_weights=init, resume=resume)
        return

    # supervisor: relaunch (resuming) until the child exits cleanly or the restart budget is spent
    with RunLock(run_dir):
        t0 = time.time()
        for attempt in range(MAX_RESTARTS + 1):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--epochs", str(args.epochs),
                   "--batch", str(args.batch), "--workers", str(args.workers), "--name", args.name,
                   "--fraction", str(args.fraction), "--pretrained", args.pretrained]
            print(f"[train] attempt {attempt + 1}/{MAX_RESTARTS + 1}  (elapsed {time.time() - t0:.0f}s)", flush=True)
            rc = subprocess.call(cmd)
            if rc == 0:
                print(f"[train] finished cleanly after {time.time() - t0:.0f}s", flush=True)
                return
            if training_finished(run_dir, args.epochs):
                print(f"[train] all {args.epochs} epochs completed (child exit code {rc} came after the last epoch, "
                      f"i.e. in end-of-run housekeeping); not resuming. Elapsed {time.time() - t0:.0f}s", flush=True)
                return
            print(f"[train] child exited with code {rc} before finishing; resuming in 30 s", flush=True)
            time.sleep(30)
        raise SystemExit(f"[train] gave up after {MAX_RESTARTS + 1} attempts")


if __name__ == "__main__":
    main()

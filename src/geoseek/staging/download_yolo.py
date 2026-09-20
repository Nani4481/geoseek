"""Stage the Ultralytics YOLO-OBB checkpoints for the Phase 8F-2 object-detection track and record, in the provenance
manifest, what was staged, from where, under which licence, and WHY this detector was chosen.

This is one of the ONLY modules in geoseek allowed to touch the network (see the project-wide rule in README.md /
config.py). Everything downstream reads the local files under ``data/models/yolo_obb/``. Re-running skips the network for
files already present. Every fact below was checked live on 2026-09-20 (not recalled), by the commands noted.

Detector choice
---------------
PRIMARY: **Ultralytics YOLO26s-OBB**, fine-tuned from its DOTAv1-pretrained weights, 8 classes.

Framework / availability (``curl https://pypi.org/pypi/ultralytics/json``, GitHub API):
  * ``ultralytics`` 8.4.156 is current on PyPI (uploaded 2026-09-19); 8.4.152 is installed here. Licence AGPL-3.0
    (PyPI classifier AGPLv3+; GitHub SPDX AGPL-3.0; the LICENSE file is GNU AGPL v3 and contains section 13, "Remote
    Network Interaction").
  * The OBB docs page (docs.ultralytics.com/tasks/obb) now lists ONLY YOLO26-OBB. The ``ultralytics/assets`` release
    v8.4.0 (2026-01-13) carries yolo26{n,s,m,l,x}-obb.pt next to yolo11* and yolov8*; all are DOTAv1-pretrained.
  * A newer release DOES supersede YOLOv8-OBB: YOLO26 (Jan 2026; NMS-free dual head, DFL removed, "STAL" small-target-aware
    label assignment, a specialised OBB angle loss, MuSGD). Same package, same API.

Pretrained-checkpoint numbers (read from each .pt's own ``train_metrics``, val split of DOTAv1, 15 classes, chip level):
  yolov8n-obb 0.765 / 0.601   yolo11n-obb 0.773 / 0.610   yolo26n-obb 0.766 / 0.613   yolo26s-obb 0.789 / 0.645
  (mAP50 / mAP50-95). The docs' published figure is mAPtest50 on the DOTA server, multi-scale: YOLO26n 78.9, s 80.9.

Alternative assessed: Oriented R-CNN via MMRotate (Apache-2.0).
  * MMRotate's last tag is v0.3.4 (2023-02-01; a v1.0.0rc1 pre-release exists from 2023-01-03) and the repository's
    default branch was last pushed 2024-09-28. It needs an mmcv/mmdet(/mmengine) version lattice built for torch 1.x/2.0,
    against this project's pinned torch 2.2.2 + CUDA 12.1 on Windows.
  * MMRotate's own results table lists Oriented R-CNN R50 (1x, 1024x1024 crops) at 7.37 GB (fp16) / 8.46 GB (fp32) of
    training memory at batch 2 PER GPU. This machine's RTX 4060 Laptop has 8.59 GB total and ~7.2 GB usable (the Windows
    desktop holds ~1.3 GB).  YOLO26s-OBB was MEASURED at 6.4-6.6 GB at batch 8 (scripts/detect_bench.py) - see below.
  * Two-stage, so no NMS-free option and a heavier per-image cost; published DOTA-v1.0 test mAP 75.63-75.87 (R50).

Why YOLO26s over the alternatives, against OUR constraints (measured, scripts/detect_bench.py, real augmenting dataloader,
worst-case label density, hard torch VRAM cap):
  * 8 GB VRAM: yolo26s batch 16 does NOT fit (the driver's system-memory fallback spilled into RAM); batch 8 fits at 6.4-6.6
    GB and 20.5 samples/s. yolo26n batch 16 runs 24.2 samples/s - only ~1.35x faster for 2.3 mAP50 / 3.2 mAP50-95 less
    (checkpoint-reported), so the small model saves little wall clock and gives up real accuracy on small vehicles.
  * Offline after staging: one pip package + one 22 MB .pt (vs a 3-package MMRotate lattice).
  * Small objects (vehicles at ~15 px): YOLO26's small-target-aware assignment is aimed at exactly this.
  * The previous session's pick (YOLOv8n-obb) is staged and benchmarked for the record; its "proven recipe" argument does
    not outweigh a 2.4-point mAP50 deficit against a same-package successor that trains fine on torch 2.2.2.

LICENCE - FLAGGED, and it matters for anything beyond this prototype:
  * ultralytics AND its pretrained weights are AGPL-3.0. geoseek's own pyproject declares Apache-2.0. Linking the two means
    a distributed or network-served build of the combined work must offer its source under the AGPL (sec. 13), or use an
    Ultralytics Enterprise licence. For this non-commercial, open SIH prototype that is not a blocker.
  * Mitigated structurally: ultralytics is an OPTIONAL extra (``pip install geoseek[detect]``), imported only inside
    ``geoseek.models.yolo_obb`` behind the dependency-free ``ObjectDetectionModel`` ABC, and a test asserts that importing
    geoseek does not import it. Swapping the detector is a new subclass.
  * The fine-tuned weights are a derivative of AGPL-3.0 weights AND of DOTA (academic use only, commercial use prohibited;
    Google Earth imagery under Google's terms), so they are non-commercial as well.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_analysis_section, record_artifact

ULTRALYTICS_PACKAGE_LATEST_ON_PYPI = "8.4.156"          # verified 2026-09-20; installed: see ultralytics.__version__
PACKAGE_LICENSE = "AGPL-3.0 (open source; Ultralytics Enterprise licence available for closed-source / hosted use)"
WEIGHTS_LICENSE = ("AGPL-3.0 (same as the ultralytics package). Pretrained on DOTAv1, which is academic-use-only "
                   "(commercial use prohibited).")

RELEASE_URL = "https://github.com/ultralytics/assets/releases/download/{tag}/{name}"

CHECKPOINTS: dict[str, dict] = {
    "yolo26s-obb.pt": {"tag": "v8.4.0", "role": "PRIMARY - fine-tuned into the geoseek detector"},
    "yolo26n-obb.pt": {"tag": "v8.4.0", "role": "benchmarked alternative (smaller)"},
    "yolo11n-obb.pt": {"tag": "v8.3.0", "role": "checkpoint-metadata comparison only"},
    "yolov8n-obb.pt": {"tag": "v8.3.0", "role": "previous session's choice; benchmarked for the record"},
    "yolo26n.pt": {
        "tag": "v8.4.0",
        "role": ("NOT a geoseek model: ultralytics' check_amp() loads it to self-test that FP32 and AMP/FP16 agree on this "
                 "GPU; staged so an offline amp=True run never tries to download it"),
    },
}


def weights_dir() -> Path:
    return get_settings().models_dir / "yolo_obb"


def checkpoint_path(name: str = "yolo26s-obb.pt") -> Path:
    return weights_dir() / name


def _download(name: str, tag: str) -> tuple[Path, str]:
    dest = checkpoint_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RELEASE_URL.format(tag=tag, name=name)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[staging] {name}: already present ({dest.stat().st_size / 1e6:.1f} MB) - skipping network")
        return dest, url
    print(f"[staging] {name}: downloading {url}")
    tmp = dest.with_suffix(".pt.part")
    try:
        urllib.request.urlretrieve(url, tmp)
        shutil.move(str(tmp), str(dest))
    except Exception as e:
        raise RuntimeError(f"failed to download {url}: {e}. Not falling back to an unverified mirror.") from e
    return dest, url


def read_checkpoint_metadata(path: Path) -> dict:
    """What the checkpoint says about itself (no ultralytics import needed beyond unpickling)."""
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    model = ck.get("model")
    ta, tm = ck.get("train_args") or {}, ck.get("train_metrics") or {}
    return {
        "ultralytics_version_at_save": ck.get("version"), "saved_at": ck.get("date"),
        "task": getattr(model, "task", None), "n_classes": getattr(model, "nc", None),
        "n_params": sum(p.numel() for p in model.parameters()) if model is not None else None,
        "head": type(model.model[-1]).__name__ if model is not None else None,
        "train_data": ta.get("data"), "train_epochs": ta.get("epochs"), "train_imgsz": ta.get("imgsz"),
        "train_optimizer": ta.get("optimizer"),
        "self_reported_val": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in tm.items()
                              if k.startswith("metrics/")},
    }


def stage_yolo_obb(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== Ultralytics YOLO-OBB checkpoints -> {weights_dir()} ===")

    staged: dict[str, dict] = {}
    for name, meta in CHECKPOINTS.items():
        if force and checkpoint_path(name).exists():
            checkpoint_path(name).unlink()
        path, url = _download(name, meta["tag"])
        rec = record_artifact(name=f"ultralytics-{name.replace('.pt', '')}", source_url=url, local_path=path,
                              license=WEIGHTS_LICENSE)
        entry = {"role": meta["role"], "release_tag": meta["tag"], "source_url": url, "local_path": str(path.resolve()),
                 "sha256": rec.sha256, "byte_size": rec.byte_size, "license": WEIGHTS_LICENSE}
        if name != "yolo26n.pt":
            entry["checkpoint_metadata"] = read_checkpoint_metadata(path)
        staged[name] = entry
        print(f"[staging]   {name}: sha256={rec.sha256[:16]}... {rec.byte_size} B  ({meta['role']})")

    section = {
        "primary": "Ultralytics YOLO26s-OBB (yolo26s-obb.pt), fine-tuned to 8 classes (see detector_model_card)",
        "package": {"name": "ultralytics", "latest_on_pypi_2026-09-20": ULTRALYTICS_PACKAGE_LATEST_ON_PYPI,
                    "license": PACKAGE_LICENSE},
        "agpl_flag": ("geoseek is Apache-2.0; ultralytics and its weights are AGPL-3.0 (sec. 13 network clause). Fine for a "
                      "non-commercial open prototype; a closed-source / hosted product would need the source released "
                      "or an Ultralytics Enterprise licence. Mitigation: optional extra, lazy import in "
                      "geoseek.models.yolo_obb only, dependency-free ObjectDetectionModel ABC, a test that importing "
                      "geoseek does not import ultralytics."),
        "newer_release_supersedes_yolov8_obb": ("YES - YOLO26-OBB (Jan 2026; docs list only YOLO26-OBB). Same package."),
        "alternative_assessed": {
            "name": "Oriented R-CNN via MMRotate (Apache-2.0)",
            "mmrotate_last_tag": "v0.3.4 (2023-02-01); v1.0.0rc1 pre-release 2023-01-03; default branch last pushed 2024-09-28",
            "training_memory_gb_batch2_per_gpu_1024_crops": {"fp16": 7.37, "fp32": 8.46},
            "published_dota_v1_0_test_map": {"R50": "75.63-75.87", "source": "MMRotate configs/oriented_rcnn README; arXiv 2108.05699"},
            "rejected_because": ("weights 7.4-8.5 GB at batch 2 vs ~7.2 GB usable here; stale dependency lattice vs torch 2.2.2; "
                                 "two-stage (no NMS-free option); heavier per-image cost"),
        },
        "gpu": "NVIDIA GeForce RTX 4060 Laptop, 8.59 GB total, ~7.2 GB usable (desktop holds ~1.3 GB)",
        "checkpoints_staged": staged,
        "corrections_to_the_2026-09-15_version_of_this_note": [
            "MMRotate v0.3.4 is dated 2023-02-01 (the earlier note read the date as 2023-01-02)",
            "ultralytics latest is 8.4.156 (earlier note: 8.4.152, which is what is installed)",
            "the earlier 'DOTAv1 train split, incl. our staged train/val imagery' licence text was garbled",
        ],
    }
    record_analysis_section("yolo_obb_detector", section)
    print(f"[staging] Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-yolo-obb")
    p.add_argument("--force", action="store_true", help="re-download even if already staged")
    args = p.parse_args(argv)
    stage_yolo_obb(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

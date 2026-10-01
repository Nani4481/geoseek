"""Capture the measurement environment: GPU, driver, torch, power source, git state, timestamp.

Capability benchmarks must be run on AC power and the power source recorded - on this laptop an
unplugged RTX 4060 + CPU roughly halves sustained throughput (docs/EVALUATION_REPORT.md section 4).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from geoseek.config import PROJECT_ROOT


def power_source() -> dict:
    """``{"source": "AC"|"battery"|"unknown", "battery_percent": int|None}`` from the OS (no shell-out)."""
    if sys.platform != "win32":
        return {"source": "unknown", "battery_percent": None, "detail": "non-Windows host"}

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = _Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return {"source": "unknown", "battery_percent": None, "detail": "GetSystemPowerStatus failed"}
    source = {1: "AC", 0: "battery"}.get(st.ACLineStatus, "unknown")
    pct = None if st.BatteryLifePercent == 255 else int(st.BatteryLifePercent)
    return {"source": source, "battery_percent": pct}


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 20.0) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def nvidia_smi_info() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.draw,clocks.sm,clocks.max.sm",
                "--format=csv,noheader,nounits"])
    if not out:
        return {}
    name, driver, mem, power, clk, clk_max = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"name": name, "driver_version": driver, "memory_total_mib": float(mem),
            "power_draw_w_at_capture": float(power) if power not in ("[N/A]", "") else None,
            "sm_clock_mhz_at_capture": float(clk) if clk not in ("[N/A]", "") else None,
            "sm_clock_max_mhz": float(clk_max) if clk_max not in ("[N/A]", "") else None}


def git_info(root: Path | None = None) -> dict:
    root = root or PROJECT_ROOT
    sha = _run(["git", "rev-parse", "HEAD"], root)
    status = _run(["git", "status", "--porcelain"], root)
    diff = _run(["git", "diff", "HEAD"], root, timeout=60.0) or ""
    changed = [ln[3:] for ln in (status or "").splitlines()]
    return {"sha": sha, "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
            "dirty": bool(status), "n_changed_paths": len(changed), "changed_paths": changed,
            "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8", "replace")).hexdigest() if diff else None}


def capture_environment() -> dict:
    import numpy as np
    import torch

    gpu = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu.update({"name": props.name, "vram_total_mib": round(props.total_memory / 2**20, 1),
                    "compute_capability": f"{props.major}.{props.minor}", "cuda_runtime": torch.version.cuda})
    gpu.update({f"nvidia_smi_{k}": v for k, v in nvidia_smi_info().items()})
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for mod in ("faiss", "open_clip", "rasterio", "ultralytics"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu,
        "power": power_source(),
        "versions": versions,
        "git": git_info(),
        "host": {"platform": platform.platform(), "cpu_logical_cores": os.cpu_count()},
        "vram_cap_fraction": 0.80,
    }

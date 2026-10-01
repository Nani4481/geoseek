"""CPU microbenchmark; no ingestion, training, index rebuild or source DB writes.

Run from repo root: python deploy/gcp/benchmark.py
Local CPU timings are not Cloud Run latency guarantees.
"""
import json
import os
import platform
import statistics
import time
from pathlib import Path

os.environ["GEOSEEK_DEVICE"] = "cpu"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch

torch.set_num_threads(2)
from geoseek.config import get_settings
from geoseek.models.remoteclip import RemoteCLIPEmbeddingModel
from geoseek.change.models.fc_siam_diff_model import FCSiamDiffChangeModel
from geoseek.vectorindex import FaissFlatIPIndex


def measure(fn, count=10):
    fn()
    values = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(values), "max_ms": max(values), "samples": count}


def main():
    settings = get_settings()
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()}
    start = time.perf_counter()
    remote = RemoteCLIPEmbeddingModel()
    remote.load()
    result["remoteclip_load_seconds"] = time.perf_counter() - start
    result["remoteclip_text"] = measure(lambda: remote.encode_text("a river with sandbars"))
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    result["remoteclip_image_synthetic"] = measure(lambda: remote.encode_image(rgb))
    index = FaissFlatIPIndex(settings.faiss_index_path)
    result["index_vectors"] = index.count()
    query = remote.encode_text("a river")
    result["faiss_exact_all_vectors"] = measure(lambda: index.search(query, index.count()))
    change = FCSiamDiffChangeModel(device="cpu")
    change.load()
    model = change._loaded.model
    x = torch.zeros(1, len(change._loaded.bands), 256, 256)
    with torch.inference_mode():
        result["fc_siam_diff_synthetic_256_pair"] = measure(lambda: model(x, x))
    # Peak resident memory includes both models + index; OS-specific units.
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        m = Memory()
        m.cb = ctypes.sizeof(m)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        if not memory_info(process(), ctypes.byref(m), m.cb):
            raise ctypes.WinError()
        result["peak_rss_bytes"] = m.PeakWorkingSetSize
    else:
        import resource
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    out = Path("deploy/gcp/benchmark-results.json")
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

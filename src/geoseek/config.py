"""Central configuration: filesystem layout + device selection for geoseek.

Hard rule: geoseek runs fully offline after staging. Nothing outside of
``geoseek.staging`` may open a network connection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

# ViT-B-32 (and RemoteCLIP's ViT-B-32 checkpoint) produces 512-d embeddings.
EMBEDDING_DIM = 512

# Repo layout is fixed relative to this file: src/geoseek/config.py -> geoseek/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    datasets_dir: Path = PROJECT_ROOT / "data" / "datasets"
    tiles_dir: Path = PROJECT_ROOT / "data" / "tiles"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    provenance_manifest_path: Path = PROJECT_ROOT / "data" / "provenance_manifest.json"

    device: str = field(default="cpu")
    embedding_dim: int = EMBEDDING_DIM

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.datasets_dir, self.tiles_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


def select_device() -> str:
    """Pick cuda if available, else cpu. No user override magic — explicit and predictable."""
    override = os.environ.get("GEOSEEK_DEVICE")
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings(device=select_device())
    settings.ensure_dirs()
    return settings


def gpu_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def print_startup_banner(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    cuda_available = torch.cuda.is_available()
    name = gpu_name() or "N/A (running on CPU)"

    banner = f"""
{'=' * 60}
 geoseek - offline geospatial ML platform
{'=' * 60}
 torch version   : {torch.__version__}
 CUDA available  : {cuda_available}
 GPU name        : {name}
 selected device : {settings.device}
 embedding dim   : {settings.embedding_dim}
 data dir        : {settings.data_dir}
{'=' * 60}
""".strip("\n")
    print(banner)


if __name__ == "__main__":
    print_startup_banner()

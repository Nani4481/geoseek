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

    @property
    def faiss_index_path(self) -> Path:
        return Path(os.environ.get("FAISS_INDEX_PATH") or self.index_dir / "tiles.faiss")

    @property
    def database_path(self) -> Path:
        url = os.environ.get("DATABASE_URL")
        if not url:
            return self.index_dir / "tiles.sqlite"
        if not url.startswith("sqlite:///") or not url.removeprefix("sqlite:///"):
            raise ValueError("DATABASE_URL must be sqlite:/// followed by a local file path")
        return Path(url.removeprefix("sqlite:///"))

    @property
    def model_path(self) -> Path:
        return Path(os.environ.get("MODEL_PATH") or self.models_dir / "RemoteCLIP-ViT-B-32.pt")

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
    # Empty variables retain the original local layout. GCS is mounted/synced
    # by deployment tooling; application paths always remain filesystem paths.
    data = Path(os.environ.get("GEOSEEK_DATA_DIR") or PROJECT_ROOT / "data")
    settings = Settings(
        device=select_device(), data_dir=data,
        models_dir=Path(os.environ.get("GEOSEEK_MODELS_DIR") or data / "models"),
        datasets_dir=Path(os.environ.get("GEOSEEK_DATASETS_DIR") or data / "datasets"),
        tiles_dir=Path(os.environ.get("GEOSEEK_TILES_DIR") or data / "tiles"),
        index_dir=Path(os.environ.get("GEOSEEK_INDEX_DIR") or data / "index"),
        provenance_manifest_path=Path(os.environ.get("GEOSEEK_PROVENANCE_PATH") or data / "provenance_manifest.json"),
    )
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

"""Environment + offline-loading smoke tests.

These tests must pass with the network OFF, provided staging has already run
once (`python -m geoseek.staging.download_models`). They never call
hf_hub_download or the HF API — only geoseek.staging.download_models.load_remote_clip,
which reads the checkpoint straight off disk.
"""

import json

import pytest
import torch

from geoseek.config import get_settings
from geoseek.staging.download_models import (
    CHECKPOINT_FILENAME,
    encode_dummy_image,
    load_remote_clip,
)


def test_torch_imports():
    assert torch.__version__


def test_cuda_available():
    assert torch.cuda.is_available(), (
        "CUDA not visible to torch. Target hardware is an RTX 4060 — "
        "check the torch build (needs a CUDA wheel / conda pytorch-cuda, not CPU-only)."
    )


def test_remoteclip_loads_from_local_file_and_encodes_to_512():
    settings = get_settings()
    checkpoint_path = settings.models_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        pytest.fail(
            f"Staged checkpoint not found at {checkpoint_path}. "
            "Run `python -m geoseek.staging.download_models` once (with network on) before testing."
        )

    # load_remote_clip only ever does torch.load() on a local path — no network involved.
    model = load_remote_clip(checkpoint_path, settings.device)
    embed_dim = encode_dummy_image(model, settings.device)

    assert embed_dim == 512
    assert embed_dim == settings.embedding_dim


def test_provenance_manifest_is_valid():
    settings = get_settings()
    manifest_path = settings.provenance_manifest_path
    if not manifest_path.is_file():
        pytest.fail(f"Provenance manifest missing at {manifest_path}. Run staging first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    assert artifacts, "provenance_manifest.json has no recorded artifacts"

    record = next((a for a in artifacts if a["name"] == "RemoteCLIP-ViT-B-32"), None)
    assert record is not None, "RemoteCLIP-ViT-B-32 not recorded in provenance manifest"

    assert len(record["sha256"]) == 64
    int(record["sha256"], 16)  # raises ValueError if not valid hex
    assert record["byte_size"] > 0
    assert record["license"]
    assert record["source_url"].startswith("https://")
    assert record["staged_at"]

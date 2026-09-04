"""Stage RemoteCLIP (ViT-B-32, OpenCLIP format) weights for fully-offline use.

This is the ONLY module in geoseek that is allowed to touch the network.
Everything downstream (config, inference, tests) must load exclusively from
the local file staged here under data/models/.

Checkpoint provenance
----------------------
The exact repo id and filename below were NOT recalled from memory. They were
confirmed by searching Hugging Face directly against the official RemoteCLIP
paper repo (https://github.com/ChenDelong1999/RemoteCLIP, "RemoteCLIP: A
Vision-Language Foundation Model for Remote Sensing", IEEE TGRS 2023):

    Hugging Face repo : chendelong/RemoteCLIP
    File              : RemoteCLIP-ViT-B-32.pt   (OpenCLIP-compatible state_dict)
    License           : Apache-2.0 (per ChenDelong1999/RemoteCLIP GitHub LICENSE;
                         the HF repo itself carries no separate license tag)

At staging time this script re-verifies the filename against the *live* HF
repo listing via the Hub API before downloading, instead of trusting a
hardcoded URL blindly.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import open_clip
import torch
from huggingface_hub import HfApi, hf_hub_download

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_artifact

HF_REPO_ID = "chendelong/RemoteCLIP"
CHECKPOINT_FILENAME = "RemoteCLIP-ViT-B-32.pt"
MODEL_LICENSE = "Apache-2.0"
OPEN_CLIP_ARCH = "ViT-B-32"


def verify_checkpoint_exists_on_hf(repo_id: str, filename: str) -> str:
    """Query the live HF repo listing and confirm `filename` is really there.

    Fails loudly instead of silently falling back to a guessed URL or to
    plain (non-remote-sensing) CLIP weights.
    """
    print(f"[staging] Querying Hugging Face for repo '{repo_id}' ...")
    try:
        info = HfApi().model_info(repo_id)
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Hugging Face to verify checkpoint repo '{repo_id}'. "
            f"Staging requires network access on first run. Original error: {e}"
        ) from e

    filenames = [s.rfilename for s in info.siblings]
    if filename not in filenames:
        raise RuntimeError(
            f"Expected checkpoint file '{filename}' not found in HF repo '{repo_id}'. "
            f"Files present there: {filenames}. Refusing to guess a URL — aborting."
        )

    print(f"[staging] Confirmed: {repo_id}/{filename} exists on Hugging Face.")
    source_url = f"https://huggingface.co/{repo_id}/blob/main/{filename}"
    print(f"[staging] Source URL: {source_url}")
    return source_url


def download_checkpoint(settings) -> tuple[Path, str]:
    dest = settings.models_dir / CHECKPOINT_FILENAME
    default_source_url = f"https://huggingface.co/{HF_REPO_ID}/blob/main/{CHECKPOINT_FILENAME}"

    if dest.is_file():
        print(f"[staging] Found existing staged checkpoint at {dest} - skipping network, using local file.")
        return dest, default_source_url

    source_url = verify_checkpoint_exists_on_hf(HF_REPO_ID, CHECKPOINT_FILENAME)

    print(f"[staging] Downloading {CHECKPOINT_FILENAME} from {HF_REPO_ID} ...")
    try:
        cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILENAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download RemoteCLIP checkpoint from Hugging Face "
            f"({HF_REPO_ID}/{CHECKPOINT_FILENAME}). No fallback to plain CLIP weights "
            f"will be used. Original error: {e}"
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached_path, dest)
    print(f"[staging] Staged checkpoint to {dest}")
    return dest, source_url


def load_remote_clip(checkpoint_path: Path, device: str):
    print(f"[staging] Building open_clip architecture '{OPEN_CLIP_ARCH}' (random init) ...")
    try:
        model, _, _preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_ARCH, pretrained=None)
    except Exception as e:
        raise RuntimeError(f"Failed to construct open_clip architecture '{OPEN_CLIP_ARCH}': {e}") from e

    print(f"[staging] Loading RemoteCLIP weights from local file {checkpoint_path} ...")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        load_result = model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load RemoteCLIP state_dict from {checkpoint_path}. "
            "Refusing to silently fall back to plain CLIP weights. "
            f"Original error: {e}"
        ) from e
    print(f"[staging] load_state_dict result: {load_result}")

    return model.to(device).eval()


def encode_dummy_image(model, device: str) -> int:
    dummy = torch.rand(1, 3, 224, 224, device=device)
    with torch.no_grad():
        features = model.encode_image(dummy)
    return int(features.shape[-1])


VANILLA_CACHE_DIRNAME = "openclip_vanilla_cache"
VANILLA_PRETRAINED_TAG = "openai"
VANILLA_LICENSE = "MIT (OpenAI CLIP ViT-B/32 weights, served via open_clip's pretrained='openai')"


def stage_vanilla_openclip() -> Path:
    """Stage vanilla (non-remote-sensing) OpenCLIP ViT-B-32 - Step C control model only.

    Never used by the real search engine. Exists purely so scripts/prove_semantic.py
    can show that RemoteCLIP's staged weights behave visibly differently (better)
    on satellite queries than an off-the-shelf natural-image CLIP - proof the
    staged checkpoint really is the remote-sensing-tuned model, not vanilla CLIP.
    Cached under data/models/ so later runs never touch the network again.
    """
    settings = get_settings()
    cache_dir = settings.models_dir / VANILLA_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    already_cached = any(cache_dir.rglob("*"))
    if already_cached:
        print(f"[staging] Vanilla OpenCLIP cache already populated at {cache_dir} - skipping network.")
    else:
        print(f"[staging] Staging vanilla OpenCLIP ({OPEN_CLIP_ARCH}, pretrained='{VANILLA_PRETRAINED_TAG}') "
              f"into {cache_dir} ...")

    import open_clip  # local import: only needed for this optional control-model path

    # force_quick_gelu=True: matches how the 'openai' checkpoint was actually trained
    # (see the matching comment in geoseek.ingest.embed.load_vanilla_clip_once).
    model, _, _ = open_clip.create_model_and_transforms(
        OPEN_CLIP_ARCH, pretrained=VANILLA_PRETRAINED_TAG, cache_dir=str(cache_dir), force_quick_gelu=True,
    )
    del model
    print(f"[staging] Vanilla OpenCLIP staged/cached at {cache_dir}.")

    for f in sorted(cache_dir.rglob("*")):
        if f.is_file():
            record_artifact(
                name=f"vanilla-openclip-{OPEN_CLIP_ARCH}-{f.name}",
                source_url=f"open_clip pretrained='{VANILLA_PRETRAINED_TAG}' ({OPEN_CLIP_ARCH})",
                local_path=f,
                license=VANILLA_LICENSE,
            )
    print(f"[staging] Manifest: {settings.provenance_manifest_path}")
    return cache_dir


def main() -> None:
    settings = get_settings()
    print_startup_banner(settings)

    checkpoint_path, source_url = download_checkpoint(settings)
    model = load_remote_clip(checkpoint_path, settings.device)

    embed_dim = encode_dummy_image(model, settings.device)
    print(f"[staging] Encoded dummy 224x224x3 image -> embedding dim: {embed_dim}")
    if embed_dim != settings.embedding_dim:
        raise RuntimeError(
            f"RemoteCLIP produced embedding dim {embed_dim}, expected {settings.embedding_dim}. Aborting."
        )

    record = record_artifact(
        name="RemoteCLIP-ViT-B-32",
        source_url=source_url,
        local_path=checkpoint_path,
        license=MODEL_LICENSE,
    )
    print(f"[staging] Wrote provenance record: sha256={record.sha256} size={record.byte_size}B")
    print(f"[staging] Manifest: {settings.provenance_manifest_path}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[staging] Done. GPU: {gpu} | embedding dim: {embed_dim}")

    if "--vanilla" in sys.argv:
        stage_vanilla_openclip()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

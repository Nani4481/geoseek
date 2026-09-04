"""Turn a tile's raw reflectance bands into a RemoteCLIP embedding.

CRITICAL CORRECTNESS NOTE
--------------------------
Sentinel-2 L2A stores calibrated surface reflectance (roughly 0-4000+,
technically up to 65535, uint16), NOT an 8-bit display image. RemoteCLIP
(like all CLIP-family models) was trained on natural 8-bit RGB images and
expects its own normalization on top of that. Feeding raw reflectance
straight into ``model.encode_image`` produces embeddings that reflect
sensor units, not the natural-color appearance a CLIP model recognizes -
effectively garbage.

The correct pipeline, applied per tile:
    1. Stack B4 (red), B3 (green), B2 (blue) -> HxWx3 reflectance.
    2. FIXED Sentinel-2 true-color stretch: subtract the per-scene BOA additive
       offset (DN), divide by 10000 to get surface reflectance, clip to a
       FIXED 0 - 0.30 reflectance window (~DN 0 - 3000, the standard S2
       true-color range), scale to 8-bit, optional fixed gamma. The SAME
       bounds are applied to every tile and every date, so equal ground
       reflectance -> equal 8-bit value everywhere. This replaced an earlier
       per-tile 2-98 percentile stretch, which adapted to each tile's own
       value range and therefore made tiles (and dates) radiometrically
       incomparable - bad for both retrieval consistency and change
       detection. See ``RADIOMETRY_CONFIG`` below; it is written into the
       provenance manifest on every ingest run.
    3. Feed the resulting natural-looking RGB image through OpenCLIP's own
       preprocessing transform (resize to 224 + CLIP mean/std normalization).

Sentinel-2 baseline / BOA_ADD_OFFSET note
-----------------------------------------
Processing baseline 04.00+ introduced BOA_ADD_OFFSET = -1000: raw DN then
encodes ``reflectance*10000 + 1000``. Both staged scenes are served by Earth
Search v1 (collection ``sentinel-2-l2a``) as baseline >= 05.00 with
``earthsearch:boa_offset_applied = true`` and identical ``raster:bands``
(scale 1e-4, offset -0.1). Empirically the COG pixels already hold
``reflectance*10000`` for BOTH dates (2024: min ~1, p2 ~254; 2019: median
~1030 - which, read as raw-offset DN, would imply an impossible ~0.003 scene
reflectance), i.e. neither shows the raw-DN "~1000 dark floor". So the
corrective offset here is 0 for both dates and they are already on one
radiometric scale; ``BOA_OFFSET_DN`` stays 0 but is kept as an explicit,
per-scene-overridable hook for future scenes staged on an older baseline.

The model (and its preprocessing transform) is loaded exactly ONCE at first
use and cached at module scope - never reloaded per tile.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from geoseek.config import EMBEDDING_DIM, get_settings
from geoseek.staging.download_models import CHECKPOINT_FILENAME, OPEN_CLIP_ARCH, load_remote_clip

_MODEL = None
_PREPROCESS = None
_TOKENIZER = None
_DEVICE = None
_LOAD_COUNT = 0  # proof-of-single-load counter, inspectable from tests

VANILLA_CACHE_DIRNAME = "openclip_vanilla_cache"
VANILLA_PRETRAINED_TAG = "openai"

_VANILLA_MODEL = None
_VANILLA_PREPROCESS = None
_VANILLA_LOAD_COUNT = 0  # separate counter - this is a distinct model instance, for the Step C control


# --- FIXED Sentinel-2 true-color parameters (same for every tile AND every date) ---
S2_REFLECTANCE_SCALE = 10000.0        # S2 L2A: reflectance = (DN - BOA_offset_DN) / 10000
TRUE_COLOR_MIN_REFLECTANCE = 0.0      # lower clip
TRUE_COLOR_MAX_REFLECTANCE = 0.30     # upper clip ~ DN 3000 - standard S2 true-color
TRUE_COLOR_GAMMA = 1.0               # fixed display gamma (1.0 = identity); recorded for reproducibility
DEFAULT_BOA_OFFSET_DN = 0.0          # DN to subtract before scaling; 0 for both staged dates (see module docstring)

RADIOMETRY_CONFIG = {
    "method": "fixed_true_color_bounds",
    "replaces": "per-tile 2-98 percentile stretch",
    "reflectance_scale_dn_per_unit": S2_REFLECTANCE_SCALE,
    "clip_reflectance": [TRUE_COLOR_MIN_REFLECTANCE, TRUE_COLOR_MAX_REFLECTANCE],
    "clip_dn_equivalent": [
        int(round(TRUE_COLOR_MIN_REFLECTANCE * S2_REFLECTANCE_SCALE)),
        int(round(TRUE_COLOR_MAX_REFLECTANCE * S2_REFLECTANCE_SCALE)),
    ],
    "gamma": TRUE_COLOR_GAMMA,
    "boa_add_offset_dn_subtracted": {
        "S2B_44RPQ_20190330_1_L2A": DEFAULT_BOA_OFFSET_DN,
        "S2A_44RPQ_20240308_0_L2A": DEFAULT_BOA_OFFSET_DN,
        "_default": DEFAULT_BOA_OFFSET_DN,
    },
    "boa_offset_rationale": (
        "Earth Search v1 sentinel-2-l2a: both items baseline>=05.00, "
        "earthsearch:boa_offset_applied=true, identical raster:bands (scale 1e-4, offset -0.1). "
        "COG pixels already encode reflectance*10000 for both dates (verified against DN "
        "distributions over stable surfaces); neither shows the raw-DN ~1000 dark floor, so "
        "no -1000 correction is applied. Dates are harmonized by using identical fixed bounds."
    ),
    "same_bounds_all_tiles_and_dates": True,
}


def boa_offset_dn_for_scene(scene_id: str) -> float:
    """Per-scene BOA additive offset (DN) to SUBTRACT from raw values before scaling.

    Both currently staged dates are already offset-applied upstream (Earth
    Search v1, boa_offset_applied=true) so this is 0. Kept as the single place
    to special-case a future scene staged on an older processing baseline
    (where raw DN would encode ``reflectance*10000 + 1000``).
    """
    return float(RADIOMETRY_CONFIG["boa_add_offset_dn_subtracted"].get(scene_id, DEFAULT_BOA_OFFSET_DN))


def _fixed_true_color_stretch_to_uint8(
    band: np.ndarray, nodata: float | None, boa_offset_dn: float = DEFAULT_BOA_OFFSET_DN
) -> np.ndarray:
    """One band of raw S2 L2A DN -> 8-bit via FIXED reflectance bounds (no per-tile adaptation)."""
    refl = (band.astype(np.float32) - float(boa_offset_dn)) / S2_REFLECTANCE_SCALE
    lo, hi = TRUE_COLOR_MIN_REFLECTANCE, TRUE_COLOR_MAX_REFLECTANCE
    norm = np.clip((refl - lo) / (hi - lo), 0.0, 1.0)
    if TRUE_COLOR_GAMMA != 1.0:
        norm = np.power(norm, 1.0 / TRUE_COLOR_GAMMA)
    out = (norm * 255.0).round().astype(np.uint8)
    if nodata is not None:
        out[band == nodata] = 0  # keep nodata black regardless of the fixed floor
    return out


def make_true_color_uint8(
    bands: dict[str, np.ndarray],
    nodata: float | None = None,
    boa_offset_dn: float = DEFAULT_BOA_OFFSET_DN,
) -> np.ndarray:
    """B04,B03,B02 surface-reflectance DN -> HxWx3 8-bit true-color RGB with FIXED bounds.

    The stretch is identical for every tile and every acquisition date (see
    ``RADIOMETRY_CONFIG``), so equal ground reflectance maps to equal 8-bit
    value - required for cross-date retrieval and change detection.
    """
    r = _fixed_true_color_stretch_to_uint8(bands["B04"], nodata, boa_offset_dn)
    g = _fixed_true_color_stretch_to_uint8(bands["B03"], nodata, boa_offset_dn)
    b = _fixed_true_color_stretch_to_uint8(bands["B02"], nodata, boa_offset_dn)
    return np.stack([r, g, b], axis=-1)


def load_model_once():
    """Load RemoteCLIP + its preprocessing transform once; return the cached instance on every later call."""
    global _MODEL, _PREPROCESS, _DEVICE, _LOAD_COUNT
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _DEVICE

    settings = get_settings()
    checkpoint_path = settings.models_dir / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RemoteCLIP checkpoint not staged at {checkpoint_path}. "
            "Run `python -m geoseek.staging.download_models` first (needs network once)."
        )

    print(f"[embed] Loading RemoteCLIP ({OPEN_CLIP_ARCH}) ONCE from {checkpoint_path} ...")
    t0 = time.time()
    model = load_remote_clip(checkpoint_path, settings.device)
    _, _, preprocess = open_clip.create_model_and_transforms(OPEN_CLIP_ARCH, pretrained=None)
    dt = time.time() - t0

    _MODEL, _PREPROCESS, _DEVICE = model, preprocess, settings.device
    _LOAD_COUNT += 1
    print(f"[embed] Model loaded once in {dt:.2f}s on device={settings.device} "
          f"(load_count={_LOAD_COUNT}) - will be reused for every tile.")
    return _MODEL, _PREPROCESS, _DEVICE


def embed_tile_rgb_uint8(rgb_uint8: np.ndarray) -> tuple[np.ndarray, float]:
    """Embed one already-stretched HxWx3 uint8 RGB tile -> (unit-norm 512-d float32 vector, latency_ms)."""
    model, preprocess, device = load_model_once()
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    tensor = preprocess(pil_img).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    latency_ms = (time.time() - t0) * 1000.0

    vec = feats.squeeze(0).to("cpu").numpy().astype(np.float32)
    return vec, latency_ms


def embed_tiles_batch(
    rgb_uint8_list: list[np.ndarray], batch_size: int = 64
) -> tuple[np.ndarray, list[float]]:
    """Batch-embed many tiles (one forward pass per batch, not per tile).

    Used at ingest time for scale: hundreds/thousands of individual
    model.encode_image(batch=1) calls waste GPU launch overhead. batch_size
    caps VRAM use - ViT-B-32 batches are tiny in memory (a batch of 64
    224x224x3 float32 images is ~38MB) so the default is deliberately
    generous while staying nowhere near the 8GB VRAM budget.

    Returns (vectors (N, EMBEDDING_DIM) float32 unit-norm, per-tile latency_ms
    - each tile in a batch is credited the batch's forward-pass time divided
    evenly across the batch, consistent with the single-tile latency metric).
    """
    model, preprocess, device = load_model_once()
    n = len(rgb_uint8_list)
    vectors = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    latencies_ms: list[float] = []

    for start in range(0, n, batch_size):
        chunk = rgb_uint8_list[start:start + batch_size]
        tensors = torch.stack([preprocess(Image.fromarray(im, mode="RGB")) for im in chunk]).to(device)

        t0 = time.time()
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        dt_ms = (time.time() - t0) * 1000.0

        vectors[start:start + len(chunk)] = feats.to("cpu").numpy().astype(np.float32)
        latencies_ms.extend([dt_ms / len(chunk)] * len(chunk))

    return vectors, latencies_ms


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = open_clip.get_tokenizer(OPEN_CLIP_ARCH)
    return _TOKENIZER


def embed_text(query: str) -> tuple[np.ndarray, float]:
    """Encode a text query with RemoteCLIP's TEXT tower -> (unit-norm 512-d float32 vector, latency_ms)."""
    model, _, device = load_model_once()
    tokenizer = _get_tokenizer()
    tokens = tokenizer([query]).to(device)

    t0 = time.time()
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    latency_ms = (time.time() - t0) * 1000.0

    vec = feats.squeeze(0).to("cpu").numpy().astype(np.float32)
    return vec, latency_ms


def save_sample_png(rgb_uint8: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8, mode="RGB").save(out_path)


# --------------------------------------------------------------------------
# Vanilla (non-remote-sensing) OpenCLIP - Step C control model only. Never
# used by the real search engine; exists purely to prove RemoteCLIP's
# staged weights are actually the remote-sensing-tuned checkpoint by showing
# a visibly worse result on the same satellite queries.
# --------------------------------------------------------------------------


def load_vanilla_clip_once():
    """Load vanilla OpenCLIP (pretrained='openai') once, from its local staged cache only."""
    global _VANILLA_MODEL, _VANILLA_PREPROCESS, _VANILLA_LOAD_COUNT
    if _VANILLA_MODEL is not None:
        return _VANILLA_MODEL, _VANILLA_PREPROCESS

    settings = get_settings()
    cache_dir = settings.models_dir / VANILLA_CACHE_DIRNAME
    if not cache_dir.is_dir() or not any(cache_dir.rglob("*")):
        raise FileNotFoundError(
            f"Vanilla OpenCLIP not staged at {cache_dir}. Run "
            "`python -m geoseek.staging.download_models --vanilla` first (needs network once)."
        )

    # open_clip's pretrained loader pulls timm/HF-hub-backed weights via
    # huggingface_hub, which - even with a populated local cache - still
    # reaches out to the Hub by default to check for a newer revision. This
    # module is not geoseek.staging: it must never touch the network. Forcing
    # HF_HUB_OFFLINE here makes huggingface_hub read the cache only and fail
    # loudly (not silently phone home) if something is actually missing.
    os.environ["HF_HUB_OFFLINE"] = "1"

    print(f"[embed] Loading VANILLA OpenCLIP ({OPEN_CLIP_ARCH}, pretrained='{VANILLA_PRETRAINED_TAG}') "
          f"ONCE from local cache {cache_dir} (HF_HUB_OFFLINE=1) ...")
    t0 = time.time()
    # force_quick_gelu=True: the original OpenAI CLIP checkpoints were trained with
    # QuickGELU activations; open_clip's plain 'ViT-B-32' config defaults to standard
    # GELU and only *warns* ("QuickGELU mismatch...") rather than correcting it. Without
    # this the vanilla control model runs with the wrong activation function throughout -
    # not just "not remote-sensing-tuned" but silently miscalibrated, which would make the
    # Step C comparison unfair (vanilla CLIP handicapped for an unrelated reason).
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPEN_CLIP_ARCH, pretrained=VANILLA_PRETRAINED_TAG, cache_dir=str(cache_dir), force_quick_gelu=True,
    )
    model = model.to(settings.device).eval()
    dt = time.time() - t0

    _VANILLA_MODEL, _VANILLA_PREPROCESS = model, preprocess
    _VANILLA_LOAD_COUNT += 1
    print(f"[embed] Vanilla OpenCLIP loaded once in {dt:.2f}s (load_count={_VANILLA_LOAD_COUNT}).")
    return _VANILLA_MODEL, _VANILLA_PREPROCESS


def embed_tile_rgb_uint8_vanilla(rgb_uint8: np.ndarray) -> np.ndarray:
    """Same interface as embed_tile_rgb_uint8, but through the vanilla OpenCLIP control model."""
    model, preprocess = load_vanilla_clip_once()
    device = get_settings().device
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).to("cpu").numpy().astype(np.float32)


def embed_text_vanilla(query: str) -> np.ndarray:
    model, _ = load_vanilla_clip_once()
    device = get_settings().device
    tokenizer = open_clip.get_tokenizer(OPEN_CLIP_ARCH)
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).to("cpu").numpy().astype(np.float32)


def embed_tiles_batch_vanilla(rgb_uint8_list: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """Same batching strategy as embed_tiles_batch, through the vanilla control model."""
    model, preprocess = load_vanilla_clip_once()
    device = get_settings().device
    n = len(rgb_uint8_list)
    vectors = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    for start in range(0, n, batch_size):
        chunk = rgb_uint8_list[start:start + batch_size]
        tensors = torch.stack([preprocess(Image.fromarray(im, mode="RGB")) for im in chunk]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors[start:start + len(chunk)] = feats.to("cpu").numpy().astype(np.float32)

    return vectors

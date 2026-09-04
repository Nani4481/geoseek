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

The fixed global bounds are for the EMBEDDINGS. For DISPLAY (the analyst UI
thumbnails + detail imagery) the three staged L2A products do not correspond
linearly - B02 (blue) inter-date correlation is only 0.24
(provenance_manifest.radiometric_normalization), 2019 (drought March) land is
~1.7x brighter than 2021/2024, and ~19% of the 2024 AOI has R/B > 2.5 (vs ~6%
in 2021) from brighter dry-season VIS-red/green plus a suppressed blue band. One
global stretch therefore renders 2019 washed-out and 2024 saturated-yellow, so
``make_true_color_uint8(..., per_band_bounds_dn=true_color_bounds_for_scene(id))``
stretches EACH observation on its own robust 2/98 percentile per band (the
standard multi-date EO display). This does not touch the embeddings.

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


# --- Phase 6: per-observation true-color DISPLAY stretch ------------------
# The FIXED global bounds above are right for the EMBEDDINGS (equal DN -> equal
# input, cross-date comparable). They are wrong for DISPLAY: the three staged
# L2A products do not correspond linearly (provenance_manifest:
# radiometric_normalization B02 inter_date_corr only 0.24), 2019 (drought March)
# land is ~1.7x brighter than 2021/2024, and over bright ground 2024 carries a
# non-linear Sen2Cor bias (higher VIS-red/green, lower blue) that no single
# global gain or offset removes - it just renders 2024 saturated-yellow.
# So the analyst UI stretches EACH observation on its own robust 2nd/98th
# percentile per band, over an SCL land+water mask (the standard way any GIS /
# EO browser shows a multi-date stack): every date reads as natural terrain.
# Cross-date *analysis* stays on the normalized-reflectance frame the change
# pipeline uses; this is display only and does not touch the FAISS embeddings.
# Bounds from a one-time full-AOI decimated sample (~1.7-2.0 M px/date).
PER_OBS_TRUE_COLOR_DN_BOUNDS = {
    "S2B_44RPQ_20190330_1_L2A": {"B04": (724.0, 1830.0), "B03": (787.0, 1594.0), "B02": (656.0, 1312.0)},
    "S2A_44RPQ_20210304_1_L2A": {"B04": (227.0, 1908.0), "B03": (355.0, 1608.0), "B02": (73.0, 1194.0)},
    "S2A_44RPQ_20240308_0_L2A": {"B04": (257.0, 2122.0), "B03": (435.0, 1838.0), "B02": (227.0, 1346.0)},
}
# Kept OUT of RADIOMETRY_CONFIG on purpose: RADIOMETRY_CONFIG is the analysis
# radiometry entry (embeddings + change model + spectral indices), recorded to
# the manifest under "radiometry". This display stretch is recorded separately
# under the manifest's "true_color_display_stretch" key - see
# record_true_color_display_stretch() / geoseek.ingest.pipeline.
TRUE_COLOR_DISPLAY_STRETCH_CONFIG = {
    "method": "per-band robust 2nd/98th percentile over an SCL land+water mask, per observation",
    "sample": "one-time full-AOI decimated read (~1.7-2.0 M px per date)",
    "bounds_dn": {k: {b: list(v) for b, v in bands.items()}
                  for k, bands in PER_OBS_TRUE_COLOR_DN_BOUNDS.items()},
    "applies_to": "true-color rendering only (thumbnails + detail imagery); NOT the FAISS embeddings",
    "why": "the 3 L2A products don't correspond linearly (B02 inter_date_corr 0.24); one "
           "global stretch renders 2019 washed-out bright and 2024 saturated-yellow",
}


def true_color_bounds_for_scene(scene_id: str) -> dict[str, tuple[float, float]] | None:
    """Per-band (lo_dn, hi_dn) display stretch for one observation, or ``None``
    for an unknown scene (caller then falls back to the fixed global bounds)."""
    return PER_OBS_TRUE_COLOR_DN_BOUNDS.get(scene_id)


def true_color_offsets_for_scene(scene_id: str) -> dict[str, float]:
    """Per-band DN to SUBTRACT before the stretch: just the BOA baseline offset
    (0 for every staged date). Cross-date display harmonisation is done by
    per-observation bounds (:func:`true_color_bounds_for_scene`)."""
    boa = boa_offset_dn_for_scene(scene_id)
    return {b: boa for b in ("B04", "B03", "B02")}


def _fixed_true_color_stretch_to_uint8(
    band: np.ndarray, nodata: float | None, boa_offset_dn: float = DEFAULT_BOA_OFFSET_DN,
    dn_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """One band of raw S2 L2A DN -> 8-bit.

    Default: the FIXED global reflectance bounds (equal DN -> equal value for
    every tile and date; used for the embeddings). ``dn_bounds`` = (lo, hi) in DN
    overrides them with a per-observation display stretch (analyst UI only).
    """
    b = band.astype(np.float32) - float(boa_offset_dn)
    if dn_bounds is not None:
        lo, hi = float(dn_bounds[0]), float(dn_bounds[1])
        norm = np.clip((b - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        refl = b / S2_REFLECTANCE_SCALE
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
    per_band_offset_dn: dict[str, float] | None = None,
    per_band_bounds_dn: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """B04,B03,B02 raw S2 L2A DN -> HxWx3 8-bit true-color RGB.

    Default: the FIXED global reflectance bounds - identical for every tile and
    date, used for the embeddings. ``per_band_bounds_dn`` (from
    ``true_color_bounds_for_scene``) overrides them with a per-observation
    display stretch so a multi-date stack reads naturally in the analyst UI (the
    three products don't correspond linearly - one global stretch renders 2019
    washed-out and 2024 saturated-yellow). ``per_band_offset_dn`` carries the BOA
    baseline offset (0 for every staged date); scalar ``boa_offset_dn`` is the
    per-band fallback.
    """
    def off(b: str) -> float:
        if per_band_offset_dn is not None:
            return float(per_band_offset_dn.get(b, boa_offset_dn))
        return float(boa_offset_dn)

    def bnd(b: str) -> tuple[float, float] | None:
        return None if per_band_bounds_dn is None else per_band_bounds_dn.get(b)

    r = _fixed_true_color_stretch_to_uint8(bands["B04"], nodata, off("B04"), bnd("B04"))
    g = _fixed_true_color_stretch_to_uint8(bands["B03"], nodata, off("B03"), bnd("B03"))
    b = _fixed_true_color_stretch_to_uint8(bands["B02"], nodata, off("B02"), bnd("B02"))
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

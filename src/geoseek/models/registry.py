"""Model registry: the ONE place a weights path, hash, licence and preprocessing config live.

A declarative mapping ``key -> ModelSpec``. Every spec carries the interface implementation
(a lazy ``"module:Class"`` string, so importing the registry never imports torch / ultralytics),
the weights location, the weights SHA256, the preprocessing configuration, the licence, the
source URL and (for embedding models) the embedding dimension.

``load_model(key)`` verifies the weights SHA256 BEFORE constructing the model and raises
:class:`WeightsIntegrityError` on any mismatch - there is no "warn and continue" path. The
registry is additive: it does not rewire any existing call site (``ingest.embed``,
``FCSiamDiffChangeModel``, ``analyst.detections`` keep their own defaults); a new model is a new
entry here plus a new interface subclass, and every migration tool resolves weights through
``resolve_weights_path`` rather than hardcoding a path.

Adding a model::

    REGISTRY["my-model"] = ModelSpec(key="my-model", kind="embedding", implementation="pkg.mod:Cls",
                                     weights_relpath="my/weights.pt", weights_root="models",
                                     weights_sha256="<64 hex>", ...)
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from geoseek.config import get_settings

KINDS = ("embedding", "change", "detector")
_HEX64 = frozenset("0123456789abcdef")

# The fixed Sentinel-2 true-colour stretch every embedding model sees. Duplicated here (literally,
# not imported) so the registry stays import-light; tests/test_registry.py asserts it equals
# geoseek.ingest.embed.RADIOMETRY_CONFIG so the two cannot drift apart silently.
S2_TRUE_COLOUR_PREPROCESSING = {
    "bands": ["B04", "B03", "B02"],
    "radiometry": "fixed_true_color_bounds",
    "clip_reflectance": [0.0, 0.30],
    "reflectance_scale_dn_per_unit": 10000.0,
    "gamma": 1.0,
    "boa_offset_dn": 0.0,
    "output": "HxWx3 uint8 RGB",
    "model_input": "open_clip transform: resize 224 (bicubic) + center crop + CLIP mean/std normalisation",
}
MAXAR_RGB_PREPROCESSING = {
    "bands": ["R", "G", "B"],
    "radiometry": "none (delivered 8-bit visual product used as-is)",
    "output": "HxWx3 uint8 RGB",
    "model_input": S2_TRUE_COLOUR_PREPROCESSING["model_input"],
}


class RegistryError(RuntimeError):
    """Base class for registry failures."""


class UnknownModelError(RegistryError, KeyError):
    pass


class WeightsMissingError(RegistryError, FileNotFoundError):
    pass


class WeightsIntegrityError(RegistryError):
    """The weights file on disk does not hash to the SHA256 the registry pins."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str                          # one of KINDS
    implementation: str                # "package.module:ClassName" (lazily imported)
    weights_relpath: str               # relative to ``weights_root``; may contain ONE glob (HF snapshot dirs)
    weights_sha256: str
    preprocessing: Mapping[str, Any]
    licence: str
    source_url: str
    embedding_dim: int | None = None   # embedding models only
    weights_root: str = "models"       # "models" -> Settings.models_dir, "data" -> Settings.data_dir
    weights_env_override: str | None = None   # env var that redirects the path (mirrors Settings)
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.key}: kind must be one of {KINDS}, got {self.kind!r}")
        if len(self.weights_sha256) != 64 or set(self.weights_sha256) - _HEX64:
            raise ValueError(f"{self.key}: weights_sha256 must be 64 lowercase hex chars")
        if self.kind == "embedding" and not self.embedding_dim:
            raise ValueError(f"{self.key}: embedding models must declare embedding_dim")
        if self.weights_root not in ("models", "data"):
            raise ValueError(f"{self.key}: weights_root must be 'models' or 'data'")

    def as_dict(self) -> dict:
        return {
            "key": self.key, "kind": self.kind, "implementation": self.implementation,
            "weights_relpath": self.weights_relpath, "weights_root": self.weights_root,
            "weights_sha256": self.weights_sha256, "preprocessing": dict(self.preprocessing),
            "licence": self.licence, "source_url": self.source_url,
            "embedding_dim": self.embedding_dim, "notes": self.notes,
        }


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec, *, replace: bool = False) -> ModelSpec:
    if spec.key in REGISTRY and not replace:
        raise RegistryError(f"model key {spec.key!r} is already registered")
    REGISTRY[spec.key] = spec
    return spec


# --------------------------------------------------------------------------
# the current models
# --------------------------------------------------------------------------

register(ModelSpec(
    key="remoteclip-vitb32",
    kind="embedding",
    implementation="geoseek.models.remoteclip:RemoteCLIPEmbeddingModel",
    weights_relpath="RemoteCLIP-ViT-B-32.pt",
    weights_root="models",
    weights_env_override="MODEL_PATH",
    weights_sha256="60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
    preprocessing=S2_TRUE_COLOUR_PREPROCESSING,
    licence="Apache-2.0",
    source_url="https://huggingface.co/chendelong/RemoteCLIP (file RemoteCLIP-ViT-B-32.pt)",
    embedding_dim=512,
    notes="Production retrieval model (zero fine-tuning). Text + image towers; cosine == inner product.",
))

register(ModelSpec(
    key="openclip-vitb32-openai",
    kind="embedding",
    implementation="geoseek.models.vanilla_clip:VanillaClipEmbeddingModel",
    weights_relpath="openclip_vanilla_cache/models--timm--vit_base_patch32_clip_224.openai/snapshots/*/open_clip_model.safetensors",
    weights_root="models",
    weights_sha256="e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31",
    preprocessing={**S2_TRUE_COLOUR_PREPROCESSING, "force_quick_gelu": True},
    licence="MIT (OpenAI CLIP ViT-B/32 weights)",
    source_url="https://huggingface.co/timm/vit_base_patch32_clip_224.openai",
    embedding_dim=512,
    notes="Evaluation CONTROL only - never in the serving path. Loaded with force_quick_gelu=True so the "
          "control is not handicapped by an activation mismatch.",
))

register(ModelSpec(
    key="fc-siam-diff-oscd",
    kind="change",
    implementation="geoseek.change.models.fc_siam_diff_model:FCSiamDiffChangeModel",
    weights_relpath="change_model/fc_siam_diff.pt",
    weights_root="data",
    weights_sha256="452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab",
    preprocessing={
        "bands": ["B02", "B03", "B04", "B08", "B11"],
        "input": "reflectance (DN / 10000), per-band standardised with data/change_model/norm_stats.json",
        "operating_threshold": 0.80,
    },
    licence="Apache-2.0 (code) / CC-BY-NC-SA-4.0 (weights are a derivative of OSCD - non-commercial, ShareAlike)",
    source_url="in-repo: scripts/train_change.py (architecture: FC-Siam-diff, Daudt et al. 2018); "
               "training data OSCD, https://doi.org/10.21227/asqe-7s69",
    notes="1.085 M parameters; trained on the 11 OSCD fit regions, threshold frozen on 3 validation regions.",
))

register(ModelSpec(
    key="yolo26s-obb-dota15",
    kind="detector",
    implementation="geoseek.models.yolo_obb:YoloObbDetectionModel",
    weights_relpath="detector/geoseek_obb_v15_yolo26s.pt",
    weights_root="models",
    weights_sha256="67f617170b608dd1541d63f4aedbd3b4752cbda9f67702e0d63f4aaaae31e113",
    preprocessing={
        "input": "RGB, 1024x1024 (larger inputs windowed with 200 px overlap)",
        "classes": ["small-vehicle", "large-vehicle", "ship", "plane", "helicopter", "storage-tank", "harbor", "bridge"],
        "operating_confidence": 0.525,
    },
    licence="AGPL-3.0 (derivative of Ultralytics YOLO26s-OBB) AND academic-use-only / non-commercial (DOTA v1.5)",
    source_url="in-repo: scripts/train_detector.py; initialisation https://github.com/ultralytics/ultralytics "
               "(yolo26s-obb.pt); data DOTA v1.5",
    notes="Optional-extra seam: ultralytics is imported lazily by the implementation, never by the registry.",
))


# --------------------------------------------------------------------------
# resolution + verification
# --------------------------------------------------------------------------


def list_models(kind: str | None = None) -> list[str]:
    return sorted(k for k, s in REGISTRY.items() if kind is None or s.kind == kind)


def get_spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise UnknownModelError(f"unknown model key {key!r}; registered: {', '.join(sorted(REGISTRY))}") from None


def resolve_weights_path(spec_or_key: ModelSpec | str, *, root: Path | None = None) -> Path:
    """Absolute weights path for a spec. ``root`` overrides the settings-derived root (tests)."""
    spec = get_spec(spec_or_key) if isinstance(spec_or_key, str) else spec_or_key
    override = os.environ.get(spec.weights_env_override) if spec.weights_env_override and root is None else None
    if override:
        return Path(override)
    settings = get_settings()
    base = root if root is not None else (settings.models_dir if spec.weights_root == "models" else settings.data_dir)
    if "*" in spec.weights_relpath:
        matches = sorted(Path(base).glob(spec.weights_relpath))
        if len(matches) != 1:
            raise WeightsMissingError(
                f"{spec.key}: expected exactly one file matching {spec.weights_relpath!r} under {base}, "
                f"found {len(matches)}")
        return matches[0]
    return Path(base) / spec.weights_relpath


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path, *, chunk: int = 8 * 1024 * 1024) -> str:
    """Streaming SHA256, memoised per (path, size, mtime) so repeated loads in one process hash once."""
    path = Path(path)
    st = path.stat()
    ck = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    hit = _SHA_CACHE.get(ck)
    if hit:
        return hit
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    _SHA_CACHE[ck] = h.hexdigest()
    return _SHA_CACHE[ck]


def verify_weights(key: str, *, root: Path | None = None) -> Path:
    """Resolve the weights file and check its SHA256 against the registry. Returns the path.

    Raises :class:`WeightsMissingError` if the file is absent and :class:`WeightsIntegrityError` if
    it hashes to anything other than the pinned value. Never downgrades to a warning.
    """
    spec = get_spec(key)
    path = resolve_weights_path(spec, root=root)
    if not path.is_file():
        raise WeightsMissingError(f"{key}: weights not found at {path}")
    actual = sha256_file(path)
    if actual != spec.weights_sha256:
        raise WeightsIntegrityError(
            f"{key}: SHA256 mismatch for {path}\n  registry pins : {spec.weights_sha256}\n  file hashes to: {actual}\n"
            "Refusing to load. Either the weights were replaced (update the registry entry deliberately, "
            "with new provenance) or the file is corrupt.")
    return path


def _import_implementation(spec: ModelSpec):
    module_name, _, cls_name = spec.implementation.partition(":")
    return getattr(importlib.import_module(module_name), cls_name)


def load_model(key: str, *, verify: bool = True, root: Path | None = None, **kwargs):
    """Verify the weights, then construct the interface implementation for ``key``.

    ``kwargs`` are forwarded to the implementation constructor (e.g. ``min_score`` for the detector).
    Embedding implementations are path-agnostic (they read the staged location from Settings), so for
    them the registry additionally refuses to proceed if the path it verified is not the path the
    implementation will actually open - otherwise it would verify one file and load another.
    """
    spec = get_spec(key)
    path = verify_weights(key, root=root) if verify else resolve_weights_path(spec, root=root)
    cls = _import_implementation(spec)
    if spec.kind == "change":
        return cls(checkpoint_path=path, **kwargs)
    if spec.kind == "detector":
        return cls(weights_path=path, **kwargs)
    if key == "remoteclip-vitb32" and root is None and Path(get_settings().model_path).resolve() != Path(path).resolve():
        raise RegistryError(f"{key}: registry resolved {path} but the implementation opens {get_settings().model_path}")
    return cls(**kwargs)

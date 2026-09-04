"""RemoteCLIP behind the :class:`EmbeddingModel` interface.

Thin adapter over :mod:`geoseek.ingest.embed` (which owns the single-load model
cache and the fixed true-color preprocessing). The search engine and the ingest
pipeline talk to this interface, not to the module functions directly, so a
different embedding backend is a one-line swap.
"""

from __future__ import annotations

import numpy as np

from geoseek.config import EMBEDDING_DIM
from geoseek.ingest.embed import (
    embed_text,
    embed_tile_rgb_uint8,
    embed_tiles_batch,
    load_model_once,
)
from geoseek.models.base import EmbeddingModel


class RemoteCLIPEmbeddingModel(EmbeddingModel):
    """RemoteCLIP ViT-B-32 (staged weights) - text + image towers, 512-d unit-norm."""

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def load(self) -> None:
        load_model_once()

    def encode_text(self, text: str) -> np.ndarray:
        vec, _ = embed_text(text)
        return vec

    def encode_image(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        vec, _ = embed_tile_rgb_uint8(image_rgb_uint8)
        return vec

    def encode_images(self, images_rgb_uint8: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        vectors, _ = embed_tiles_batch(images_rgb_uint8, batch_size=batch_size)
        return vectors

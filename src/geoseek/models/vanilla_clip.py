"""Vanilla OpenAI CLIP ViT-B/32 behind the :class:`EmbeddingModel` interface.

This is the evaluation CONTROL model (never in the serving path): it exists so the
retrieval evaluation can show RemoteCLIP beating a generic CLIP, and so a new
embedding model can be re-embedded through exactly the same pipeline as the
production one. Thin adapter over the vanilla helpers in :mod:`geoseek.ingest.embed`
(which own the single-load cache and ``force_quick_gelu=True``).
"""

from __future__ import annotations

import numpy as np

from geoseek.config import EMBEDDING_DIM
from geoseek.ingest.embed import (
    embed_text_vanilla,
    embed_tile_rgb_uint8_vanilla,
    embed_tiles_batch_vanilla,
    load_vanilla_clip_once,
)
from geoseek.models.base import EmbeddingModel


class VanillaClipEmbeddingModel(EmbeddingModel):
    """OpenAI CLIP ViT-B/32 (staged offline cache) - text + image towers, 512-d unit-norm."""

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def load(self) -> None:
        load_vanilla_clip_once()

    def encode_text(self, text: str) -> np.ndarray:
        return embed_text_vanilla(text)

    def encode_image(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        return embed_tile_rgb_uint8_vanilla(image_rgb_uint8)

    def encode_images(self, images_rgb_uint8: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        return embed_tiles_batch_vanilla(images_rgb_uint8, batch_size=batch_size)

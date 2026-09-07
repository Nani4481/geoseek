"""VectorIndex: the swap-point for the embedding search backend.

The current implementation is a brute-force FAISS ``IndexFlatIP`` (exact cosine
on unit-norm vectors). The interface is deliberately small so an approximate
index (HNSW, IVF-PQ) can drop in later without touching the search engine or
the ingest pipeline.

Id model: ``add`` returns the integer id assigned to each vector. The flat
implementation assigns them positionally (0, 1, 2, ... == insertion order), so
``get_vector(i)`` reconstructs the i-th vector added. Callers persist that id
on the tile row (``tiles.faiss_id``) and must not assume a particular id
scheme beyond "``add`` tells you the ids".
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorIndexValidation:
    """Result of :meth:`VectorIndex.validate`."""

    ok: bool
    count: int
    expected_count: int
    dim: int
    expected_dim: int
    detail: str = ""


class VectorIndex(abc.ABC):
    """Append-first vector index over unit-norm float32 embeddings."""

    @abc.abstractmethod
    def add(self, vectors: np.ndarray) -> list[int]:
        """Append ``vectors`` (N x dim, float32). Return the id assigned to each, in order."""

    @abc.abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` by inner product. Return ``(scores[k], ids[k])`` for a single query vector.

        ``k`` may exceed :meth:`count`; implementations return whatever exists
        (FAISS pads missing slots with id ``-1``, which callers already skip).
        """

    @abc.abstractmethod
    def get_vector(self, vector_id: int) -> np.ndarray:
        """Reconstruct the stored vector for ``vector_id`` (float32, dim,)."""

    def reconstruct_all(self) -> np.ndarray:
        """Reconstruct every stored vector as one ``(count, dim)`` float32 array.

        Batch jobs (clustering, discovery) need the whole matrix at once. The
        default reconstructs id-by-id via :meth:`get_vector`; implementations
        backed by a contiguous store should override with a bulk read.
        """
        n = self.count()
        out = np.zeros((n, self._reconstruct_dim()), dtype=np.float32)
        for i in range(n):
            out[i] = self.get_vector(i)
        return out

    def _reconstruct_dim(self) -> int:
        return int(getattr(self, "dim", self.get_vector(0).shape[0]) if self.count() else 0)

    @abc.abstractmethod
    def delete(self, vector_ids: list[int]) -> None:
        """Remove vectors by id.

        The positional flat index is append-only - deleting would renumber every
        later id and break the tile <-> vector mapping - so ``FaissFlatIPIndex``
        raises ``NotImplementedError`` here. The method is on the interface so a
        future id-mapped / HNSW index can support it without a signature change.
        """

    @abc.abstractmethod
    def persist(self) -> Path:
        """Write the index to its backing file. Return that path."""

    @abc.abstractmethod
    def load(self) -> None:
        """(Re)load the index from its backing file, replacing in-memory state."""

    @abc.abstractmethod
    def count(self) -> int:
        """Number of vectors currently in the index."""

    @abc.abstractmethod
    def validate(self, expected_count: int, expected_dim: int) -> VectorIndexValidation:
        """Check the vector count matches the catalog and the dimensionality is right."""

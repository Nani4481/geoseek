"""FAISS ``IndexFlatIP`` implementation of :class:`VectorIndex`.

This is the ONLY module in geoseek that imports ``faiss``.

Exact brute-force cosine similarity on unit-norm vectors (inner product ==
cosine). At a few thousand vectors a full scan is sub-millisecond, so there is
no approximate structure to tune and results are deterministic. Vector ids are
positional: the i-th vector added has id ``i``. Persisted as a single
``.faiss`` file via ``faiss.write_index`` / ``read_index`` - byte-identical
across reloads.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from geoseek.config import EMBEDDING_DIM
from geoseek.vectorindex.base import VectorIndex, VectorIndexValidation


class FaissFlatIPIndex(VectorIndex):
    def __init__(self, index_path: Path | str, *, dim: int = EMBEDDING_DIM):
        self.index_path = Path(index_path)
        self.dim = dim
        if self.index_path.is_file():
            self._index = faiss.read_index(str(self.index_path))
        else:
            self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> list[int]:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}) vectors, got {vectors.shape}")
        start = self._index.ntotal
        self._index.add(vectors)  # append-only; prior vectors untouched
        return list(range(start, self._index.ntotal))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        scores, ids = self._index.search(q, k)
        return scores[0], ids[0]

    def get_vector(self, vector_id: int) -> np.ndarray:
        return self._index.reconstruct(int(vector_id))

    def delete(self, vector_ids: list[int]) -> None:
        raise NotImplementedError(
            "FaissFlatIPIndex is append-only: positional ids cannot be deleted without "
            "renumbering every later vector and breaking the tile <-> faiss_id mapping. "
            "Use an id-mapped / HNSW VectorIndex implementation when deletion is required."
        )

    def persist(self) -> Path:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        return self.index_path

    def load(self) -> None:
        if self.index_path.is_file():
            self._index = faiss.read_index(str(self.index_path))
        else:
            self._index = faiss.IndexFlatIP(self.dim)

    def count(self) -> int:
        return int(self._index.ntotal)

    def validate(self, expected_count: int, expected_dim: int) -> VectorIndexValidation:
        count, dim = self.count(), int(self._index.d)
        ok = count == expected_count and dim == expected_dim
        detail = (
            f"faiss ntotal={count} (catalog expects {expected_count}); "
            f"dim={dim} (expects {expected_dim})"
        )
        return VectorIndexValidation(ok, count, expected_count, dim, expected_dim, detail)

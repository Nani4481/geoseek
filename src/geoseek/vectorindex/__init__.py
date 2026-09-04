"""geoseek vector-search seam.

:class:`geoseek.vectorindex.base.VectorIndex` is the interface every embedding
index implements; :class:`geoseek.vectorindex.faiss_flat.FaissFlatIPIndex` is
the current implementation and the ONLY code in geoseek that imports ``faiss``.
Swapping in an HNSW / IVF index later is a new ``VectorIndex`` subclass - no
caller changes.
"""

from geoseek.vectorindex.base import VectorIndex, VectorIndexValidation
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

__all__ = ["VectorIndex", "VectorIndexValidation", "FaissFlatIPIndex"]

"""Model interfaces + their current implementations.

  EmbeddingModel        - text/image -> unit-norm vector.  Impl: RemoteCLIPEmbeddingModel
  QualityEstimator      - tile bands -> cloud / usable fraction.  Impl: SclQualityEstimator
  ChangeDetectionModel  - an ObservationPair -> ChangeResult.  INTERFACE ONLY (Phase 3b)

Swapping a model is a new subclass; callers depend only on the interface.
"""

from geoseek.models.base import (
    ChangeDetectionModel,
    ChangeResult,
    EmbeddingModel,
    QualityEstimator,
    QualityReport,
)
from geoseek.models.quality import SclQualityEstimator
from geoseek.models.remoteclip import RemoteCLIPEmbeddingModel

__all__ = [
    "EmbeddingModel",
    "QualityEstimator",
    "QualityReport",
    "ChangeDetectionModel",
    "ChangeResult",
    "RemoteCLIPEmbeddingModel",
    "SclQualityEstimator",
]

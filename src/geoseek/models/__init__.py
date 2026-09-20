"""Model interfaces + their current implementations.

  EmbeddingModel        - text/image -> unit-norm vector.  Impl: RemoteCLIPEmbeddingModel
  QualityEstimator      - tile bands -> cloud / usable fraction.  Impl: SclQualityEstimator
  ChangeDetectionModel  - an ObservationPair -> ChangeResult.  Impl: geoseek.change.models (Phase 3b)
  ObjectDetectionModel  - an RGB tile -> oriented Detections.  Impl: YoloObbDetectionModel (Phase 8F-2; the only
                          place the AGPL-3.0 ultralytics package is imported, and only lazily)

Swapping a model is a new subclass; callers depend only on the interface.
"""

from geoseek.models.base import (
    ChangeDetectionModel,
    ChangeResult,
    Detection,
    EmbeddingModel,
    ObjectDetectionModel,
    QualityEstimator,
    QualityReport,
    TileGeoRef,
)
from geoseek.models.quality import SclQualityEstimator
from geoseek.models.remoteclip import RemoteCLIPEmbeddingModel
from geoseek.models.yolo_obb import YoloObbDetectionModel

__all__ = [
    "EmbeddingModel",
    "QualityEstimator",
    "QualityReport",
    "ChangeDetectionModel",
    "ChangeResult",
    "ObjectDetectionModel",
    "Detection",
    "TileGeoRef",
    "YoloObbDetectionModel",
    "RemoteCLIPEmbeddingModel",
    "SclQualityEstimator",
]

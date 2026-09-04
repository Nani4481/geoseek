"""Model interfaces for geoseek.

Only :class:`EmbeddingModel` and :class:`QualityEstimator` have implementations
today. :class:`ChangeDetectionModel` is defined here as an interface only - its
input contract is fixed now so Phase 3b slots in without reshaping anything:
``predict_change`` consumes an :class:`geoseek.temporal.contract.ObservationPair`
straight from :class:`geoseek.temporal.matcher.TemporalObservationMatcher`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for annotations
    from geoseek.catalog.entities import TileRecord
    from geoseek.temporal.contract import ObservationPair


# --------------------------------------------------------------------------
# EmbeddingModel
# --------------------------------------------------------------------------


class EmbeddingModel(abc.ABC):
    """Encodes text and imagery into a shared unit-norm embedding space."""

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int: ...

    @abc.abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """One text string -> (embedding_dim,) float32 unit-norm vector."""

    @abc.abstractmethod
    def encode_image(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """One HxWx3 uint8 RGB image -> (embedding_dim,) float32 unit-norm vector."""

    @abc.abstractmethod
    def encode_images(self, images_rgb_uint8: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """N images -> (N, embedding_dim) float32 unit-norm vectors (batched)."""

    def load(self) -> None:
        """Optional: make the model resident before the first call. Default: no-op."""


# --------------------------------------------------------------------------
# QualityEstimator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityReport:
    cloud_fraction: float
    usable_fraction: float
    flags: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"cloud_fraction": self.cloud_fraction, "usable_fraction": self.usable_fraction,
                "flags": self.flags}


class QualityEstimator(abc.ABC):
    """Per-tile (or per-window) usability estimate from the raw bands."""

    @abc.abstractmethod
    def estimate(self, bands: dict[str, np.ndarray]) -> QualityReport:
        """``bands`` maps band name -> 2-D array; returns cloud / usable fractions + flags."""


# --------------------------------------------------------------------------
# ChangeDetectionModel  (INTERFACE ONLY - Phase 3b)
# --------------------------------------------------------------------------


@dataclass
class ChangeResult:
    """Output shape a Phase 3b change-detection model returns for one observation pair.

    Defined now so the matcher -> change-detection boundary is stable. All
    per-tile maps are keyed by ``tile_id`` of the *later* observation.
    """

    earlier_observation_id: str
    later_observation_id: str
    method: str
    change_score_by_tile: dict[str, float] = field(default_factory=dict)
    change_label_by_tile: dict[str, str] = field(default_factory=dict)     # e.g. "construction", "water_loss"
    change_mask_path: str | None = None                                    # referenced by path, not blobbed
    confidence: float = 0.0
    comparable: bool = True                                                # copied from the input pair's verdict
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "earlier_observation_id": self.earlier_observation_id,
            "later_observation_id": self.later_observation_id,
            "method": self.method,
            "n_tiles_scored": len(self.change_score_by_tile),
            "change_mask_path": self.change_mask_path,
            "confidence": self.confidence,
            "comparable": self.comparable,
            "notes": self.notes,
        }


class ChangeDetectionModel(abc.ABC):
    """Predicts change between two observations of the same location.

    The input is an :class:`geoseek.temporal.contract.ObservationPair` exactly as
    :class:`geoseek.temporal.matcher.TemporalObservationMatcher` emits it -
    ``pair.earlier`` / ``pair.later`` carry the full provenance chain, the AOI
    footprint, and the radiometry / co-registration params recorded per
    observation, and ``pair.comparability`` says whether (and why) the pair is
    safe to compare. An implementation SHOULD refuse, or return a low-confidence
    result flagged in ``notes``, when ``pair.comparable`` is ``False``.
    """

    @abc.abstractmethod
    def predict_change(
        self,
        pair: "ObservationPair",
        *,
        tiles: "list[TileRecord] | None" = None,
        aoi_wkt: str | None = None,
        bands: list[str] | None = None,
    ) -> ChangeResult:
        """Compare ``pair.earlier`` and ``pair.later``.

        ``tiles`` optionally restricts the comparison to a tile subset (e.g. the
        matcher's spatially-overlapping tiles); ``aoi_wkt`` optionally restricts
        it to a sub-AOI; ``bands`` selects the spectral bands to use.
        """

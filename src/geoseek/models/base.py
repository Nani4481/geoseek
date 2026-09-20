"""Model interfaces for geoseek.

:class:`EmbeddingModel`, :class:`QualityEstimator`, :class:`ChangeDetectionModel` (Phase 3b) and
:class:`ObjectDetectionModel` (Phase 8F-2) have implementations. :class:`ChangeDetectionModel`'s input
contract was fixed in Phase 3.5 so Phase 3b slotted in without reshaping anything: ``predict_change``
consumes an :class:`geoseek.temporal.contract.ObservationPair` straight from
:class:`geoseek.temporal.matcher.TemporalObservationMatcher`.

:class:`ObjectDetectionModel` was only a design sketch (``docs/ARCHITECTURE.md`` FW-5) until Phase 8F-2 - there
was no ABC in code before it. It is deliberately dependency-free (numpy only): the one concrete implementation,
:class:`geoseek.models.yolo_obb.YoloObbDetectionModel`, is the ONLY place the AGPL-3.0 ``ultralytics`` package is
touched, so swapping the detector (e.g. for a permissively licensed one) is a new subclass and nothing else.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

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


# --------------------------------------------------------------------------
# ObjectDetectionModel  (Phase 8F-2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TileGeoRef:
    """Where a tile sits on the ground, so pixel detections can be turned into lon/lat footprints.

    ``transform`` is the affine tile-pixel -> CRS map as rasterio orders it, ``(a, b, c, d, e, f)`` with
    ``x = a*col + b*row + c`` and ``y = d*col + e*row + f`` (pixel-corner origin); ``crs`` is anything pyproj
    accepts (``"EPSG:32645"``). Kept as plain values so this module needs neither rasterio nor pyproj.
    """

    transform: tuple[float, float, float, float, float, float]
    crs: str


@dataclass(frozen=True)
class Detection:
    """One oriented detection in a tile. Pixel geometry is always present; the lon/lat footprint only with a
    :class:`TileGeoRef`."""

    class_name: str
    class_id: int
    score: float
    obb_px: tuple[float, float, float, float, float]          # (cx, cy, w, h, angle_rad) in tile pixels (image coords)
    polygon_px: tuple[tuple[float, float], ...]                # the 4 corners of that rectangle, tile pixels
    geom_wkt_4326: str | None = None                           # oriented footprint, POLYGON in EPSG:4326 (lon lat)

    @property
    def long_side_px(self) -> float:
        return max(self.obb_px[2], self.obb_px[3])

    @property
    def heading_deg(self) -> float:
        """Direction of the long axis in [0, 180) degrees, image coordinates (y down)."""
        cx, cy, w, h, r = self.obb_px
        a = math.degrees(r) + (0.0 if w >= h else 90.0)
        return a % 180.0

    def as_dict(self) -> dict:
        return {
            "class": self.class_name, "class_id": self.class_id, "score": round(self.score, 4),
            "obb_px": [round(v, 2) for v in self.obb_px],
            "polygon_px": [[round(x, 2), round(y, 2)] for x, y in self.polygon_px],
            "geom_wkt_4326": self.geom_wkt_4326,
        }


class ObjectDetectionModel(abc.ABC):
    """Finds and classifies oriented objects (vehicles, aircraft, vessels, tanks, ...) in one true-colour tile.

    It consumes the same ``HxWx3 uint8 RGB`` tile arrays the embedding pipeline already produces, and stays OUT of the
    retrieval and change seams - a parallel enrichment, not a dependency. Outputs are meant to be registered as
    ``DerivedProduct(kind="detection")`` (a per-observation GeoJSON, referenced by path) so they inherit the provenance
    chain and show up in ``/export`` with no new plumbing.
    """

    @property
    @abc.abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """The classes this model can emit, in class-id order."""

    @abc.abstractmethod
    def detect(
        self,
        tile_rgb_uint8: np.ndarray,
        *,
        classes: Sequence[str] | None = None,
        min_score: float | None = None,
        geo: TileGeoRef | None = None,
    ) -> list[Detection]:
        """Detect objects in one RGB tile.

        ``classes`` restricts the output to those class names (None = all); ``min_score`` overrides the model's
        default operating threshold; with ``geo`` each detection also carries its lon/lat footprint. Tiles larger
        than the model's native window are handled by the implementation (windowed inference + cross-window
        de-duplication), not by the caller."""

    def detect_batch(self, tiles: Sequence[np.ndarray], **kwargs) -> list[list[Detection]]:
        """Default: one tile at a time. Implementations may override with true batching."""
        return [self.detect(t, **kwargs) for t in tiles]

    def load(self) -> None:
        """Optional: make the model resident before the first call. Default: no-op."""

    @property
    def info(self) -> dict:
        """Provenance of the running model (name, weights checksum, operating point). Default: empty."""
        return {}

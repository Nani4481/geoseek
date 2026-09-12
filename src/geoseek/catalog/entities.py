"""Catalog entities: plain, storage-agnostic dataclasses.

Nothing here imports sqlite3 / faiss / a DB driver. The repository layer maps
these to and from rows; a PostGIS implementation would map the same objects.
All geometry is carried as WKT in EPSG:4326.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Collection:
    """A source image collection, e.g. Sentinel-2 L2A."""

    collection_id: str                    # "sentinel-2-l2a"
    sensor: str                           # "MSI"
    platform: str                         # "Sentinel-2" (the constellation)
    bands: tuple[str, ...]                # ("B04","B03","B02","B08","B11","SCL")
    native_gsd_m: float                   # 10.0
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scene:
    """One source product / scene as published by the data provider."""

    scene_id: str                         # source product id, e.g. "S2B_44RPQ_20190330_1_L2A"
    collection_id: str                    # -> Collection.collection_id
    platform: str                         # "Sentinel-2A" / "Sentinel-2B"
    acquired_at: str                      # ISO 8601 acquisition datetime (date-only accepted)
    footprint_wkt_4326: str               # scene footprint geometry
    processing_baseline: str | None = None
    source_url: str | None = None
    license: str | None = None
    crs: str | None = None                # native CRS, e.g. "EPSG:32644"
    checksums: dict = field(default_factory=dict)   # {band: sha256}
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """A scene observed over our AOI at one acquisition time.

    ``observation_id`` doubles as the on-disk dataset directory name under
    ``data/datasets/`` (that is where its band GeoTIFFs live).
    """

    observation_id: str                   # e.g. "S2B_44RPQ_20190330_1_L2A_scaled"
    scene_id: str                         # -> Scene.scene_id
    acquired_at: str                      # copied from the scene, for range queries
    footprint_wkt_4326: str               # scene ∩ AOI footprint
    aoi_name: str | None = None
    dataset_dir: str | None = None        # dir name under data/datasets/ (defaults to observation_id)
    quality_summary: dict = field(default_factory=dict)   # cloud fraction / usable px over the AOI
    radiometry: dict = field(default_factory=dict)        # radiometry + normalization params used
    coregistration: dict = field(default_factory=dict)    # co-registration status vs a reference obs
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Tile:
    """A 256x256 tile belonging to one observation."""

    tile_id: str
    observation_id: str                   # -> Observation.observation_id
    row: int
    col: int
    geom_wkt_4326: str
    cloud_fraction: float
    quality_flags: dict = field(default_factory=dict)
    faiss_id: int | None = None           # vector id in the embedding index (None if not embedded)
    embedding_ref: str | None = None      # which vector index the faiss_id belongs to
    indices_ref: str | None = None        # path to per-tile spectral indices, if computed
    processing_history: list = field(default_factory=list)


@dataclass(frozen=True)
class DerivedProduct:
    """A per-observation or per-tile derived raster/table, referenced by path."""

    derived_id: str
    kind: str                             # "NDVI" | "NDWI" | "NDBI" | "mask" | "thumbnail" | ...
    path: str                             # filesystem path; NOT blobbed into the DB
    observation_id: str | None = None
    tile_id: str | None = None
    params: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class TileProvenance:
    """A tile plus its full upward chain: tile -> observation -> scene -> collection."""

    tile: Tile
    observation: Observation
    scene: Scene
    collection: Collection

    def as_dict(self) -> dict:
        return {
            "tile_id": self.tile.tile_id,
            "observation_id": self.observation.observation_id,
            "scene_id": self.scene.scene_id,
            "collection_id": self.collection.collection_id,
            "acquired_at": self.observation.acquired_at,
            "sensor": self.collection.sensor,
            "platform": self.scene.platform,
            "source_url": self.scene.source_url,
            "license": self.scene.license,
            "processing_baseline": self.scene.processing_baseline,
            "checksums": self.scene.checksums,
            "radiometry": self.observation.radiometry,
            "coregistration": self.observation.coregistration,
            "row": self.tile.row,
            "col": self.tile.col,
            "geom_wkt_4326": self.tile.geom_wkt_4326,
            "cloud_fraction": self.tile.cloud_fraction,
            "faiss_id": self.tile.faiss_id,
            "processing_history": self.tile.processing_history,
        }


@dataclass(frozen=True)
class AnalystDecision:
    """One analyst confirm/reject on a change candidate (PS 2.2.5 audit trail).

    Append-only: a decision row is never overwritten. Re-deciding a candidate
    writes a NEW row with a fresh ``decision_id``; the full history is the
    ordered list of rows for that ``candidate_id``. Every row snapshots the
    exact model / weights / code state and the confidence AND evidence *at the
    time the decision was made*, so the audit log stands on its own even after
    the pipeline is re-run.
    """

    decision_id: str                      # uuid4, assigned on write
    candidate_id: str                     # -> the change candidate this decides
    decision: str                         # "confirm" | "reject"
    analyst_note: str = ""
    analyst: str = ""                     # who decided (free text / login)
    created_at: str = ""                  # ISO-8601 UTC, assigned on write
    model_version: str = ""               # e.g. "FCSiamDiff@0.80"
    weights_sha256: str = ""              # model weights checksum at decision time
    git_commit: str = ""                  # code revision at decision time
    pipeline_version: str = ""            # geoseek package version at decision time
    confidence_at_decision: float | None = None   # queue confidence when decided
    evidence_snapshot: dict = field(default_factory=dict)   # frozen evidence/provenance blob

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "analyst_note": self.analyst_note,
            "analyst": self.analyst,
            "created_at": self.created_at,
            "model_version": self.model_version,
            "weights_sha256": self.weights_sha256,
            "git_commit": self.git_commit,
            "pipeline_version": self.pipeline_version,
            "confidence_at_decision": self.confidence_at_decision,
            "evidence_snapshot": self.evidence_snapshot,
        }


@dataclass(frozen=True)
class WatchArea:
    """A standing watch: an analyst-defined AOI + filters, evaluated against
    every new observation's change candidates (Phase 8 Step C).

    Unlike :class:`AnalystDecision` this is a normal, editable/deletable
    record - an operational definition, not an audit log entry - so the
    repository exposes update/delete for it (see
    :meth:`geoseek.catalog.repository.MetadataRepository.update_watch_area`).
    """

    watch_id: str
    name: str
    bbox: tuple[float, float, float, float] | None = None   # (west, south, east, north), EPSG:4326
    polygon_wkt_4326: str | None = None                      # optional finer AOI; bbox is always set too
    text_query: str = ""                                     # optional; simple keyword match, not semantic search
    change_types: tuple[str, ...] = ()                       # empty = any change type
    min_confidence: float | None = None
    active: bool = True
    created_at: str = ""
    created_by: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict:
        return {
            "watch_id": self.watch_id, "name": self.name, "bbox": list(self.bbox) if self.bbox else None,
            "polygon_wkt_4326": self.polygon_wkt_4326, "text_query": self.text_query,
            "change_types": list(self.change_types), "min_confidence": self.min_confidence,
            "active": self.active, "created_at": self.created_at, "created_by": self.created_by,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class WatchNotification:
    """One firing of a :class:`WatchArea`: the new candidates it matched the
    first time each was seen, produced when the change pipeline is re-run
    after a new observation is ingested (see ``geoseek.watch.evaluator``)."""

    notification_id: str
    watch_id: str
    observation_id: str                       # the newest observation in the run that produced this
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = ""
    seen: bool = False

    def as_dict(self) -> dict:
        return {
            "notification_id": self.notification_id, "watch_id": self.watch_id,
            "observation_id": self.observation_id, "candidate_ids": list(self.candidate_ids),
            "created_at": self.created_at, "seen": self.seen,
        }


@dataclass(frozen=True)
class TileRecord:
    """Flat tile + provenance projection returned by catalog queries.

    This is the shape search / ranking code consumes - a denormalised join of
    tile, observation, scene and collection with just the fields those callers
    filter and display on.
    """

    tile_id: str
    observation_id: str
    scene_id: str
    collection_id: str
    sensor: str
    platform: str
    acq_date: str
    geom_wkt_4326: str
    cloud_fraction: float
    faiss_id: int | None

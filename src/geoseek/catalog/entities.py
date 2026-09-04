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

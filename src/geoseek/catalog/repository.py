"""MetadataRepository: the swap-point for the geospatial / metadata catalog.

The concrete store today is
:class:`geoseek.catalog.sqlite_repository.SQLiteMetadataRepository`. A PostGIS
implementation would subclass this same ABC and be swapped in without touching
a single caller - so the interface stays storage-neutral: it takes and returns
the plain entities from :mod:`geoseek.catalog.entities`, never rows, cursors or
SQL, and expresses spatial filters as bboxes / points rather than backend
geometry predicates.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator

from geoseek.catalog.entities import (
    Collection,
    DerivedProduct,
    Observation,
    Scene,
    Tile,
    TileProvenance,
    TileRecord,
)


class MetadataRepository(abc.ABC):
    """Catalog of collections -> scenes -> observations -> tiles (+ derived products)."""

    # -- registration / writes -------------------------------------------------

    @abc.abstractmethod
    def register_collection(self, collection: Collection) -> None:
        """Upsert a collection (sensor / platform / bands / native GSD)."""

    @abc.abstractmethod
    def register_scene(self, scene: Scene) -> None:
        """Upsert a source scene (product id, footprint, acquisition, baseline, URL, license, checksums)."""

    @abc.abstractmethod
    def register_observation(self, observation: Observation) -> None:
        """Upsert an observation (a scene's AOI intersection at one acquisition time)."""

    @abc.abstractmethod
    def add_tiles(self, tiles: list[Tile]) -> None:
        """Insert tiles. Each tile references an already-registered observation."""

    @abc.abstractmethod
    def upsert_derived(self, derived: DerivedProduct) -> None:
        """Upsert a per-observation / per-tile derived product (referenced by path)."""

    # -- point reads ---------------------------------------------------------

    @abc.abstractmethod
    def get_collection(self, collection_id: str) -> Collection | None: ...

    @abc.abstractmethod
    def get_scene(self, scene_id: str) -> Scene | None: ...

    @abc.abstractmethod
    def get_observation(self, observation_id: str) -> Observation | None: ...

    @abc.abstractmethod
    def get_tile(self, tile_id: str) -> Tile | None: ...

    @abc.abstractmethod
    def get_tile_provenance(self, tile_id: str) -> TileProvenance:
        """The full upward chain for a tile: tile -> observation -> scene -> collection.

        Raises if any link is missing.
        """

    # -- collection reads ----------------------------------------------------

    @abc.abstractmethod
    def list_collections(self) -> list[Collection]: ...

    @abc.abstractmethod
    def list_scenes(self) -> list[Scene]: ...

    @abc.abstractmethod
    def list_tiles(self, *, observation_id: str | None = None) -> list[Tile]: ...

    @abc.abstractmethod
    def list_derived(
        self, *, observation_id: str | None = None, tile_id: str | None = None, kind: str | None = None
    ) -> list[DerivedProduct]: ...

    @abc.abstractmethod
    def list_observations(
        self,
        *,
        location: tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
        collection: str | None = None,
    ) -> list[Observation]:
        """Observations, optionally filtered by a point (lon, lat), bbox, date window, or collection."""

    # -- tile queries ------------------------------------------------------

    @abc.abstractmethod
    def query_tiles(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
        sensor: str | None = None,
        platform: str | None = None,
        max_cloud: float | None = None,
        collection: str | None = None,
    ) -> list[TileRecord]:
        """Tiles (with provenance projected on) matching bbox + date range + sensor + max cloud."""

    @abc.abstractmethod
    def iter_tile_records(self) -> Iterator[TileRecord]:
        """Every tile as a flat provenance record, in ``faiss_id`` order (search-engine refresh)."""

    # -- small aggregates ---------------------------------------------------

    @abc.abstractmethod
    def tile_faiss_pairs(self) -> list[tuple[int | None, str]]:
        """``(faiss_id, tile_id)`` for every tile, ordered by ``faiss_id`` (index<->catalog checks)."""

    @abc.abstractmethod
    def count_tiles(self) -> int: ...

    @abc.abstractmethod
    def close(self) -> None: ...

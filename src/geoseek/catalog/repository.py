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
    AnalystDecision,
    Collection,
    DerivedProduct,
    Observation,
    Scene,
    Tile,
    TileProvenance,
    TileRecord,
    WatchArea,
    WatchNotification,
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

    def count_tiles_by_observation(self) -> dict[str, int]:
        """``{observation_id: tile count}``. Non-abstract on purpose: a backend with a cheaper aggregate overrides it,
        any other still satisfies the seam through :meth:`list_tiles`."""
        counts: dict[str, int] = {}
        for t in self.list_tiles():
            counts[t.observation_id] = counts.get(t.observation_id, 0) + 1
        return counts

    # -- per-tile spectral descriptor (Phase 10) ------------------------------
    # Non-abstract on purpose: a backend without spectral support still satisfies the seam.

    def upsert_tile_spectral(self, rows: "Sequence[Mapping[str, object]]") -> int:
        """Insert/update descriptor rows (``tile_id`` + the fields in ``geoseek.spectral.fields``). Returns rows written."""
        raise NotImplementedError

    def set_tile_spectral_context(self, rows: "Sequence[tuple[str, float | None, float | None]]") -> int:
        """Set ``dist_river_m`` / ``dist_water_m`` for already-described tiles: ``(tile_id, dist_river, dist_water)``."""
        raise NotImplementedError

    def get_tile_spectral(self, tile_id: str) -> "dict | None":
        raise NotImplementedError

    def list_tile_spectral(self, tile_ids: "Sequence[str] | None" = None) -> "dict[str, dict]":
        """``{tile_id: descriptor dict}`` for the given tiles (all described tiles when None)."""
        raise NotImplementedError

    def spectral_tile_ids(self, *, version: "int | None" = None) -> "set[str]":
        raise NotImplementedError

    def table_fingerprints(self, tables: "Sequence[str] | None" = None) -> "dict[str, str]":
        """SHA256 of each catalog table's rows in primary-key order - proves a migration left existing rows untouched."""
        raise NotImplementedError

    # -- analyst audit trail (PS 2.2.5) -----------------------------------
    # Append-only. ``record_analyst_decision`` only ever INSERTs; there is no
    # update or delete method on the interface by design.

    @abc.abstractmethod
    def record_analyst_decision(self, decision: AnalystDecision) -> AnalystDecision:
        """Append one analyst confirm/reject. Returns the row as stored (with
        ``decision_id`` / ``created_at`` filled in if they were blank)."""

    @abc.abstractmethod
    def list_analyst_decisions(
        self, *, candidate_id: str | None = None, limit: int | None = None
    ) -> list[AnalystDecision]:
        """Decisions in write order (oldest first), optionally for one candidate."""

    @abc.abstractmethod
    def get_analyst_decision(self, decision_id: str) -> AnalystDecision | None: ...

    @abc.abstractmethod
    def latest_decision_by_candidate(self) -> dict[str, AnalystDecision]:
        """``{candidate_id: most-recent AnalystDecision}`` - the current verdict per candidate."""

    # -- standing watch areas (Phase 8 Step C) -----------------------------
    # Unlike the audit trail above, watch areas ARE editable/deletable - an
    # analyst-maintained operational definition, not a log.

    @abc.abstractmethod
    def create_watch_area(self, watch: WatchArea) -> WatchArea:
        """Insert a new watch area. Raises if ``watch.watch_id`` already exists."""

    @abc.abstractmethod
    def update_watch_area(self, watch: WatchArea) -> WatchArea:
        """Replace an existing watch area's definition (by ``watch_id``)."""

    @abc.abstractmethod
    def delete_watch_area(self, watch_id: str) -> None: ...

    @abc.abstractmethod
    def get_watch_area(self, watch_id: str) -> WatchArea | None: ...

    @abc.abstractmethod
    def list_watch_areas(self, *, active_only: bool = False) -> list[WatchArea]: ...

    @abc.abstractmethod
    def record_notification(self, notification: WatchNotification) -> WatchNotification:
        """Append one watch-area firing (a set of newly-matched candidates)."""

    @abc.abstractmethod
    def list_notifications(
        self, *, watch_id: str | None = None, unseen_only: bool = False
    ) -> list[WatchNotification]:
        """Notifications newest-first, optionally for one watch area / unseen only."""

    @abc.abstractmethod
    def mark_notification_seen(self, notification_id: str) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

"""SQLite implementation of the metadata catalog.

This is the ONLY module in geoseek that imports ``sqlite3``. Everything else
goes through :class:`geoseek.catalog.repository.MetadataRepository` (the ABC
this class satisfies) so the store can be swapped for PostGIS without touching
callers.

Spatial predicates (bbox / point-in-footprint) are evaluated in Python with
shapely over the candidate rows - the same post-filter strategy the search
engine already uses, and fine at this scale (a few thousand tiles). A PostGIS
implementation would push these into SQL instead.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt as shapely_wkt
from shapely.geometry import Point, box

from geoseek.catalog.entities import (
    AnalystDecision,
    Collection,
    DerivedProduct,
    Observation,
    Scene,
    Tile,
    TileProvenance,
    TileRecord,
)
from geoseek.catalog.repository import MetadataRepository
from geoseek.catalog.schema import CATALOG_TABLES, SCHEMA_SQL
from geoseek.config import get_settings


class CatalogError(RuntimeError):
    """Raised for catalog integrity problems (missing parent, broken chain)."""


def _loads(s: str | None) -> dict:
    return json.loads(s) if s else {}


class SQLiteMetadataRepository(MetadataRepository):
    """:class:`MetadataRepository` backed by a single SQLite file.

    Built to be swapped for a PostGIS implementation: callers depend only on the
    ABC, and the SQLite-specific bits (spatial predicates evaluated in Python,
    the ``connection`` escape hatch) do not leak into the interface.
    """

    def __init__(self, db_path: Path | str | None = None, *, connection: sqlite3.Connection | None = None):
        if connection is not None:
            self.db_path = Path(getattr(connection, "_geoseek_path", ":memory:"))
            self._conn = connection
        else:
            self.db_path = Path(db_path) if db_path is not None else get_settings().index_dir / "tiles.sqlite"
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False + an explicit lock: a SearchEngine backing a
            # server is queried from a threadpool. The flag only lifts the
            # same-thread restriction; the lock provides the serialization.
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._ensure_schema()

    # -- lifecycle -------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """Escape hatch for the migration only. Application code must not use this."""
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteMetadataRepository":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes -------------------------------------------------------------

    def register_collection(self, c: Collection) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO collections "
                "(collection_id, sensor, platform, bands_json, native_gsd_m, description, metadata_json) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(collection_id) DO UPDATE SET "
                "sensor=excluded.sensor, platform=excluded.platform, bands_json=excluded.bands_json, "
                "native_gsd_m=excluded.native_gsd_m, description=excluded.description, "
                "metadata_json=excluded.metadata_json",
                (c.collection_id, c.sensor, c.platform, json.dumps(list(c.bands)),
                 c.native_gsd_m, c.description, json.dumps(c.metadata)),
            )
            self._conn.commit()

    def register_scene(self, s: Scene) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO scenes "
                "(scene_id, collection_id, platform, acquired_at, footprint_wkt_4326, processing_baseline, "
                " source_url, license, crs, checksums_json, metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scene_id) DO UPDATE SET "
                "collection_id=excluded.collection_id, platform=excluded.platform, "
                "acquired_at=excluded.acquired_at, footprint_wkt_4326=excluded.footprint_wkt_4326, "
                "processing_baseline=excluded.processing_baseline, source_url=excluded.source_url, "
                "license=excluded.license, crs=excluded.crs, checksums_json=excluded.checksums_json, "
                "metadata_json=excluded.metadata_json",
                (s.scene_id, s.collection_id, s.platform, s.acquired_at, s.footprint_wkt_4326,
                 s.processing_baseline, s.source_url, s.license, s.crs,
                 json.dumps(s.checksums), json.dumps(s.metadata)),
            )
            self._conn.commit()

    def register_observation(self, o: Observation) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO observations "
                "(observation_id, scene_id, acquired_at, footprint_wkt_4326, aoi_name, dataset_dir, "
                " quality_summary_json, radiometry_json, coregistration_json, metadata_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(observation_id) DO UPDATE SET "
                "scene_id=excluded.scene_id, acquired_at=excluded.acquired_at, "
                "footprint_wkt_4326=excluded.footprint_wkt_4326, aoi_name=excluded.aoi_name, "
                "dataset_dir=excluded.dataset_dir, quality_summary_json=excluded.quality_summary_json, "
                "radiometry_json=excluded.radiometry_json, coregistration_json=excluded.coregistration_json, "
                "metadata_json=excluded.metadata_json",
                (o.observation_id, o.scene_id, o.acquired_at, o.footprint_wkt_4326, o.aoi_name,
                 o.dataset_dir or o.observation_id, json.dumps(o.quality_summary),
                 json.dumps(o.radiometry), json.dumps(o.coregistration), json.dumps(o.metadata)),
            )
            self._conn.commit()

    def add_tiles(self, tiles: list[Tile]) -> None:
        if not tiles:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO tiles "
                "(tile_id, observation_id, row, col, geom_wkt_4326, cloud_fraction, quality_flags_json, "
                " faiss_id, embedding_ref, indices_ref, processing_history_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (t.tile_id, t.observation_id, t.row, t.col, t.geom_wkt_4326, float(t.cloud_fraction),
                     json.dumps(t.quality_flags), t.faiss_id, t.embedding_ref, t.indices_ref,
                     json.dumps(t.processing_history))
                    for t in tiles
                ],
            )
            self._conn.commit()

    def upsert_derived(self, d: DerivedProduct) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO derived (derived_id, kind, path, observation_id, tile_id, params_json, created_at) "
                "VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(derived_id) DO UPDATE SET "
                "kind=excluded.kind, path=excluded.path, observation_id=excluded.observation_id, "
                "tile_id=excluded.tile_id, params_json=excluded.params_json, created_at=excluded.created_at",
                (d.derived_id, d.kind, d.path, d.observation_id, d.tile_id,
                 json.dumps(d.params), d.created_at),
            )
            self._conn.commit()

    # -- row -> entity ----------------------------------------------------------

    @staticmethod
    def _collection(r: tuple) -> Collection:
        return Collection(
            collection_id=r[0], sensor=r[1], platform=r[2], bands=tuple(json.loads(r[3])),
            native_gsd_m=r[4], description=r[5], metadata=_loads(r[6]),
        )

    @staticmethod
    def _scene(r: tuple) -> Scene:
        return Scene(
            scene_id=r[0], collection_id=r[1], platform=r[2], acquired_at=r[3],
            footprint_wkt_4326=r[4], processing_baseline=r[5], source_url=r[6], license=r[7],
            crs=r[8], checksums=_loads(r[9]), metadata=_loads(r[10]),
        )

    @staticmethod
    def _observation(r: tuple) -> Observation:
        return Observation(
            observation_id=r[0], scene_id=r[1], acquired_at=r[2], footprint_wkt_4326=r[3],
            aoi_name=r[4], dataset_dir=r[5], quality_summary=_loads(r[6]), radiometry=_loads(r[7]),
            coregistration=_loads(r[8]), metadata=_loads(r[9]),
        )

    @staticmethod
    def _tile(r: tuple) -> Tile:
        return Tile(
            tile_id=r[0], observation_id=r[1], row=r[2], col=r[3], geom_wkt_4326=r[4],
            cloud_fraction=r[5], quality_flags=_loads(r[6]), faiss_id=r[7], embedding_ref=r[8],
            indices_ref=r[9], processing_history=json.loads(r[10]) if r[10] else [],
        )

    _COLLECTION_COLS = "collection_id, sensor, platform, bands_json, native_gsd_m, description, metadata_json"
    _SCENE_COLS = ("scene_id, collection_id, platform, acquired_at, footprint_wkt_4326, processing_baseline, "
                   "source_url, license, crs, checksums_json, metadata_json")
    _OBS_COLS = ("observation_id, scene_id, acquired_at, footprint_wkt_4326, aoi_name, dataset_dir, "
                 "quality_summary_json, radiometry_json, coregistration_json, metadata_json")
    _TILE_COLS = ("tile_id, observation_id, row, col, geom_wkt_4326, cloud_fraction, quality_flags_json, "
                  "faiss_id, embedding_ref, indices_ref, processing_history_json")

    # -- reads ------------------------------------------------------------------

    def get_collection(self, collection_id: str) -> Collection | None:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._COLLECTION_COLS} FROM collections WHERE collection_id=?", (collection_id,)
            ).fetchone()
        return self._collection(r) if r else None

    def get_scene(self, scene_id: str) -> Scene | None:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._SCENE_COLS} FROM scenes WHERE scene_id=?", (scene_id,)
            ).fetchone()
        return self._scene(r) if r else None

    def get_observation(self, observation_id: str) -> Observation | None:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._OBS_COLS} FROM observations WHERE observation_id=?", (observation_id,)
            ).fetchone()
        return self._observation(r) if r else None

    def get_tile(self, tile_id: str) -> Tile | None:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._TILE_COLS} FROM tiles WHERE tile_id=?", (tile_id,)
            ).fetchone()
        return self._tile(r) if r else None

    def list_tiles(self, *, observation_id: str | None = None) -> list[Tile]:
        sql = f"SELECT {self._TILE_COLS} FROM tiles"
        params: list = []
        if observation_id is not None:
            sql += " WHERE observation_id=?"
            params.append(observation_id)
        sql += " ORDER BY faiss_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._tile(r) for r in rows]

    def list_collections(self) -> list[Collection]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {self._COLLECTION_COLS} FROM collections").fetchall()
        return [self._collection(r) for r in rows]

    def list_scenes(self) -> list[Scene]:
        with self._lock:
            rows = self._conn.execute(f"SELECT {self._SCENE_COLS} FROM scenes ORDER BY acquired_at").fetchall()
        return [self._scene(r) for r in rows]

    def list_observations(
        self,
        *,
        location: tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
        collection: str | None = None,
    ) -> list[Observation]:
        """Observations, optionally filtered by a point / bbox / date window / collection.

        ``location`` is (lon, lat); an observation matches when its footprint
        contains that point. ``date_range`` is (start, end), inclusive,
        "YYYY-MM-DD" (either side may be None).
        """
        sql = f"SELECT {self._OBS_COLS} FROM observations"
        clauses: list[str] = []
        params: list = []
        if collection is not None:
            sql = (f"SELECT o.observation_id, o.scene_id, o.acquired_at, o.footprint_wkt_4326, o.aoi_name, "
                   f"o.dataset_dir, o.quality_summary_json, o.radiometry_json, o.coregistration_json, "
                   f"o.metadata_json FROM observations o JOIN scenes s ON s.scene_id = o.scene_id")
            clauses.append("s.collection_id = ?")
            params.append(collection)
        if date_range is not None:
            start, end = date_range
            if start:
                clauses.append("o.acquired_at >= ?" if collection else "acquired_at >= ?")
                params.append(start)
            if end:
                clauses.append("o.acquired_at <= ?" if collection else "acquired_at <= ?")
                params.append(end)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY o.acquired_at" if collection else " ORDER BY acquired_at"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        obs = [self._observation(r) for r in rows]

        if location is not None:
            pt = Point(location[0], location[1])
            obs = [o for o in obs if shapely_wkt.loads(o.footprint_wkt_4326).contains(pt)]
        if bbox is not None:
            b = box(*bbox)
            obs = [o for o in obs if shapely_wkt.loads(o.footprint_wkt_4326).intersects(b)]
        return obs

    def get_tile_provenance(self, tile_id: str) -> TileProvenance:
        tile = self.get_tile(tile_id)
        if tile is None:
            raise CatalogError(f"tile {tile_id!r} not found")
        obs = self.get_observation(tile.observation_id)
        if obs is None:
            raise CatalogError(f"tile {tile_id!r} -> observation {tile.observation_id!r} missing")
        scene = self.get_scene(obs.scene_id)
        if scene is None:
            raise CatalogError(f"observation {obs.observation_id!r} -> scene {obs.scene_id!r} missing")
        collection = self.get_collection(scene.collection_id)
        if collection is None:
            raise CatalogError(f"scene {scene.scene_id!r} -> collection {scene.collection_id!r} missing")
        return TileProvenance(tile=tile, observation=obs, scene=scene, collection=collection)

    _TILE_RECORD_SQL = (
        "SELECT t.tile_id, t.observation_id, s.scene_id, c.collection_id, c.sensor, s.platform, "
        "       o.acquired_at, t.geom_wkt_4326, t.cloud_fraction, t.faiss_id "
        "FROM tiles t "
        "JOIN observations o ON o.observation_id = t.observation_id "
        "JOIN scenes s       ON s.scene_id       = o.scene_id "
        "JOIN collections c  ON c.collection_id  = s.collection_id "
    )

    @staticmethod
    def _tile_record(r: tuple) -> TileRecord:
        return TileRecord(
            tile_id=r[0], observation_id=r[1], scene_id=r[2], collection_id=r[3], sensor=r[4],
            platform=r[5], acq_date=r[6], geom_wkt_4326=r[7], cloud_fraction=r[8], faiss_id=r[9],
        )

    def iter_tile_records(self) -> Iterator[TileRecord]:
        with self._lock:
            rows = self._conn.execute(self._TILE_RECORD_SQL + "ORDER BY t.faiss_id").fetchall()
        for r in rows:
            yield self._tile_record(r)

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
        """Tiles (with provenance) matching bbox + date range + sensor + max cloud."""
        sql = self._TILE_RECORD_SQL
        clauses: list[str] = []
        params: list = []
        if date_range is not None:
            start, end = date_range
            if start:
                clauses.append("o.acquired_at >= ?")
                params.append(start)
            if end:
                clauses.append("o.acquired_at <= ?")
                params.append(end)
        if sensor is not None:
            clauses.append("c.sensor = ?")
            params.append(sensor)
        if platform is not None:
            clauses.append("s.platform = ?")
            params.append(platform)
        if collection is not None:
            clauses.append("c.collection_id = ?")
            params.append(collection)
        if max_cloud is not None:
            clauses.append("t.cloud_fraction <= ?")
            params.append(float(max_cloud))
        if clauses:
            sql += "WHERE " + " AND ".join(clauses) + " "
        sql += "ORDER BY t.faiss_id"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        records = [self._tile_record(r) for r in rows]
        if bbox is not None:
            b = box(*bbox)
            records = [rec for rec in records if shapely_wkt.loads(rec.geom_wkt_4326).intersects(b)]
        return records

    def list_derived(
        self, *, observation_id: str | None = None, tile_id: str | None = None, kind: str | None = None
    ) -> list[DerivedProduct]:
        sql = ("SELECT derived_id, kind, path, observation_id, tile_id, params_json, created_at FROM derived")
        clauses, params = [], []
        if observation_id is not None:
            clauses.append("observation_id = ?")
            params.append(observation_id)
        if tile_id is not None:
            clauses.append("tile_id = ?")
            params.append(tile_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            DerivedProduct(derived_id=r[0], kind=r[1], path=r[2], observation_id=r[3], tile_id=r[4],
                           params=_loads(r[5]), created_at=r[6])
            for r in rows
        ]

    # -- small aggregates used by verification / reports ----------------------

    def tile_faiss_pairs(self) -> list[tuple[int | None, str]]:
        with self._lock:
            rows = self._conn.execute("SELECT faiss_id, tile_id FROM tiles ORDER BY faiss_id").fetchall()
        return [(r[0], r[1]) for r in rows]

    def count_tiles(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0])

    # -- analyst audit trail (PS 2.2.5) -------------------------------------
    # INSERT only. The schema's BEFORE UPDATE / BEFORE DELETE triggers on
    # analyst_decisions reject any rewrite at the storage layer, so this log is
    # append-only by construction, not merely by convention.

    _DECISION_COLS = (
        "decision_id, candidate_id, decision, analyst_note, analyst, created_at, model_version, "
        "weights_sha256, git_commit, pipeline_version, confidence_at_decision, evidence_snapshot_json"
    )

    @staticmethod
    def _analyst_decision(r: tuple) -> AnalystDecision:
        return AnalystDecision(
            decision_id=r[0], candidate_id=r[1], decision=r[2], analyst_note=r[3], analyst=r[4],
            created_at=r[5], model_version=r[6], weights_sha256=r[7], git_commit=r[8],
            pipeline_version=r[9], confidence_at_decision=r[10], evidence_snapshot=_loads(r[11]),
        )

    def record_analyst_decision(self, d: AnalystDecision) -> AnalystDecision:
        stored = AnalystDecision(
            decision_id=d.decision_id or f"dec_{uuid.uuid4().hex}",
            candidate_id=d.candidate_id,
            decision=d.decision,
            analyst_note=d.analyst_note,
            analyst=d.analyst,
            created_at=d.created_at or datetime.now(timezone.utc).isoformat(),
            model_version=d.model_version,
            weights_sha256=d.weights_sha256,
            git_commit=d.git_commit,
            pipeline_version=d.pipeline_version,
            confidence_at_decision=d.confidence_at_decision,
            evidence_snapshot=d.evidence_snapshot,
        )
        if stored.decision not in ("confirm", "reject"):
            raise CatalogError(f"decision must be 'confirm' or 'reject', got {stored.decision!r}")
        with self._lock:
            self._conn.execute(
                f"INSERT INTO analyst_decisions ({self._DECISION_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (stored.decision_id, stored.candidate_id, stored.decision, stored.analyst_note,
                 stored.analyst, stored.created_at, stored.model_version, stored.weights_sha256,
                 stored.git_commit, stored.pipeline_version, stored.confidence_at_decision,
                 json.dumps(stored.evidence_snapshot)),
            )
            self._conn.commit()
        return stored

    def list_analyst_decisions(
        self, *, candidate_id: str | None = None, limit: int | None = None
    ) -> list[AnalystDecision]:
        sql = f"SELECT {self._DECISION_COLS} FROM analyst_decisions"
        params: list = []
        if candidate_id is not None:
            sql += " WHERE candidate_id = ?"
            params.append(candidate_id)
        sql += " ORDER BY created_at, decision_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._analyst_decision(r) for r in rows]

    def get_analyst_decision(self, decision_id: str) -> AnalystDecision | None:
        with self._lock:
            r = self._conn.execute(
                f"SELECT {self._DECISION_COLS} FROM analyst_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        return self._analyst_decision(r) if r else None

    def latest_decision_by_candidate(self) -> dict[str, AnalystDecision]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {self._DECISION_COLS} FROM analyst_decisions ORDER BY created_at, decision_id"
            ).fetchall()
        latest: dict[str, AnalystDecision] = {}
        for r in rows:                      # rows are ascending; last write wins
            d = self._analyst_decision(r)
            latest[d.candidate_id] = d
        return latest

    def table_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                t: int(self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                for t in CATALOG_TABLES
            }

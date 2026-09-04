"""SQLite DDL for the catalog.

Kept deliberately close to standard SQL (foreign keys, no SQLite-only types)
so the same shape ports to PostGIS: there, geometry columns become
``geometry(Polygon, 4326)`` and the bbox / point predicates the SQLite
repository does in Python become ``ST_Intersects`` / ``&&`` in SQL.
"""

from __future__ import annotations

# Legacy flat table (Phase <=3a). Renamed, not dropped, by the migration.
LEGACY_FLAT_TILES_TABLE = "_migration_legacy_flat_tiles"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collections (
    collection_id   TEXT PRIMARY KEY,
    sensor          TEXT NOT NULL,
    platform        TEXT NOT NULL,
    bands_json      TEXT NOT NULL,
    native_gsd_m    REAL,
    description     TEXT NOT NULL DEFAULT '',
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id            TEXT PRIMARY KEY,
    collection_id       TEXT NOT NULL REFERENCES collections(collection_id),
    platform            TEXT NOT NULL,
    acquired_at         TEXT NOT NULL,
    footprint_wkt_4326  TEXT NOT NULL,
    processing_baseline TEXT,
    source_url          TEXT,
    license             TEXT,
    crs                 TEXT,
    checksums_json      TEXT NOT NULL DEFAULT '{}',
    metadata_json       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id       TEXT PRIMARY KEY,
    scene_id             TEXT NOT NULL REFERENCES scenes(scene_id),
    acquired_at          TEXT NOT NULL,
    footprint_wkt_4326   TEXT NOT NULL,
    aoi_name             TEXT,
    dataset_dir          TEXT,
    quality_summary_json TEXT NOT NULL DEFAULT '{}',
    radiometry_json      TEXT NOT NULL DEFAULT '{}',
    coregistration_json  TEXT NOT NULL DEFAULT '{}',
    metadata_json        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tiles (
    tile_id                 TEXT PRIMARY KEY,
    observation_id          TEXT NOT NULL REFERENCES observations(observation_id),
    row                     INTEGER NOT NULL,
    col                     INTEGER NOT NULL,
    geom_wkt_4326           TEXT NOT NULL,
    cloud_fraction          REAL NOT NULL,
    quality_flags_json      TEXT NOT NULL DEFAULT '{}',
    faiss_id                INTEGER,
    embedding_ref           TEXT,
    indices_ref             TEXT,
    processing_history_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS derived (
    derived_id      TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    path            TEXT NOT NULL,
    observation_id  TEXT REFERENCES observations(observation_id),
    tile_id         TEXT REFERENCES tiles(tile_id),
    params_json     TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS ix_scenes_collection      ON scenes(collection_id);
CREATE INDEX IF NOT EXISTS ix_observations_scene     ON observations(scene_id);
CREATE INDEX IF NOT EXISTS ix_observations_acquired  ON observations(acquired_at);
CREATE INDEX IF NOT EXISTS ix_tiles_observation      ON tiles(observation_id);
CREATE INDEX IF NOT EXISTS ix_tiles_faiss            ON tiles(faiss_id);
CREATE INDEX IF NOT EXISTS ix_tiles_rowcol           ON tiles(row, col);
CREATE INDEX IF NOT EXISTS ix_derived_observation    ON derived(observation_id);
CREATE INDEX IF NOT EXISTS ix_derived_tile           ON derived(tile_id);
"""

CATALOG_TABLES = ("collections", "scenes", "observations", "tiles", "derived")

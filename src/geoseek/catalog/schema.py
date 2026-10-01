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

-- PS 2.2.5 audit trail. APPEND-ONLY: the two triggers below reject every
-- UPDATE and DELETE at the storage layer, so the decision history can never be
-- silently rewritten. Re-deciding a candidate inserts a NEW row.
CREATE TABLE IF NOT EXISTS analyst_decisions (
    decision_id            TEXT PRIMARY KEY,
    candidate_id           TEXT NOT NULL,
    decision               TEXT NOT NULL CHECK (decision IN ('confirm', 'reject', 'reopen')),
    analyst_note           TEXT NOT NULL DEFAULT '',
    analyst                TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL,
    model_version          TEXT NOT NULL DEFAULT '',
    weights_sha256         TEXT NOT NULL DEFAULT '',
    git_commit             TEXT NOT NULL DEFAULT '',
    pipeline_version       TEXT NOT NULL DEFAULT '',
    confidence_at_decision REAL,
    evidence_snapshot_json TEXT NOT NULL DEFAULT '{}'
);

-- Phase 8 Step C: standing watch areas. A normal (editable/deletable)
-- operational table - NOT append-only like analyst_decisions, since a watch
-- area's definition is meant to be edited and a stale one deleted.
CREATE TABLE IF NOT EXISTS watch_areas (
    watch_id            TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    bbox_json           TEXT,                              -- [west,south,east,north] or NULL
    polygon_wkt_4326    TEXT,
    text_query          TEXT NOT NULL DEFAULT '',
    change_types_json   TEXT NOT NULL DEFAULT '[]',
    min_confidence      REAL,
    active              INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    created_by          TEXT NOT NULL DEFAULT '',
    updated_at          TEXT NOT NULL DEFAULT ''
);

-- One row per watch area per evaluation run that found a NEW match (new
-- relative to that watch area's own notification history - see
-- geoseek.watch.evaluator). `seen` is the one mutable field (an analyst
-- dismissing a notification), toggled via a dedicated method, not a general
-- UPDATE surface.
CREATE TABLE IF NOT EXISTS watch_notifications (
    notification_id     TEXT PRIMARY KEY,
    watch_id             TEXT NOT NULL REFERENCES watch_areas(watch_id),
    observation_id       TEXT NOT NULL,
    candidate_ids_json   TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL,
    seen                 INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS ix_watch_notifications_watch ON watch_notifications(watch_id);
CREATE INDEX IF NOT EXISTS ix_watch_notifications_time  ON watch_notifications(created_at);

CREATE INDEX IF NOT EXISTS ix_scenes_collection      ON scenes(collection_id);
CREATE INDEX IF NOT EXISTS ix_observations_scene     ON observations(scene_id);
CREATE INDEX IF NOT EXISTS ix_observations_acquired  ON observations(acquired_at);
CREATE INDEX IF NOT EXISTS ix_tiles_observation      ON tiles(observation_id);
CREATE INDEX IF NOT EXISTS ix_tiles_faiss            ON tiles(faiss_id);
CREATE INDEX IF NOT EXISTS ix_tiles_rowcol           ON tiles(row, col);
CREATE INDEX IF NOT EXISTS ix_derived_observation    ON derived(observation_id);
CREATE INDEX IF NOT EXISTS ix_derived_tile           ON derived(tile_id);
CREATE INDEX IF NOT EXISTS ix_analyst_decisions_cand ON analyst_decisions(candidate_id);
CREATE INDEX IF NOT EXISTS ix_analyst_decisions_time ON analyst_decisions(created_at);

CREATE TRIGGER IF NOT EXISTS trg_analyst_decisions_no_update
BEFORE UPDATE ON analyst_decisions
BEGIN
    SELECT RAISE(ABORT, 'analyst_decisions is append-only: UPDATE is not permitted');
END;

CREATE TRIGGER IF NOT EXISTS trg_analyst_decisions_no_delete
BEFORE DELETE ON analyst_decisions
BEGIN
    SELECT RAISE(ABORT, 'analyst_decisions is append-only: DELETE is not permitted');
END;
"""

# --- spatial index for tile footprints -------------------------------------
# Created and maintained by SQLiteMetadataRepository, NOT part of SCHEMA_SQL:
#   * SCHEMA_SQL is kept standard-SQL so it ports verbatim to PostGIS; a virtual
#     table is SQLite-only.
#   * not every SQLite build ships the R*Tree module, so the repository creates
#     this under a guard and falls back to the Python WKT scan if it is absent.
# `tile_rtree` holds one bounding box per tile, keyed by tiles.rowid, turning the
# bbox / point-in-AOI lookup in query_tiles() from an O(N) shapely scan over
# every footprint into an O(log N + k) index probe. SQLite's R*Tree rounds
# stored bounds outward (min down, max up) so it never drops a true match; the
# exact shapely .intersects() post-filter still runs on the (now tiny) candidate
# set, so results are identical to the brute-force path. The PostGIS port
# replaces this with a GiST index on the geometry column - callers are
# unaffected either way, the predicate stays a bbox on MetadataRepository.
TILE_RTREE_TABLE = "tile_rtree"
TILE_RTREE_SQL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS tile_rtree USING rtree("
    "id, min_lon, max_lon, min_lat, max_lat)"
)

CATALOG_TABLES = ("collections", "scenes", "observations", "tiles", "derived")

# The audit table is deliberately NOT in CATALOG_TABLES: it is not part of the
# scene->tile lineage the Phase 3.5 migration verifies, it is an independent
# append-only log written by the analyst UI.
AUDIT_TABLES = ("analyst_decisions",)

# Phase 8 Step C: also independent of the scene->tile lineage.
WATCH_TABLES = ("watch_areas", "watch_notifications")


# --- per-tile spectral descriptor (Phase 10) ---------------------------------
# A MIGRATION, not a rewrite: CREATE TABLE IF NOT EXISTS leaves every existing table and row untouched, and the
# repository applies it on open (like the R*Tree). One row per tile that could be described (Sentinel-2 tiles with
# NIR+SWIR on disk); Maxar / SAR tiles simply have no row. Standard SQL, ports to PostGIS unchanged.
from geoseek.spectral.fields import ALL_FIELDS, HEADER_FIELDS, INDEXED_FIELDS  # noqa: E402

# REAL columns = every descriptor field except the integer header fields (valid_frac is a real fraction)
_REAL_FIELDS = [f for f in ALL_FIELDS if f not in HEADER_FIELDS or f == "valid_frac"]
SPECTRAL_TABLE = "tile_spectral"
SPECTRAL_SQL = (
    "CREATE TABLE IF NOT EXISTS tile_spectral (\n"
    "    tile_id            TEXT PRIMARY KEY REFERENCES tiles(tile_id),\n"
    "    descriptor_version INTEGER NOT NULL,\n"
    "    usable             INTEGER NOT NULL,\n"
    "    n_valid            INTEGER,\n"
    + "".join(f"    {f} REAL,\n" for f in _REAL_FIELDS)
    + "    computed_at        TEXT NOT NULL DEFAULT ''\n);\n"
    + "".join(f"CREATE INDEX IF NOT EXISTS ix_tile_spectral_{f} ON tile_spectral({f});\n" for f in INDEXED_FIELDS)
)

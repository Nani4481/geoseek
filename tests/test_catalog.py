"""Tests for the Phase 3.5 catalog: entities, SQLiteMetadataRepository, migration.

Unit tests use synthetic data in a tmp DB. A few integration checks run
against the real migrated production catalog (data/index/tiles.sqlite) and
skip if it hasn't been migrated yet.
"""

from __future__ import annotations

import sqlite3

import pytest

from geoseek.catalog.entities import Collection, DerivedProduct, Observation, Scene, Tile
from geoseek.catalog.migrate import migrate, verify_migration
from geoseek.catalog.naming import base_scene_id, scene_source_url
from geoseek.catalog.repository import MetadataRepository
from geoseek.catalog.sqlite_repository import CatalogError, SQLiteMetadataRepository
from geoseek.config import get_settings


def test_sqlite_repo_satisfies_the_metadata_repository_interface():
    assert issubclass(SQLiteMetadataRepository, MetadataRepository)
    # no abstract method left unimplemented
    assert not getattr(SQLiteMetadataRepository, "__abstractmethods__", set())

# a small square footprint near the real AOI, and two tiles inside it
_AOI_WKT = "POLYGON ((82.0 26.5, 82.5 26.5, 82.5 27.0, 82.0 27.0, 82.0 26.5))"
_T1_WKT = "POLYGON ((82.10 26.60, 82.12 26.60, 82.12 26.62, 82.10 26.62, 82.10 26.60))"
_T2_WKT = "POLYGON ((82.30 26.80, 82.32 26.80, 82.32 26.82, 82.30 26.82, 82.30 26.80))"


def _seed_repo(db_path):
    repo = SQLiteMetadataRepository(db_path)
    repo.register_collection(Collection(
        collection_id="sentinel-2-l2a", sensor="MSI", platform="Sentinel-2",
        bands=("B04", "B03", "B02", "B08", "B11", "SCL"), native_gsd_m=10.0,
    ))
    repo.register_scene(Scene(
        scene_id="S2B_44RPQ_20190330_1_L2A", collection_id="sentinel-2-l2a", platform="Sentinel-2B",
        acquired_at="2019-03-30", footprint_wkt_4326=_AOI_WKT, source_url="http://x/", license="Copernicus",
        crs="EPSG:32644", checksums={"B04": "deadbeef"},
    ))
    repo.register_observation(Observation(
        observation_id="S2B_44RPQ_20190330_1_L2A_scaled", scene_id="S2B_44RPQ_20190330_1_L2A",
        acquired_at="2019-03-30", footprint_wkt_4326=_AOI_WKT, aoi_name="ayodhya",
        dataset_dir="S2B_44RPQ_20190330_1_L2A_scaled",
        quality_summary={"n_tiles": 2, "cloud_fraction_mean": 0.05},
        radiometry={"role": "subject"}, coregistration={"residual_magnitude_px": 0.16},
    ))
    repo.add_tiles([
        Tile(tile_id="S2B_44RPQ_20190330_1_L2A_scaled_r000_c000",
             observation_id="S2B_44RPQ_20190330_1_L2A_scaled", row=0, col=0,
             geom_wkt_4326=_T1_WKT, cloud_fraction=0.0, faiss_id=0, embedding_ref="tiles.faiss",
             processing_history=[{"step": "read", "at": "t0"}]),
        Tile(tile_id="S2B_44RPQ_20190330_1_L2A_scaled_r010_c012",
             observation_id="S2B_44RPQ_20190330_1_L2A_scaled", row=10, col=12,
             geom_wkt_4326=_T2_WKT, cloud_fraction=0.5, faiss_id=1, embedding_ref="tiles.faiss",
             processing_history=[]),
    ])
    return repo


# --------------------------------------------------------------------------
# naming helpers
# --------------------------------------------------------------------------


def test_base_scene_id_strips_aoi_suffix():
    assert base_scene_id("S2B_44RPQ_20190330_1_L2A_scaled") == "S2B_44RPQ_20190330_1_L2A"
    assert base_scene_id("S2B_44RPQ_20190330_1_L2A") == "S2B_44RPQ_20190330_1_L2A"
    assert base_scene_id("S1A_IW_GRDH_1SDV_20190329T123812_20190329T123837_026553_02F9E3_grd") == \
        "S1A_IW_GRDH_1SDV_20190329T123812_20190329T123837_026553_02F9E3"


def test_scene_source_url_reconstructs_cog_folder():
    url = scene_source_url("S2B_44RPQ_20190330_1_L2A")
    assert url.endswith("/2019/3/S2B_44RPQ_20190330_1_L2A/")
    assert url.startswith("https://sentinel-cogs.s3.")


# --------------------------------------------------------------------------
# repository round-trips
# --------------------------------------------------------------------------


def test_repo_registers_and_reads_back_the_chain(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    c = repo.get_collection("sentinel-2-l2a")
    assert c.bands == ("B04", "B03", "B02", "B08", "B11", "SCL") and c.native_gsd_m == 10.0
    s = repo.get_scene("S2B_44RPQ_20190330_1_L2A")
    assert s.platform == "Sentinel-2B" and s.crs == "EPSG:32644" and s.checksums == {"B04": "deadbeef"}
    o = repo.get_observation("S2B_44RPQ_20190330_1_L2A_scaled")
    assert o.scene_id == "S2B_44RPQ_20190330_1_L2A" and o.coregistration["residual_magnitude_px"] == 0.16
    assert repo.count_tiles() == 2
    repo.close()


def test_repo_tile_provenance_walks_tile_to_collection(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    prov = repo.get_tile_provenance("S2B_44RPQ_20190330_1_L2A_scaled_r010_c012")
    assert prov.tile.row == 10 and prov.tile.col == 12
    assert prov.observation.observation_id == "S2B_44RPQ_20190330_1_L2A_scaled"
    assert prov.scene.scene_id == "S2B_44RPQ_20190330_1_L2A"
    assert prov.collection.collection_id == "sentinel-2-l2a"
    d = prov.as_dict()
    assert d["sensor"] == "MSI" and d["platform"] == "Sentinel-2B" and d["faiss_id"] == 1
    repo.close()


def test_repo_tile_provenance_missing_tile_raises(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    with pytest.raises(CatalogError):
        repo.get_tile_provenance("nope")
    repo.close()


def test_repo_foreign_keys_enforced(tmp_path):
    repo = SQLiteMetadataRepository(tmp_path / "cat.sqlite")
    # tile with no parent observation must be rejected by the FK constraint
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_tiles([Tile(tile_id="orphan", observation_id="ghost", row=0, col=0,
                             geom_wkt_4326=_T1_WKT, cloud_fraction=0.0)])
    repo.close()


def test_repo_iter_tile_records_shape(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    recs = list(repo.iter_tile_records())
    assert [r.faiss_id for r in recs] == [0, 1]
    r = recs[0]
    assert r.scene_id == "S2B_44RPQ_20190330_1_L2A"
    assert r.observation_id == "S2B_44RPQ_20190330_1_L2A_scaled"
    assert r.sensor == "MSI" and r.platform == "Sentinel-2B" and r.acq_date == "2019-03-30"
    repo.close()


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------


def test_query_tiles_by_bbox_date_and_cloud(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")

    # bbox around T1 only
    hits = repo.query_tiles(bbox=(82.09, 26.59, 82.13, 26.63))
    assert [h.tile_id for h in hits] == ["S2B_44RPQ_20190330_1_L2A_scaled_r000_c000"]

    # max cloud excludes the 0.5 tile
    hits = repo.query_tiles(max_cloud=0.1)
    assert {h.tile_id for h in hits} == {"S2B_44RPQ_20190330_1_L2A_scaled_r000_c000"}

    # date range that misses 2019
    assert repo.query_tiles(date_range=("2024-01-01", "2024-12-31")) == []
    # sensor + platform
    assert len(repo.query_tiles(sensor="MSI")) == 2
    assert len(repo.query_tiles(platform="Sentinel-2B")) == 2
    assert repo.query_tiles(platform="Sentinel-2A") == []
    repo.close()


def test_list_observations_by_location_and_date(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    assert [o.observation_id for o in repo.list_observations(location=(82.2, 26.75))] == \
        ["S2B_44RPQ_20190330_1_L2A_scaled"]
    assert repo.list_observations(location=(0.0, 0.0)) == []
    assert len(repo.list_observations(date_range=("2019-01-01", "2019-12-31"))) == 1
    assert repo.list_observations(date_range=("2020-01-01", None)) == []
    assert len(repo.list_observations(collection="sentinel-2-l2a")) == 1
    repo.close()


def test_upsert_derived_and_list(tmp_path):
    repo = _seed_repo(tmp_path / "cat.sqlite")
    repo.upsert_derived(DerivedProduct(
        derived_id="obs:NDVI", kind="NDVI", path="/x/NDVI.tif",
        observation_id="S2B_44RPQ_20190330_1_L2A_scaled", created_at="t0"))
    repo.upsert_derived(DerivedProduct(  # same id -> update, not duplicate
        derived_id="obs:NDVI", kind="NDVI", path="/x/NDVI_v2.tif",
        observation_id="S2B_44RPQ_20190330_1_L2A_scaled", created_at="t1"))
    d = repo.list_derived(observation_id="S2B_44RPQ_20190330_1_L2A_scaled")
    assert len(d) == 1 and d[0].path == "/x/NDVI_v2.tif"
    repo.close()


# --------------------------------------------------------------------------
# migration on a synthetic legacy DB
# --------------------------------------------------------------------------


_LEGACY_DDL = """
CREATE TABLE tiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id TEXT UNIQUE NOT NULL, scene_id TEXT NOT NULL, sensor TEXT NOT NULL,
    acq_date TEXT NOT NULL, geom_wkt_4326 TEXT NOT NULL, cloud_fraction REAL NOT NULL,
    faiss_id INTEGER NOT NULL, processing_history_json TEXT NOT NULL
);
"""


def _make_legacy_db(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(_LEGACY_DDL)
    rows = []
    fid = 0
    for scene, sensor, date in (("S2B_44RPQ_20190330_1_L2A_scaled", "Sentinel-2B", "2019-03-30"),
                                ("S2A_44RPQ_20240308_0_L2A_scaled", "Sentinel-2A", "2024-03-08")):
        for r in range(2):
            for c in range(2):
                lon, lat = 82.0 + 0.02 * c, 26.5 + 0.02 * r
                wkt = (f"POLYGON (({lon} {lat}, {lon + 0.02} {lat}, {lon + 0.02} {lat + 0.02}, "
                       f"{lon} {lat + 0.02}, {lon} {lat}))")
                rows.append((f"{scene}_r{r:03d}_c{c:03d}", scene, sensor, date, wkt, 0.01 * fid, fid,
                             '[{"step": "read", "at": "t0"}, {"step": "indexed", "at": "t1"}]'))
                fid += 1
    conn.executemany("INSERT INTO tiles (tile_id, scene_id, sensor, acq_date, geom_wkt_4326, "
                     "cloud_fraction, faiss_id, processing_history_json) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return rows


def test_migrate_synthetic_legacy_db(tmp_path):
    db = tmp_path / "tiles.sqlite"
    legacy_rows = _make_legacy_db(db)

    report = migrate(db, datasets_dir=tmp_path / "datasets", manifest_path=tmp_path / "none.json")
    assert not report.already_migrated
    assert report.backup_path and (tmp_path / "tiles.sqlite.pre_phase35.bak").is_file()
    assert report.table_counts == {"collections": 1, "scenes": 2, "observations": 2,
                                   "tiles": 8, "derived": 0}
    assert report.verification.ok, report.verification.checks

    # legacy table kept, not dropped
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT COUNT(*) FROM _migration_legacy_flat_tiles").fetchone()[0] == 8

    # geometry + faiss_id preserved verbatim
    new = dict(conn.execute("SELECT tile_id, geom_wkt_4326 FROM tiles"))
    new_faiss = dict(conn.execute("SELECT tile_id, faiss_id FROM tiles"))
    for tile_id, scene, sensor, date, wkt, cf, fid, ph in legacy_rows:
        assert new[tile_id] == wkt
        assert new_faiss[tile_id] == fid
    # chain: tile -> observation -> scene (suffix stripped) -> the one collection
    row = conn.execute(
        "SELECT o.scene_id, s.collection_id FROM tiles t "
        "JOIN observations o ON o.observation_id = t.observation_id "
        "JOIN scenes s ON s.scene_id = o.scene_id "
        "WHERE t.tile_id = 'S2A_44RPQ_20240308_0_L2A_scaled_r001_c001'").fetchone()
    assert row == ("S2A_44RPQ_20240308_0_L2A", "sentinel-2-l2a")
    conn.close()


def test_migrate_is_idempotent(tmp_path):
    db = tmp_path / "tiles.sqlite"
    _make_legacy_db(db)
    migrate(db, datasets_dir=tmp_path / "d", manifest_path=tmp_path / "none.json")
    again = migrate(db, datasets_dir=tmp_path / "d", manifest_path=tmp_path / "none.json")
    assert again.already_migrated
    assert again.verification.ok
    assert again.table_counts["tiles"] == 8


def test_migrate_rolls_back_on_failure(tmp_path, monkeypatch):
    db = tmp_path / "tiles.sqlite"
    _make_legacy_db(db)

    import geoseek.catalog.migrate as m

    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "parse_tile_row_col", _boom)
    with pytest.raises(RuntimeError):
        migrate(db, datasets_dir=tmp_path / "d", manifest_path=tmp_path / "none.json")

    # DB restored: the flat `tiles` table is back, no new schema left behind
    conn = sqlite3.connect(str(db))
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "tiles" in names and "observations" not in names
    assert conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0] == 8
    conn.close()


# --------------------------------------------------------------------------
# integration: the real migrated production catalog
# --------------------------------------------------------------------------


def _production_db():
    db = get_settings().index_dir / "tiles.sqlite"
    if not db.is_file():
        pytest.skip("no production catalog DB")
    conn = sqlite3.connect(str(db))
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    if "observations" not in names:
        pytest.skip("production catalog not migrated yet - run `python -m geoseek.catalog.migrate`")
    return db


def test_production_catalog_verifies():
    db = _production_db()
    rep = verify_migration(db)
    assert rep.ok, {k: v for k, v in rep.checks.items() if not v["ok"]}
    assert rep.table_counts["tiles"] >= 2178
    assert rep.table_counts["observations"] >= 2
    # >= 1: Phase 5 may add sentinel-1-grd as a second, non-embedded collection
    assert rep.table_counts["collections"] >= 1


def test_production_catalog_provenance_chain_for_every_observation():
    db = _production_db()
    repo = SQLiteMetadataRepository(db)
    obs = repo.list_observations()
    assert obs
    for o in obs:
        s = repo.get_scene(o.scene_id)
        assert s is not None and repo.get_collection(s.collection_id) is not None
        tiles = repo.list_tiles(observation_id=o.observation_id)
        assert tiles
        prov = repo.get_tile_provenance(tiles[0].tile_id)
        assert prov.scene.scene_id == base_scene_id(o.observation_id)
        assert prov.scene.source_url and prov.scene.license
    repo.close()


def test_production_catalog_has_the_third_date_with_alignment_provenance():
    db = _production_db()
    repo = SQLiteMetadataRepository(db)
    s2_obs = repo.list_observations(collection="sentinel-2-l2a")
    obs_by_date = {o.acquired_at: o for o in s2_obs}
    if "2021-03-04" not in obs_by_date:
        pytest.skip("third date not staged/ingested yet")
    # 3 Sentinel-2 observations, time-ordered, ~1089 tiles each
    assert sorted(obs_by_date) == ["2019-03-30", "2021-03-04", "2024-03-08"]
    s2_tiles = sum(len(repo.list_tiles(observation_id=o.observation_id)) for o in s2_obs)
    assert s2_tiles == 3267
    third = obs_by_date["2021-03-04"]
    # normalized against the SAME reference date as the original pair
    assert third.coregistration.get("reference_scene") == "S2A_44RPQ_20240308_0_L2A_scaled"
    assert third.coregistration["median_magnitude_px"] < 0.5
    assert "reference" in third.radiometry.get("role", "")
    assert set(third.radiometry["relative_normalization"]["per_band_dn"]) == \
        {"B04", "B03", "B02", "B08", "B11"}
    repo.close()

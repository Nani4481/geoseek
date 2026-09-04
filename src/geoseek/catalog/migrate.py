"""One-time migration: flat ``tiles`` table -> collections/scenes/observations/tiles/derived.

    python -m geoseek.catalog.migrate           # migrate data/index/tiles.sqlite in place, then verify
    python -m geoseek.catalog.migrate --verify  # verify only (no writes)

The migration is NON-DESTRUCTIVE:
  * the sqlite file is copied to ``<db>.pre_phase35.bak`` first;
  * the legacy ``tiles`` table is RENAMED to ``_migration_legacy_flat_tiles``
    (kept in the same DB as an in-place backup and the verification baseline),
    never dropped;
  * tile rows are copied across verbatim - same tile_id, same geometry WKT
    string, same faiss_id, same processing history. No re-ingest, no re-embed.

It is idempotent: re-running detects the completed migration and only verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt as shapely_wkt
from shapely.geometry import box

from geoseek.catalog.entities import DerivedProduct, Tile
from geoseek.catalog.naming import (
    AOI_NAME,
    COLLECTION_ID,
    DATA_LICENSE,
    base_scene_id,
    platform_from_scene_id,
    scene_source_url,
)
from geoseek.catalog.schema import CATALOG_TABLES, LEGACY_FLAT_TILES_TABLE, SCHEMA_SQL
from geoseek.config import get_settings
from geoseek.ingest.tiler import parse_tile_row_col

LEGACY_TABLE = LEGACY_FLAT_TILES_TABLE

# spot-check point from the Phase 3.5 brief
SPOT_LON, SPOT_LAT = 82.185, 26.809

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    cols = ["tile_id", "scene_id", "sensor", "acq_date", "geom_wkt_4326",
            "cloud_fraction", "faiss_id", "processing_history_json"]
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY faiss_id").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _manifest_checksums(manifest: dict, observation_id: str) -> dict[str, str]:
    prefix = f"sentinel2-{observation_id}-"
    out: dict[str, str] = {}
    for a in manifest.get("artifacts", []):
        name = a.get("name", "")
        if name.startswith(prefix):
            out[name[len(prefix):]] = a.get("sha256", "")
    return out


def _footprint_wkt(geoms: list[str]) -> str:
    minx = miny = math.inf
    maxx = maxy = -math.inf
    for g in geoms:
        x0, y0, x1, y1 = shapely_wkt.loads(g).bounds
        minx, miny = min(minx, x0), min(miny, y0)
        maxx, maxy = max(maxx, x1), max(maxy, y1)
    return box(minx, miny, maxx, maxy).wkt


def _quality_summary(cloud_fractions: list[float]) -> dict:
    n = len(cloud_fractions)
    s = sorted(cloud_fractions)
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return {
        "n_tiles": n,
        "cloud_fraction_mean": round(sum(cloud_fractions) / n, 6) if n else None,
        "cloud_fraction_median": round(median, 6) if n else None,
        "cloud_fraction_max": round(max(cloud_fractions), 6) if n else None,
        "n_clear_tiles_cf_le_0p05": sum(1 for cf in cloud_fractions if cf <= 0.05),
    }


def _read_crs(dataset_dir: Path) -> str | None:
    ref = dataset_dir / "B04.tif"
    if not ref.is_file():
        return None
    try:
        import rasterio

        with rasterio.open(ref) as ds:
            return str(ds.crs) if ds.crs else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------


@dataclass
class VerificationReport:
    ok: bool = True
    checks: dict = field(default_factory=dict)
    table_counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks[name] = {"ok": bool(ok), "detail": detail}
        self.ok = self.ok and bool(ok)


@dataclass
class MigrationReport:
    already_migrated: bool
    backup_path: str | None
    table_counts: dict = field(default_factory=dict)
    verification: VerificationReport | None = None


# ----------------------------------------------------------------------------
# migration
# ----------------------------------------------------------------------------


def _is_migrated(conn: sqlite3.Connection) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"collections", "scenes", "observations", "tiles"}.issubset(names):
        return False
    if LEGACY_TABLE not in names:
        return False
    try:
        n_obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        # new `tiles` must have the new column layout
        tile_cols = {r[1] for r in conn.execute("PRAGMA table_info(tiles)")}
    except sqlite3.OperationalError:
        return False
    return n_obs > 0 and "observation_id" in tile_cols


def migrate(
    db_path: Path | str | None = None,
    *,
    datasets_dir: Path | None = None,
    manifest_path: Path | None = None,
    verify: bool = True,
) -> MigrationReport:
    settings = get_settings()
    db_path = Path(db_path) if db_path is not None else settings.index_dir / "tiles.sqlite"
    datasets_dir = Path(datasets_dir) if datasets_dir is not None else settings.datasets_dir
    manifest_path = Path(manifest_path) if manifest_path is not None else settings.provenance_manifest_path
    faiss_path = db_path.parent / "tiles.faiss"

    if not db_path.is_file():
        raise FileNotFoundError(f"no catalog DB at {db_path} - nothing to migrate")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")

    if _is_migrated(conn):
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in CATALOG_TABLES}
        conn.close()
        report = MigrationReport(already_migrated=True, backup_path=None, table_counts=counts)
        if verify:
            report.verification = verify_migration(db_path, faiss_path=faiss_path)
        return report

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "tiles" not in tables:
        conn.close()
        raise RuntimeError(f"{db_path} has no legacy `tiles` table to migrate")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    # --- backup the whole file first --------------------------------------
    backup_path = db_path.with_suffix(db_path.suffix + ".pre_phase35.bak")
    shutil.copy2(db_path, backup_path)

    legacy = _legacy_rows(conn, "tiles")
    if not legacy:
        conn.close()
        raise RuntimeError("legacy `tiles` table is empty - refusing to migrate an empty catalog")

    # Atomicity here is the file backup taken above: on ANY failure the original
    # DB is copied back verbatim. (sqlite3.executescript force-commits, so a
    # single wrapping transaction would not actually cover the schema creation.)
    try:
        conn.execute(f"ALTER TABLE tiles RENAME TO {LEGACY_TABLE}")
        conn.executescript(SCHEMA_SQL)

        # group legacy rows by their (observation-level) scene_id
        by_obs: dict[str, list[dict]] = {}
        for r in legacy:
            by_obs.setdefault(r["scene_id"], []).append(r)

        # --- collection (one, for the Sentinel-2 L2A COGs) ------------------
        bands_present: set[str] = set()
        for obs_id in by_obs:
            d = datasets_dir / obs_id
            if d.is_dir():
                bands_present.update(p.stem for p in d.glob("*.tif")
                                     if p.stem in {"B02", "B03", "B04", "B08", "B11", "B12", "B8A", "SCL"})
        band_order = [b for b in ("B04", "B03", "B02", "B08", "B11", "SCL") if b in bands_present] or \
                     ["B04", "B03", "B02", "SCL"]
        conn.execute(
            "INSERT INTO collections (collection_id, sensor, platform, bands_json, native_gsd_m, "
            "description, metadata_json) VALUES (?,?,?,?,?,?,?)",
            (COLLECTION_ID, "MSI", "Sentinel-2", json.dumps(band_order), 10.0,
             "Sentinel-2 L2A surface reflectance COGs (Earth Search v1 / AWS Open Data).",
             json.dumps({"provider": "Element 84 Earth Search v1",
                         "stac_root": "https://earth-search.aws.element84.com/v1",
                         "bands_native_gsd_m": {"B04": 10, "B03": 10, "B02": 10, "B08": 10,
                                                "B11": 20, "SCL": 20},
                         "note": "B11 & SCL resampled to the 10m grid at staging time"})),
        )

        norm_section = manifest.get("radiometric_normalization", {})
        coreg_section = manifest.get("coregistration", {})
        radiometry_fixed = manifest.get("radiometry", {})
        subject_obs = norm_section.get("subject_scene")
        reference_obs = norm_section.get("reference_scene")

        derived_rows: list[DerivedProduct] = []
        tiles_out: list[Tile] = []
        scene_seen: set[str] = set()

        for obs_id, rows in sorted(by_obs.items()):
            base_id = base_scene_id(obs_id)
            dataset_dir = datasets_dir / obs_id
            geoms = [r["geom_wkt_4326"] for r in rows]
            footprint = _footprint_wkt(geoms)
            acq_date = rows[0]["acq_date"]
            platform = platform_from_scene_id(base_id)
            checksums = _manifest_checksums(manifest, obs_id)

            if base_id not in scene_seen:
                scene_seen.add(base_id)
                conn.execute(
                    "INSERT INTO scenes (scene_id, collection_id, platform, acquired_at, "
                    "footprint_wkt_4326, processing_baseline, source_url, license, crs, checksums_json, "
                    "metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (base_id, COLLECTION_ID, platform, acq_date, footprint, None,
                     scene_source_url(base_id), DATA_LICENSE, _read_crs(dataset_dir),
                     json.dumps(checksums),
                     json.dumps({
                         "footprint_source": "observation extent (true full-scene footprint not retained offline)",
                         "acquired_at_precision": "date only (time-of-day not retained from STAC)",
                         "earthsearch:boa_offset_applied": True,
                         "processing_baseline_note": "baseline >= 05.00 per staging manifest 'radiometry' / 'extra_bands'",
                     })),
                )

            # observation radiometry / co-registration provenance
            radiometry = {"fixed_true_color": radiometry_fixed}
            coregistration: dict = {}
            if obs_id == subject_obs:
                radiometry["relative_normalization"] = norm_section
                radiometry["role"] = "subject (adjusted onto the reference date)"
                coregistration = dict(coreg_section)
            elif obs_id == reference_obs:
                radiometry["role"] = "reference (held fixed)"
                coregistration = {"role": "reference", "shift_px": {"dy": 0.0, "dx": 0.0},
                                  "residual_magnitude_px": 0.0, "correction_applied": False}

            conn.execute(
                "INSERT INTO observations (observation_id, scene_id, acquired_at, footprint_wkt_4326, "
                "aoi_name, dataset_dir, quality_summary_json, radiometry_json, coregistration_json, "
                "metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (obs_id, base_id, acq_date, footprint, AOI_NAME, obs_id,
                 json.dumps(_quality_summary([float(r["cloud_fraction"]) for r in rows])),
                 json.dumps(radiometry), json.dumps(coregistration),
                 json.dumps({"aoi_bounds_4326": [82.01237, 26.36132, 82.84594, 27.10975],
                             "tile_grid": "256px, partial edge tiles kept, fully-nodata tiles skipped"})),
            )

            # tiles - verbatim copy
            idx_csv = db_path.parent / "spectral_indices_per_tile.csv"
            indices_ref = str(idx_csv) if idx_csv.is_file() else None
            for r in rows:
                try:
                    row_i, col_i = parse_tile_row_col(r["tile_id"])
                except ValueError:
                    row_i, col_i = -1, -1
                tiles_out.append(Tile(
                    tile_id=r["tile_id"], observation_id=obs_id, row=row_i, col=col_i,
                    geom_wkt_4326=r["geom_wkt_4326"], cloud_fraction=float(r["cloud_fraction"]),
                    quality_flags={}, faiss_id=int(r["faiss_id"]), embedding_ref="tiles.faiss",
                    indices_ref=indices_ref,
                    processing_history=json.loads(r["processing_history_json"]),
                ))

            # derived rasters that already exist on disk for this observation
            for kind in ("NDVI", "NDWI", "NDBI"):
                p = dataset_dir / f"{kind}.tif"
                if p.is_file():
                    derived_rows.append(DerivedProduct(
                        derived_id=f"{obs_id}:{kind}", kind=kind, path=str(p), observation_id=obs_id,
                        params={"computed_on": "normalized reflectance (Phase 3a change.prep)"},
                        created_at=_now_iso(),
                    ))
            if indices_ref:
                derived_rows.append(DerivedProduct(
                    derived_id=f"{obs_id}:spectral_indices_csv", kind="spectral_indices_csv",
                    path=indices_ref, observation_id=obs_id,
                    params={"per_tile": True, "indices": ["NDVI", "NDWI", "NDBI"]}, created_at=_now_iso(),
                ))

        conn.executemany(
            "INSERT INTO tiles (tile_id, observation_id, row, col, geom_wkt_4326, cloud_fraction, "
            "quality_flags_json, faiss_id, embedding_ref, indices_ref, processing_history_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(t.tile_id, t.observation_id, t.row, t.col, t.geom_wkt_4326, t.cloud_fraction,
              json.dumps(t.quality_flags), t.faiss_id, t.embedding_ref, t.indices_ref,
              json.dumps(t.processing_history)) for t in tiles_out],
        )
        conn.executemany(
            "INSERT INTO derived (derived_id, kind, path, observation_id, tile_id, params_json, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [(d.derived_id, d.kind, d.path, d.observation_id, d.tile_id, json.dumps(d.params),
              d.created_at) for d in derived_rows],
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        finally:
            conn.close()
        shutil.copy2(backup_path, db_path)  # restore the pre-migration file verbatim
        raise

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in CATALOG_TABLES}
    conn.close()

    report = MigrationReport(already_migrated=False, backup_path=str(backup_path), table_counts=counts)
    if verify:
        report.verification = verify_migration(db_path, faiss_path=faiss_path)
    return report


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------


def _legacy_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r['faiss_id']}|{r['tile_id']}|{r['scene_id']}|{r['sensor']}|{r['acq_date']}|"
                 f"{r['cloud_fraction']}|{r['geom_wkt_4326']}|{r['processing_history_json']}".encode())
    return h.hexdigest()


def verify_migration(db_path: Path | str, *, faiss_path: Path | None = None) -> VerificationReport:
    db_path = Path(db_path)
    faiss_path = faiss_path or (db_path.parent / "tiles.faiss")
    rep = VerificationReport()
    conn = sqlite3.connect(str(db_path))

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    has_legacy = LEGACY_TABLE in tables
    rep.table_counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        for t in CATALOG_TABLES if t in tables}

    legacy = _legacy_rows(conn, LEGACY_TABLE) if has_legacy else []
    n_new = conn.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
    new_tile_ids = {r[0] for r in conn.execute("SELECT tile_id FROM tiles")}

    # 1. every pre-migration tile is still present (later incremental ingests may
    #    have ADDED tiles - that is fine; none may have been lost or altered).
    if has_legacy:
        missing = [r["tile_id"] for r in legacy if r["tile_id"] not in new_tile_ids]
        added = n_new - len(legacy)
        rep.add("all_legacy_tiles_migrated_and_present", not missing and len(legacy) > 0,
                f"{len(legacy)} legacy tiles all present; {added} tile(s) added since migration"
                + (f"; MISSING {missing[:5]}" if missing else ""))
    else:
        rep.notes.append("legacy table absent - count checked against 2178 baseline only")
        rep.add("all_legacy_tiles_migrated_and_present", n_new >= 2178, f"{n_new} tiles (>= 2178 baseline)")

    # 2. FK chain: every tile -> observation -> scene -> collection resolves
    orphans_obs = conn.execute(
        "SELECT COUNT(*) FROM tiles t LEFT JOIN observations o ON o.observation_id=t.observation_id "
        "WHERE o.observation_id IS NULL").fetchone()[0]
    orphans_scene = conn.execute(
        "SELECT COUNT(*) FROM observations o LEFT JOIN scenes s ON s.scene_id=o.scene_id "
        "WHERE s.scene_id IS NULL").fetchone()[0]
    orphans_coll = conn.execute(
        "SELECT COUNT(*) FROM scenes s LEFT JOIN collections c ON c.collection_id=s.collection_id "
        "WHERE c.collection_id IS NULL").fetchone()[0]
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    rep.add("every_tile_linked_tile_to_observation", orphans_obs == 0, f"{orphans_obs} orphan tiles")
    rep.add("every_observation_linked_to_scene", orphans_scene == 0, f"{orphans_scene} orphan observations")
    rep.add("every_scene_linked_to_collection", orphans_coll == 0, f"{orphans_coll} orphan scenes")
    rep.add("no_foreign_key_violations", not fk_violations, f"{len(fk_violations)} violations")

    # 3. per-tile scene/observation correctness vs legacy (observation_id == legacy scene_id;
    #    base scene_id is that with the AOI suffix stripped)
    if has_legacy:
        mism = 0
        cur = conn.execute(
            "SELECT t.tile_id, t.observation_id, o.scene_id FROM tiles t "
            "JOIN observations o ON o.observation_id=t.observation_id")
        newmap = {tid: (oid, sid) for tid, oid, sid in cur}
        for r in legacy:
            oid, sid = newmap.get(r["tile_id"], (None, None))
            if oid != r["scene_id"] or sid != base_scene_id(r["scene_id"]):
                mism += 1
        rep.add("tile_scene_and_observation_correct", mism == 0, f"{mism} mislinked tiles")

    # 4. every legacy tile's geometry + row content is byte-identical after migration
    #    (checked over the legacy tile_ids only - tiles added later are not in scope)
    if has_legacy:
        legacy_geom = {r["tile_id"]: r["geom_wkt_4326"] for r in legacy}
        new_by_id = {
            tid: (fid, oid, plat, ad, cf, g, ph)
            for tid, fid, oid, plat, ad, cf, g, ph in conn.execute(
                "SELECT t.tile_id, t.faiss_id, t.observation_id, s.platform, o.acquired_at, "
                "t.cloud_fraction, t.geom_wkt_4326, t.processing_history_json FROM tiles t "
                "JOIN observations o ON o.observation_id=t.observation_id "
                "JOIN scenes s ON s.scene_id=o.scene_id")
        }
        geom_diffs = sum(1 for r in legacy
                         if r["tile_id"] not in new_by_id
                         or new_by_id[r["tile_id"]][5] != r["geom_wkt_4326"])
        rep.add("legacy_geometries_byte_identical", geom_diffs == 0, f"{geom_diffs} geometry strings changed")
        # row-content digest over the legacy tiles, reconstructed from the new schema.
        # legacy `scene_id` == observation_id, legacy `sensor` == the platform string.
        recon = []
        for r in sorted(legacy, key=lambda x: int(x["faiss_id"])):
            fid, oid, plat, ad, cf, g, ph = new_by_id.get(r["tile_id"], (None,) * 7)
            recon.append({"faiss_id": fid, "tile_id": r["tile_id"], "scene_id": oid, "sensor": plat,
                          "acq_date": ad, "cloud_fraction": cf, "geom_wkt_4326": g,
                          "processing_history_json": ph})
        rep.add("legacy_row_content_digest_preserved",
                _legacy_digest(recon) == _legacy_digest(legacy),
                "sha256 of (faiss_id|tile_id|observation|platform|date|cloud|geom|history) over legacy tiles")

    # 5. centroid spot-check ~82.185E / 26.809N  (over legacy tiles when present -
    #    the invariant is that a PRE-migration tile there is unchanged)
    spot = None
    best_d = math.inf
    spot_rows = ([(r["tile_id"], r["geom_wkt_4326"]) for r in sorted(legacy, key=lambda x: x["tile_id"])]
                 if has_legacy
                 else list(conn.execute("SELECT tile_id, geom_wkt_4326 FROM tiles ORDER BY tile_id")))
    for tid, g in spot_rows:
        cx, cy = shapely_wkt.loads(g).centroid.coords[0]
        d = math.hypot(cx - SPOT_LON, cy - SPOT_LAT)
        if d < best_d:
            best_d, spot = d, (tid, cx, cy, g)
    # invariant asserted here: the tile nearest the brief's spot point has, in the
    # migrated schema, the exact geometry string it had pre-migration. The distance
    # itself is a soft AOI sanity signal, reported but not gated.
    spot_ok = True
    detail = f"nearest tile {spot[0]} centroid ({spot[1]:.6f}, {spot[2]:.6f}), {best_d * 111:.2f} km from point"
    if has_legacy:
        new_geom_for_spot = new_by_id.get(spot[0], (None,) * 7)[5]
        spot_ok = new_geom_for_spot == spot[3]  # spot[3] is the legacy geom
        detail += f"; migrated geometry identical to pre-migration: {spot_ok}; near AOI: {best_d < 0.05}"
    rep.add("centroid_spot_check_82.185E_26.809N", spot_ok, detail)

    # 6. FAISS <-> tile mapping intact
    faiss_ok, faiss_detail = _verify_faiss(conn, faiss_path, legacy if has_legacy else None)
    rep.add("faiss_id_to_tile_mapping_intact", faiss_ok, faiss_detail)

    conn.close()

    # 7. search parity against the pre-refactor fixture
    parity_ok, parity_detail = _verify_search_parity(db_path)
    rep.add("search_results_identical_to_pre_refactor", parity_ok, parity_detail)

    return rep


def _verify_faiss(conn, faiss_path: Path, legacy: list[dict] | None) -> tuple[bool, str]:
    # only embedded tiles carry a faiss_id; a non-embedded collection (e.g. Sentinel-1
    # SAR, added as corroborating evidence) legitimately has faiss_id = NULL.
    pairs = conn.execute(
        "SELECT faiss_id, tile_id FROM tiles WHERE faiss_id IS NOT NULL ORDER BY faiss_id").fetchall()
    ids = [p[0] for p in pairs]
    contiguous = ids == list(range(len(ids)))
    if legacy is not None:
        # every pre-migration (faiss_id, tile_id) pair must survive unchanged;
        # later incremental ingests append further pairs, which is expected.
        new_pairs = {(int(a), b) for a, b in pairs}
        legacy_pairs = {(int(r["faiss_id"]), r["tile_id"]) for r in legacy}
        mapping_same = legacy_pairs <= new_pairs
    else:
        mapping_same = True
    if not faiss_path.is_file():
        return contiguous and mapping_same, f"{len(ids)} pairs, contiguous={contiguous}, faiss file absent"
    # go through the VectorIndex seam - migrate.py never touches faiss directly
    from geoseek.vectorindex import FaissFlatIPIndex

    val = FaissFlatIPIndex(faiss_path).validate(len(ids), get_settings().embedding_dim)
    ok = contiguous and mapping_same and val.ok
    return ok, (f"{val.detail}; {len(ids)} tile rows; contiguous 0..N={contiguous}; "
                f"every legacy (faiss_id,tile_id) preserved: {mapping_same}")


def _verify_search_parity(db_path: Path) -> tuple[bool, str]:
    fixture = FIXTURES_DIR / "search_baseline.json"
    if not fixture.is_file():
        return True, "no search_baseline.json fixture - skipped"
    if not (db_path.parent / "tiles.faiss").is_file():
        return True, "no production faiss index - skipped"
    try:
        from geoseek.search.engine import SearchEngine

        from geoseek.search.engine import SearchFilters

        base = json.loads(fixture.read_text(encoding="utf-8"))
        eng = SearchEngine(index_dir=db_path.parent)
        k = base.get("k", 15)
        max_dscore = 0.0
        mismatches: list[str] = []

        def _cmp(tag, got, expected):
            nonlocal max_dscore
            if [g.tile_id for g in got] != [e["tile_id"] for e in expected]:
                mismatches.append(tag)
                return
            for g, e in zip(got, expected):
                max_dscore = max(max_dscore, abs(float(g.score) - e["score"]))

        for q, expected in base.get("text", {}).items():
            _cmp(f"text:{q}", eng.search_text(q, k=k)[0], expected)
        for label, spec in base.get("filtered", {}).items():
            got, _ = eng.search_text(spec["query"], k=k, filters=SearchFilters(**spec["filters"]))
            _cmp(f"filtered:{label}", got, spec["results"])
        for tid, expected in base.get("image", {}).items():
            _cmp(f"image:{tid}", eng.search_image(tile_id=tid, k=k)[0], expected)
        eng.close()
        ok = not mismatches and max_dscore < 1e-4
        return ok, (f"{len(base.get('text', {}))} text + {len(base.get('filtered', {}))} filtered + "
                    f"{len(base.get('image', {}))} image queries; ordering identical: {not mismatches}; "
                    f"max |Δscore| = {max_dscore:.2e}" + (f"; MISMATCHES {mismatches}" if mismatches else ""))
    except Exception as e:
        return False, f"parity check raised: {e!r}"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _print_report(mr: MigrationReport) -> None:
    print("=" * 78)
    print("geoseek Phase 3.5 - Step 1: scene -> observation -> tile migration")
    print("=" * 78)
    if mr.already_migrated:
        print("  status : ALREADY MIGRATED (idempotent no-op) - verifying current state")
    else:
        print("  status : MIGRATED")
        print(f"  backup : {mr.backup_path}")
    print("\n  row counts per table:")
    for t, n in mr.table_counts.items():
        print(f"    {t:14s} {n:>6d}")

    v = mr.verification
    if v is None:
        return
    print("\n  verification:")
    for name, res in v.checks.items():
        print(f"    [{'PASS' if res['ok'] else 'FAIL'}] {name}")
        if res["detail"]:
            print(f"           {res['detail']}")
    for n in v.notes:
        print(f"    note: {n}")
    print("\n  " + ("ALL CHECKS PASSED" if v.ok else "!! SOME CHECKS FAILED"))
    print("=" * 78)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-catalog-migrate")
    p.add_argument("--db", type=Path, default=None, help="path to tiles.sqlite (default: data/index/tiles.sqlite)")
    p.add_argument("--verify", action="store_true", help="verify only; do not migrate")
    args = p.parse_args(argv)

    settings = get_settings()
    db_path = args.db or (settings.index_dir / "tiles.sqlite")

    if args.verify:
        v = verify_migration(db_path)
        mr = MigrationReport(already_migrated=True, backup_path=None,
                             table_counts=v.table_counts, verification=v)
        _print_report(mr)
        raise SystemExit(0 if v.ok else 2)

    mr = migrate(db_path)
    _print_report(mr)
    raise SystemExit(0 if (mr.verification is None or mr.verification.ok) else 2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[catalog.migrate] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

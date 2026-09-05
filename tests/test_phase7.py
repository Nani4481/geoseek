"""Phase 7a: the tile-geometry spatial index (Step A) and the retrieval
evaluation harness (Step B).

Step A adds a SQLite R*Tree bbox prefilter to
``SQLiteMetadataRepository.query_tiles`` behind the ``MetadataRepository``
seam. These tests pin the two properties that make that safe to ship:

  * results are byte-identical to the pre-index brute-force path (the R*Tree
    only *prefilters*; the exact shapely ``.intersects()`` post-filter is
    unchanged), and
  * the index self-heals - it is created and backfilled on open, and
    ``add_tiles`` keeps it in lockstep.

Step B's numbers are produced by scripts (``scripts/eval_retrieval_*.py``) and
recorded to ``data/eval_retrieval/`` + the provenance manifest; here we just
sanity-check that the on-disk evaluation artifacts are internally consistent
when they are present (skipped in a fresh checkout).
"""

from __future__ import annotations

import json
import shutil

import pytest
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings

# a spread of query boxes over the real AOI: a sliver, a quadrant, the whole
# AOI (every tile), a zero-area point box, and a box well outside it (no hits).
_BBOXES = [
    (82.10, 26.60, 82.20, 26.70),
    (82.00, 26.60, 82.30, 26.90),
    (81.90, 26.40, 82.60, 27.10),
    (82.1234, 26.7654, 82.1234, 26.7654),
    (80.00, 25.00, 80.10, 25.10),
]


def _production_repo() -> SQLiteMetadataRepository:
    db = get_settings().index_dir / "tiles.sqlite"
    if not db.is_file():
        pytest.skip("production catalog not built yet")
    repo = SQLiteMetadataRepository(db)
    if repo.count_tiles() < 100:
        repo.close()
        pytest.skip("production catalog too small")
    return repo


def _bruteforce_bbox(repo: SQLiteMetadataRepository, bbox) -> list[str]:
    """query_tiles with no bbox -> filter in Python. Independent of the R*Tree
    path (no bbox arg means the prefilter branch is never taken)."""
    b = box(*bbox)
    return [
        r.tile_id
        for r in repo.query_tiles()
        if shapely_wkt.loads(r.geom_wkt_4326).intersects(b)
    ]


# --------------------------------------------------------------------------
# Step A - spatial index
# --------------------------------------------------------------------------


def test_rtree_module_is_available_on_this_build():
    """The prefilter falls back gracefully, but the reference build ships the
    R*Tree module and we want the fast path exercised in CI."""
    repo = SQLiteMetadataRepository(":memory:")
    try:
        assert repo._has_rtree is True
    finally:
        repo.close()


def test_rtree_bbox_matches_bruteforce_on_production_catalog():
    repo = _production_repo()
    try:
        assert repo._has_rtree is True
        for bbox in _BBOXES:
            fast = [r.tile_id for r in repo.query_tiles(bbox=bbox)]
            slow = _bruteforce_bbox(repo, bbox)
            assert fast == slow, f"mismatch for {bbox}: {len(fast)} vs {len(slow)}"
    finally:
        repo.close()


def test_rtree_bbox_combined_with_other_filters_matches_bruteforce():
    repo = _production_repo()
    try:
        bbox = (82.00, 26.60, 82.30, 26.90)
        b = box(*bbox)
        fast = [r.tile_id for r in repo.query_tiles(bbox=bbox, collection="sentinel-2-l2a")]
        slow = [
            r.tile_id
            for r in repo.query_tiles(collection="sentinel-2-l2a")
            if shapely_wkt.loads(r.geom_wkt_4326).intersects(b)
        ]
        assert fast == slow and len(fast) > 0
    finally:
        repo.close()


def test_prefilter_on_and_off_agree(tmp_path):
    """Force the fallback (``_has_rtree = False``) on a copy of the production
    catalog and confirm identical results - proves the prefilter changes
    latency, not semantics."""
    src = get_settings().index_dir / "tiles.sqlite"
    if not src.is_file():
        pytest.skip("production catalog not built yet")
    dst = tmp_path / "copy.sqlite"
    shutil.copy2(src, dst)

    with_idx = SQLiteMetadataRepository(dst)
    without_idx = SQLiteMetadataRepository(dst)
    without_idx._has_rtree = False
    try:
        if with_idx._has_rtree is not True:
            pytest.skip("R*Tree not available on this build")
        for bbox in _BBOXES:
            a = [r.tile_id for r in with_idx.query_tiles(bbox=bbox)]
            b = [r.tile_id for r in without_idx.query_tiles(bbox=bbox)]
            assert a == b
    finally:
        with_idx.close()
        without_idx.close()


def test_rtree_backfills_when_missing(tmp_path):
    src = get_settings().index_dir / "tiles.sqlite"
    if not src.is_file():
        pytest.skip("production catalog not built yet")
    dst = tmp_path / "copy.sqlite"
    shutil.copy2(src, dst)

    repo = SQLiteMetadataRepository(dst)
    if not repo._has_rtree:
        repo.close()
        pytest.skip("R*Tree not available on this build")
    # wipe the index, reopen -> _ensure_tile_rtree must rebuild it
    with repo._lock:
        repo._conn.execute("DELETE FROM tile_rtree")
        repo._conn.commit()
    repo.close()

    repo2 = SQLiteMetadataRepository(dst)
    try:
        n_tiles = repo2.count_tiles()
        (n_idx,) = repo2._conn.execute("SELECT COUNT(*) FROM tile_rtree").fetchone()
        assert n_idx == n_tiles
        bbox = (82.00, 26.60, 82.30, 26.90)
        assert [r.tile_id for r in repo2.query_tiles(bbox=bbox)] == _bruteforce_bbox(repo2, bbox)
    finally:
        repo2.close()


def test_add_tiles_keeps_rtree_in_lockstep(tmp_path):
    repo = SQLiteMetadataRepository(tmp_path / "cat.sqlite")
    try:
        if not repo._has_rtree:
            pytest.skip("R*Tree not available on this build")
        repo.register_collection(Collection(
            collection_id="sentinel-2-l2a", sensor="MSI", platform="Sentinel-2",
            bands=("B04", "B03", "B02"), native_gsd_m=10.0,
        ))
        repo.register_scene(Scene(
            scene_id="S2X", collection_id="sentinel-2-l2a", platform="Sentinel-2B",
            acquired_at="2019-03-30", footprint_wkt_4326="POLYGON ((82 26, 83 26, 83 27, 82 27, 82 26))",
            source_url="http://x/", license="Copernicus", crs="EPSG:32644",
        ))
        repo.register_observation(Observation(
            observation_id="S2X_scaled", scene_id="S2X", acquired_at="2019-03-30",
            footprint_wkt_4326="POLYGON ((82 26, 83 26, 83 27, 82 27, 82 26))", aoi_name="ayodhya",
            dataset_dir="S2X_scaled",
        ))
        repo.add_tiles([
            Tile(tile_id="S2X_scaled_r000_c000", observation_id="S2X_scaled", row=0, col=0,
                 geom_wkt_4326="POLYGON ((82.1 26.1, 82.2 26.1, 82.2 26.2, 82.1 26.2, 82.1 26.1))",
                 cloud_fraction=0.0, faiss_id=0, embedding_ref="t.faiss"),
        ])
        repo.add_tiles([
            Tile(tile_id="S2X_scaled_r000_c001", observation_id="S2X_scaled", row=0, col=1,
                 geom_wkt_4326="POLYGON ((82.8 26.8, 82.9 26.8, 82.9 26.9, 82.8 26.9, 82.8 26.8))",
                 cloud_fraction=0.0, faiss_id=1, embedding_ref="t.faiss"),
        ])
        (n_idx,) = repo._conn.execute("SELECT COUNT(*) FROM tile_rtree").fetchone()
        assert n_idx == repo.count_tiles() == 2
        hits = [r.tile_id for r in repo.query_tiles(bbox=(82.05, 26.05, 82.25, 26.25))]
        assert hits == ["S2X_scaled_r000_c000"]
    finally:
        repo.close()


# --------------------------------------------------------------------------
# Step B - retrieval evaluation artifacts (present only after the scripts run)
# --------------------------------------------------------------------------


def _eval_dir():
    d = get_settings().data_dir / "eval_retrieval"
    if not (d / "report.json").is_file():
        pytest.skip("retrieval evaluation not run yet (scripts/eval_retrieval_*.py)")
    return d


def test_eval_report_is_internally_consistent():
    d = _eval_dir()
    report = json.loads((d / "report.json").read_text())
    queries = json.loads((d / "queries.json").read_text())
    judgments = json.loads((d / "judgments.json").read_text())
    pools = json.loads((d / "pools.json").read_text())

    assert 15 <= len(queries) <= 20
    assert report["num_queries"] == len(queries)
    # both systems scored on the identical judgement set
    for q in queries:
        assert q in judgments and q in pools
        judged = set(judgments[q])
        assert judged == set(pools[q]["pool_tile_ids"])
        assert all(v in (0, 1, 2) for v in judgments[q].values())
        # every item either system ranks in its top-20 is judged (no unjudged
        # item can silently enter a metric)
        assert set(pools[q]["remoteclip_ranked"]) <= judged
        assert set(pools[q]["vanilla_ranked"]) <= judged

    for system in ("remoteclip", "vanilla"):
        for k in ("1", "5", "10", "20"):
            m = report["macro_averaged_metrics"][system][k]
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["ndcg"] <= 1.0


def test_eval_judgments_are_independent_of_the_retrieval_model():
    """The judgement file must declare a non-RemoteCLIP provenance for every
    grade - the whole evaluation is circular otherwise."""
    d = _eval_dir()
    rationale = json.loads((d / "judgments_rationale.json").read_text())
    method = rationale["methodology"]
    assert "remoteclip" not in method["judge_signal"].lower()
    assert method["judge_is_the_system_under_test"] is False
    # every judged tile carries a physical-criterion trace
    for q, per_tile in rationale["per_query"].items():
        for tid, entry in per_tile.items():
            assert "criterion" in entry and entry["criterion"]

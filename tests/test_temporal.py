"""Tests for the TemporalObservationMatcher and its output contract."""

from __future__ import annotations

import sqlite3

import pytest

from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.models import ChangeDetectionModel, ChangeResult
from geoseek.temporal.contract import ObservationPair, ObservationSequence
from geoseek.temporal.matcher import TemporalObservationMatcher

_AOI = "POLYGON ((82.0 26.5, 82.6 26.5, 82.6 27.1, 82.0 27.1, 82.0 26.5))"
_AOI_FAR = "POLYGON ((10.0 10.0, 10.2 10.0, 10.2 10.2, 10.0 10.2, 10.0 10.0))"
_PT = (82.3, 26.8)


def _repo(tmp_path):
    return SQLiteMetadataRepository(tmp_path / "cat.sqlite")


def _add_collection(repo, cid="sentinel-2-l2a", sensor="MSI", gsd=10.0):
    repo.register_collection(Collection(collection_id=cid, sensor=sensor, platform="Sentinel-2",
                                        bands=("B04", "B08"), native_gsd_m=gsd))


def _add_observation(repo, obs_id, date, *, platform="Sentinel-2A", collection="sentinel-2-l2a",
                     footprint=_AOI, cloud_mean=0.0, cloud_max=0.0, n_tiles=100, coreg=None):
    base = obs_id.replace("_scaled", "")
    if repo.get_scene(base) is None:
        repo.register_scene(Scene(scene_id=base, collection_id=collection, platform=platform,
                                  acquired_at=date, footprint_wkt_4326=footprint))
    repo.register_observation(Observation(
        observation_id=obs_id, scene_id=base, acquired_at=date, footprint_wkt_4326=footprint,
        aoi_name="x", dataset_dir=obs_id,
        quality_summary={"n_tiles": n_tiles, "cloud_fraction_mean": cloud_mean,
                         "cloud_fraction_max": cloud_max, "n_clear_tiles_cf_le_0p05": n_tiles},
        coregistration=coreg or {},
    ))


# --------------------------------------------------------------------------
# sequencing
# --------------------------------------------------------------------------


def test_orders_observations_and_builds_consecutive_pairs(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "C_scaled", "2022-02-20")
    _add_observation(repo, "A_scaled", "2019-03-30", platform="Sentinel-2B")
    _add_observation(repo, "B_scaled", "2024-03-08")

    seq = TemporalObservationMatcher(repo).match(location=_PT)
    assert [o.observation_id for o in seq.observations] == ["A_scaled", "C_scaled", "B_scaled"]
    assert [(p.earlier.observation_id, p.later.observation_id) for p in seq.pairs] == \
        [("A_scaled", "C_scaled"), ("C_scaled", "B_scaled")]
    assert isinstance(seq, ObservationSequence)
    repo.close()


def test_all_pairs_mode(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    for oid, d in (("A_scaled", "2019-03-30"), ("C_scaled", "2022-02-20"), ("B_scaled", "2024-03-08")):
        _add_observation(repo, oid, d)
    seq = TemporalObservationMatcher(repo).match(location=_PT, all_pairs=True)
    assert len(seq.pairs) == 3
    repo.close()


def test_single_observation_yields_no_pairs(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30")
    seq = TemporalObservationMatcher(repo).match(location=_PT)
    assert seq.pairs == []
    assert any("only 1 observation" in n for n in seq.notes)
    repo.close()


def test_match_needs_a_query_region(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError):
        TemporalObservationMatcher(repo).match()
    repo.close()


# --------------------------------------------------------------------------
# comparability verdicts
# --------------------------------------------------------------------------


def test_clean_pair_is_comparable_on_every_criterion(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30", platform="Sentinel-2B",
                     coreg={"reference_scene": "B_scaled", "median_magnitude_px": 0.16,
                            "correction_applied": False})
    _add_observation(repo, "B_scaled", "2024-03-08")
    seq = TemporalObservationMatcher(repo).match(location=_PT)
    pc = seq.pairs[0].comparability
    assert pc.comparable
    names = {c.name for c in pc.criteria}
    assert names == {"spatial_overlap", "temporal_separation", "sensor_compatibility",
                     "collection_compatibility", "resolution_compatibility", "quality", "coregistration"}
    assert pc.spatial_overlap_fraction == 1.0
    assert pc.temporal_separation_days == 1805
    assert pc.sensor_compatible and pc.collection_compatible and pc.resolution_compatible
    assert "0.16" in pc.coregistration_status
    repo.close()


def test_disjoint_footprints_block_comparability(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30", footprint=_AOI)
    _add_observation(repo, "B_scaled", "2024-03-08", footprint=_AOI_FAR)
    # both must be found at a location - use a bbox spanning both instead
    seq = TemporalObservationMatcher(repo).match(bbox=(9.0, 9.0, 83.0, 28.0))
    pc = seq.pairs[0].comparability
    assert not pc.comparable
    assert any("spatial_overlap" in r for r in pc.blocking_reasons)
    repo.close()


def test_different_collections_block_comparability(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo, cid="sentinel-2-l2a", gsd=10.0)
    _add_collection(repo, cid="landsat-c2-l2", sensor="OLI", gsd=30.0)
    _add_observation(repo, "A_scaled", "2019-03-30", collection="sentinel-2-l2a")
    _add_observation(repo, "B_scaled", "2024-03-08", collection="landsat-c2-l2", platform="Landsat-9")
    seq = TemporalObservationMatcher(repo).match(location=_PT)
    pc = seq.pairs[0].comparability
    assert not pc.comparable
    blocking = " ".join(pc.blocking_reasons)
    assert "collection_compatibility" in blocking and "resolution_compatibility" in blocking
    repo.close()


def test_cloudy_pair_warns_but_stays_comparable(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30", cloud_mean=0.45, cloud_max=0.9)
    _add_observation(repo, "B_scaled", "2024-03-08", cloud_mean=0.05, cloud_max=0.1)
    pc = TemporalObservationMatcher(repo).match(location=_PT).pairs[0].comparability
    assert pc.comparable  # quality is a warning, not blocking
    assert pc.quality_ok is False
    repo.close()


def test_unknown_coregistration_is_flagged_not_blocking(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30")   # no coreg info
    _add_observation(repo, "B_scaled", "2024-03-08")
    pc = TemporalObservationMatcher(repo).match(location=_PT).pairs[0].comparability
    assert pc.comparable
    assert pc.coregistration_status == "unknown"
    repo.close()


def test_transitive_coregistration_via_common_reference(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30",
                     coreg={"reference_scene": "REF_scaled", "median_magnitude_px": 0.15})
    _add_observation(repo, "C_scaled", "2022-02-20",
                     coreg={"reference_scene": "REF_scaled", "median_magnitude_px": 0.20})
    pc = TemporalObservationMatcher(repo).match(location=_PT).pairs[0].comparability
    crit = next(c for c in pc.criteria if c.name == "coregistration")
    assert crit.passed and "REF_scaled" in crit.detail
    repo.close()


def test_too_close_in_time_warns(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2024-03-05")
    _add_observation(repo, "B_scaled", "2024-03-08")
    crit = next(c for c in TemporalObservationMatcher(repo).match(location=_PT).pairs[0].comparability.criteria
                if c.name == "temporal_separation")
    assert crit.passed is False and "3 days" in crit.detail
    repo.close()


# --------------------------------------------------------------------------
# output contract -> ChangeDetectionModel
# --------------------------------------------------------------------------


def test_matcher_output_feeds_change_detection_model_unchanged(tmp_path):
    repo = _repo(tmp_path)
    _add_collection(repo)
    _add_observation(repo, "A_scaled", "2019-03-30", platform="Sentinel-2B")
    _add_observation(repo, "B_scaled", "2024-03-08")
    seq = TemporalObservationMatcher(repo).match(location=_PT)
    pair = seq.comparable_pairs[0]
    assert isinstance(pair, ObservationPair)

    class _Stub(ChangeDetectionModel):
        def predict_change(self, pair, *, tiles=None, aoi_wkt=None, bands=None):
            return ChangeResult(pair.earlier.observation_id, pair.later.observation_id,
                                method="stub", comparable=pair.comparable)

    res = _Stub().predict_change(pair, aoi_wkt=seq.aoi_wkt)
    assert res.earlier_observation_id == "A_scaled" and res.comparable is True

    d = seq.to_dict()
    assert d["pairs"][0]["comparability"]["comparable"] is True
    assert "criteria" in d["pairs"][0]["comparability"]
    repo.close()


# --------------------------------------------------------------------------
# integration: the real production catalog
# --------------------------------------------------------------------------


def test_matcher_on_production_catalog_ayodhya_point():
    db = get_settings().index_dir / "tiles.sqlite"
    if not db.is_file():
        pytest.skip("no production catalog")
    conn = sqlite3.connect(str(db))
    migrated = "observations" in {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    if not migrated:
        pytest.skip("production catalog not migrated")

    repo = SQLiteMetadataRepository(db)
    seq = TemporalObservationMatcher(repo).match(location=(82.1998, 26.7922))
    assert len(seq.observations) >= 2
    assert len(seq.pairs) >= 1
    first = seq.pairs[0]
    assert first.earlier.acquired_at < first.later.acquired_at
    assert first.comparability.comparable  # 2019<->2024 Ayodhya pair is comparable
    # the report renders
    txt = seq.format_report()
    assert "COMPARABLE" in txt and "coregistration" in txt
    repo.close()

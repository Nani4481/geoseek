"""Dossier sensor provenance: read from the catalog, and sun elevation / off-nadir only when the catalog really holds them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from geoseek.analyst import ui_support


def _repo(scene_meta=None, tile_flags=None):
    scenes = {"S2B_44RPQ_20190330_1_L2A": SimpleNamespace(
        scene_id="S2B_44RPQ_20190330_1_L2A", platform="Sentinel-2B", collection_id="s2", processing_baseline="05.00",
        crs="EPSG:32644", license="Copernicus", metadata=scene_meta or {"acquired_at_precision": "date only"}),
        "S2C_44RPQ_20260308_0_L2A": SimpleNamespace(
        scene_id="S2C_44RPQ_20260308_0_L2A", platform="Sentinel-2C", collection_id="s2", processing_baseline=None,
        crs="EPSG:32644", license="Copernicus", metadata={})}
    obs = {"o19": SimpleNamespace(observation_id="o19", scene_id="S2B_44RPQ_20190330_1_L2A", acquired_at="2019-03-30", metadata={}),
           "o26": SimpleNamespace(observation_id="o26", scene_id="S2C_44RPQ_20260308_0_L2A", acquired_at="2026-03-08", metadata={})}
    tiles = {"t19": SimpleNamespace(tile_id="t19", cloud_fraction=0.04, quality_flags=tile_flags or {}),
             "t26": SimpleNamespace(tile_id="t26", cloud_fraction=0.0, quality_flags={})}
    return SimpleNamespace(
        get_observation=obs.get, get_scene=scenes.get,
        get_collection=lambda cid: SimpleNamespace(sensor="MSI", collection_id=cid, native_gsd_m=10.0, metadata={}),
        query_tiles=lambda bbox, collection: [SimpleNamespace(tile_id="t19", observation_id="o19"), SimpleNamespace(tile_id="t26", observation_id="o26")],
        get_tile=tiles.get)


@pytest.fixture(autouse=True)
def _dates(monkeypatch):
    monkeypatch.setattr("geoseek.change.analyze.DATE_TO_OBS", {"2019": "o19", "2026": "o26"})


def _svc(repo):
    return SimpleNamespace(_by_id={"c1": {"candidate_id": "c1", "centroid_lonlat": [82.25, 26.63]}}, repo=repo)


def test_provenance_comes_from_the_catalog_and_missing_view_fields_are_omitted():
    d = ui_support.candidate_dossier(_svc(_repo()), "c1", "2019-03-30", "2026")
    b, a = d["observations"]
    assert (b["role"], a["role"]) == ("before", "after")
    assert b["platform"] == "Sentinel-2B" and b["acquired_at"] == "2019-03-30" and b["cloud_fraction"] == 0.04
    assert b["processing_baseline"] == "05.00" and b["native_gsd_m"] == 10.0 and b["scene_id"].startswith("S2B_44RPQ")
    assert a["platform"] == "Sentinel-2C" and a["cloud_fraction"] == 0.0 and a["processing_baseline"] is None
    for row in (b, a):                                           # not catalogued -> not present at all (no None placeholder)
        assert not {"sun_elevation_deg", "sun_azimuth_deg", "off_nadir_deg"} & set(row)
    assert d["view_fields_not_catalogued"] == ["off_nadir_deg", "sun_azimuth_deg", "sun_elevation_deg"]


def test_sun_elevation_and_off_nadir_appear_when_the_catalog_holds_them():
    repo = _repo(scene_meta={"view:sun_elevation": 41.5, "stac": {"view:off_nadir": 3.25, "flag": True}})
    d = ui_support.candidate_dossier(_svc(repo), "c1", "2019", "2026")
    b, a = d["observations"]
    assert b["sun_elevation_deg"] == 41.5 and b["off_nadir_deg"] == 3.25
    assert "sun_elevation_deg" not in a                                    # per observation: only where it is catalogued
    assert d["view_fields_not_catalogued"] == ["sun_azimuth_deg"]          # found on at least one observation -> not "missing"


def test_booleans_and_text_are_never_mistaken_for_angles():
    repo = _repo(scene_meta={"sun_elevation_known": True, "off_nadir_note": "n/a"})
    b = ui_support.candidate_dossier(_svc(repo), "c1", "2019", "2026")["observations"][0]
    assert "sun_elevation_deg" not in b and "off_nadir_deg" not in b


def test_unknown_candidate_is_none_and_unknown_year_is_rejected():
    assert ui_support.candidate_dossier(_svc(_repo()), "nope", "2019", "2026") is None
    with pytest.raises(ValueError):
        ui_support.candidate_dossier(_svc(_repo()), "c1", "2020-01-01", "2026")

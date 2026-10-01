"""scripts/stage_nir_swir_and_describe.py end to end on a tiny synthetic catalog + scene (bands already on disk, so no
network): plan selection, the single describe pass, region context, the catalog-integrity report, and resumability."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from affine import Affine

from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.spectral.fields import DESCRIPTOR_VERSION

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stage_nir_swir_and_describe.py"
POLY = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
H, W = 300, 520                                   # 2 x 3 tiles, partial edges


def _write_scene(scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True)
    transform = Affine(10, 0, 399_970.0, 0, -10, 3_000_020.0)
    prof = dict(driver="GTiff", height=H, width=W, count=1, crs="EPSG:32644", transform=transform, nodata=0)
    rng = np.random.default_rng(1)
    base = {"B02": 400, "B03": 800, "B04": 500, "B08": 3500, "B11": 1800}
    for b, v in base.items():
        a = (v * rng.uniform(0.97, 1.03, (H, W))).astype(np.uint16)
        if b in ("B08", "B03"):
            a[:, 300:] = 300 if b == "B08" else 900            # a water body on the right side
        with rasterio.open(scene_dir / f"{b}.tif", "w", dtype="uint16", **prof) as dst:
            dst.write(a, 1)
    with rasterio.open(scene_dir / "SCL.tif", "w", dtype="uint8", **prof) as dst:
        dst.write(np.full((H, W), 4, np.uint8), 1)


@pytest.fixture
def env(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    db = tmp_path / "index" / "tiles.sqlite"
    db.parent.mkdir()
    faiss = tmp_path / "index" / "tiles.faiss"
    faiss.write_bytes(b"FAISS-BYTES")
    repo = SQLiteMetadataRepository(db)
    repo.register_collection(Collection("sentinel-2-l2a", "MSI", "Sentinel-2", ("B04",), 10.0))
    repo.register_collection(Collection("maxar-opendata", "VHR", "Maxar", ("R",), 0.3))
    for sid, coll in (("sc_div", "sentinel-2-l2a"), ("sc_other", "sentinel-2-l2a"), ("sc_maxar", "maxar-opendata")):
        repo.register_scene(Scene(sid, coll, "P", "2024-01-01", POLY))
    obs = {"obs_div": ("sc_div", "kanha_44QMK_diverse"), "obs_other": ("sc_other", "someplace_else"),
           "obs_maxar": ("sc_maxar", "maxar_event")}
    for oid, (sid, aoi) in obs.items():
        repo.register_observation(Observation(oid, sid, "2024-01-01", POLY, aoi_name=aoi, dataset_dir=oid))
    tiles = [Tile(f"obs_div_r{r:03d}_c{c:03d}", "obs_div", r, c, POLY, 0.0, faiss_id=r * 3 + c) for r in range(2) for c in range(3)]
    tiles += [Tile("obs_other_r000_c000", "obs_other", 0, 0, POLY, 0.0, faiss_id=6),
              Tile("obs_div_r001_c002_noembed", "obs_div", 1, 2, POLY, 0.0, faiss_id=None),     # never described: no vector
              Tile("obs_maxar_r000_c000", "obs_maxar", 0, 0, POLY, 0.0, faiss_id=7)]
    repo.add_tiles(tiles)
    repo.close()
    _write_scene(datasets / "obs_div")

    spec = importlib.util.spec_from_file_location("stage_nir_swir_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)             # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "SETTINGS", SimpleNamespace(database_path=db, faiss_index_path=faiss, datasets_dir=datasets,
                                                          data_dir=tmp_path))
    monkeypatch.setattr(mod, "OUT_DIR", out)
    monkeypatch.setattr(mod, "power_source", lambda: {"source": "AC", "battery_percent": 100})
    return SimpleNamespace(mod=mod, db=db, out=out, datasets=datasets, faiss=faiss)


def test_plan_selects_only_diverse_and_ayodhya_sentinel2_observations_with_embedded_tiles(env):
    repo = SQLiteMetadataRepository(env.db)
    plan = env.mod.build_plan(repo, None)
    assert [e.observation_id for e in plan] == ["obs_div"] and len(plan[0].tiles) == 6      # no Maxar, no other AOI, no un-embedded tile
    assert env.mod.build_plan(repo, ["nothing"]) == [] and len(env.mod.build_plan(repo, ["kanha"])) == 1
    repo.close()


def test_the_pass_describes_every_tile_writes_the_mask_and_context_and_leaves_the_catalog_untouched(env):
    assert env.mod.main([]) == 0
    repo = SQLiteMetadataRepository(env.db)
    try:
        ids = repo.spectral_tile_ids(version=DESCRIPTOR_VERSION)
        assert ids == {f"obs_div_r{r:03d}_c{c:03d}" for r in range(2) for c in range(3)}
        left, right = repo.get_tile_spectral("obs_div_r000_c000"), repo.get_tile_spectral("obs_div_r000_c002")
        assert left["ndvi_mean"] > 0.6 and left["water_frac"] == 0 and right["water_frac"] == 1
        assert left["dist_river_m"] is not None and right["dist_river_m"] == 0.0        # context was computed in the same run
        assert repo.get_tile_spectral("obs_other_r000_c000") is None and repo.get_tile_spectral("obs_maxar_r000_c000") is None
    finally:
        repo.close()
    report = json.loads((env.out / "staging_report.json").read_text())
    ci = report["catalog_integrity"]
    assert ci["all_identical"] and ci["faiss_unchanged"] and ci["tile_spectral_rows"] == 6
    assert (env.datasets / "obs_div" / "WATER_DEC4.npz").is_file()
    assert (env.db.with_name("tiles.sqlite.pre_phase10.bak")).is_file(), "the catalog is backed up before it is first opened"
    assert report["scenes"][0]["tiles"] == 6 and report["scenes"][0]["staged_now"] == {"B08": False, "B11": False}


def test_a_second_run_finds_nothing_to_do(env, capsys):
    env.mod.main([])
    capsys.readouterr()
    assert env.mod.main([]) == 0
    assert "1 already complete; 0 to do" in capsys.readouterr().out
    assert len((env.out / "ledger.jsonl").read_text().splitlines()) == 1             # the first run's single scene only


def test_a_changed_existing_table_or_faiss_is_reported_as_an_integrity_failure(env):
    env.mod.main([])
    repo = SQLiteMetadataRepository(env.db)
    repo.connection.execute("UPDATE tiles SET cloud_fraction = 0.5 WHERE tile_id = 'obs_div_r000_c000'")   # tamper
    repo.connection.commit()
    repo.close()
    assert env.mod.main(["--context-only"]) == 3
    ci = json.loads((env.out / "staging_report.json").read_text())["catalog_integrity"]
    assert ci["tables_identical"]["tiles"] is False and ci["tables_identical"]["scenes"] is True and not ci["all_identical"]


def test_dry_run_plans_against_a_copy_and_never_touches_the_catalog(env, capsys):
    before = env.db.read_bytes()
    assert env.mod.main(["--dry-run"]) == 0
    assert env.db.read_bytes() == before and not env.db.with_name("tiles.sqlite.pre_phase10.bak").exists()
    assert "to do" in capsys.readouterr().out

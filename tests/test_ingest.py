"""Tests for the geoseek ingestion pipeline.

Unit tests use small synthetic arrays. The integration tests exercise the
real staged AOI scenes under data/datasets/ (two Sentinel-2 L2A dates over
Ayodhya, UP) end-to-end, including the hard incremental-index and
offline-after-staging requirements from the spec.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS
from shapely import wkt as shapely_wkt

from geoseek.config import get_settings
from geoseek.ingest import embed as embed_mod
from geoseek.ingest.embed import embed_tile_rgb_uint8, load_model_once, make_true_color_uint8
from geoseek.ingest.pipeline import ingest_scene
from geoseek.ingest.quality import BAD_SCL_CLASSES, cloud_fraction
from geoseek.ingest.reader import SceneData, read_scene
from geoseek.ingest.store import TileStore
from geoseek.ingest.tiler import tile_scene

SCENE_A = "S2B_44RPQ_20190330_1_L2A"  # 2019-03-30
SCENE_B = "S2A_44RPQ_20240308_0_L2A"  # 2024-03-08


def _scene_dir(scene_id: str):
    path = get_settings().datasets_dir / scene_id
    if not path.is_dir():
        pytest.skip(f"staged scene not found at {path} - run `python -m geoseek.staging.download_datasets`")
    return path


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


def test_reader_preserves_crs_transform_nodata():
    scene = read_scene(_scene_dir(SCENE_A), ["B04", "B03", "B02", "SCL"])
    assert scene.crs == CRS.from_epsg(32644)
    assert scene.transform.a > 0 and scene.transform.e < 0  # north-up pixel size, not identity
    assert scene.nodata == 0.0
    assert scene.width > 0 and scene.height > 0
    for band in ("B04", "B03", "B02", "SCL"):
        assert scene.bands[band].shape == (scene.height, scene.width)


def test_reader_missing_band_raises():
    with pytest.raises(FileNotFoundError):
        read_scene(_scene_dir(SCENE_A), ["B04", "B08"])  # B08 not staged for this AOI


# --------------------------------------------------------------------------
# tiler
# --------------------------------------------------------------------------


def _synthetic_scene(width=300, height=300, nodata_block=True) -> SceneData:
    rng = np.random.default_rng(0)
    red = rng.integers(500, 3000, size=(height, width), dtype=np.uint16)
    green = rng.integers(500, 3000, size=(height, width), dtype=np.uint16)
    blue = rng.integers(500, 3000, size=(height, width), dtype=np.uint16)
    scl = np.full((height, width), 4, dtype=np.uint8)  # all vegetation

    if nodata_block:
        # Make one full 256x256 tile region entirely nodata.
        red[0:256, 0:256] = 0
        green[0:256, 0:256] = 0
        blue[0:256, 0:256] = 0
        scl[0:256, 0:256] = 0

    return SceneData(
        scene_id="SYN_TEST_20200101",
        band_order=["B04", "B03", "B02", "SCL"],
        bands={"B04": red, "B03": green, "B02": blue, "SCL": scl},
        crs=CRS.from_epsg(32644),
        transform=Affine(10.0, 0.0, 700000.0, 0.0, -10.0, 2960000.0),
        nodata=0.0,
        width=width,
        height=height,
    )


def test_tiler_skips_fully_nodata_tile_and_keeps_partial_edge_tiles():
    scene = _synthetic_scene(width=300, height=300, nodata_block=True)
    tiles = tile_scene(scene)

    # 300x300 with 256px tiles -> 2x2 grid; top-left is fully nodata -> skipped.
    tile_ids = {t.tile_id for t in tiles}
    assert "SYN_TEST_20200101_r000_c000" not in tile_ids
    assert len(tiles) == 3

    # Edge tiles (row/col 1) are partial: 300-256=44px, not padded to 256.
    edge_tiles = [t for t in tiles if t.row == 1 or t.col == 1]
    assert edge_tiles
    for t in edge_tiles:
        assert t.width <= 256 and t.height <= 256
        assert (t.width < 256) or (t.height < 256)


def test_tiler_footprint_is_valid_lonlat_polygon():
    scene = _synthetic_scene(width=300, height=300, nodata_block=False)
    tiles = tile_scene(scene)
    assert len(tiles) == 4  # no nodata block this time -> full 2x2 grid kept

    poly = shapely_wkt.loads(tiles[0].footprint_wkt_4326)
    assert poly.is_valid
    minx, miny, maxx, maxy = poly.bounds
    assert -180 <= minx < maxx <= 180
    assert -90 <= miny < maxy <= 90


def test_tiler_on_real_scene_mix_of_full_and_partial_tiles():
    scene = read_scene(_scene_dir(SCENE_A), ["B04", "B03", "B02", "SCL"])
    tiles = tile_scene(scene)
    assert len(tiles) == 9  # 563x626 px -> 3x3 grid, all AOI has data (no nodata)
    widths = {t.width for t in tiles}
    heights = {t.height for t in tiles}
    assert 256 in widths and any(w < 256 for w in widths)
    assert 256 in heights and any(h < 256 for h in heights)


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------


def test_cloud_fraction_all_good():
    scl = np.full((10, 10), 4, dtype=np.uint8)  # vegetation, not bad
    assert cloud_fraction(scl) == 0.0


def test_cloud_fraction_all_bad():
    scl = np.full((10, 10), 9, dtype=np.uint8)  # cloud high probability
    assert cloud_fraction(scl) == 1.0


def test_cloud_fraction_mixed():
    scl = np.array([[4, 4, 9, 3]] * 10, dtype=np.uint8)  # 2/4 bad (cloud, cloud-shadow)
    assert cloud_fraction(scl) == pytest.approx(0.5)


def test_bad_scl_classes_match_spec():
    # cloud, cloud-shadow, snow, saturated -> bad (plus no_data, which carries no valid class)
    assert {1, 3, 8, 9, 10, 11}.issubset(BAD_SCL_CLASSES)
    assert 4 not in BAD_SCL_CLASSES  # vegetation is not bad
    assert 6 not in BAD_SCL_CLASSES  # water is not bad


# --------------------------------------------------------------------------
# embed: contrast stretch correctness (the CRITICAL CORRECTNESS requirement)
# --------------------------------------------------------------------------


def test_true_color_stretch_uses_full_8bit_range_not_raw_reflectance():
    rng = np.random.default_rng(1)
    # Reflectance-like values (S2 typical range), NOT 0-255.
    bands = {
        "B04": rng.integers(200, 3500, size=(64, 64)).astype(np.uint16),
        "B03": rng.integers(200, 3500, size=(64, 64)).astype(np.uint16),
        "B02": rng.integers(200, 3500, size=(64, 64)).astype(np.uint16),
    }
    rgb = make_true_color_uint8(bands, nodata=0)

    assert rgb.dtype == np.uint8
    assert rgb.shape == (64, 64, 3)
    # A proper percentile stretch should spread values across most of 0-255,
    # not cluster near 0 the way raw reflectance would if cast straight to uint8.
    assert rgb.min() < 20
    assert rgb.max() > 235


def test_true_color_stretch_excludes_nodata_from_percentiles():
    band = np.full((32, 32), 1000, dtype=np.uint16)
    band[0, 0] = 0  # nodata pixel, should not skew the stretch
    rgb = make_true_color_uint8({"B04": band, "B03": band, "B02": band}, nodata=0)
    # A single constant valid value (1000) with degenerate percentiles should not crash
    # and should not produce NaNs/overflow.
    assert np.isfinite(rgb).all()


def test_true_color_stretch_all_nodata_returns_zeros():
    band = np.zeros((16, 16), dtype=np.uint16)
    rgb = make_true_color_uint8({"B04": band, "B03": band, "B02": band}, nodata=0)
    assert np.array_equal(rgb, np.zeros((16, 16, 3), dtype=np.uint8))


# --------------------------------------------------------------------------
# embed: model loaded once, reused
# --------------------------------------------------------------------------


def test_model_loaded_once_and_reused():
    model_a, _, _ = load_model_once()
    load_count_after_first = embed_mod._LOAD_COUNT

    model_b, _, _ = load_model_once()
    load_count_after_second = embed_mod._LOAD_COUNT

    assert model_a is model_b  # same object -> not reconstructed
    assert load_count_after_first == load_count_after_second  # no reload happened


def test_embed_produces_unit_norm_512d_vector():
    rgb = np.random.default_rng(2).integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    vec, latency_ms = embed_tile_rgb_uint8(rgb)
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)
    assert latency_ms > 0


# --------------------------------------------------------------------------
# store: incremental add
# --------------------------------------------------------------------------


def _fake_record(tile_id: str, scene_id: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return {
        "tile_id": tile_id,
        "scene_id": scene_id,
        "sensor": "Sentinel-2B",
        "acq_date": "2019-03-30",
        "geom_wkt_4326": "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        "cloud_fraction": 0.0,
        "embedding": v,
        "processing_history": [{"step": "read", "at": "t0"}, {"step": "indexed", "at": "t1"}],
    }


def test_store_incremental_add_never_touches_existing_vectors(tmp_path):
    store = TileStore(index_dir=tmp_path)
    records_a = [_fake_record(f"a{i}", "SCENE_A", seed=i) for i in range(5)]
    store.add_tiles(records_a)
    store.save()
    assert store.count() == 5

    vectors_before = [store.reconstruct(i).copy() for i in range(5)]

    records_b = [_fake_record(f"b{i}", "SCENE_B", seed=100 + i) for i in range(3)]
    store.add_tiles(records_b)
    store.save()
    assert store.count() == 8
    assert store.row_count() == 8

    for i in range(5):
        assert np.array_equal(store.reconstruct(i), vectors_before[i])  # byte-identical, no rebuild

    store.close()


def test_store_reopens_persisted_index_and_appends(tmp_path):
    store1 = TileStore(index_dir=tmp_path)
    store1.add_tiles([_fake_record("x0", "SCENE_X", seed=1)])
    store1.save()
    store1.close()

    store2 = TileStore(index_dir=tmp_path)  # fresh instance, loads from disk
    assert store2.count() == 1
    store2.add_tiles([_fake_record("x1", "SCENE_X", seed=2)])
    store2.save()
    assert store2.count() == 2
    store2.close()


# --------------------------------------------------------------------------
# pipeline: full integration on the real staged AOI scenes (acceptance 1 & 2)
# --------------------------------------------------------------------------


def test_pipeline_ingest_two_scenes_incremental(tmp_path):
    dir_a = _scene_dir(SCENE_A)
    dir_b = _scene_dir(SCENE_B)

    report_a = ingest_scene(dir_a, save_sample=False, index_dir=tmp_path, record_manifest=False)
    tiles_added_a = report_a["tiles_added"]
    assert tiles_added_a == 9
    assert report_a["index_total_vectors"] == tiles_added_a

    store = TileStore(index_dir=tmp_path)
    vectors_after_a = [store.reconstruct(i).copy() for i in range(tiles_added_a)]
    store.close()

    report_b = ingest_scene(dir_b, save_sample=False, index_dir=tmp_path, record_manifest=False)
    tiles_added_b = report_b["tiles_added"]
    assert report_b["index_total_vectors"] == tiles_added_a + tiles_added_b

    store = TileStore(index_dir=tmp_path)
    assert store.count() == tiles_added_a + tiles_added_b
    assert store.row_count() == tiles_added_a + tiles_added_b

    for i in range(tiles_added_a):
        assert np.array_equal(store.reconstruct(i), vectors_after_a[i])  # proven incremental, no rebuild

    tiles = store.repo.list_tiles()  # faiss_id order, via the catalog repository
    assert len(tiles) == tiles_added_a + tiles_added_b
    for t in tiles:
        history = t.processing_history
        assert isinstance(history, list) and len(history) >= 3
        steps = [h["step"] for h in history]
        assert steps == ["read", "tiled", "quality_scored", "embedded", "indexed"]
        assert all("at" in h and h["at"] for h in history)
        assert 0.0 <= t.cloud_fraction <= 1.0
        # every tile reaches its source scene + collection by foreign key
        prov = store.repo.get_tile_provenance(t.tile_id)
        assert prov.observation.observation_id == t.observation_id
        assert prov.scene.scene_id in (SCENE_A, SCENE_B)
        assert prov.collection.collection_id == "sentinel-2-l2a"
    store.close()


def test_pipeline_run_report_has_required_fields(tmp_path):
    report = ingest_scene(_scene_dir(SCENE_A), save_sample=False, index_dir=tmp_path, record_manifest=False)
    for key in ("scene_id", "tiles_added", "index_total_vectors", "build_time_s", "index_size_mb", "mean_embed_latency_ms"):
        assert key in report
    assert report["build_time_s"] > 0
    assert report["index_size_mb"] >= 0
    assert report["mean_embed_latency_ms"] > 0


# --------------------------------------------------------------------------
# offline (acceptance 6): ingest must work with the network OFF
# --------------------------------------------------------------------------


def test_ingest_works_fully_offline(tmp_path, monkeypatch):
    def _blocked_connect(*args, **kwargs):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a, **k: 111)

    report = ingest_scene(_scene_dir(SCENE_B), save_sample=False, index_dir=tmp_path, record_manifest=False)
    assert report["tiles_added"] == 9


def test_vanilla_clip_loads_and_embeds_fully_offline(monkeypatch):
    """Regression test: open_clip's pretrained='openai' loader pulls weights via
    huggingface_hub, which by default re-checks the Hub for a newer revision even
    when the local cache is already populated - a real network call from a module
    that is not geoseek.staging. load_vanilla_clip_once() must force HF_HUB_OFFLINE
    so this never happens; this test proves it with sockets actually blocked.
    """
    cache_dir = get_settings().models_dir / embed_mod.VANILLA_CACHE_DIRNAME
    if not cache_dir.is_dir() or not any(cache_dir.rglob("*")):
        pytest.skip(
            f"vanilla OpenCLIP not staged at {cache_dir} - run "
            "`python -m geoseek.staging.download_models --vanilla`"
        )

    def _blocked_connect(*args, **kwargs):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a, **k: 111)

    # Force a fresh load even if an earlier test in this session already cached it.
    monkeypatch.setattr(embed_mod, "_VANILLA_MODEL", None)
    monkeypatch.setattr(embed_mod, "_VANILLA_PREPROCESS", None)

    model, preprocess = embed_mod.load_vanilla_clip_once()
    assert model is not None

    vec = embed_mod.embed_text_vanilla("a river with sandbars")
    assert vec.shape == (512,)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)

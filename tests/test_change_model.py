"""Phase 3b: the trained change-detection model.

Two layers:
  * pure ``FCSiamDiff`` architecture unit tests - always run (torch only).
  * ``FCSiamDiffChangeModel`` seam tests - hermetic: a tiny freshly-initialised
    net saved to a tmp checkpoint + synthetic 5-band GeoTIFFs, no network, no
    dependence on the real trained weights or on OSCD being staged.

An integration check against the real trained checkpoint runs only if
``data/change_model/fc_siam_diff.pt`` exists (skips cleanly otherwise).
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geoseek.catalog.entities import Observation
from geoseek.change.models import FCSiamDiff, FCSiamDiffChangeModel, count_parameters
from geoseek.change.models.fc_siam_diff import DEFAULT_BANDS
from geoseek.models.base import ChangeDetectionModel, ChangeResult
from geoseek.temporal.contract import ComparabilityCriterion, ObservationPair, PairComparability

BANDS = ("B02", "B03", "B04", "B08", "B11")


# ==========================================================================
# FCSiamDiff - architecture
# ==========================================================================


def test_default_param_count_is_small_1_to_2M():
    m = FCSiamDiff(in_channels=5, base_channels=24, depth=4)
    n = m.num_parameters()
    assert 0.5e6 < n < 2.0e6, f"{n:,} params - outside the ~1-2M 'small' target"


@pytest.mark.parametrize("hw", [(96, 96), (64, 96), (97, 131), (256, 256), (40, 40)])
def test_forward_shape_matches_input(hw):
    h, w = hw
    m = FCSiamDiff(in_channels=5, base_channels=16, depth=4).eval()
    x1 = torch.randn(2, 5, h, w)
    x2 = torch.randn(2, 5, h, w)
    y = m(x1, x2)
    assert y.shape == (2, 1, h, w)


def test_forward_is_order_invariant():
    """|f1 - f2| skip combination -> f(a, b) == f(b, a) exactly."""
    m = FCSiamDiff(in_channels=5, base_channels=16, depth=4).eval()
    a = torch.randn(2, 5, 96, 96)
    b = torch.randn(2, 5, 96, 96)
    assert torch.equal(m(a, b), m(b, a))


def test_encoder_is_shared_siamese_no_second_branch():
    m = FCSiamDiff(in_channels=5, base_channels=16, depth=4)
    total = m.num_parameters()
    parts = (count_parameters(m.enc_blocks) + count_parameters(m.upconvs)
             + count_parameters(m.dec_blocks) + count_parameters(m.classifier))
    # if the two dates went through *separate* encoders, total would exceed the
    # sum of one encoder + decoder + head.
    assert total == parts
    feats = m.encode(torch.randn(1, 5, 96, 96))
    assert [f.shape[1] for f in feats] == m.channels


def test_identical_dates_give_zero_change_logits_before_training():
    m = FCSiamDiff(in_channels=5, base_channels=16, depth=4).eval()
    x = torch.randn(1, 5, 96, 96)
    out = m(x, x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_predict_proba_in_unit_interval():
    m = FCSiamDiff(in_channels=5, base_channels=16, depth=4).eval()
    p = m.predict_proba(torch.randn(1, 5, 64, 64), torch.randn(1, 5, 64, 64))
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_default_bands_are_geoseek_production_bands():
    assert DEFAULT_BANDS == BANDS


# ==========================================================================
# FCSiamDiffChangeModel - the ChangeDetectionModel seam
# ==========================================================================


def _write_bands(dirpath, seed, h=192, w=160, bands=BANDS, change_block=None):
    import rasterio
    from affine import Affine

    dirpath.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)
    for bi, b in enumerate(bands):
        arr = rng.integers(600, 3200, (h, w)).astype(np.uint16)
        if change_block is not None:
            ys, xs = change_block
            arr[ys, xs] = rng.integers(3200, 6000, arr[ys, xs].shape).astype(np.uint16)
        profile = dict(driver="GTiff", height=h, width=w, count=1, dtype="uint16",
                       crs="EPSG:32644", transform=transform, nodata=0)
        with rasterio.open(dirpath / f"{b}.tif", "w", **profile) as dst:
            dst.write(arr, 1)


def _tiny_checkpoint(path, *, base_channels=8, depth=3):
    net = FCSiamDiff(in_channels=5, base_channels=base_channels, depth=depth, dropout=0.0)
    card = {
        "architecture": {"name": "FCSiamDiff", "in_channels": 5, "base_channels": base_channels,
                         "depth": depth, "channels": net.channels, "parameters": net.num_parameters()},
        "bands": list(BANDS),
        "norm_stats": {"bands": list(BANDS), "mean": [0.12] * 5, "std": [0.05] * 5,
                       "space": "reflectance (DN / 10000)"},
        "provenance": {"weights_sha256": "deadbeef" * 8},
        "eval": {"chosen_threshold_precision_favouring": 0.5},
    }
    torch.save({"state_dict": net.state_dict(), "model_card": card}, path)
    return path


def _obs(obs_id, date, dataset_dir):
    return Observation(observation_id=obs_id, scene_id=obs_id.replace("_scaled", ""),
                       acquired_at=date, footprint_wkt_4326="POLYGON ((0 0,0 1,1 1,1 0,0 0))",
                       aoi_name="x", dataset_dir=str(dataset_dir),
                       coregistration={"reference_observation": "ref", "median_magnitude_px": 0.1})


def _pair(tmp_path, comparable=True):
    e_dir, l_dir = tmp_path / "obsA", tmp_path / "obsB"
    _write_bands(e_dir, seed=1)
    _write_bands(l_dir, seed=2, change_block=(slice(20, 80), slice(30, 90)))
    crit = [ComparabilityCriterion(name="spatial_overlap", passed=True, blocking=True, detail="ok", value=1.0)]
    pc = PairComparability(comparable=comparable, criteria=crit)
    return ObservationPair(earlier=_obs("obsA", "2019-03-30", e_dir),
                           later=_obs("obsB", "2024-03-08", l_dir), comparability=pc)


@pytest.fixture()
def model(tmp_path):
    ckpt = _tiny_checkpoint(tmp_path / "tiny.pt")
    return FCSiamDiffChangeModel(checkpoint_path=ckpt, device="cpu", threshold=0.5, tile_px=64)


def test_is_a_change_detection_model(model):
    assert isinstance(model, ChangeDetectionModel)
    assert not getattr(type(model), "__abstractmethods__", set())


def test_consumes_observation_pair_and_returns_change_result(tmp_path, model):
    pair = _pair(tmp_path)
    res = model.predict_change(pair, bands=list(BANDS))
    assert isinstance(res, ChangeResult)
    assert res.earlier_observation_id == "obsA" and res.later_observation_id == "obsB"
    assert res.method == "fc_siam_diff"
    assert res.comparable is True
    assert res.change_score_by_tile, "expected per-tile change scores"
    assert all(k.startswith("obsB_r") for k in res.change_score_by_tile), "tiles keyed by the LATER obs"
    assert set(res.change_label_by_tile.values()) <= {"change", "no_change"}
    assert 0.0 <= res.confidence <= 1.0
    from pathlib import Path
    assert res.change_mask_path and Path(res.change_mask_path).is_file()


def test_zero_reshaping_from_matcher_observation_pair(tmp_path):
    """An ObservationPair straight from TemporalObservationMatcher feeds in unchanged."""
    from geoseek.catalog.entities import Collection, Scene
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    from geoseek.temporal.matcher import TemporalObservationMatcher

    repo = SQLiteMetadataRepository(tmp_path / "cat.sqlite")
    repo.register_collection(Collection(collection_id="sentinel-2-l2a", sensor="MSI", platform="Sentinel-2",
                                        bands=tuple(BANDS), native_gsd_m=10.0))
    aoi = "POLYGON ((82.0 26.5, 82.6 26.5, 82.6 27.1, 82.0 27.1, 82.0 26.5))"
    for oid, date, seed, cb in (("A_scaled", "2019-03-30", 1, None),
                                ("B_scaled", "2024-03-08", 2, (slice(20, 80), slice(30, 90)))):
        ddir = tmp_path / oid
        _write_bands(ddir, seed=seed, change_block=cb)
        repo.register_scene(Scene(scene_id=oid.replace("_scaled", ""), collection_id="sentinel-2-l2a",
                                  platform="Sentinel-2A", acquired_at=date, footprint_wkt_4326=aoi))
        repo.register_observation(Observation(
            observation_id=oid, scene_id=oid.replace("_scaled", ""), acquired_at=date,
            footprint_wkt_4326=aoi, aoi_name="ayodhya", dataset_dir=str(ddir),
            quality_summary={"n_tiles": 100, "cloud_fraction_mean": 0.0, "cloud_fraction_max": 0.0,
                             "n_clear_tiles_cf_le_0p05": 100},
            coregistration={"reference_observation": "B_scaled", "median_magnitude_px": 0.16,
                            "correction_applied": False}))
    seq = TemporalObservationMatcher(repo).match(location=(82.3, 26.8))
    repo.close()
    pair = seq.comparable_pairs[0]

    model = FCSiamDiffChangeModel(checkpoint_path=_tiny_checkpoint(tmp_path / "t.pt"),
                                  device="cpu", threshold=0.5, tile_px=64)
    res = model.predict_change(pair, aoi_wkt=seq.aoi_wkt)   # <- no reshaping at the boundary
    assert isinstance(res, ChangeResult)
    assert res.earlier_observation_id == "A_scaled" and res.later_observation_id == "B_scaled"
    assert res.comparable is True and res.change_score_by_tile


def test_not_comparable_pair_is_refused_with_low_confidence(tmp_path, model):
    pair = _pair(tmp_path, comparable=False)
    res = model.predict_change(pair)
    assert res.comparable is False
    assert res.confidence == 0.0
    assert any("not comparable" in n for n in res.notes)
    assert res.change_mask_path is None


def test_band_not_in_training_set_raises(tmp_path, model):
    pair = _pair(tmp_path)
    with pytest.raises(ValueError):
        model.predict_change(pair, bands=["B02", "B12"])  # B12 was never trained on


def test_runs_fully_offline(tmp_path, monkeypatch, model):
    def _blocked(*a, **k):
        raise OSError("network disabled for offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", lambda *a, **k: 111)
    res = model.predict_change(_pair(tmp_path))
    assert isinstance(res, ChangeResult) and res.change_score_by_tile


def test_restricting_to_a_tile_subset(tmp_path, model):
    pair = _pair(tmp_path)

    class _T:
        def __init__(self, r, c):
            self.row, self.col = r, c

    res = model.predict_change(pair, tiles=[_T(0, 0), _T(1, 1)])
    assert 0 < len(res.change_score_by_tile) <= 2


# ==========================================================================
# integration: the real trained checkpoint (skips if not trained yet)
# ==========================================================================


def test_real_trained_checkpoint_loads_and_reports_operating_point():
    from geoseek.change.models.fc_siam_diff_model import DEFAULT_CHECKPOINT

    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("real checkpoint not trained yet - run scripts/train_change.py")
    m = FCSiamDiffChangeModel(device="cpu")
    m.load()
    assert m._loaded.model.num_parameters() > 0
    assert 0.0 < m.threshold < 1.0
    assert list(m._loaded.bands) == list(BANDS)

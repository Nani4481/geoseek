"""Tests for the model interfaces: EmbeddingModel, QualityEstimator, ChangeDetectionModel."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from geoseek.catalog.entities import Observation
from geoseek.models import (
    ChangeDetectionModel,
    ChangeResult,
    EmbeddingModel,
    QualityEstimator,
    QualityReport,
    RemoteCLIPEmbeddingModel,
    SclQualityEstimator,
)
from geoseek.temporal.contract import ObservationPair, PairComparability


# --------------------------------------------------------------------------
# EmbeddingModel  (RemoteCLIP - integration, needs the staged weights)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emb():
    m = RemoteCLIPEmbeddingModel()
    m.load()
    return m


def test_remoteclip_is_an_embedding_model(emb):
    assert isinstance(emb, EmbeddingModel)
    assert emb.embedding_dim == 512


def test_encode_text_is_unit_norm_512(emb):
    v = emb.encode_text("a river with sandbars")
    assert v.shape == (512,) and v.dtype == np.float32
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


def test_encode_image_and_batch_match(emb):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
    single = emb.encode_image(imgs[1])
    batch = emb.encode_images(imgs, batch_size=2)
    assert batch.shape == (3, 512)
    assert np.allclose(single, batch[1], atol=1e-4)
    assert np.linalg.norm(batch[0]) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# QualityEstimator  (SCL)
# --------------------------------------------------------------------------


def test_scl_quality_all_vegetation_is_fully_usable():
    q = SclQualityEstimator()
    rep = q.estimate({"SCL": np.full((32, 32), 4, dtype=np.uint8)})
    assert isinstance(q, QualityEstimator)
    assert rep.cloud_fraction == 0.0 and rep.usable_fraction == 1.0
    assert rep.flags["class_fractions"]["vegetation"] == 1.0


def test_scl_quality_all_cloud_is_unusable():
    rep = SclQualityEstimator().estimate({"SCL": np.full((10, 10), 9, dtype=np.uint8)})
    assert rep.cloud_fraction == 1.0 and rep.usable_fraction == 0.0


def test_scl_quality_mixed_and_water_flag():
    scl = np.full((10, 10), 4, dtype=np.uint8)
    scl[:2, :] = 9   # 20% cloud
    scl[8:, :] = 6   # 20% water (not "bad", but flagged)
    rep = SclQualityEstimator().estimate({"SCL": scl})
    assert rep.cloud_fraction == pytest.approx(0.2)
    assert rep.flags["water_fraction"] == pytest.approx(0.2)


def test_scl_quality_requires_scl_band():
    with pytest.raises(ValueError):
        SclQualityEstimator().estimate({"B04": np.zeros((4, 4))})


def test_quality_report_as_dict():
    d = QualityReport(0.1, 0.9, {"x": 1}).as_dict()
    assert d == {"cloud_fraction": 0.1, "usable_fraction": 0.9, "flags": {"x": 1}}


# --------------------------------------------------------------------------
# ChangeDetectionModel  (interface only - Phase 3b)
# --------------------------------------------------------------------------


def test_change_detection_model_is_interface_only():
    assert inspect.isabstract(ChangeDetectionModel)
    with pytest.raises(TypeError):
        ChangeDetectionModel()  # cannot instantiate the ABC


def _obs(obs_id, date):
    return Observation(observation_id=obs_id, scene_id=obs_id.replace("_scaled", ""),
                       acquired_at=date, footprint_wkt_4326="POLYGON ((0 0,0 1,1 1,1 0,0 0))",
                       aoi_name="x", dataset_dir=obs_id)


def test_change_detection_model_consumes_matcher_observation_pair_unchanged():
    """A Phase 3b model slots straight onto TemporalObservationMatcher output."""

    class _StubChangeModel(ChangeDetectionModel):
        def predict_change(self, pair, *, tiles=None, aoi_wkt=None, bands=None):
            # the input is exactly an ObservationPair from geoseek.temporal
            return ChangeResult(
                earlier_observation_id=pair.earlier.observation_id,
                later_observation_id=pair.later.observation_id,
                method="stub",
                comparable=pair.comparable,
                notes=list(pair.comparability.blocking_reasons),
            )

    pair = ObservationPair(
        earlier=_obs("A_scaled", "2019-03-30"),
        later=_obs("B_scaled", "2024-03-08"),
        comparability=PairComparability(comparable=True, criteria=[]),
    )
    res = _StubChangeModel().predict_change(pair, bands=["B04", "B08"])
    assert res.earlier_observation_id == "A_scaled" and res.later_observation_id == "B_scaled"
    assert res.comparable is True
    assert res.as_dict()["method"] == "stub"

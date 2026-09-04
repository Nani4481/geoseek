"""Phase 5: SAR backscatter + corroboration, fusion ranking, discovery clustering.

Hermetic - synthetic arrays + stubs. No network, no staged S1/S2 needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.change.confidence import compute_confidence
from geoseek.fusion.ranker import FusionRanker, _rescale_semantic, fusion_score
from geoseek.sar.backscatter import db_change, lee_filter
from geoseek.sar.evidence import sar_factor


# --------------------------------------------------------------------------
# SAR backscatter (Step C)
# --------------------------------------------------------------------------


def test_lee_filter_smooths_homogeneous_keeps_edge():
    rng = np.random.default_rng(0)
    img = np.full((80, 80), 100.0)
    img[:, 40:] = 400.0                                  # a hard edge
    speck = img * rng.gamma(4.4, 1 / 4.4, img.shape)     # multiplicative speckle, mean 1
    out = lee_filter(speck)
    left = out[20:60, 5:35]
    right = out[20:60, 45:75]
    assert left.std() < speck[20:60, 5:35].std() * 0.6          # speckle knocked down
    assert abs(np.median(left) - 100) < 25 and abs(np.median(right) - 400) < 60
    assert np.median(right) - np.median(left) > 200             # edge preserved


def test_db_change_zero_and_known_ratio_and_nodata():
    a = np.full((16, 16), 200, np.uint16)
    db, valid = db_change(a, a, speckle_filter=False)
    assert np.nanmax(np.abs(db[valid])) < 1e-6                  # identical -> 0 dB
    b = np.full((16, 16), 400, np.uint16)                       # 2x amplitude -> 4x intensity
    db2, _ = db_change(a, b, speckle_filter=False)
    assert np.nanmedian(db2) == pytest.approx(10 * np.log10(4.0), abs=0.05)   # ~+6.02 dB
    c = a.copy(); c[0, 0] = 0
    db3, v3 = db_change(c, b, speckle_filter=False)
    assert np.isnan(db3[0, 0]) and not v3[0, 0]


# --------------------------------------------------------------------------
# SAR corroboration factor (Step C)
# --------------------------------------------------------------------------


def test_sar_factor_water_gain_expects_backscatter_drop():
    # scene VV trend +4 dB; a real new water body drops well below that
    up, _ = sar_factor("water_gain", dvv_db=-2.0, dvh_db=-1.0, scene_dvv_db=+4.0)
    assert up > 1.0                                              # anomaly -6 dB -> strong agreement
    down, _ = sar_factor("water_gain", dvv_db=+7.0, dvh_db=+3.0, scene_dvv_db=+4.0)
    assert down < 1.0                                            # anomaly +3 dB -> disagreement


def test_sar_factor_construction_expects_rise():
    up, d = sar_factor("construction", dvv_db=+5.0, dvh_db=+3.0, scene_dvv_db=+0.5)
    assert up > 1.0 and "rise" in d


def test_sar_factor_neutral_when_unavailable_or_no_expectation():
    assert sar_factor("water_gain", None, None, available=False)[0] == 1.0
    assert sar_factor("road", -5.0, -5.0)[0] == 1.0
    assert sar_factor("other", +5.0, +5.0)[0] == 1.0
    # matches the scene trend -> within noise -> neutral
    f, _ = sar_factor("construction", dvv_db=0.6, dvh_db=0.6, scene_dvv_db=0.5)
    assert f == 1.0


def test_confidence_folds_sar_factor_and_clamps():
    base = dict(candidate_id="c", model_prob=0.9, bad_scl_fraction=0.0, valid_fraction=1.0,
                coreg_residual_px=0.15, radiometric_reliability=1.0, persistence="persistent",
                persistence_confidence=0.95, spectral_agreement=0.9, change_type="water_gain")
    neutral = compute_confidence(**base).confidence
    boosted = compute_confidence(**base, sar_corroboration=1.10, sar_detail="VV drop").confidence
    hurt = compute_confidence(**base, sar_corroboration=0.80, sar_detail="VV rose").confidence
    assert hurt < neutral <= boosted <= 1.0
    assert any("SAR corroboration" in b for b in
               compute_confidence(**base, sar_corroboration=0.8, sar_detail="x").breakdown)


# --------------------------------------------------------------------------
# fusion ranking (Step D)
# --------------------------------------------------------------------------


def test_fusion_score_geometric_and_query_optional():
    no_q = fusion_score(0.9, 0.5)
    with_q = fusion_score(0.9, 0.5, semantic=0.9)
    assert with_q > no_q                                        # a strong semantic match lifts it
    assert fusion_score(0.9, 0.5, semantic=0.1) < no_q          # a poor match drags it down
    # at equal significance + semantic, higher change confidence wins (w_c 3 > w_s 1.5, w_q 2)
    assert fusion_score(0.92, 0.6, semantic=0.7) > fusion_score(0.60, 0.6, semantic=0.7)
    # semantic alone cannot carry it: a perfect match with weak change evidence stays modest
    assert fusion_score(0.30, 0.5, semantic=1.0) < 0.6
    # ... but a perfect semantic match with solid change evidence rises to the top
    assert fusion_score(0.85, 0.7, semantic=1.0) > fusion_score(0.85, 0.7, semantic=0.3)


def test_semantic_rescale_band():
    assert _rescale_semantic(0.15) == pytest.approx(0.0)
    assert _rescale_semantic(0.32) == pytest.approx(1.0)
    assert 0.0 < _rescale_semantic(0.24) < 1.0


def test_fusion_ranker_without_engine_ranks_by_change_evidence():
    cands = [
        {"candidate_id": "big", "confidence": 0.80, "significance": 0.95, "change_type": "water_gain",
         "centroid_lonlat": (82.2, 26.8), "area_m2": 3e5, "later_obs": "L"},
        {"candidate_id": "sure", "confidence": 0.93, "significance": 0.55, "change_type": "construction",
         "centroid_lonlat": (82.3, 26.7), "area_m2": 6e4, "later_obs": "L"},
        {"candidate_id": "weak", "confidence": 0.60, "significance": 0.60, "change_type": "other",
         "centroid_lonlat": (82.4, 26.6), "area_m2": 5e4, "later_obs": "L"},
    ]
    ranked, meta = FusionRanker(None).rank(cands, query=None)
    assert meta["query"] is None
    assert [r.candidate_id for r in ranked][-1] == "weak"
    assert all(r.semantic is None for r in ranked)
    assert ranked[0].fusion_score > ranked[-1].fusion_score


# --------------------------------------------------------------------------
# discovery: HDBSCAN clustering (Step E)
# --------------------------------------------------------------------------


class _StubEmbedder:
    """encode_text -> a deterministic unit vector per string."""

    def encode_text(self, s: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(s)) % (2**32))
        v = rng.standard_normal(512).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def test_cluster_embeddings_finds_blobs_and_labels_them():
    from geoseek.discovery.cluster import cluster_embeddings

    rng = np.random.default_rng(1)
    centres = rng.standard_normal((3, 512)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    blobs = []
    for k in range(3):
        b = centres[k] + 0.03 * rng.standard_normal((120, 512)).astype(np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        blobs.append(b)
    X = np.vstack(blobs)
    tids = [f"t_{i}" for i in range(len(X))]
    res = cluster_embeddings(X, tids, _StubEmbedder(), min_cluster_size=25)
    assert res.n_clusters >= 2
    assert sum(res.sizes.values()) + res.noise_count == len(X)
    for cid in res.sizes:
        assert len(res.cluster_concepts[cid]) == 3
    d = res.as_dict()
    assert d["n_clusters"] == res.n_clusters and "params" in d

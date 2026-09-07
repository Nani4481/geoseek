"""Phase 7b follow-up: vector-index bulk reconstruct, result-set refinements
(region filter / score threshold / near-dup suppression), and sample+assign
clustering for large corpora.

Hermetic - synthetic arrays and a tiny stub embedding model. No FAISS file on
disk beyond tmp_path, no catalog, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.config import EMBEDDING_DIM
from geoseek.discovery.cluster import (cluster_sample_and_assign, load_all_vectors,
                                       stratified_sample_indices)
from geoseek.search.rerank import (apply_score_threshold, best_threshold_by_f1,
                                   filter_by_region, region_key, suppress_near_duplicates)
from geoseek.vectorindex import FaissFlatIPIndex


def _unit(rng, n, d=EMBEDDING_DIM):
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# --------------------------------------------------------------------------
# VectorIndex.reconstruct_all
# --------------------------------------------------------------------------

def test_reconstruct_all_matches_per_vector_and_load_all_vectors(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    rng = np.random.default_rng(0)
    vecs = _unit(rng, 37)
    idx.add(vecs)

    allv = idx.reconstruct_all()
    assert allv.shape == (37, EMBEDDING_DIM) and allv.dtype == np.float32
    for i in range(37):
        assert np.array_equal(allv[i], idx.get_vector(i))
    # discovery helper now routes through the bulk path
    assert np.array_equal(load_all_vectors(idx), allv)


def test_reconstruct_all_empty(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    assert idx.reconstruct_all().shape == (0, EMBEDDING_DIM)


def test_reconstruct_all_survives_reopen(tmp_path):
    p = tmp_path / "v.faiss"
    idx = FaissFlatIPIndex(p)
    rng = np.random.default_rng(1)
    idx.add(_unit(rng, 10))
    idx.persist()
    reopened = FaissFlatIPIndex(p)
    assert np.array_equal(reopened.reconstruct_all(), idx.reconstruct_all())


# --------------------------------------------------------------------------
# region_key
# --------------------------------------------------------------------------

@pytest.mark.parametrize("aoi,expect", [
    ("dehradun_43RGP_diverse", "dehradun"),
    ("dehradun_44RKU_diverse", "dehradun"),
    ("jaisalmer_42RYR_diverse", "jaisalmer"),
    ("kerala_backwaters_43PFL_diverse", "kerala_backwaters"),
    ("delhi_ncr_43RFM_diverse", "delhi_ncr"),
    ("ayodhya_44RPQ_scaled_82km", "ayodhya"),
    ("ayodhya-82km", "ayodhya"),
    ("sundarbans_45QXE_diverse", "sundarbans"),
    (None, "unknown"),
    ("", "unknown"),
])
def test_region_key(aoi, expect):
    assert region_key(aoi) == expect


# --------------------------------------------------------------------------
# result-set refinements
# --------------------------------------------------------------------------

def test_apply_score_threshold_is_a_prefix_cut():
    ranked = [("a", 0.9), ("b", 0.5), ("c", 0.49), ("d", 0.8)]
    assert apply_score_threshold(ranked, 0.5) == [("a", 0.9), ("b", 0.5)]
    assert apply_score_threshold(ranked, 0.95) == []
    assert apply_score_threshold(ranked, 0.0) == ranked


def test_filter_by_region_preserves_order():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
    reg = {"a": "ay", "b": "kanha", "c": "ay", "d": "kutch"}
    assert filter_by_region(ranked, reg.__getitem__, ["ay"]) == [("a", 0.9), ("c", 0.7)]


def test_suppress_near_duplicates_greedy_keeps_first():
    v = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([1.0, 0.02, 0.0]),   # ~identical to a
        "c": np.array([0.0, 1.0, 0.0]),
        "d": np.array([0.0, 1.0, 0.03]),   # ~identical to c
        "e": np.array([0.4, 0.4, 0.82]),
    }
    v = {k: x / np.linalg.norm(x) for k, x in v.items()}
    ranked = [("a", 0.9), ("b", 0.89), ("c", 0.7), ("d", 0.69), ("e", 0.6)]
    kept, suppressed = suppress_near_duplicates(ranked, v.__getitem__, tau_dup=0.99)
    assert [k for k, _ in kept] == ["a", "c", "e"]
    assert suppressed == ["b", "d"]
    # a stricter threshold than the ~0.9998 duplicate cosine suppresses nothing
    kept2, sup2 = suppress_near_duplicates(ranked, v.__getitem__, tau_dup=0.9999)
    assert sup2 == [] and len(kept2) == 5


def test_best_threshold_by_f1_picks_the_separating_cut():
    # relevant items score high, irrelevant score low -> a mid threshold is best
    r1 = [("p", 0.40), ("q", 0.38), ("x", 0.20), ("y", 0.18)]
    j1 = {"p": 2, "q": 1, "x": 0, "y": 0}
    r2 = [("s", 0.42), ("t", 0.15), ("u", 0.12)]
    j2 = {"s": 2, "t": 0, "u": 0}
    tau, curve = best_threshold_by_f1([(r1, j1), (r2, j2)],
                                      taus=[0.10, 0.25, 0.30, 0.45], k=4)
    assert tau in (0.25, 0.30)                       # cuts the irrelevant tail, keeps the relevant head
    assert curve[0.30]["f1"] >= curve[0.10]["f1"]


# --------------------------------------------------------------------------
# stratified sampling
# --------------------------------------------------------------------------

def test_stratified_sample_respects_floor_proportional_and_cap():
    groups = ["A"] * 5000 + ["B"] * 3000 + ["C"] * 120
    idx = stratified_sample_indices(groups, 2000, seed=3, min_per_group=200)
    got = np.asarray(groups)[idx]
    n = {g: int((got == g).sum()) for g in ("A", "B", "C")}
    assert len(idx) <= 2000
    assert n["C"] == 120                              # small group taken whole (below floor)
    assert n["A"] >= 200 and n["B"] >= 200            # floor honoured
    assert n["A"] > n["B"]                            # remainder split proportionally
    # deterministic
    idx2 = stratified_sample_indices(groups, 2000, seed=3, min_per_group=200)
    assert np.array_equal(idx, idx2)


def test_stratified_sample_smaller_than_corpus_when_asked_for_more():
    groups = ["A"] * 100 + ["B"] * 50
    idx = stratified_sample_indices(groups, 10_000, seed=0, min_per_group=200)
    assert len(idx) == 150                            # capped at corpus size


# --------------------------------------------------------------------------
# sample + assign clustering
# --------------------------------------------------------------------------

class _StubEmbed:
    """Deterministic 'text tower': maps a concept string to a fixed unit vector."""

    def __init__(self, dim):
        self.dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


def test_cluster_sample_and_assign_recovers_blobs_and_labels_everything():
    dim = 16
    rng = np.random.default_rng(7)
    centers = _unit(rng, 3, dim)
    blocks, groups = [], []
    for bi, c in enumerate(centers):
        pts = c + 0.02 * rng.standard_normal((900, dim)).astype(np.float32)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        blocks.append(pts)
        groups += [f"region{bi}"] * 900
    vecs = np.concatenate(blocks)
    tile_ids = [f"t{i}" for i in range(len(vecs))]

    res = cluster_sample_and_assign(
        vecs, tile_ids, groups, _StubEmbed(dim),
        sample_size=900, min_cluster_size=30, core_dist_n_jobs=1, seed=1, min_per_group=100)

    assert res.n_clusters == 3
    assert res.labels.shape == (2700,)
    assert res.noise_count == 0                         # hard nearest-centroid labels everyone
    assert set(res.labels.tolist()) == set(res.centroid_ids)
    # centroids are unit-norm
    assert np.allclose(np.linalg.norm(res.centroids, axis=1), 1.0, atol=1e-5)
    # each true blob ended up dominated by a single assigned label
    for bi in range(3):
        lab = res.labels[bi * 900:(bi + 1) * 900]
        top = np.bincount(lab - lab.min()).max()
        assert top >= 870                              # >= 96% of the blob agrees
    # abstention relabels low-similarity points as noise
    res_ab = cluster_sample_and_assign(
        vecs, tile_ids, groups, _StubEmbed(dim), sample_size=900, min_cluster_size=30,
        core_dist_n_jobs=1, seed=1, min_per_group=100, abstain_below_sim=1.01)
    assert res_ab.noise_count == 2700

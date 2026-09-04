"""Tests for the VectorIndex seam (FaissFlatIPIndex)."""

from __future__ import annotations

import numpy as np
import pytest

from geoseek.config import EMBEDDING_DIM
from geoseek.vectorindex import FaissFlatIPIndex, VectorIndex


def _unit(rng, n):
    v = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_faissflat_is_a_vectorindex():
    assert issubclass(FaissFlatIPIndex, VectorIndex)


def test_add_returns_contiguous_positional_ids(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    rng = np.random.default_rng(0)
    assert idx.add(_unit(rng, 5)) == [0, 1, 2, 3, 4]
    assert idx.add(_unit(rng, 3)) == [5, 6, 7]
    assert idx.count() == 8


def test_add_rejects_wrong_dim(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    with pytest.raises(ValueError):
        idx.add(np.zeros((2, EMBEDDING_DIM - 1), dtype=np.float32))


def test_search_ranks_descending_and_self_match_is_one(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    rng = np.random.default_rng(1)
    vecs = _unit(rng, 20)
    idx.add(vecs)
    scores, ids = idx.search(vecs[7], k=20)
    assert list(scores) == sorted(scores, reverse=True)
    assert ids[0] == 7 and scores[0] == pytest.approx(1.0, abs=1e-4)


def test_get_vector_roundtrips(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    rng = np.random.default_rng(2)
    vecs = _unit(rng, 4)
    idx.add(vecs)
    assert np.allclose(idx.get_vector(2), vecs[2], atol=1e-6)


def test_delete_is_not_supported_on_the_flat_index(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    idx.add(_unit(np.random.default_rng(3), 2))
    with pytest.raises(NotImplementedError):
        idx.delete([0])


def test_persist_and_load_roundtrip_is_byte_identical(tmp_path):
    p = tmp_path / "v.faiss"
    idx = FaissFlatIPIndex(p)
    rng = np.random.default_rng(4)
    vecs = _unit(rng, 10)
    idx.add(vecs)
    idx.persist()
    assert p.is_file()

    reopened = FaissFlatIPIndex(p)  # loads from disk
    assert reopened.count() == 10
    for i in range(10):
        assert np.array_equal(reopened.get_vector(i), idx.get_vector(i))

    idx.add(_unit(rng, 2))
    idx.load()  # discard the un-persisted append
    assert idx.count() == 10


def test_validate_reports_count_and_dim(tmp_path):
    idx = FaissFlatIPIndex(tmp_path / "v.faiss")
    idx.add(_unit(np.random.default_rng(5), 6))
    good = idx.validate(expected_count=6, expected_dim=EMBEDDING_DIM)
    assert good.ok and good.count == 6 and good.dim == EMBEDDING_DIM
    bad = idx.validate(expected_count=99, expected_dim=EMBEDDING_DIM)
    assert not bad.ok and "99" in bad.detail

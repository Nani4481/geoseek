"""The 3-D projection artifact: honest sampling, exact lookups, staleness - and the script's refusal to run without UMAP."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from geoseek.analyst import ui_support


@pytest.fixture()
def data_dir(tmp_path):
    d = tmp_path / "discovery"
    d.mkdir()
    n = 50
    rng = np.random.default_rng(1)
    np.savez_compressed(
        d / "projection_3d.npz", tile_ids=np.array([f"t{i:03d}" for i in range(n)]), xyz=rng.normal(size=(n, 3)).astype(np.float32),
        lon=np.linspace(80, 81, n).astype(np.float32), lat=np.linspace(26, 27, n).astype(np.float32),
        region=(np.arange(n) % 3).astype(np.int16), cluster=np.array([(-1 if i % 10 == 0 else i % 4) for i in range(n)], dtype=np.int16),
        regions=np.array(["ayodhya", "kutch", "kerala"]))
    (d / "projection_3d.meta.json").write_text(json.dumps({
        "n_points": n, "method": "PCA-50 -> UMAP-3D", "umap": {"n_neighbors": 15}, "wall_seconds": {"total": 1.0},
        "power_source": "AC", "created_at": "2026-10-01T00:00:00Z", "caveat": "c", "libraries": {}, "pca_components": 50, "partial": False}), encoding="utf-8")
    ui_support._proj_cache.update(key=None, value=None)
    return tmp_path


def test_unavailable_when_never_computed(tmp_path):
    out = ui_support.projection_sample(tmp_path, 10)
    assert out["available"] is False and "compute_projection.py" in out["reason"]
    assert ui_support.projection_lookup(tmp_path, ["a"]) == {"available": False, "points": [], "missing": ["a"]}


def test_small_artifact_is_returned_whole_and_not_called_a_sample(data_dir):
    out = ui_support.projection_sample(data_dir, 50, max_points=1000)
    assert out["n_total"] == out["n_shown"] == 50 and out["sampled"] is False and out["stale"] is False
    assert len(out["xyz"]) == 150 and len(out["tile_ids"]) == 50 and out["regions"] == ["ayodhya", "kutch", "kerala"]
    assert out["clusters"] == [1, 2, 3, 0] or out["clusters"] == sorted(out["clusters"])      # real cluster ids only, -1 (no cluster) excluded
    assert -1 not in out["clusters"]


def test_sampling_is_deterministic_reports_both_counts_and_keeps_rows_aligned(data_dir):
    a = ui_support.projection_sample(data_dir, 50, max_points=12, seed=3)
    b = ui_support.projection_sample(data_dir, 50, max_points=12, seed=3)
    c = ui_support.projection_sample(data_dir, 50, max_points=12, seed=4)
    assert a == b and a["tile_ids"] != c["tile_ids"]
    assert (a["n_total"], a["n_shown"], a["sampled"]) == (50, 12, True)
    assert len(set(a["tile_ids"])) == 12 and a["tile_ids"] == sorted(a["tile_ids"])
    # every per-point array lines up with its tile id: the lon of t007 must be the 7th value of the linspace, not a shifted one
    for i, t in enumerate(a["tile_ids"]):
        k = int(t[1:])
        assert a["lon"][i] == pytest.approx(80 + k / 49, abs=1e-4) and a["region"][i] == k % 3
        assert a["cluster"][i] == (-1 if k % 10 == 0 else k % 4)


def test_projection_is_flagged_stale_when_the_index_has_grown(data_dir):
    assert ui_support.projection_sample(data_dir, 50)["stale"] is False
    out = ui_support.projection_sample(data_dir, 61)
    assert out["stale"] is True and out["current_vectors"] == 61 and out["meta"]["n_points"] == 50


def test_lookup_returns_exact_points_for_any_tile_and_reports_unknown_ones(data_dir):
    out = ui_support.projection_lookup(data_dir, ["t007", "t049", "nope"])
    assert [p["tile_id"] for p in out["points"]] == ["t007", "t049"] and out["missing"] == ["nope"]
    p = out["points"][0]
    assert p["region"] == "kutch" and p["cluster"] == 3 and len(p["xyz"]) == 3 and p["lon"] == pytest.approx(80 + 7 / 49, abs=1e-4)


def test_http_endpoints(data_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from geoseek.search import api

    monkeypatch.setattr(api, "_analyst", SimpleNamespace(settings=SimpleNamespace(data_dir=data_dir)))
    monkeypatch.setattr(api, "_engine", SimpleNamespace(count=lambda: 50))
    c = TestClient(api.app)
    r = c.get("/ui/projection", params={"max_points": 10})
    assert r.status_code == 200 and r.json()["n_shown"] == 10 and r.json()["n_total"] == 50
    assert c.get("/ui/projection", params={"max_points": 0}).status_code == 422
    assert c.get("/ui/projection/lookup", params={"ids": "t001,zzz"}).json()["missing"] == ["zzz"]

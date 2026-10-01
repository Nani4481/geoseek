"""A catalogued tile whose band rasters are not staged on this machine must be a 404, not a 500.

Found while auditing the React console: GET /tile/{id}/thumbnail returned 500 for the staged Maxar Van Nuys tiles because
``<datasets>/maxar/.../B04.tif`` is not on disk (rasterio raised RasterioIOError, which nothing mapped). Absence is now
translated to the KeyError the API already maps to 404; a file that exists but cannot be read stays a genuine 500.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geoseek.search import api as api_module
from geoseek.search.engine import SearchEngine

BANDS = ("B04", "B03", "B02")


def _engine(datasets_dir, *, dataset_dir="scene_a", tile_known=True, obs_known=True):
    tile = SimpleNamespace(tile_id="T1", observation_id="obs1", row=0, col=0)
    obs = SimpleNamespace(observation_id="obs1", scene_id="scene_a", dataset_dir=dataset_dir)
    repo = SimpleNamespace(
        get_tile=lambda tid: tile if (tile_known and tid == "T1") else None,
        get_observation=lambda oid: obs if obs_known else None,
        get_scene=lambda sid: SimpleNamespace(scene_id=sid),
    )
    eng = object.__new__(SearchEngine)          # bypass __init__: no model / index load needed here
    eng.repo = repo
    eng.settings = SimpleNamespace(datasets_dir=datasets_dir)
    return eng


def _write_band(path, value=1500, size=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint16",
                       crs="EPSG:32644", transform=from_origin(0, size * 10, 10, 10), nodata=0) as ds:
        ds.write(np.full((size, size), value, dtype="uint16"), 1)


def test_engine_raises_keyerror_when_band_rasters_are_not_staged(tmp_path):
    eng = _engine(tmp_path)                                  # datasets dir exists but holds no scene at all
    with pytest.raises(KeyError, match="not staged"):
        eng.get_tile_thumbnail_png("T1")


def test_engine_names_the_missing_bands_when_only_some_are_staged(tmp_path):
    _write_band(tmp_path / "scene_a" / "B03.tif")
    _write_band(tmp_path / "scene_a" / "B02.tif")
    with pytest.raises(KeyError) as e:
        _engine(tmp_path).get_tile_thumbnail_png("T1")
    assert "B04" in str(e.value) and "B03" not in str(e.value).split("missing")[1]


def test_unknown_tile_is_still_a_keyerror(tmp_path):
    with pytest.raises(KeyError, match="not found"):
        _engine(tmp_path, tile_known=False).get_tile_thumbnail_png("nope")


def test_staged_tile_still_renders_a_jpeg(tmp_path):
    """Positive control: the existence check must not break the happy path."""
    from PIL import Image

    for b in BANDS:
        _write_band(tmp_path / "scene_a" / f"{b}.tif")
    out = _engine(tmp_path).get_tile_thumbnail_png("T1")
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and img.size == (256, 256)


def test_present_but_unreadable_band_is_not_disguised_as_404(tmp_path):
    for b in BANDS:
        (tmp_path / "scene_a").mkdir(exist_ok=True)
        (tmp_path / "scene_a" / f"{b}.tif").write_bytes(b"this is not a tiff")
    with pytest.raises(rasterio.errors.RasterioIOError):     # a corrupt file is a real server fault
        _engine(tmp_path, dataset_dir="scene_a").get_tile_thumbnail_png("T1")


def test_http_returns_404_with_a_reason_for_unstaged_imagery_and_200_when_staged(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_module, "_engine", _engine(tmp_path))
    c = TestClient(api_module.app)                           # no `with`: lifespan (model load) deliberately not started

    r = c.get("/tile/T1/thumbnail")
    assert r.status_code == 404, r.text
    assert "not staged" in r.json()["detail"] and "B04" in r.json()["detail"]
    assert c.get("/tile/does-not-exist/thumbnail").status_code == 404

    for b in BANDS:
        _write_band(tmp_path / "scene_a" / f"{b}.tif")
    ok = c.get("/tile/T1/thumbnail")
    assert ok.status_code == 200 and ok.headers["content-type"].startswith("image/")

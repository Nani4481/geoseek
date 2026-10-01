from pathlib import Path

import pytest

from geoseek.config import PROJECT_ROOT, get_settings


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith("GEOSEEK_") or key in ("MODEL_PATH", "FAISS_INDEX_PATH", "DATABASE_URL"):
            monkeypatch.delenv(key)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_layout():
    s = get_settings()
    assert s.data_dir == PROJECT_ROOT / "data"
    assert s.database_path == s.index_dir / "tiles.sqlite"
    assert s.faiss_index_path == s.index_dir / "tiles.faiss"
    assert s.model_path == s.models_dir / "RemoteCLIP-ViT-B-32.pt"


def test_paths_and_device(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOSEEK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEOSEEK_DATASETS_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("GEOSEEK_DEVICE", "cpu")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + str(tmp_path / "catalog.sqlite"))
    monkeypatch.setenv("FAISS_INDEX_PATH", str(tmp_path / "search.faiss"))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "remoteclip.pt"))
    s = get_settings()
    assert s.datasets_dir == tmp_path / "archive"
    assert s.models_dir == tmp_path / "data/models"
    assert s.datasets_dir.is_dir()
    assert s.device == "cpu"
    assert s.database_path == tmp_path / "catalog.sqlite"
    assert s.faiss_index_path == tmp_path / "search.faiss"
    assert s.model_path == tmp_path / "remoteclip.pt"


def test_reject_non_sqlite(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unsupported")
    with pytest.raises(ValueError, match="sqlite"):
        _ = get_settings().database_path


def test_portable_paths_do_not_rewrite_urls(tmp_path):
    import importlib.util
    path = PROJECT_ROOT / "deploy/gcp/prepare_release.py"
    spec = importlib.util.spec_from_file_location("prepare_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {"path": str(tmp_path / "change_model/report.json"), "url": "https://example.org/data/a.tif"}
    result = module.portable(values, tmp_path)
    assert result["path"] == "/app/data/change_model/report.json"
    assert result["url"] == values["url"]

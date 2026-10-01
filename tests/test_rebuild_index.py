"""scripts/rebuild_index.py must never delete the production index unless --force is given."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_index.py"


@pytest.fixture
def rebuild(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("rebuild_index_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / mod.INDEX_FILENAME).write_bytes(b"PRODUCTION-FAISS")
    (index_dir / mod.DB_FILENAME).write_bytes(b"PRODUCTION-SQLITE")
    calls = []

    def fake_ingest(scene_dir, **kw):
        calls.append({"scene": Path(scene_dir).name, **kw})
        return {"index_total_vectors": 7}

    monkeypatch.setattr(mod, "get_settings", lambda: SimpleNamespace(index_dir=index_dir, datasets_dir=tmp_path / "ds"))
    monkeypatch.setattr(mod, "ingest_scene", fake_ingest)
    monkeypatch.setattr(mod, "load_model_once", lambda: None)
    return mod, index_dir, calls


def test_default_run_leaves_the_production_index_untouched_and_writes_elsewhere(rebuild):
    mod, index_dir, calls = rebuild
    assert mod.main([]) == 0
    assert (index_dir / mod.INDEX_FILENAME).read_bytes() == b"PRODUCTION-FAISS"
    assert (index_dir / mod.DB_FILENAME).read_bytes() == b"PRODUCTION-SQLITE"
    assert len(calls) == 2
    for c in calls:
        assert c["index_dir"] is not None and c["index_dir"] != index_dir and c["index_dir"].parent == index_dir
        assert c["record_manifest"] is False, "a scratch rebuild must not rewrite the provenance manifest"


def test_explicit_out_dir_is_used_and_must_be_new(rebuild, tmp_path):
    mod, index_dir, calls = rebuild
    out = tmp_path / "fresh"
    assert mod.main(["--out-dir", str(out)]) == 0 and {c["index_dir"] for c in calls} == {out}
    out.mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="already exists"):
        mod.main(["--out-dir", str(out)])
    assert (index_dir / mod.INDEX_FILENAME).read_bytes() == b"PRODUCTION-FAISS"


def test_force_is_the_only_way_to_delete_and_it_rebuilds_in_place_and_records_provenance(rebuild):
    mod, index_dir, calls = rebuild
    assert mod.main(["--force"]) == 0
    assert not (index_dir / mod.INDEX_FILENAME).exists() and not (index_dir / mod.DB_FILENAME).exists()   # the (fake) ingest wrote nothing
    assert all(c["index_dir"] is None and c["record_manifest"] is True for c in calls)


def test_force_cannot_be_combined_with_out_dir(rebuild, tmp_path):
    mod, index_dir, calls = rebuild
    with pytest.raises(SystemExit, match="cannot be combined"):
        mod.main(["--force", "--out-dir", str(tmp_path / "x")])
    assert (index_dir / mod.INDEX_FILENAME).exists() and not calls

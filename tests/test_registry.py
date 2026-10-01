"""Model registry: every current model is registered, weights are SHA256-verified on load, and a
mismatch fails loudly BEFORE any model is constructed."""

from __future__ import annotations

import hashlib
import os
import sys
import types

import pytest

from geoseek.models import registry as reg
from geoseek.models.registry import (
    REGISTRY,
    ModelSpec,
    RegistryError,
    UnknownModelError,
    WeightsIntegrityError,
    WeightsMissingError,
)

PAYLOAD = b"geoseek-test-weights\x00\x01\x02" * 1000
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Impl:
    constructed = 0

    def __init__(self, **kw):
        type(self).constructed += 1
        self.kw = kw


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """A throwaway registry with one embedding model whose weights live under tmp_path."""
    (tmp_path / "w.bin").write_bytes(PAYLOAD)
    mod = types.ModuleType("geoseek_test_impl")
    mod.Impl = _Impl
    monkeypatch.setitem(sys.modules, "geoseek_test_impl", mod)
    monkeypatch.setattr(reg, "REGISTRY", {})
    _Impl.constructed = 0

    def make(sha=PAYLOAD_SHA, **kw):
        spec = ModelSpec(key="t", kind=kw.pop("kind", "embedding"), implementation="geoseek_test_impl:Impl",
                         weights_relpath="w.bin", weights_sha256=sha, preprocessing={"bands": ["B04"]},
                         licence="MIT", source_url="x", embedding_dim=4, **kw)
        return reg.register(spec, replace=True)

    return tmp_path, make


def test_all_four_current_models_are_registered():
    assert {"remoteclip-vitb32", "openclip-vitb32-openai", "fc-siam-diff-oscd", "yolo26s-obb-dota15"} <= set(REGISTRY)
    for key, spec in REGISTRY.items():
        assert spec.licence and spec.source_url and len(spec.weights_sha256) == 64, key
        assert (spec.embedding_dim == 512) == (spec.kind == "embedding"), key


def test_registry_shas_match_the_documented_provenance():
    # pinned in docs/EVALUATION_REPORT.md section 5; a typo here would make every load fail loudly
    assert REGISTRY["remoteclip-vitb32"].weights_sha256 == "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4"
    assert REGISTRY["openclip-vitb32-openai"].weights_sha256 == "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
    assert REGISTRY["fc-siam-diff-oscd"].weights_sha256 == "452ac062e3b3d56f7997b9a5bbf81bab41d148b525fc28621e964cbbd0a3bcab"


def test_embedding_preprocessing_matches_the_real_radiometry_config():
    from geoseek.ingest.embed import RADIOMETRY_CONFIG

    p = REGISTRY["remoteclip-vitb32"].preprocessing
    assert p["clip_reflectance"] == RADIOMETRY_CONFIG["clip_reflectance"]
    assert p["reflectance_scale_dn_per_unit"] == RADIOMETRY_CONFIG["reflectance_scale_dn_per_unit"]
    assert p["gamma"] == RADIOMETRY_CONFIG["gamma"]
    assert p["bands"] == ["B04", "B03", "B02"]


def test_matching_hash_loads_and_constructs(fake_registry):
    root, make = fake_registry
    make()
    assert reg.verify_weights("t", root=root) == root / "w.bin"
    obj = reg.load_model("t", root=root, flag=1)
    assert isinstance(obj, _Impl) and obj.kw == {"flag": 1}


def test_hash_mismatch_fails_loudly_and_never_constructs_the_model(fake_registry):
    root, make = fake_registry
    make(sha="0" * 64)
    with pytest.raises(WeightsIntegrityError) as ei:
        reg.load_model("t", root=root)
    msg = str(ei.value)
    assert "SHA256 mismatch" in msg and "0" * 64 in msg and PAYLOAD_SHA in msg
    assert _Impl.constructed == 0, "implementation must not be instantiated when the hash is wrong"


def test_one_flipped_byte_is_detected(fake_registry):
    root, make = fake_registry
    make()
    reg.verify_weights("t", root=root)                          # ok, and now memoised
    corrupted = bytearray(PAYLOAD)
    corrupted[17] ^= 0x01
    (root / "w.bin").write_bytes(bytes(corrupted))              # same size; new mtime -> the memo must not hide it
    st = (root / "w.bin").stat()
    os.utime(root / "w.bin", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(WeightsIntegrityError):
        reg.verify_weights("t", root=root)


def test_missing_weights_and_unknown_key(fake_registry):
    root, make = fake_registry
    make()
    (root / "w.bin").unlink()
    with pytest.raises(WeightsMissingError):
        reg.verify_weights("t", root=root)
    with pytest.raises(UnknownModelError, match="registered"):
        reg.get_spec("nope")


def test_glob_weights_path_requires_exactly_one_match(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "REGISTRY", {})
    spec = ModelSpec(key="g", kind="embedding", implementation="a:B", weights_relpath="snap/*/w.bin",
                     weights_sha256=PAYLOAD_SHA, preprocessing={}, licence="x", source_url="x", embedding_dim=2)
    reg.register(spec)
    with pytest.raises(WeightsMissingError):
        reg.resolve_weights_path("g", root=tmp_path)
    for d in ("a", "b"):
        (tmp_path / "snap" / d).mkdir(parents=True)
        (tmp_path / "snap" / d / "w.bin").write_bytes(PAYLOAD)
    with pytest.raises(WeightsMissingError, match="found 2"):
        reg.resolve_weights_path("g", root=tmp_path)


def test_spec_validation_rejects_bad_entries():
    with pytest.raises(ValueError):
        ModelSpec(key="x", kind="embedding", implementation="a:B", weights_relpath="w", weights_sha256="abc",
                  preprocessing={}, licence="x", source_url="x", embedding_dim=2)
    with pytest.raises(ValueError):
        ModelSpec(key="x", kind="embedding", implementation="a:B", weights_relpath="w", weights_sha256="0" * 64,
                  preprocessing={}, licence="x", source_url="x")                       # no embedding_dim
    with pytest.raises(RegistryError):
        reg.register(REGISTRY["remoteclip-vitb32"])                                    # duplicate key


@pytest.mark.skipif(not reg.resolve_weights_path("remoteclip-vitb32").is_file(), reason="RemoteCLIP not staged")
def test_staged_remoteclip_weights_verify():
    assert reg.verify_weights("remoteclip-vitb32").is_file()

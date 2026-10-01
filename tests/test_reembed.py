"""Re-embedding pipeline: shard checkpointing, resume correctness, finalize guards.

The model and the tile reader are deterministic fakes (a vector derived only from the tile's id), so
these tests need no GPU, no imagery and no weights - they pin the pipeline's own guarantees.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoseek.catalog.embedding_map import read_mapping
from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.ingest import reembed as R
from geoseek.models.registry import ModelSpec, sha256_file
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

DIM = 8
N_TILES = 23            # 5 shards of 5 with a short last shard (3) - exercises the partial-shard path
SHARD, BATCH = 5, 2     # batch does not divide the shard size either


class FakeReader:
    def __init__(self, fail_on_faiss_id: int | None = None):
        self.fail_on, self.closed = fail_on_faiss_id, False

    def read_rgb(self, ref):
        if self.fail_on is not None and ref.faiss_id == self.fail_on:
            raise RuntimeError("simulated crash / OOM")
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[..., 0], img[..., 1] = ref.faiss_id % 256, ref.faiss_id // 256
        return img

    def close(self):
        self.closed = True


class FakeModel:
    """Per-tile deterministic: the vector depends only on the pixels, never on batch composition."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def encode_images(self, images, batch_size=64):
        out = []
        for im in images:
            v = np.random.RandomState(int(im[0, 0, 0]) + 256 * int(im[0, 0, 1])).randn(DIM)
            out.append((v / np.linalg.norm(v) * self.scale).astype(np.float32))
        return np.stack(out)


def make_refs(n=N_TILES):
    return [R.TileRef(faiss_id=i + 100, tile_id=f"scene_r{i // 5:03d}_c{i % 5:03d}", observation_id=f"obs{i // 10}",
                      collection_id="sentinel-2-l2a", dataset_dir=None, row=i // 5, col=i % 5) for i in range(n)]


SPEC = ModelSpec(key="fake", kind="embedding", implementation="x:Y", weights_relpath="w", weights_sha256="a" * 64,
                 preprocessing={"bands": ["B04", "B03", "B02"]}, licence="MIT", source_url="x", embedding_dim=DIM)


def header(refs, *, batch=BATCH, shard=SHARD):
    return R.build_header(SPEC, "a" * 64, refs, shard_size=shard, batch_size=batch, selection={"collection": None})


def run(refs, d, *, resume=False, max_shards=None, reader=None, model=None, **hk):
    return R.run_embedding(model or FakeModel(), reader or FakeReader(), refs, d, header(refs, **hk),
                           resume=resume, max_shards=max_shards, log=lambda _m: None, track_vram=False)


def shard_bytes(d):
    return {p.name: p.read_bytes() for p in sorted(d.glob("shard_*.npy"))}


# ---------------------------------------------------------------- catalog selection


def test_select_tiles_reads_the_catalog_ordered_by_faiss_id(tmp_path):
    repo = SQLiteMetadataRepository(tmp_path / "c.sqlite")
    poly = "POLYGON((0 0,1 0,1 1,0 1,0 0))"
    repo.register_collection(Collection("sentinel-2-l2a", "MSI", "Sentinel-2", ("B04",), 10.0))
    repo.register_collection(Collection("sentinel-1-grd", "SAR", "Sentinel-1", ("VV",), 10.0))
    for scene, coll in (("s2", "sentinel-2-l2a"), ("s1", "sentinel-1-grd")):
        repo.register_scene(Scene(scene, coll, "P", "2024-01-01", poly))
        repo.register_observation(Observation(f"o_{scene}", scene, "2024-01-01", poly, dataset_dir=f"dir_{scene}"))
    # inserted OUT of faiss order, plus an un-embedded SAR tile that must never be selected
    repo.add_tiles([Tile("t_c", "o_s2", 0, 2, poly, 0.0, faiss_id=7), Tile("t_a", "o_s2", 0, 0, poly, 0.0, faiss_id=3),
                    Tile("t_b", "o_s2", 0, 1, poly, 0.0, faiss_id=5), Tile("t_sar", "o_s1", 0, 0, poly, 0.0, faiss_id=None)])
    refs = R.select_tiles(repo)
    assert [r.tile_id for r in refs] == ["t_a", "t_b", "t_c"]
    assert [r.faiss_id for r in refs] == [3, 5, 7]
    assert {r.collection_id for r in refs} == {"sentinel-2-l2a"} and refs[0].dataset_dir == "dir_s2"
    assert [r.faiss_id for r in R.select_tiles(repo, max_faiss_id=6)] == [3, 5]
    assert [r.faiss_id for r in R.select_tiles(repo, limit=1)] == [3]
    assert R.select_tiles(repo, collection="sentinel-1-grd") == []          # SAR has no vectors
    repo.close()


# ---------------------------------------------------------------- resume correctness


def test_interrupt_and_resume_is_byte_identical_to_an_uninterrupted_run(tmp_path):
    refs = make_refs()
    full, part = tmp_path / "full", tmp_path / "part"

    s_full = run(refs, full)
    assert s_full["complete"] and s_full["new_shards"] == 5

    s1 = run(refs, part, max_shards=2)                      # "crash" after 2 shards
    assert not s1["complete"] and s1["new_shards"] == 2
    s2 = run(refs, part, resume=True)                       # resume
    assert s2["complete"] and s2["new_shards"] == 3, "resume must only compute the shards that are missing"

    assert shard_bytes(full) == shard_bytes(part), "resumed shards must be byte-identical to an uninterrupted run"
    m_full, m_part = (json.loads((d / "manifest.json").read_text()) for d in (full, part))
    assert [(s["index"], s["sha256"], s["count"]) for s in m_full["shards"]] == \
           [(s["index"], s["sha256"], s["count"]) for s in m_part["shards"]]

    # ... and so is the final index
    out_full, out_part = tmp_path / "idx_full.faiss", tmp_path / "idx_part.faiss"
    R.finalize(refs, full, header(refs), out_full, log=lambda _m: None)
    R.finalize(refs, part, header(refs), out_part, log=lambda _m: None)
    a, b = (FaissFlatIPIndex(p, dim=DIM).reconstruct_all() for p in (out_full, out_part))
    assert a.tobytes() == b.tobytes() and a.shape == (N_TILES, DIM)
    assert sha256_file(out_full) == sha256_file(out_part)


def test_resume_does_not_recompute_completed_shards(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    run(refs, d, max_shards=3)
    seen = []

    class Spy(FakeModel):
        def encode_images(self, images, batch_size=64):
            seen.append(len(images))
            return super().encode_images(images, batch_size)

    run(refs, d, resume=True, model=Spy())
    assert sum(seen) == N_TILES - 3 * SHARD            # only the 2 missing shards (5 + 3 tiles) were embedded


def test_a_crash_mid_shard_leaves_no_half_written_shard_and_resumes(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    with pytest.raises(RuntimeError, match="simulated"):
        run(refs, d, reader=FakeReader(fail_on_faiss_id=100 + 12))      # dies inside shard 2
    manifest = json.loads((d / "manifest.json").read_text())
    assert [s["index"] for s in manifest["shards"]] == [0, 1]
    assert not (d / "shard_00002.npy").exists() and not list(d.glob("*.tmp"))
    assert run(refs, d, resume=True)["complete"]
    ref_dir = tmp_path / "ref"
    run(refs, ref_dir)
    assert shard_bytes(d) == shard_bytes(ref_dir)


def test_resume_reverifies_shard_hashes_and_redoes_a_corrupted_one(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    run(refs, d)
    good = (d / "shard_00001.npy").read_bytes()
    (d / "shard_00001.npy").write_bytes(good[:-4] + b"\x00\x00\x00\x00")       # silent corruption, same size
    (d / "shard_00003.npy").unlink()                                          # a vanished shard
    s = run(refs, d, resume=True)
    assert s["new_shards"] == 2 and s["complete"]
    assert (d / "shard_00001.npy").read_bytes() == good


def test_an_existing_run_needs_explicit_resume_and_a_matching_configuration(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    run(refs, d, max_shards=1)
    with pytest.raises(R.ReembedError, match="--resume"):
        run(refs, d)                                                           # would silently mix / overwrite
    with pytest.raises(R.ReembedError, match="batch_size"):
        run(refs, d, resume=True, batch=BATCH + 1)                             # different batching => different bytes
    with pytest.raises(R.ReembedError, match="n_tiles|tile_list_sha256"):
        run(make_refs(N_TILES - 1), d, resume=True)
    with pytest.raises(R.ReembedError, match="shard_size"):
        run(refs, d, resume=True, shard=SHARD + 1)


def test_embeddings_that_are_not_unit_norm_are_rejected_before_anything_is_written(tmp_path):
    d = tmp_path / "s"
    with pytest.raises(R.ReembedError, match="unit-norm"):
        run(make_refs(), d, model=FakeModel(scale=2.0))
    assert not list(d.glob("shard_*.npy"))


# ---------------------------------------------------------------- finalize


def test_finalize_writes_a_new_index_and_a_correct_mapping_and_never_overwrites(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    run(refs, d)
    out = tmp_path / "cand" / "m.faiss"
    info = R.finalize(refs, d, header(refs), out, log=lambda _m: None)
    assert out.is_file() and info["n_vectors"] == N_TILES

    rows, meta = read_mapping(out.with_name("m.mapping.sqlite"))
    assert [r[0] for r in rows] == list(range(N_TILES))
    assert [(r[1], r[2], r[3]) for r in rows] == [(x.tile_id, x.faiss_id, x.observation_id) for x in refs]
    assert meta["model_key"] == "fake" and int(meta["n_tiles"]) == N_TILES
    idx = FaissFlatIPIndex(out, dim=DIM)
    expect = FakeModel().encode_images([FakeReader().read_rgb(r) for r in refs])
    assert idx.reconstruct_all().tobytes() == expect.tobytes()
    assert json.loads((d / "manifest.json").read_text())["finalized"]["index_sha256"] == sha256_file(out)

    with pytest.raises(R.ReembedError, match="already exists"):
        R.finalize(refs, d, header(refs), out, log=lambda _m: None)           # second run must not clobber
    assert not list(out.parent.glob("*.partial"))


def test_finalize_refuses_the_production_index_and_incomplete_runs(tmp_path):
    from geoseek.config import get_settings

    refs, d = make_refs(), tmp_path / "s"
    run(refs, d, max_shards=2)
    with pytest.raises(R.ReembedError, match="production"):
        R.finalize(refs, d, header(refs), get_settings().faiss_index_path, log=lambda _m: None)
    with pytest.raises(R.ReembedError, match="missing or corrupt"):
        R.finalize(refs, d, header(refs), tmp_path / "x.faiss", log=lambda _m: None)
    assert not (tmp_path / "x.faiss").exists() and not list(tmp_path.glob("x.faiss*"))
    with pytest.raises(R.ReembedError, match="no manifest"):
        R.finalize(refs, tmp_path / "empty", header(refs), tmp_path / "y.faiss", log=lambda _m: None)


def test_manifest_records_model_weights_preprocessing_bands_and_per_shard_sha(tmp_path):
    refs, d = make_refs(), tmp_path / "s"
    run(refs, d)
    m = json.loads((d / "manifest.json").read_text())
    assert m["model_key"] == "fake" and m["weights_sha256"] == "a" * 64
    assert m["preprocessing"] == {"bands": ["B04", "B03", "B02"]}
    assert m["bands_by_collection"] == {"sentinel-2-l2a": ["B04", "B03", "B02"]}
    assert m["n_tiles"] == N_TILES and m["shard_size"] == SHARD and m["batch_size"] == BATCH
    assert [s["count"] for s in m["shards"]] == [5, 5, 5, 5, 3]
    for s in m["shards"]:
        assert s["sha256"] == sha256_file(d / s["file"]) and s["tiles_per_s"] > 0


def test_compare_embeddings_reports_identity_and_noise():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 16)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    same = R.compare_embeddings(a, a.copy(), n_queries=50)
    assert same["fraction_byte_identical"] == 1.0 and same["max_abs_diff"] == 0.0 and same["top1_agreement"] == 1.0
    b = a + 1e-6 * rng.standard_normal(a.shape).astype(np.float32)
    near = R.compare_embeddings(a, b, n_queries=50)
    assert near["fraction_byte_identical"] == 0.0 and near["min_cosine"] > 0.99999
    assert near["mean_top10_neighbour_overlap"] > 0.95
    other = rng.standard_normal(a.shape).astype(np.float32)
    other /= np.linalg.norm(other, axis=1, keepdims=True)
    assert R.compare_embeddings(a, other, n_queries=50)["mean_top10_neighbour_overlap"] < 0.2

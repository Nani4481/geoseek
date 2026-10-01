"""Resumable, model-agnostic re-embedding of every catalogued tile.

Used by ``scripts/reembed.py``. The production index is NEVER mutated: vectors are written to
checkpointed shards under ``<shards_root>/<model-key>/`` and only :func:`finalize` assembles them
into a NEW FAISS file (+ a faiss_id mapping database) at a path that must not already exist.

Design contract
---------------
* The tile list comes from the SQLite catalog (through the ``MetadataRepository`` seam), ordered by
  the production ``faiss_id``; position ``i`` of that list is vector ``i`` of the candidate index.
* Shards are ``shard_size`` consecutive tiles. A shard is complete only once its file is renamed into
  place AND its SHA256 is recorded in ``manifest.json`` (itself replaced atomically), so a crash at
  any instant leaves either a fully valid shard or nothing for that shard.
* Resume re-verifies every recorded shard's SHA256, drops any that fail, and continues from the first
  missing one. Because resume happens only at shard boundaries, every batch is composed exactly as in
  an uninterrupted run - same ``batch_size``, same tiles per batch - so the vectors are byte-identical
  (the manifest pins ``batch_size`` and refuses to resume under a different one).
* Tile pixels are streamed in ``batch_size`` chunks with one chunk of read-ahead; a whole shard of
  pixels is never held in memory (1024 px Maxar tiles would be ~3 MB each).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from geoseek.catalog.embedding_map import write_mapping
from geoseek.catalog.repository import MetadataRepository
from geoseek.config import get_settings
from geoseek.models.registry import ModelSpec, get_spec, sha256_file
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

SCHEMA_VERSION = 1
DEFAULT_SHARD_SIZE = 5000
DEFAULT_BATCH_SIZE = 64
VRAM_FRACTION = 0.80          # the hard torch cap used by detect/train.py and scripts/detect_bench.py
MANIFEST_NAME = "manifest.json"

COLLECTION_S2 = "sentinel-2-l2a"
COLLECTION_MAXAR = "maxar-opendata"


class ReembedError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# tile selection (catalog -> ordered list)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TileRef:
    faiss_id: int
    tile_id: str
    observation_id: str
    collection_id: str
    dataset_dir: str | None
    row: int
    col: int


def select_tiles(repo: MetadataRepository, *, collection: str | None = None,
                 max_faiss_id: int | None = None, limit: int | None = None) -> list[TileRef]:
    """Every embedded tile (``faiss_id`` not NULL), ordered by production ``faiss_id``.

    ``collection`` keeps one collection; ``max_faiss_id`` keeps ``faiss_id < max_faiss_id`` (the
    frozen Phase 7b corpus is ``max_faiss_id=101911``); ``limit`` keeps the first N after ordering.
    """
    collection_of_scene = {s.scene_id: s.collection_id for s in repo.list_scenes()}
    refs: list[TileRef] = []
    for obs in repo.list_observations(collection=collection):
        coll = collection_of_scene.get(obs.scene_id, "")
        for t in repo.list_tiles(observation_id=obs.observation_id):
            if t.faiss_id is None:
                continue
            if max_faiss_id is not None and t.faiss_id >= max_faiss_id:
                continue
            refs.append(TileRef(int(t.faiss_id), t.tile_id, obs.observation_id, coll, obs.dataset_dir, t.row, t.col))
    refs.sort(key=lambda r: r.faiss_id)
    if limit is not None:
        refs = refs[:limit]
    return refs


def tile_list_sha256(refs: Sequence[TileRef]) -> str:
    h = hashlib.sha256()
    for r in refs:
        h.update(r.tile_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# tile pixel readers
# --------------------------------------------------------------------------


class TileReader(Protocol):
    def read_rgb(self, ref: TileRef) -> np.ndarray: ...
    def close(self) -> None: ...


def resolve_scene_dir(ref: TileRef) -> Path:
    """Directory holding the per-band GeoTIFFs of ``ref``'s observation.

    Sentinel-2 ``dataset_dir`` is a name under ``datasets_dir``; Maxar stores an absolute path from the
    machine that ingested it, so fall back to a search under ``datasets_dir/maxar`` if it moved.
    """
    datasets = get_settings().datasets_dir
    if ref.dataset_dir:
        p = Path(ref.dataset_dir)
        cand = p if p.is_absolute() else datasets / p
        if cand.is_dir():
            return cand
    cand = datasets / ref.observation_id
    if cand.is_dir():
        return cand
    for found in (datasets / "maxar").glob(f"*/{ref.observation_id}"):
        if found.is_dir():
            return found
    raise ReembedError(f"cannot locate imagery directory for observation {ref.observation_id!r} "
                       f"(dataset_dir={ref.dataset_dir!r})")


class _WindowReader:
    """Holds the band GeoTIFFs of ONE observation open; reopens when the observation changes.

    Not thread-safe by design: the pipeline keeps exactly one read in flight.
    """

    bands: tuple[str, ...] = ()
    tile_size: int = 0

    def __init__(self) -> None:
        self._obs: str | None = None
        self._ds: dict = {}

    def _open(self, ref: TileRef) -> dict:
        import rasterio

        if ref.observation_id != self._obs:
            self.close()
            scene_dir = resolve_scene_dir(ref)
            self._ds = {b: rasterio.open(scene_dir / f"{b}.tif") for b in self.bands}
            self._obs = ref.observation_id
            self._scene_dir_name = scene_dir.name
        return self._ds

    def _read_bands(self, ref: TileRef) -> tuple[dict[str, np.ndarray], float | None]:
        from rasterio.windows import Window

        ds = self._open(ref)
        ref_ds = ds[self.bands[0]]
        col0, row0 = ref.col * self.tile_size, ref.row * self.tile_size
        if col0 >= ref_ds.width or row0 >= ref_ds.height:
            raise ReembedError(f"tile {ref.tile_id} (row={ref.row}, col={ref.col}) is outside its scene")
        w = min(self.tile_size, ref_ds.width - col0)
        h = min(self.tile_size, ref_ds.height - row0)
        win = Window(col0, row0, w, h)
        return {b: d.read(1, window=win) for b, d in ds.items()}, ref_ds.nodata

    def close(self) -> None:
        for d in self._ds.values():
            d.close()
        self._ds, self._obs = {}, None


class Sentinel2Reader(_WindowReader):
    """Sentinel-2 L2A -> the fixed true-colour 8-bit RGB the production embedder sees (same code path)."""

    bands = ("B04", "B03", "B02")
    tile_size = 256

    def read_rgb(self, ref: TileRef) -> np.ndarray:
        from geoseek.ingest.embed import boa_offset_dn_for_scene, make_true_color_uint8

        arrays, nodata = self._read_bands(ref)
        return make_true_color_uint8(arrays, nodata=nodata,
                                     boa_offset_dn=boa_offset_dn_for_scene(self._scene_dir_name))


class MaxarReader(_WindowReader):
    """Maxar Open Data visual product: the delivered 8-bit R/G/B bands are the model input as-is."""

    bands = ("R", "G", "B")
    tile_size = 1024

    def read_rgb(self, ref: TileRef) -> np.ndarray:
        arrays, _ = self._read_bands(ref)
        return np.stack([arrays["R"], arrays["G"], arrays["B"]], axis=-1)


class CollectionDispatchReader:
    """Routes each tile to the reader for its collection (a run may span Sentinel-2 and Maxar)."""

    _FACTORIES: dict[str, Callable[[], TileReader]] = {COLLECTION_S2: Sentinel2Reader, COLLECTION_MAXAR: MaxarReader}

    def __init__(self) -> None:
        self._readers: dict[str, TileReader] = {}

    def read_rgb(self, ref: TileRef) -> np.ndarray:
        r = self._readers.get(ref.collection_id)
        if r is None:
            factory = self._FACTORIES.get(ref.collection_id)
            if factory is None:
                raise ReembedError(f"no tile reader for collection {ref.collection_id!r} (tile {ref.tile_id})")
            r = self._readers[ref.collection_id] = factory()
        return r.read_rgb(ref)

    def close(self) -> None:
        for r in self._readers.values():
            r.close()


def bands_by_collection(refs: Sequence[TileRef]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for coll in sorted({r.collection_id for r in refs}):
        factory = CollectionDispatchReader._FACTORIES.get(coll)
        out[coll] = list(getattr(factory, "bands", ())) if factory else []
    return out


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

_HEADER_KEYS = ("schema_version", "model_key", "weights_sha256", "preprocessing", "bands_by_collection",
                "shard_size", "batch_size", "n_tiles", "tile_list_sha256", "embedding_dim")


def build_header(spec: ModelSpec, weights_sha256: str, refs: Sequence[TileRef], *, shard_size: int,
                 batch_size: int, selection: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": spec.key,
        "weights_sha256": weights_sha256,
        "weights_relpath": spec.weights_relpath,
        "preprocessing": dict(spec.preprocessing),
        "bands_by_collection": bands_by_collection(refs),
        "embedding_dim": spec.embedding_dim,
        "licence": spec.licence,
        "source_url": spec.source_url,
        "shard_size": shard_size,
        "batch_size": batch_size,
        "n_tiles": len(refs),
        "tile_list_sha256": tile_list_sha256(refs),
        "selection": selection,
        "source_catalog": str(get_settings().database_path),
    }


def manifest_path(shards_dir: Path) -> Path:
    return Path(shards_dir) / MANIFEST_NAME


def shard_filename(index: int) -> str:
    return f"shard_{index:05d}.npy"


def n_shards_for(n_tiles: int, shard_size: int) -> int:
    return -(-n_tiles // shard_size)


def load_manifest(shards_dir: Path) -> dict | None:
    p = manifest_path(shards_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_manifest(shards_dir: Path, manifest: dict) -> None:
    _atomic_write_text(manifest_path(shards_dir), json.dumps(manifest, indent=1, sort_keys=False))


def check_header_matches(manifest: dict, header: dict) -> None:
    diffs = [k for k in _HEADER_KEYS if manifest.get(k) != header.get(k)]
    if diffs:
        detail = "; ".join(f"{k}: manifest={manifest.get(k)!r:.80} vs now={header.get(k)!r:.80}" for k in diffs)
        raise ReembedError(
            "existing shards were produced under a different configuration - refusing to mix them "
            f"({detail}). Use a different --shards-root, or restore the original settings.")


def valid_shards(shards_dir: Path, manifest: dict, log: Callable[[str], None] = print) -> dict[int, dict]:
    """Manifest shard records whose file exists with the recorded SHA256. Bad ones are reported and dropped."""
    good: dict[int, dict] = {}
    for rec in manifest.get("shards", []):
        f = Path(shards_dir) / rec["file"]
        if not f.is_file():
            log(f"[reembed] shard {rec['index']} listed in manifest but file is missing - will redo")
        elif sha256_file(f) != rec["sha256"]:
            log(f"[reembed] shard {rec['index']} SHA256 mismatch - will redo")
        else:
            good[rec["index"]] = rec
    return good


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


def apply_vram_cap(fraction: float = VRAM_FRACTION) -> bool:
    """Hard torch VRAM cap (Windows otherwise spills CUDA into system RAM and throughput collapses)."""
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return True


def _vram_reset() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_peaks_mb() -> dict | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return {"peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)}


def embed_refs(model, reader: TileReader, refs: Sequence[TileRef], batch_size: int, dim: int) -> np.ndarray:
    """Embed ``refs`` in order -> (n, dim) float32. One chunk of read-ahead; one forward per chunk."""
    out = np.empty((len(refs), dim), dtype=np.float32)
    chunks = [(s, refs[s:s + batch_size]) for s in range(0, len(refs), batch_size)]
    if not chunks:
        return out

    def read_chunk(chunk: Sequence[TileRef]) -> list[np.ndarray]:
        return [reader.read_rgb(r) for r in chunk]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(read_chunk, chunks[0][1])
        for k, (start, chunk) in enumerate(chunks):
            images = fut.result()
            if k + 1 < len(chunks):
                fut = ex.submit(read_chunk, chunks[k + 1][1])
            vecs = np.asarray(model.encode_images(images, batch_size=batch_size), dtype=np.float32)
            if vecs.shape != (len(chunk), dim):
                raise ReembedError(f"model returned {vecs.shape}, expected {(len(chunk), dim)}")
            out[start:start + len(chunk)] = vecs
    return out


def _check_vectors(vecs: np.ndarray, shard_index: int) -> None:
    if not np.isfinite(vecs).all():
        raise ReembedError(f"shard {shard_index}: non-finite values in embeddings")
    norms = np.linalg.norm(vecs, axis=1)
    if np.abs(norms - 1.0).max() > 1e-3:
        raise ReembedError(f"shard {shard_index}: embeddings are not unit-norm "
                           f"(norm range {norms.min():.4f}..{norms.max():.4f})")


def write_shard_atomic(path: Path, vecs: np.ndarray) -> tuple[str, int]:
    """Write ``vecs`` as .npy via tmp + rename; returns (sha256, bytes)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(vecs, dtype=np.float32))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256_file(path), path.stat().st_size


def run_embedding(model, reader: TileReader, refs: Sequence[TileRef], shards_dir: Path, header: dict, *,
                  resume: bool = False, max_shards: int | None = None,
                  log: Callable[[str], None] = print, track_vram: bool = True) -> dict:
    """Embed every shard not yet completed. Returns a summary dict.

    ``max_shards`` stops after that many NEW shards (deliberate interruption, used by tests and drills).
    """
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_size, batch_size, dim = header["shard_size"], header["batch_size"], header["embedding_dim"]
    n_total = len(refs)
    n_shards = n_shards_for(n_total, shard_size)

    manifest = load_manifest(shards_dir)
    if manifest is None:
        manifest = {**header, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "shards": [], "finalized": None}
    else:
        if not resume:
            raise ReembedError(f"{shards_dir} already holds a run ({len(manifest.get('shards', []))} shard(s) recorded); "
                               "pass --resume to continue it, or choose a different --shards-root")
        check_header_matches(manifest, header)
    done = valid_shards(shards_dir, manifest, log)
    manifest["shards"] = [done[i] for i in sorted(done)]
    save_manifest(shards_dir, manifest)

    todo = [i for i in range(n_shards) if i not in done]
    log(f"[reembed] model={header['model_key']} tiles={n_total} shards={n_shards} (size {shard_size}, "
        f"batch {batch_size}); {len(done)} already complete, {len(todo)} to do")

    session_tiles, session_seconds, new_shards = 0, 0.0, 0
    for i in todo:
        if max_shards is not None and new_shards >= max_shards:
            break
        start, stop = i * shard_size, min((i + 1) * shard_size, n_total)
        if track_vram:
            _vram_reset()
        t0 = time.perf_counter()
        vecs = embed_refs(model, reader, refs[start:stop], batch_size, dim)
        _check_vectors(vecs, i)
        sha, nbytes = write_shard_atomic(shards_dir / shard_filename(i), vecs)
        dt = time.perf_counter() - t0
        rec = {"index": i, "file": shard_filename(i), "start": start, "count": stop - start, "sha256": sha,
               "bytes": nbytes, "seconds": round(dt, 3), "tiles_per_s": round((stop - start) / dt, 2),
               "vram": _vram_peaks_mb() if track_vram else None,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        manifest["shards"] = sorted([*manifest["shards"], rec], key=lambda r: r["index"])
        save_manifest(shards_dir, manifest)
        new_shards += 1
        session_tiles += stop - start
        session_seconds += dt
        remaining = sum(min((j + 1) * shard_size, n_total) - j * shard_size for j in todo if j > i)
        eta = remaining / (session_tiles / session_seconds) if session_seconds else float("nan")
        vr = rec["vram"]
        log(f"[reembed] shard {i + 1}/{n_shards}  {rec['count']} tiles  {rec['tiles_per_s']:.1f} tiles/s  "
            + (f"peak VRAM {vr['peak_allocated_mb']:.0f} MB alloc / {vr['peak_reserved_mb']:.0f} MB reserved  " if vr else "")
            + f"ETA {eta / 60:.1f} min")

    complete = len(valid_shards(shards_dir, load_manifest(shards_dir), lambda _m: None)) == n_shards
    return {"complete": complete, "shards_total": n_shards, "new_shards": new_shards,
            "session_tiles": session_tiles, "session_seconds": round(session_seconds, 3),
            "session_tiles_per_s": round(session_tiles / session_seconds, 2) if session_seconds else None}


# --------------------------------------------------------------------------
# finalize + verification
# --------------------------------------------------------------------------


def load_all_shards(shards_dir: Path, manifest: dict, n_tiles: int, dim: int) -> np.ndarray:
    good = valid_shards(shards_dir, manifest, lambda _m: None)
    n_shards = n_shards_for(n_tiles, manifest["shard_size"])
    missing = [i for i in range(n_shards) if i not in good]
    if missing:
        raise ReembedError(f"cannot finalize: {len(missing)} shard(s) missing or corrupt (first: {missing[:5]})")
    arr = np.concatenate([np.load(Path(shards_dir) / good[i]["file"]) for i in range(n_shards)], axis=0)
    if arr.shape != (n_tiles, dim):
        raise ReembedError(f"assembled shape {arr.shape} != expected {(n_tiles, dim)}")
    return arr


def finalize(refs: Sequence[TileRef], shards_dir: Path, header: dict, index_out: Path, *,
             log: Callable[[str], None] = print) -> dict:
    """Assemble the shards into a NEW FAISS file + mapping database. Never overwrites anything."""
    shards_dir, index_out = Path(shards_dir), Path(index_out)
    production = Path(get_settings().faiss_index_path)
    if index_out.resolve() == production.resolve():
        raise ReembedError(f"refusing to write to the production index {production}")
    mapping_path = index_out.with_name(index_out.stem + ".mapping.sqlite")
    for p in (index_out, mapping_path):
        if p.exists():
            raise ReembedError(f"{p} already exists; a candidate index is always written to a new path")
    manifest = load_manifest(shards_dir)
    if manifest is None:
        raise ReembedError(f"no manifest in {shards_dir}; nothing to finalize")
    check_header_matches(manifest, header)

    vecs = load_all_shards(shards_dir, manifest, len(refs), header["embedding_dim"])
    partial = index_out.with_name(index_out.name + ".partial")
    if partial.exists():
        partial.unlink()
    idx = FaissFlatIPIndex(partial, dim=header["embedding_dim"])
    ids = idx.add(vecs)
    if ids != list(range(len(refs))):
        raise ReembedError("index assigned non-positional ids; mapping would be wrong")
    check = idx.validate(len(refs), header["embedding_dim"])
    if not check.ok:
        raise ReembedError(f"candidate index failed validation: {check.detail}")
    idx.persist()
    # reload from disk and confirm it round-trips the shard bytes exactly
    back = FaissFlatIPIndex(partial, dim=header["embedding_dim"]).reconstruct_all()
    if back.tobytes() != np.ascontiguousarray(vecs).tobytes():
        raise ReembedError("persisted index does not round-trip the shard vectors byte-for-byte")
    os.replace(partial, index_out)

    write_mapping(mapping_path, ((i, r.tile_id, r.faiss_id, r.observation_id) for i, r in enumerate(refs)),
                  {"model_key": header["model_key"], "weights_sha256": header["weights_sha256"],
                   "tile_list_sha256": header["tile_list_sha256"], "n_tiles": len(refs),
                   "source_catalog": header["source_catalog"], "shards_dir": str(shards_dir)})
    info = {"index_path": str(index_out), "index_sha256": sha256_file(index_out), "index_bytes": index_out.stat().st_size,
            "mapping_path": str(mapping_path), "mapping_sha256": sha256_file(mapping_path), "n_vectors": len(refs),
            "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    manifest["finalized"] = info
    save_manifest(shards_dir, manifest)
    _atomic_write_text(index_out.with_name(index_out.name + ".finalize.json"), json.dumps(info, indent=1))
    log(f"[reembed] finalized {len(refs)} vectors -> {index_out} ({info['index_bytes'] / 1e6:.1f} MB) + {mapping_path.name}")
    return info


def compare_embeddings(new: np.ndarray, source: np.ndarray, *, n_queries: int = 200, k: int = 10, seed: int = 7) -> dict:
    """How closely ``new`` reproduces ``source`` (row-aligned). Used to prove the pipeline against the
    production index when the SAME model is re-embedded. Reports exact-byte fraction, max |diff|,
    cosine, and mean top-``k`` neighbour-set overlap over seeded query rows."""
    if new.shape != source.shape:
        raise ReembedError(f"shape mismatch {new.shape} vs {source.shape}")
    diff = np.abs(new - source)
    rowdiff = diff.max(axis=1)
    cos = np.einsum("ij,ij->i", new, source) / (np.linalg.norm(new, axis=1) * np.linalg.norm(source, axis=1))
    rng = np.random.default_rng(seed)
    q = rng.choice(len(new), size=min(n_queries, len(new)), replace=False)
    overlaps, top1 = [], 0
    for qi in q:
        s_new, s_src = new @ new[qi], source @ source[qi]
        s_new[qi] = s_src[qi] = -np.inf          # exclude the query itself (trivially its own neighbour)
        a = np.argpartition(-s_new, k)[:k]
        b = np.argpartition(-s_src, k)[:k]
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / k)
        top1 += int(np.argmax(s_new) == np.argmax(s_src))
    return {"n": int(len(new)), "rows_byte_identical": int((rowdiff == 0).sum()),
            "fraction_byte_identical": float((rowdiff == 0).mean()),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
            f"mean_top{k}_neighbour_overlap": float(np.mean(overlaps)), "top1_agreement": top1 / len(q),
            "n_queries": int(len(q))}

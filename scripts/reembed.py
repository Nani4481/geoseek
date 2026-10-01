"""Resumable, model-agnostic re-embedding of the catalogued tiles. Never mutates the production index.

    # embed (resumable; Ctrl-C / crash / OOM safe - re-run with --resume)
    python scripts/reembed.py --model openclip-vitb32-openai --index-out data/index/candidates/vanilla.faiss
    python scripts/reembed.py --model openclip-vitb32-openai --index-out data/index/candidates/vanilla.faiss --resume
    # assemble the shards into a NEW FAISS file + faiss_id mapping database
    python scripts/reembed.py --model openclip-vitb32-openai --index-out data/index/candidates/vanilla.faiss --resume --finalize

The tile list is read from the SQLite catalog (ordered by production faiss_id), not from the filesystem.
Vectors go to ``data/index/shards/<model-key>/shard_NNNNN.npy`` (``--shard-size`` tiles each, default 5000)
with a ``manifest.json`` recording the model key, the verified weights SHA256, the preprocessing config,
the band selection per collection and every shard's SHA256. The weights SHA256 is verified against the
model registry before anything runs.

Scope: ``--collection sentinel-2-l2a`` and/or ``--max-faiss-id 101911`` (the frozen Phase 7b corpus);
default is every embedded tile in the catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.config import get_settings  # noqa: E402
from geoseek.eval.env import power_source  # noqa: E402
from geoseek.ingest import reembed as R  # noqa: E402
from geoseek.models.registry import get_spec, list_models, load_model, verify_weights  # noqa: E402
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=f"registry key; one of: {', '.join(list_models('embedding'))}")
    ap.add_argument("--index-out", required=True, type=Path, help="NEW faiss file to assemble at --finalize "
                    "(must not exist; the production index is never written)")
    ap.add_argument("--batch-size", type=int, default=R.DEFAULT_BATCH_SIZE)
    ap.add_argument("--shard-size", type=int, default=R.DEFAULT_SHARD_SIZE)
    ap.add_argument("--resume", action="store_true", help="continue an existing run in the shard directory")
    ap.add_argument("--finalize", action="store_true", help="after embedding, assemble shards -> faiss + mapping db")
    ap.add_argument("--shards-root", type=Path, default=None, help="default: <index_dir>/shards")
    ap.add_argument("--catalog", type=Path, default=None, help="default: the production catalog (read through the repository seam)")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--max-faiss-id", type=int, default=None, help="keep production faiss_id < N (frozen corpus: 101911)")
    ap.add_argument("--limit", type=int, default=None, help="first N tiles after ordering (smoke tests)")
    ap.add_argument("--stop-after-shards", type=int, default=None, help="stop after N NEW shards (interruption drill)")
    ap.add_argument("--verify-against", type=Path, default=None,
                    help="faiss file to compare the finalized index with (same-model sanity check, e.g. production)")
    ap.add_argument("--report-json", type=Path, default=None, help="write a throughput/VRAM summary here")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    spec = get_spec(args.model)
    if spec.kind != "embedding":
        raise SystemExit(f"{args.model!r} is a {spec.kind} model; re-embedding needs an embedding model")

    power = power_source()
    print(f"[reembed] power source: {power['source']}" + (f" ({power['battery_percent']}%)" if power.get("battery_percent") is not None else ""))
    if power["source"] != "AC":
        print("[reembed] WARNING: not on AC power - throughput will be roughly halved and is not a capability number")

    weights = verify_weights(args.model)               # raises loudly on a hash mismatch
    weights_sha = spec.weights_sha256
    print(f"[reembed] weights verified: {weights} sha256={weights_sha[:16]}...")

    settings = get_settings()
    catalog = args.catalog or settings.database_path
    repo = SQLiteMetadataRepository(catalog)
    try:
        refs = R.select_tiles(repo, collection=args.collection, max_faiss_id=args.max_faiss_id, limit=args.limit)
    finally:
        repo.close()
    if not refs:
        raise SystemExit("catalog selection is empty")
    print(f"[reembed] catalog {catalog}: {len(refs)} tiles selected "
          f"(faiss_id {refs[0].faiss_id}..{refs[-1].faiss_id}), collections {sorted({r.collection_id for r in refs})}")

    shards_root = args.shards_root or (settings.index_dir / "shards")
    shards_dir = shards_root / args.model
    header = R.build_header(spec, weights_sha, refs, shard_size=args.shard_size, batch_size=args.batch_size,
                            selection={"collection": args.collection, "max_faiss_id": args.max_faiss_id, "limit": args.limit})

    capped = R.apply_vram_cap()
    print(f"[reembed] VRAM cap {R.VRAM_FRACTION:.2f}: {'applied' if capped else 'n/a (no CUDA)'}")
    model = load_model(args.model)
    model.load()
    reader = R.CollectionDispatchReader()
    t0 = time.perf_counter()
    try:
        summary = R.run_embedding(model, reader, refs, shards_dir, header, resume=args.resume,
                                  max_shards=args.stop_after_shards)
    finally:
        reader.close()
    summary["wall_s"] = round(time.perf_counter() - t0, 2)
    summary["power_source"] = power["source"]
    print(f"[reembed] session: {summary}")

    result: dict = {"summary": summary, "shards_dir": str(shards_dir)}
    if args.finalize:
        if not summary["complete"]:
            print("[reembed] not all shards complete - skipping --finalize (re-run with --resume)")
            code = 2
        else:
            info = R.finalize(refs, shards_dir, header, args.index_out)
            result["finalize"] = info
            if args.verify_against is not None:
                src = FaissFlatIPIndex(args.verify_against).reconstruct_all()
                new = FaissFlatIPIndex(args.index_out).reconstruct_all()
                aligned = src[[r.faiss_id for r in refs]]
                result["verify_against"] = {"source": str(args.verify_against), **R.compare_embeddings(new, aligned)}
                print(f"[reembed] vs {args.verify_against.name}: {result['verify_against']}")
            code = 0
    else:
        code = 0 if summary["complete"] else 2
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

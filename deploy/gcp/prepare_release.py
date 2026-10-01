"""Create a portable serving snapshot; never change the source catalog/reports.

Run with the geoseek Python environment from the repository root. No imagery
is copied here: upload data/datasets separately. Stop ingestion while staging
so the FAISS file and SQLite snapshot represent the same generation.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def portable(value, source_data, target="/app/data"):
    """Rebase only paths anchored under this source data tree; retain URLs."""
    if isinstance(value, dict):
        return {k: portable(v, source_data, target) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, source_data, target) for v in value]
    if isinstance(value, str):
        normal = value.replace("\\", "/")
        root = source_data.as_posix().rstrip("/")
        if normal.lower().startswith(root.lower() + "/"):
            return target + normal[len(root):]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/cloud-release"))
    args = parser.parse_args()
    source = Path("data").resolve()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}; use a new --output directory")
    required = ["index/tiles.faiss", "index/tiles.sqlite", "index/tile_clusters.json",
                "models/RemoteCLIP-ViT-B-32.pt", "change_model/fc_siam_diff.pt",
                "change_model/ayodhya_change_report.json",
                "change_model/ayodhya_change_ranked_detail.json", "provenance_manifest.json"]
    for name in required:
        if not (source / name).is_file():
            raise SystemExit(f"Required staged artifact missing: {source / name}")
    out.mkdir(parents=True)
    for name in ("change_model", "detections", "discovery"):
        if (source / name).exists():
            shutil.copytree(source / name, out / name)
    (out / "index").mkdir()
    for p in (source / "index").iterdir():
        if p.is_file() and p.suffix in (".faiss", ".json", ".csv"):
            shutil.copy2(p, out / "index" / p.name)
    (out / "models").mkdir()
    shutil.copy2(source / "models/RemoteCLIP-ViT-B-32.pt", out / "models/RemoteCLIP-ViT-B-32.pt")
    # Only the deployed YOLO26s checkpoint/card; exclude rejected/evaluation caches.
    if (source / "models/detector").is_dir():
        shutil.copytree(source / "models/detector", out / "models/detector")
    shutil.copy2(source / "provenance_manifest.json", out / "provenance_manifest.json")
    with sqlite3.connect((source / "index/tiles.sqlite").as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(out / "index/tiles.sqlite") as dest:
            src.backup(dest)
            # Maxar paths are absolute; Sentinel paths are already relative.
            for oid, path in dest.execute("SELECT observation_id, dataset_dir FROM observations").fetchall():
                rebased = portable(path, source)
                if rebased != path:
                    dest.execute("UPDATE observations SET dataset_dir=? WHERE observation_id=?", (rebased, oid))
            if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Snapshot failed integrity check")
    # Deployment copies retain original provenance alongside rebased references.
    for p in out.rglob("*.json"):
        value = json.loads(p.read_text(encoding="utf-8"))
        rebased = portable(value, source)
        if rebased != value:
            p.write_text(json.dumps(rebased, indent=2) + "\n", encoding="utf-8")
    records = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            records.append({"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size,
                            "sha256": digest.hexdigest()})
    manifest = {"source_data_root": str(source), "runtime_data_root": "/app/data",
                "note": "Source files unchanged; only deployment-copy path references rebased.",
                "files": records, "bytes": sum(r["bytes"] for r in records)}
    (out / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} files, {manifest['bytes'] / 2**30:.3f} GiB at {out}")


if __name__ == "__main__":
    main()

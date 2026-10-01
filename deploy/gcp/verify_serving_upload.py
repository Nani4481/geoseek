"""Verify every SQLite-referenced imagery file exists in GCS with matching size."""
from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

BUCKET_PREFIX = "gs://geoseek-510206-archive/datasets"


def catalog_directories(root: Path) -> list[Path]:
    with sqlite3.connect("data/index/tiles.sqlite") as con:
        rows = con.execute("SELECT dataset_dir FROM observations").fetchall()
    result = []
    for (raw,) in rows:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        elif not str(path).lower().startswith(str(root).lower()):
            parts = list(path.parts)
            lowered = [part.lower() for part in parts]
            path = root.joinpath(*parts[lowered.index("datasets") + 1 :])
        path = path.resolve()
        if path not in result:
            result.append(path)
    return result


def main() -> None:
    root = Path("data/datasets").resolve()
    expected: dict[str, int] = {}
    for directory in catalog_directories(root):
        if not directory.is_dir():
            raise SystemExit(f"Missing local serving directory: {directory}")
        for file in directory.rglob("*"):
            if file.is_file():
                expected[file.relative_to(root).as_posix()] = file.stat().st_size

    gcloud = shutil.which("gcloud.cmd" if __import__("os").name == "nt" else "gcloud")
    if not gcloud:
        raise SystemExit("gcloud CLI not found")
    result = subprocess.run(
        [gcloud, "storage", "ls", "--long", "--recursive", BUCKET_PREFIX],
        check=True, text=True, capture_output=True,
    )
    remote: dict[str, int] = {}
    pattern = re.compile(r"^\s*(\d+)\s+\S+\s+gs://[^/]+/datasets/(.+)$")
    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match:
            remote[match.group(2)] = int(match.group(1))

    missing = sorted(set(expected) - set(remote))
    mismatched = sorted(
        path for path, size in expected.items()
        if path in remote and remote[path] != size
    )
    expected_bytes = sum(expected.values())
    verified_bytes = sum(expected[path] for path in expected if path in remote and path not in mismatched)
    print(f"expected_files={len(expected)} expected_bytes={expected_bytes}")
    print(f"verified_files={len(expected)-len(missing)-len(mismatched)} verified_bytes={verified_bytes}")
    if missing:
        print("missing:")
        print("\n".join(missing[:50]))
    if mismatched:
        print("size_mismatch:")
        print("\n".join(mismatched[:50]))
    if missing or mismatched:
        raise SystemExit(1)
    print("SERVING_UPLOAD_VERIFIED")


if __name__ == "__main__":
    main()

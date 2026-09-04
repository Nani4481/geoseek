"""Provenance manifest: records every artifact staged from the network.

Every staged file (model weights, datasets, ...) must be registered here with
its source URL, SHA256, byte size, license, and staging timestamp, so the
project can be audited offline and artifacts can be verified for integrity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from geoseek.config import get_settings

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    source_url: str
    local_path: str
    sha256: str
    byte_size: int
    license: str
    staged_at: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record(*, name: str, source_url: str, local_path: Path, license: str) -> ArtifactRecord:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Cannot build provenance record — staged file missing: {local_path}")
    return ArtifactRecord(
        name=name,
        source_url=source_url,
        local_path=str(local_path.resolve()),
        sha256=sha256_of(local_path),
        byte_size=local_path.stat().st_size,
        license=license,
        staged_at=datetime.now(timezone.utc).isoformat(),
    )


def load_manifest(manifest_path: Path | None = None) -> dict:
    manifest_path = manifest_path or get_settings().provenance_manifest_path
    if not manifest_path.is_file():
        return {"artifacts": []}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(manifest: dict, manifest_path: Path | None = None) -> Path:
    manifest_path = manifest_path or get_settings().provenance_manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    return manifest_path


def append_ingest_run(run_report: dict, manifest_path: Path | None = None) -> dict:
    """Append one pipeline run report (scene ingested, tiles added, timings...) to the manifest.

    Distinct from `artifacts` (staged network-sourced files): this tracks
    processing provenance - what was ingested, when, and the performance
    numbers for that run - so the manifest doubles as an ingest run log.
    """
    manifest = load_manifest(manifest_path)
    runs = manifest.setdefault("ingest_runs", [])
    runs.append(run_report)
    write_manifest(manifest, manifest_path)
    return run_report


def record_analysis_section(key: str, config: dict, manifest_path: Path | None = None) -> dict:
    """Upsert a top-level analysis/provenance section into the manifest by key.

    Generic sibling of ``record_radiometry_config`` for the Phase 3a temporal
    pair-prep sections (``extra_bands``, ``coregistration``,
    ``radiometric_normalization``, ``spectral_indices``): the exact parameters
    that made a given date pair comparable are reproducible from the manifest
    alone.
    """
    manifest = load_manifest(manifest_path)
    manifest[key] = config
    write_manifest(manifest, manifest_path)
    return config


def record_radiometry_config(config: dict, manifest_path: Path | None = None) -> dict:
    """Record the true-color / radiometry harmonization config used at ingest.

    Stored under the top-level ``radiometry`` key so the exact reflectance
    bounds, gamma, and per-scene BOA offset handling that produced the index
    are reproducible from the manifest alone.
    """
    manifest = load_manifest(manifest_path)
    manifest["radiometry"] = config
    write_manifest(manifest, manifest_path)
    return config


def record_artifact(
    *,
    name: str,
    source_url: str,
    local_path: Path,
    license: str,
    manifest_path: Path | None = None,
) -> ArtifactRecord:
    """Compute provenance for a staged file and upsert it into the manifest by name."""
    record = build_record(name=name, source_url=source_url, local_path=local_path, license=license)

    manifest = load_manifest(manifest_path)
    artifacts = [a for a in manifest.get("artifacts", []) if a.get("name") != name]
    artifacts.append(asdict(record))
    manifest["artifacts"] = artifacts

    write_manifest(manifest, manifest_path)
    return record

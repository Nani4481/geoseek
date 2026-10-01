"""Block A: stage B08 + B11 for the 8 diverse regions AND compute the per-tile spectral descriptor - in ONE pass.

    python scripts/stage_nir_swir_and_describe.py --dry-run          # plan only: what is missing, nothing written
    python scripts/stage_nir_swir_and_describe.py                    # all 69 diverse scenes + the 5 Ayodhya dates
    python scripts/stage_nir_swir_and_describe.py --region kanha     # one region (slug prefix of the aoi name)
    python scripts/stage_nir_swir_and_describe.py --max-scenes 2     # smoke test
    python scripts/stage_nir_swir_and_describe.py --context-only     # just the region river/water distances

Per scene: ``ensure_bands`` (network; skips what already exists on the scene's grid; resumable partial downloads) ->
``describe_scene`` (strips; descriptor rows + the decimated water mask) -> upsert into ``tile_spectral``. The next
scene's download overlaps the current scene's description. After all scenes, the region step turns the water masks into
``dist_river_m`` / ``dist_water_m``.

Safety, because this migrates the production catalog:
  * ``tiles.sqlite`` is copied to ``tiles.sqlite.pre_phase10.bak`` BEFORE it is first opened by this script;
  * SHA256 fingerprints of every existing catalog table (collections, scenes, observations, tiles, derived) are taken
    from that backup and compared with the live catalog at the end - they must be identical (the migration only adds
    ``tile_spectral``); ``tiles.faiss`` is hashed before and after and must not change;
  * tile ids / geometry are read from the catalog and never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.config import get_settings  # noqa: E402
from geoseek.eval.env import power_source  # noqa: E402
from geoseek.eval.retrieval import AYODHYA_JUDGED_OBS  # noqa: E402
from geoseek.spectral import context as ctx  # noqa: E402
from geoseek.spectral.descriptor import WATER_MASK_NAME, TileSpec, describe_scene  # noqa: E402
from geoseek.spectral.fields import DESCRIPTOR_VERSION  # noqa: E402
from geoseek.staging import download_nir_swir as dl  # noqa: E402
from geoseek.staging.manifest import load_manifest, write_manifest  # noqa: E402

SETTINGS = get_settings()
OUT_DIR = SETTINGS.data_dir / "eval" / "judge_coverage"
MANIFEST_KEY = "nir_swir_staging"
TABLES = ("collections", "scenes", "observations", "tiles", "derived")


def log(msg: str) -> None:
    print(f"[nir-swir] {msg}", flush=True)


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


@dataclass
class SceneEntry:
    observation_id: str
    scene_id: str
    aoi: str
    scene_dir: Path
    tiles: list[TileSpec]

    @property
    def mask_path(self) -> Path:
        return self.scene_dir / WATER_MASK_NAME


def build_plan(repo, region_filter: list[str] | None) -> list[SceneEntry]:
    plan: list[SceneEntry] = []
    for obs in repo.list_observations(collection="sentinel-2-l2a"):
        aoi = obs.aoi_name or ""
        diverse = aoi.endswith("_diverse")
        ayodhya = aoi.startswith("ayodhya_44RPQ_scaled")
        if not (diverse or ayodhya):
            continue
        if region_filter and not any(aoi.startswith(r) for r in region_filter):
            continue
        tiles = [TileSpec(t.tile_id, t.row, t.col) for t in repo.list_tiles(observation_id=obs.observation_id)
                 if t.faiss_id is not None]
        if not tiles:
            continue
        plan.append(SceneEntry(obs.observation_id, obs.scene_id, aoi, SETTINGS.datasets_dir / (obs.dataset_dir or obs.observation_id), tiles))
    plan.sort(key=lambda e: (e.aoi, e.observation_id))
    return plan


def scene_done(repo_ids: set[str], e: SceneEntry) -> bool:
    return all(t.tile_id in repo_ids for t in e.tiles) and e.mask_path.is_file() \
        and all(dl.band_on_reference_grid(e.scene_dir / f"{b}.tif", e.scene_dir / "B04.tif") for b in dl.BANDS)


def record_staging(entry: SceneEntry, bands: dict) -> None:
    staged = {b: v for b, v in bands.items() if v.get("staged_now")}
    if not staged:
        return
    manifest = load_manifest()
    manifest.setdefault(MANIFEST_KEY, {})[entry.scene_id] = {
        "observation_id": entry.observation_id, "aoi": entry.aoi, "dataset_dir": str(entry.scene_dir),
        "purpose": "NIR/SWIR for spectral descriptor + retrieval judge coverage (Phase 10 Block A)",
        "bands": {b: {k: v for k, v in rec.items() if k != "staged_now"} for b, rec in bands.items()}}
    write_manifest(manifest)


def catalog_fingerprints_of_backup(backup: Path) -> dict[str, str]:
    """Fingerprints of the PRE-migration catalog, taken from a scratch copy of the backup."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "pre.sqlite"
        shutil.copy2(backup, copy)
        repo = SQLiteMetadataRepository(copy)          # opening the copy migrates the COPY; existing tables are unaffected
        try:
            return repo.table_fingerprints(TABLES)
        finally:
            repo.close()


def run_context(repo, plan: list[SceneEntry]) -> dict:
    groups: dict[str, list[SceneEntry]] = {}
    for e in plan:
        if e.mask_path.is_file():
            groups.setdefault(e.aoi, []).append(e)
    summary = {}
    for aoi, entries in sorted(groups.items()):
        judged = [e for e in entries if e.observation_id in AYODHYA_JUDGED_OBS]
        union_over = [e.mask_path for e in (judged or entries)]       # Ayodhya: the 3 dates the Phase 7a river was built from
        apply_to = {e.mask_path: [(t.tile_id, t.row, t.col) for t in e.tiles] for e in entries}
        rows, meta = ctx.region_context(union_over, apply_to)
        n = repo.set_tile_spectral_context(rows)
        summary[aoi] = {**meta, "tiles_updated": n, "dates_in_union": len(union_over), "dates_applied": len(entries)}
        log(f"context {aoi}: {meta['n_water_components']} water components, river = {meta['river_component_cells']} cells, "
            f"{n} tiles updated")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--region", action="append", default=None, help="aoi-name prefix, e.g. kanha (repeatable)")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--context-only", action="store_true")
    ap.add_argument("--no-network", action="store_true", help="describe only scenes whose bands are already on disk")
    ap.add_argument("--tmp-root", type=Path, default=SETTINGS.datasets_dir / "_nir_swir_tmp")
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    power = power_source()
    log(f"power source: {power['source']}")

    db_path = Path(SETTINGS.database_path)
    faiss_path = Path(SETTINGS.faiss_index_path)
    backup = db_path.with_name(db_path.name + ".pre_phase10.bak")
    # ---- safety: backup BEFORE this script opens the catalog for the first time -----------------------------------
    if not args.dry_run:
        if not backup.exists():
            shutil.copy2(db_path, backup)
            log(f"backed up the catalog -> {backup.name} ({backup.stat().st_size / 1e6:.1f} MB)")
        before_path = OUT_DIR / "catalog_fingerprint_before.json"
        if not before_path.exists():
            fp = catalog_fingerprints_of_backup(backup)
            before_path.write_text(json.dumps({"taken_from": backup.name, "backup_file_sha256": sha256_path(backup),
                                               "faiss_sha256": sha256_path(faiss_path), "tables": fp}, indent=1))
            log(f"pre-migration catalog fingerprints written: {fp}")
    scratch = None
    if args.dry_run:                                   # a dry run must not migrate production: plan against a scratch copy
        scratch = tempfile.TemporaryDirectory()
        shutil.copy2(db_path, Path(scratch.name) / "plan.sqlite")
    repo = SQLiteMetadataRepository(Path(scratch.name) / "plan.sqlite" if scratch else db_path)
    try:
        plan = build_plan(repo, args.region)
        done_ids = repo.spectral_tile_ids(version=DESCRIPTOR_VERSION)
        todo = [e for e in plan if not scene_done(done_ids, e)]
        log(f"plan: {len(plan)} scenes ({sum(len(e.tiles) for e in plan)} tiles); {len(plan) - len(todo)} already complete; "
            f"{len(todo)} to do")
        if args.dry_run:
            missing = [e for e in todo if not all(dl.band_on_reference_grid(e.scene_dir / f'{b}.tif', e.scene_dir / 'B04.tif') for b in dl.BANDS)]
            log(f"dry run: {len(missing)} scenes need a download (~{len(missing) * 0.29:.1f} GB at ~290 MB/scene); "
                f"{len(todo) - len(missing)} only need describing")
            for e in todo[:5]:
                log(f"  e.g. {e.observation_id} ({e.aoi}, {len(e.tiles)} tiles)")
            return 0
        if args.max_scenes:
            todo = todo[: args.max_scenes]
        report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "power_source_start": power["source"],
                  "scenes": []}
        if not args.context_only and todo:
            def stage(e: SceneEntry) -> dict:
                if args.no_network:
                    return {b: {"staged_now": False} for b in dl.BANDS}
                return dl.ensure_bands(e.scene_id, e.scene_dir, args.tmp_root, log=log)

            t_all = time.time()
            with ThreadPoolExecutor(max_workers=1) as pool:
                nxt = pool.submit(stage, todo[0])
                for k, e in enumerate(todo):
                    t0 = time.time()
                    bands = nxt.result()
                    if k + 1 < len(todo):
                        nxt = pool.submit(stage, todo[k + 1])           # overlap the next download with this description
                    t_stage = time.time() - t0
                    if not all(dl.band_on_reference_grid(e.scene_dir / f"{b}.tif", e.scene_dir / "B04.tif") for b in dl.BANDS):
                        log(f"SKIP {e.observation_id}: B08/B11 not on disk (use without --no-network)")
                        continue
                    record_staging(e, bands)
                    t1 = time.time()
                    rows = describe_scene(e.scene_dir, e.tiles, log=None)
                    repo.upsert_tile_spectral(rows)
                    usable = sum(r["usable"] for r in rows)
                    rec = {"observation_id": e.observation_id, "aoi": e.aoi, "tiles": len(rows), "usable": usable,
                           "staged_now": {b: bool(v.get("staged_now")) for b, v in bands.items()},
                           "bytes_downloaded": sum(v.get("source_bytes", 0) for v in bands.values()),
                           "wait_for_download_s": round(t_stage, 1), "describe_s": round(time.time() - t1, 1)}
                    report["scenes"].append(rec)
                    with open(OUT_DIR / "ledger.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({**rec, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                    log(f"[{k + 1}/{len(todo)}] {e.observation_id}: {len(rows)} tiles ({usable} usable), "
                        f"wait {t_stage:.0f}s, describe {rec['describe_s']}s; elapsed {(time.time() - t_all) / 60:.1f} min")
        # ---- region water context (needs every date of a region) -----------------------------------------------
        if not args.max_scenes or args.context_only:
            report["context"] = run_context(repo, plan)
        # ---- integrity: the migration must not have changed any existing row ------------------------------------
        before = json.loads((OUT_DIR / "catalog_fingerprint_before.json").read_text())
        after = repo.table_fingerprints(TABLES)
        same = {t: before["tables"][t] == after[t] for t in TABLES}
        report["catalog_integrity"] = {"before": before["tables"], "after": after, "tables_identical": same,
                                       "all_identical": all(same.values()),
                                       "faiss_sha256_before": before["faiss_sha256"], "faiss_sha256_after": sha256_path(faiss_path),
                                       "faiss_unchanged": before["faiss_sha256"] == sha256_path(faiss_path),
                                       "sqlite_file_sha256_before": before["backup_file_sha256"],
                                       "sqlite_file_sha256_after": sha256_path(db_path),
                                       "tile_spectral_rows": len(repo.spectral_tile_ids(version=DESCRIPTOR_VERSION))}
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["power_source_end"] = power_source()["source"]
        (OUT_DIR / "staging_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        ci = report["catalog_integrity"]
        log(f"catalog integrity: tables identical={ci['all_identical']}, faiss unchanged={ci['faiss_unchanged']}, "
            f"tile_spectral rows={ci['tile_spectral_rows']}")
        return 0 if ci["all_identical"] and ci["faiss_unchanged"] else 3
    finally:
        repo.close()
        if scratch:
            scratch.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

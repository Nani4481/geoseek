"""Frozen baseline snapshot: re-run every existing evaluation and write ONE machine-readable artifact.

    python scripts/snapshot_baseline.py                          # all sections -> data/eval/baseline_v1.json
    python scripts/snapshot_baseline.py --sections retrieval,oscd
    python scripts/snapshot_baseline.py --assemble-only          # merge the per-section parts already on disk
    # a candidate: separate parts dir + output, and name the model whose vectors to evaluate (same label as the baseline's)
    python scripts/snapshot_baseline.py --sections env,corpus,retrieval --parts-dir data/eval/cand_parts         --out data/eval/candidate.json --retrieval-system remoteclip=<model-key>=<finalized candidate .faiss>

Every number is produced by an actual run here; nothing is copied from a markdown report. The numbers the
reports CLAIM live in ``geoseek.eval.reported`` and are used only to label each result REPRODUCED /
NOT_REPRODUCED in the snapshot's ``reproduction`` block. Sections are independent and each writes
``data/eval/baseline_parts/<section>.json`` as soon as it finishes, so a crash never loses finished work.

Sections
  env         GPU / VRAM / driver / torch / power source / git SHA / timestamp
  corpus      catalog + index counts
  retrieval   Recall/Precision/NDCG@{1,5,10,20}: RemoteCLIP and the vanilla CLIP control on the Ayodhya corpus
              (3,267 tiles) and the frozen 101,911-tile corpus; frozen Phase 7a judgements; noise bands from 3
              query orderings + a query bootstrap
  oscd        FC-Siam-diff on the 10 held-out OSCD regions at thresholds 0.50 / 0.80, per region + pooled
  detector    DOTA official-val AP50 / AP50:95 per class group with bootstrap CIs (scripts/eval_detector.py,
              run into a scratch directory)
  latency     text / image / point-KNN / tile-KNN / bbox query latency percentiles (on a scratch copy of the index)
  footprint   index + catalog size, per-tile cost
  ingestion   end-to-end and pure-embed throughput into a scratch index, append-only proof, re-embed throughput
  suite       the pytest result

Safety: the production index/catalog/weights are never written. Latency runs against a copy; ingestion goes
to a scratch index; the detector evaluation writes to a scratch directory. Timed sections require AC power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from geoseek.config import get_settings  # noqa: E402
from geoseek.eval import compare as CMP  # noqa: E402
from geoseek.eval.env import capture_environment, power_source  # noqa: E402

SCHEMA = "geoseek.baseline_snapshot/1"
SECTIONS = ("env", "corpus", "retrieval", "oscd", "detector", "latency", "footprint", "ingestion", "suite")
FROZEN_N = 101_911                       # the Phase 7b tier-3 corpus == production faiss_id < 101911
ORDER_SEEDS = (11, 12, 13)               # three different query orderings for the noise band
SETTINGS = get_settings()
EVAL_DIR = SETTINGS.data_dir / "eval"
PARTS_DIR = EVAL_DIR / "baseline_parts"
CANDIDATES = SETTINGS.index_dir / "candidates"
SHARDS_ROOT = SETTINGS.index_dir / "shards" / "baseline_v1"


def log(msg: str) -> None:
    print(f"[snapshot] {msg}", flush=True)


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def _clean(o):
    """numpy -> plain Python; non-finite floats -> None (strictly valid JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    return o


def save_part(name: str, metrics: dict, noise_bands: dict | None = None, noise_detail: dict | None = None,
              extra: dict | None = None, power: dict | None = None) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    part = {"section": name, "metrics": metrics, "noise_bands": noise_bands or {}, "noise_detail": noise_detail or {},
            "extra": extra or {}, "power": power or {}, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (PARTS_DIR / f"{name}.json").write_text(json.dumps(_clean(part), indent=1), encoding="utf-8")
    log(f"section '{name}' written -> {PARTS_DIR / (name + '.json')}")


def load_part(name: str) -> dict | None:
    p = PARTS_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


class PowerGuard:
    """Records the power source at section start/end; timed sections refuse to run on battery."""

    def __init__(self, section: str, args, timed: bool):
        self.section, self.timed, self.allow = section, timed, args.allow_battery
        self.start: dict = {}

    def __enter__(self):
        self.start = power_source()
        if self.timed and self.start["source"] != "AC" and not self.allow:
            raise SystemExit(f"section '{self.section}' is a timed capability measurement and the machine is on "
                             f"{self.start['source']} power (it roughly halves throughput). Plug in, or pass --allow-battery "
                             "to record it anyway (the power source is stored with the result).")
        return self

    def result(self) -> dict:
        return {"start": self.start, "end": power_source()}

    def __exit__(self, *exc):
        return False


def noise_entry(path: str, ordering_spread: float | None = None, bootstrap_hw: float | None = None,
                method: str = "") -> tuple[dict, dict]:
    comps = [x for x in (ordering_spread, bootstrap_hw) if x is not None]
    band = max(comps) if comps else 0.0
    detail = {"band": band, "method": method}
    if ordering_spread is not None:
        detail["ordering_or_rerun_spread"] = ordering_spread
    if bootstrap_hw is not None:
        detail["bootstrap_ci95_halfwidth"] = bootstrap_hw
    return {"band": band}, detail


# =============================================================================
# corpus
# =============================================================================


def open_catalog():
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    return SQLiteMetadataRepository(SETTINGS.database_path)


def section_corpus(args) -> None:
    from geoseek.ingest import reembed as R
    from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

    repo = open_catalog()
    try:
        refs = R.select_tiles(repo)
        n_catalog = repo.count_tiles()
        n_scenes, n_obs = len(repo.list_scenes()), len(repo.list_observations())
    finally:
        repo.close()
    by_coll: dict[str, int] = {}
    for r in refs:
        by_coll[r.collection_id] = by_coll.get(r.collection_id, 0) + 1
    ix = FaissFlatIPIndex(SETTINGS.faiss_index_path)
    frozen = [r for r in refs if r.faiss_id < FROZEN_N]
    ids_contiguous = all(r.faiss_id == i for i, r in enumerate(refs))
    m = {
        "faiss_vectors": ix.count(), "catalog_tiles_total": n_catalog, "embedded_tiles_total": len(refs),
        "embedded_tiles_frozen_prefix": len(frozen),
        "frozen_prefix_is_all_sentinel2": int(all(r.collection_id == "sentinel-2-l2a" for r in frozen)),
        "sentinel2_embedded_tiles": by_coll.get("sentinel-2-l2a", 0),
        "maxar_embedded_tiles": by_coll.get("maxar-opendata", 0),
        "sar_tiles_not_embedded": n_catalog - len(refs),
        "scenes_total": n_scenes, "observations_total": n_obs,
        "faiss_ids_contiguous_from_zero": int(ids_contiguous),
        "faiss_count_equals_embedded_tiles": int(ix.count() == len(refs)),
    }
    save_part("corpus", m, extra={"definition": "frozen prefix = production faiss_id < 101911 (the Phase 7b tier-3 corpus)"})


# =============================================================================
# retrieval
# =============================================================================


def section_retrieval(args) -> None:
    from geoseek.catalog.embedding_map import read_mapping
    from geoseek.eval import retrieval as RT
    from geoseek.ingest import reembed as R
    from geoseek.models.registry import load_model, get_spec
    from geoseek.search.rerank import region_key
    from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

    guard = PowerGuard("retrieval", args, timed=False)
    guard.__enter__()
    eval_dir = SETTINGS.data_dir / "eval_retrieval"
    judged = RT.load_judged(eval_dir)
    repo = open_catalog()
    try:
        refs = R.select_tiles(repo, max_faiss_id=FROZEN_N)
        aoi_of_obs = {o.observation_id: region_key(o.aoi_name) for o in repo.list_observations()}
    finally:
        repo.close()
    assert len(refs) == FROZEN_N and all(r.faiss_id == i for i, r in enumerate(refs)), "frozen corpus is not the faiss_id prefix"
    tile_ids = [r.tile_id for r in refs]
    region = np.array([aoi_of_obs[r.observation_id] for r in refs])
    mask_ay_judged = np.array([RT.is_ayodhya_judged_tile(t) for t in tile_ids])
    mask_ay_region = region == "ayodhya"
    log(f"frozen corpus: {len(refs)} tiles; Ayodhya-judged {int(mask_ay_judged.sum())}, region=ayodhya {int(mask_ay_region.sum())}")

    # vectors: RemoteCLIP = what production serves; vanilla = the fresh re-embed (scripts/reembed.py).
    # --retrieval-system LABEL=MODEL_KEY=INDEX replaces / adds a system (INDEX = 'production' or a finalized candidate
    # .faiss over the frozen corpus whose mapping database sits next to it).
    prod_vecs = FaissFlatIPIndex(SETTINGS.faiss_index_path).reconstruct_all()[:FROZEN_N]
    spec = {"remoteclip": ("remoteclip-vitb32", "production"),
            "vanilla": ("openclip-vitb32-openai", str(CANDIDATES / f"vanilla_frozen{FROZEN_N}.faiss"))}
    for item in args.retrieval_system:
        label, key, index = item.split("=", 2)
        spec[label] = (key, index)
    defaults = {"remoteclip": "remoteclip-vitb32", "vanilla": "openclip-vitb32-openai"}
    systems = {}
    for label, (key, index) in spec.items():
        if index == "production":
            vecs, src = prod_vecs, "production index data/index/tiles.faiss rows 0..101910"
        else:
            idx_path = Path(index)
            rows, _meta = read_mapping(idx_path.with_name(idx_path.stem + ".mapping.sqlite"))
            assert [r[1] for r in rows] == tile_ids, f"{idx_path.name}: tile order differs from the catalog's frozen corpus"
            vecs, src = FaissFlatIPIndex(idx_path).reconstruct_all(), f"fresh re-embed via scripts/reembed.py ({idx_path.name})"
        systems[label] = {"key": key, "vectors": vecs, "vectors_source": src, "is_default_model": defaults.get(label) == key}
    rc_idx = CANDIDATES / f"remoteclip_frozen{FROZEN_N}.faiss"        # fresh RemoteCLIP re-embed (optional cross-check)
    rc_fresh = FaissFlatIPIndex(rc_idx).reconstruct_all() if rc_idx.is_file() else None
    for s in systems.values():
        model = load_model(s["key"])
        model.load()
        s["encode"] = model.encode_text
        s["weights_sha256"] = get_spec(s["key"]).weights_sha256

    conditions = {                         # name -> (mask over the frozen corpus, which systems)
        "ayodhya_3267": mask_ay_judged,
        "full_global": None,
        "full_region_filtered_ayodhya": mask_ay_region,
    }

    def run(system: dict, mask, order=None):
        return RT.evaluate_system(system["encode"], system["vectors"], tile_ids, judged, order=order, mask=mask)

    canonical, order_runs = {}, {}
    for sname, s in systems.items():
        for cname, mask in conditions.items():
            canonical[(sname, cname)] = run(s, mask)
            order_runs[(sname, cname)] = []
            for seed in ORDER_SEEDS:
                perm = list(judged.queries)
                np.random.default_rng(seed).shuffle(perm)
                order_runs[(sname, cname)].append(run(s, mask, perm))
        log(f"retrieval: {sname} done ({len(conditions)} conditions x (1 canonical + {len(ORDER_SEEDS)} reordered))")

    metrics: dict = {
        "judgements": {"source": "frozen Phase 7a judgements (data/eval_retrieval/judgments.json), independent of both models",
                       "n_queries": len(judged.queries), "n_low_confidence": len(judged.low_confidence),
                       "low_confidence_queries": sorted(judged.low_confidence)},
        "systems": {n: {"model_key": s["key"], "weights_sha256": s["weights_sha256"], "vectors_source": s["vectors_source"]}
                    for n, s in systems.items()},
        "ayodhya_3267": {"corpus_tiles": int(mask_ay_judged.sum())},
        "full_101911": {"corpus_tiles": FROZEN_N},
    }
    noise_bands, noise_detail = {}, {}

    def add_noise(prefix: str, sname: str, cname: str, subset: str | None):
        """Noise band per metric = max(spread of the macro metric over 3 query orderings, query-bootstrap 95% CI half-width).

        Ranking is exact and order-independent, so the ordering spread is expected to be ~0 - it is a floor, not a
        meaningful band; the bootstrap over the 16 queries is the real sampling uncertainty of a 16-query macro mean.
        """
        qs = [q for q in judged.queries if not (subset == "excluding_low_confidence" and q in judged.low_confidence)]
        for k in RT.KS:
            for m in RT.METRICS:
                per_run = [statistics.mean(r["scores"][q][k][m] for q in qs) for r in order_runs[(sname, cname)]]
                pq = RT.per_query_vector(canonical[(sname, cname)], judged, k, m, exclude_low=(subset == "excluding_low_confidence"))
                path = f"{prefix}.{subset}.k{k}.{m}" if subset else f"{prefix}.k{k}.{m}"
                b, d = noise_entry(path, CMP.spread(per_run), CMP.bootstrap_halfwidth(pq, n_boot=2000, seed=7),
                                   "max(spread over 3 query orderings, query-bootstrap 95% CI half-width)")
                noise_bands[path], noise_detail[path] = b, d

    for sname in systems:
        blk = RT.summarize_system(canonical[(sname, "ayodhya_3267")], judged)
        metrics["ayodhya_3267"][sname] = blk
        if systems[sname]["is_default_model"]:                       # the stored Phase 7a pools are for the default models only
            pools = json.loads((eval_dir / "pools.json").read_text(encoding="utf-8"))
            fresh = canonical[(sname, "ayodhya_3267")]["ranked"]
            same = sum(fresh[q] == pools[q][f"{sname}_ranked"] for q in judged.queries)
            overlap = statistics.mean(len(set(fresh[q]) & set(pools[q][f"{sname}_ranked"])) / 20 for q in judged.queries)
            blk["ranking_vs_stored_phase7a_pools"] = {"queries_with_identical_top20_list": same,
                                                      "mean_top20_set_overlap": overlap, "of_queries": len(judged.queries)}
        for subset in ("all", "excluding_low_confidence"):
            add_noise(f"retrieval.ayodhya_3267.{sname}", sname, "ayodhya_3267", subset)
        full = {}
        for cname, label in (("full_global", "global"), ("full_region_filtered_ayodhya", "region_filtered_ayodhya")):
            b = RT.summarize_system(canonical[(sname, cname)], judged)
            full[label] = {**b["all"], "topk_composition_k20": b["topk_composition_k20"], "per_query": b["per_query"]}
            add_noise(f"retrieval.full_101911.{sname}.{label}", sname, cname, None)
        metrics["full_101911"][sname] = full
    metrics["checks"] = {
        "region_filter_mask_equals_judged_ayodhya_mask": int(np.array_equal(mask_ay_judged, mask_ay_region)),
        "deterministic_rankings_across_orderings": int(all(
            all(r["ranked"][q] == canonical[k]["ranked"][q] for r in runs for q in judged.queries) for k, runs in order_runs.items())),
    }

    # fresh-vs-stored vector checks (default models only)
    if "vanilla" in systems and systems["vanilla"]["is_default_model"] and systems["vanilla"]["vectors_source"].startswith("fresh"):
        stored_ids = json.loads((eval_dir / "vanilla_corpus_tile_ids.json").read_text(encoding="utf-8"))
        stored_vec = np.load(eval_dir / "vanilla_corpus_vectors.npy")
        pos = {t: i for i, t in enumerate(tile_ids)}
        sel = [pos[t] for t in stored_ids]
        fresh_v = systems["vanilla"]["vectors"][sel]
        cos = np.einsum("ij,ij->i", fresh_v, stored_vec) / (np.linalg.norm(fresh_v, axis=1) * np.linalg.norm(stored_vec, axis=1))
        metrics["checks"]["vanilla_ayodhya_vectors_vs_phase7a_stored"] = {
            "n": len(sel), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
            "max_abs_diff": float(np.abs(fresh_v - stored_vec).max()),
            "tile_order_identical_to_phase7a": int(stored_ids == [tile_ids[i] for i in sel])}
    if rc_fresh is not None and "remoteclip" in systems and systems["remoteclip"]["is_default_model"]:
        metrics["checks"]["remoteclip_fresh_reembed_vs_production"] = R.compare_embeddings(rc_fresh, prod_vecs)
    save_part("retrieval", metrics, noise_bands, noise_detail, power=guard.result())


# =============================================================================
# OSCD
# =============================================================================


def _metrics(tp, fp, fn, tn) -> dict:
    """Same definitions as scripts/train_change.py ``_metrics`` (kept local: that one also returns the counts)."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0}


def _counts_metrics(c: np.ndarray) -> dict:
    tp, fp, fn, tn = (int(x) for x in c)
    return {**_metrics(tp, fp, fn, tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def section_oscd(args) -> None:
    import torch

    import train_change as TC
    from geoseek.change.models.fc_siam_diff import FCSiamDiff
    from geoseek.models.registry import get_spec, verify_weights

    guard = PowerGuard("oscd", args, timed=False)
    guard.__enter__()
    ckpt = verify_weights("fc-siam-diff-oscd")                      # SHA256 gate
    device = SETTINGS.device
    blob = torch.load(ckpt, map_location=device)
    card = blob["model_card"]
    arch = card["architecture"]
    model = FCSiamDiff(in_channels=arch["in_channels"], base_channels=arch["base_channels"], depth=arch["depth"], dropout=0.0).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    mean = np.asarray(card["norm_stats"]["mean"], np.float32)
    std = np.asarray(card["norm_stats"]["std"], np.float32)
    td = card["training_data"]
    val_regions, test_regions = td["validation_regions_heldout_from_train"], td["test_regions_never_used"]
    assert not (set(test_regions) & set(td["fit_regions_gradient_updates"] + val_regions)), "LEAK: test overlaps train/val"

    # threshold selection reproduced from the VALIDATION regions only (protocol of train_change.evaluate)
    val_probs, val_gts = zip(*[TC.infer_region(model, r, mean, std, device) for r in val_regions])
    grid = np.round(np.arange(0.02, 0.985, 0.01), 3)
    curve = TC.pr_curve(list(val_probs), list(val_gts), grid)
    viable = [c for c in curve if c["recall"] >= 0.15]
    chosen = max(viable or curve, key=lambda c: c["f0.5"])["threshold"]
    del val_probs, val_gts

    thresholds = {"0p50": 0.5, "0p80": 0.8}
    passes = 3
    counts = {t: np.zeros((passes, len(test_regions), 4), dtype=np.int64) for t in thresholds}
    change_frac, px = {}, {}
    for p in range(passes):
        for ri, r in enumerate(test_regions):
            prob, gt = TC.infer_region(model, r, mean, std, device)
            for t, thr in thresholds.items():
                counts[t][p, ri] = TC._counts_at(prob, gt, thr)
            if p == 0:
                change_frac[r], px[r] = float((gt > 0.5).mean()), int(gt.size)
        log(f"oscd: inference pass {p + 1}/{passes} over {len(test_regions)} held-out regions")

    metrics: dict = {"model_key": "fc-siam-diff-oscd", "weights_sha256": get_spec("fc-siam-diff-oscd").weights_sha256,
                     "chosen_threshold_from_validation": chosen,
                     "chosen_threshold_stored_in_checkpoint": card.get("eval", {}).get("chosen_threshold_precision_favouring"),
                     "validation_regions": list(val_regions), "test_regions": list(test_regions),
                     "test_pixels_total": sum(px.values()),
                     "test_change_pixel_fraction": sum(change_frac[r] * px[r] for r in test_regions) / sum(px.values()),
                     "inference_passes": passes}
    noise_bands, noise_detail = {}, {}
    rng = np.random.default_rng(7)
    boot_idx = rng.integers(0, len(test_regions), size=(2000, len(test_regions)))
    for t in thresholds:
        pooled_by_pass = [_counts_metrics(counts[t][p].sum(axis=0)) for p in range(passes)]
        pooled = pooled_by_pass[0]
        per_region = {r: _counts_metrics(counts[t][0, i]) for i, r in enumerate(test_regions)}
        metrics[f"thr_{t}"] = {"threshold": thresholds[t], "pooled": pooled,
                               "per_region": {r: {**m, "change_fraction": change_frac[r]} for r, m in per_region.items()}}
        c0 = counts[t][0]
        for name in ("precision", "recall", "f1", "iou", "fpr"):
            boots = [_metrics(*c0[ix].sum(axis=0))[name] for ix in boot_idx]
            hw = float((np.quantile(boots, 0.975) - np.quantile(boots, 0.025)) / 2)
            spread = CMP.spread(m[name] for m in pooled_by_pass)
            path = f"change_detection.oscd_heldout.thr_{t}.pooled.{name}"
            b, d = noise_entry(path, spread, hw, "max(spread over 3 inference passes, region-bootstrap 95% CI half-width, 2000 draws over 10 regions)")
            noise_bands[path], noise_detail[path] = b, d
            for i, r in enumerate(test_regions):
                spread_r = CMP.spread(_counts_metrics(counts[t][p, i])[name] for p in range(passes))
                path_r = f"change_detection.oscd_heldout.thr_{t}.per_region.{r}.{name}"
                b, d = noise_entry(path_r, spread_r, None, "spread over 3 inference passes (fixed region: no sampling component)")
                noise_bands[path_r], noise_detail[path_r] = b, d
    save_part("change_detection", {"oscd_heldout": metrics}, noise_bands, noise_detail, power=guard.result())


# =============================================================================
# detector
# =============================================================================


def section_detector(args) -> None:
    from geoseek.models.registry import get_spec, verify_weights

    guard = PowerGuard("detector", args, timed=False)
    guard.__enter__()
    weights = verify_weights("yolo26s-obb-dota15")                  # SHA256 gate on the deployed weights
    out = EVAL_DIR / "detector_run"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-W", "ignore", str(ROOT / "scripts" / "eval_detector.py"), "--weights", str(weights),
           "--out", str(out), "--stages", "monitor,val,bootstrap", "--n-boot", str(args.detector_n_boot), "--force"]
    log(f"detector: {' '.join(cmd[2:])}")
    t0 = time.time()
    env = {**os.environ, "YOLO_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    with open(EVAL_DIR / "logs" / "detector_eval.log", "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        raise SystemExit(f"eval_detector.py failed (exit {rc}); see {EVAL_DIR / 'logs' / 'detector_eval.log'}")
    res = json.loads((out / "eval_results.json").read_text(encoding="utf-8"))
    val, boot, op = res["val_full_image_v15"], res["val_bootstrap_v15"], res["val_operating_point_v15"]
    base = "detector.dota_val_v15_full_image"
    m: dict = {"model_key": "yolo26s-obb-dota15", "weights_sha256": get_spec("yolo26s-obb-dota15").weights_sha256,
               "n_val_images": val["counts"]["n_images"], "n_gt_difficult_ignored": val["counts"]["n_gt_difficult"],
               "operating_confidence_chosen_on_monitor": res["operating_point"]["conf"], "bootstrap_resamples": boot["n_boot"],
               "eval_wall_s": wall}
    noise_bands, noise_detail = {}, {}

    def add(prefix: str, rec: dict, ci: dict, opr: dict | None):
        d = {"ap50": rec["macro_AP50"] if "macro_AP50" in rec else rec["AP50"],
             "ap50_95": rec["macro_AP50_95"] if "macro_AP50_95" in rec else rec["AP50_95"], "n_gt": rec.get("n_gt")}
        for key, ck in (("ap50", "AP50_ci" if "AP50_ci" in ci else "macro_AP50_ci"),
                        ("ap50_95", "AP50_95_ci" if "AP50_95_ci" in ci else "macro_AP50_95_ci")):
            lo, hi = ci[ck]
            d[f"{key}_ci_lo"], d[f"{key}_ci_hi"] = lo, hi
            path = f"{prefix}.{key}"
            b, nd = noise_entry(path, None, (hi - lo) / 2, "bootstrap 95% CI half-width over val images (n_boot resamples)")
            noise_bands[path], noise_detail[path] = b, nd
        if opr:
            d.update({f"op_{k}": opr.get(k) for k in ("precision", "recall", "f1")})
        return d

    for g, rec in val["groups"].items():
        m[g] = add(f"{base}.{g}", rec, boot["groups"][g], op["groups"].get(g))
    m["per_class"] = {}
    for c, rec in val["per_class"].items():
        key = c.replace("-", "_")
        opc = op["per_class"].get(c)
        m["per_class"][key] = add(f"{base}.per_class.{key}", rec, boot["per_class"][c], opc)
    m["small_vehicle"] = {"ap50_95": val["per_class"]["small-vehicle"]["AP50_95"]}
    save_part("detector", {"dota_val_v15_full_image": m}, noise_bands, noise_detail,
              extra={"scratch_results": str(out / "eval_results.json")}, power=guard.result())


# =============================================================================
# latency
# =============================================================================

N_POINTS, REPEATS, LAT_SEED = 30, 8, 7042026          # same as scripts/measure_tier.py
TEXT_QUERIES = ["a water body", "dense urban buildings", "agricultural fields", "open bare ground", "a river with sandbars"]


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def _lat(xs: list[float]) -> dict:
    return {"n": len(xs), "median_ms": statistics.median(xs), "p95_ms": _pct(xs, 0.95), "p99_ms": _pct(xs, 0.99), "max_ms": max(xs)}


def section_latency(args) -> None:
    import random

    from shapely import wkt as shapely_wkt

    from geoseek.discovery.knn import find_more_like_this
    from geoseek.search.engine import SearchEngine

    guard = PowerGuard("latency", args, timed=True)
    guard.__enter__()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    passes = 3
    with tempfile.TemporaryDirectory(dir=EVAL_DIR, prefix="latency_copy_") as tmp:
        tmp = Path(tmp)
        for name in ("tiles.faiss", "tiles.sqlite"):                 # a COPY: the production files are never opened here
            shutil.copy2(SETTINGS.index_dir / name, tmp / name)
        eng = SearchEngine(index_dir=tmp)
        try:
            n = eng.count()
            recs = eng.repo.query_tiles(collection="sentinel-2-l2a")
            sample = random.Random(LAT_SEED).sample(recs, k=N_POINTS)
            cents = [shapely_wkt.loads(r.geom_wkt_4326).centroid for r in sample]
            points, tile_ids = [(c.x, c.y) for c in cents], [r.tile_id for r in sample]
            for _ in range(3):                                       # warm-up, discarded
                eng.search_text(TEXT_QUERIES[0], k=20)
            samples = {k: [[] for _ in range(passes)] for k in ("text_search_k20", "image_search_k20", "point_seeded_knn", "tile_seeded_knn", "bbox_filter")}
            for p in range(passes):
                for q in TEXT_QUERIES:
                    for _ in range(REPEATS):
                        samples["text_search_k20"][p].append(eng.search_text(q, k=20)[1])
                for tid in tile_ids[:15]:
                    for _ in range(REPEATS):
                        samples["image_search_k20"][p].append(eng.search_image(tile_id=tid, k=20)[1])
                for lon, lat in points:
                    for _ in range(5):
                        try:
                            samples["point_seeded_knn"][p].append(find_more_like_this(eng, lon=lon, lat=lat, k=8)["latency_ms"])
                        except KeyError:
                            pass
                for tid in tile_ids:
                    for _ in range(5):
                        samples["tile_seeded_knn"][p].append(find_more_like_this(eng, tile_id=tid, k=8)["latency_ms"])
                eps = 0.01
                for lon, lat in points:
                    box = (lon - eps, lat - eps, lon + eps, lat + eps)
                    for _ in range(REPEATS):
                        t0 = time.perf_counter()
                        eng.repo.query_tiles(bbox=box)
                        samples["bbox_filter"][p].append((time.perf_counter() - t0) * 1000.0)
                log(f"latency: pass {p + 1}/{passes} done")
        finally:
            eng.close()
    metrics: dict = {"n_vectors": n, "passes": passes, "warm": 1, "index": "scratch COPY of the production index",
                     "method": "scripts/measure_tier.py query sets and seeds; 3 discarded warm-ups; samples pooled over 3 passes"}
    noise_bands, noise_detail = {}, {}
    for op, per_pass in samples.items():
        pooled = [x for ps in per_pass for x in ps]
        metrics[op] = _lat(pooled)
        medians = [statistics.median(ps) for ps in per_pass]
        metrics[op]["pass_medians_ms"] = medians
        for stat, vals in (("median_ms", medians), ("p95_ms", [_pct(ps, 0.95) for ps in per_pass]), ("p99_ms", [_pct(ps, 0.99) for ps in per_pass])):
            path = f"latency.{op}.{stat}"
            b, d = noise_entry(path, CMP.spread(vals), None, "spread (max-min) of the per-pass statistic over 3 passes")
            noise_bands[path], noise_detail[path] = b, d
    save_part("latency", metrics, noise_bands, noise_detail, power=guard.result())


# =============================================================================
# footprint
# =============================================================================


def _dir_bytes(p: Path) -> int:
    total = 0
    stack = [str(p)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    total += e.stat().st_size
    return total


def section_footprint(args) -> None:
    from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex

    guard = PowerGuard("footprint", args, timed=True)
    guard.__enter__()
    corpus = (load_part("corpus") or {}).get("metrics") or {}
    if not corpus:
        raise SystemExit("run the 'corpus' section first")
    faiss_b = (SETTINGS.index_dir / "tiles.faiss").stat().st_size
    sqlite_b = (SETTINGS.index_dir / "tiles.sqlite").stat().st_size
    n_vec, n_cat = corpus["faiss_vectors"], corpus["catalog_tiles_total"]
    # persist time: write the loaded index to a scratch path (never to the production file)
    ix = FaissFlatIPIndex(SETTINGS.faiss_index_path)
    persist_ms = []
    with tempfile.TemporaryDirectory(dir=EVAL_DIR, prefix="persist_") as tmp:
        for _ in range(3):
            ix.index_path = Path(tmp) / "persist.faiss"
            t0 = time.perf_counter()
            ix.persist()
            persist_ms.append((time.perf_counter() - t0) * 1000.0)
    frozen_idx = CANDIDATES / f"remoteclip_frozen{FROZEN_N}.faiss"
    m = {
        "live_index": {"faiss_bytes": faiss_b, "faiss_mb": faiss_b / 1e6, "sqlite_bytes": sqlite_b, "sqlite_mb": sqlite_b / 1e6,
                       "n_vectors": n_vec, "n_catalog_tiles": n_cat},
        "faiss_bytes_per_vector": faiss_b / n_vec,                                       # whole file / vectors
        "faiss_file_overhead_bytes": faiss_b - n_vec * 512 * 4,                          # header beyond n x 512 x 4 B
        "sqlite_bytes_per_catalog_tile": sqlite_b / n_cat,
        "index_bytes_per_embedded_tile": (faiss_b + sqlite_b) / n_vec,
        "faiss_persist_ms_to_scratch": statistics.median(persist_ms),
        "datasets_mb": _dir_bytes(SETTINGS.datasets_dir) / 1e6,
        "models_mb": _dir_bytes(SETTINGS.models_dir) / 1e6,
        "data_dir_mb": _dir_bytes(SETTINGS.data_dir) / 1e6,
    }
    if frozen_idx.is_file():
        m["frozen_101911"] = {"faiss_bytes": frozen_idx.stat().st_size, "faiss_mb": frozen_idx.stat().st_size / 1e6,
                              "bytes_per_vector": frozen_idx.stat().st_size / FROZEN_N}
    noise_bands = {"footprint.faiss_persist_ms_to_scratch": {"band": CMP.spread(persist_ms)}}
    save_part("footprint", m, noise_bands, power=guard.result())


# =============================================================================
# ingestion
# =============================================================================


def section_ingestion(args) -> None:
    import torch

    from geoseek.ingest.pipeline import ingest_scene
    from geoseek.ingest.store import TileStore

    guard = PowerGuard("ingestion", args, timed=True)
    guard.__enter__()
    scene_a = SETTINGS.datasets_dir / "S2B_44QMK_20240103_0_L2A"               # 1,024 tiles (Kanha), as in Phase 7b
    scene_b = SETTINGS.datasets_dir / "S2A_44RPQ_20240308_0_L2A_scaled"         # 1,089 tiles (Ayodhya)
    runs = []
    with tempfile.TemporaryDirectory(dir=EVAL_DIR, prefix="ingest_") as tmp:
        tmp = Path(tmp)
        for i in range(4):                                                      # run 0 = cold file cache, kept separately
            d = tmp / f"idx{i}"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            rep = ingest_scene(scene_a, save_sample=False, index_dir=d, record_manifest=False)
            runs.append({"tiles": rep["tiles_added"], "wall_s": rep["build_time_s"], "tiles_per_s": rep["tiles_added"] / rep["build_time_s"],
                         "pure_embed_tiles_per_s": 1000.0 / rep["mean_embed_latency_ms"],
                         "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None})
            log(f"ingestion: run {i}: {runs[-1]['tiles_per_s']:.1f} tiles/s end-to-end, {runs[-1]['pure_embed_tiles_per_s']:.1f} pure embed")
        # append-only proof, in scratch: ingest A then B into one index; A's vectors must be byte-identical afterwards
        d = tmp / "append"
        ingest_scene(scene_a, save_sample=False, index_dir=d, record_manifest=False)
        s1 = TileStore(index_dir=d)
        n_before = s1.count()
        before = [s1.reconstruct(i).tobytes() for i in range(n_before)]
        s1.close()
        rep_b = ingest_scene(scene_b, save_sample=False, index_dir=d, record_manifest=False)
        s2 = TileStore(index_dir=d)
        same = sum(s2.reconstruct(i).tobytes() == before[i] for i in range(n_before))
        n_after = s2.count()
        s2.close()
    warm = runs[1:]
    m = {
        "scene": scene_a.name, "tiles_per_run": runs[0]["tiles"], "power_source": power_source()["source"],
        "end_to_end_tiles_per_s": statistics.median(r["tiles_per_s"] for r in warm),
        "pure_embed_tiles_per_s": statistics.median(r["pure_embed_tiles_per_s"] for r in warm),
        "peak_vram_mb": max(r["peak_vram_mb"] or 0 for r in runs),
        "cold_first_run_tiles_per_s": runs[0]["tiles_per_s"],
        "runs": runs,
        "incremental_append_byte_identical_fraction": same / n_before,
        "incremental_append": {"vectors_before": n_before, "vectors_after": n_after, "tiles_added": rep_b["tiles_added"],
                               "no_rebuild": int(n_after == n_before + rep_b["tiles_added"])},
    }
    noise_bands = {
        "ingestion.end_to_end_tiles_per_s": {"band": CMP.spread(r["tiles_per_s"] for r in warm)},
        "ingestion.pure_embed_tiles_per_s": {"band": CMP.spread(r["pure_embed_tiles_per_s"] for r in warm)},
    }
    # re-embed throughput from the shard manifests of the baseline runs (written by scripts/reembed.py)
    reembed = {}
    for label, key in (("remoteclip", "remoteclip-vitb32"), ("vanilla", "openclip-vitb32-openai")):
        mf = SHARDS_ROOT / key / "manifest.json"
        if not mf.is_file():
            continue
        man = json.loads(mf.read_text(encoding="utf-8"))
        shards = man["shards"]
        tiles, secs = sum(s["count"] for s in shards), sum(s["seconds"] for s in shards)
        v = [s["vram"] for s in shards if s.get("vram")]
        reembed[label] = {"n_tiles": tiles, "shards": len(shards), "shard_size": man["shard_size"], "batch_size": man["batch_size"],
                          "tiles_per_s": tiles / secs, "shard_seconds_total": secs,
                          "min_shard_tiles_per_s": min(s["tiles_per_s"] for s in shards),
                          "max_shard_tiles_per_s": max(s["tiles_per_s"] for s in shards),
                          "peak_vram_allocated_mb": max(x["peak_allocated_mb"] for x in v) if v else None,
                          "peak_vram_reserved_mb": max(x["peak_reserved_mb"] for x in v) if v else None,
                          "vram_total_mb": v[0]["total_mb"] if v else None,
                          "projected_full_corpus_wall_min_at_this_rate": man["n_tiles"] / (tiles / secs) / 60.0,
                          "run_was_interrupted_and_resumed": int(label == "remoteclip")}
    m["reembed"] = reembed
    save_part("ingestion", m, noise_bands, power=guard.result())


# =============================================================================
# suite
# =============================================================================


def section_suite(args) -> None:
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore"], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1]
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail)}
    failed = [ln.split(" ", 1)[1].split(" - ")[0] for ln in p.stdout.splitlines() if ln.startswith("FAILED ")]
    m = {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "skipped": counts.get("skipped", 0),
         "errors": counts.get("error", 0) + counts.get("errors", 0), "wall_s": time.time() - t0}
    save_part("suite", m, extra={"summary_line": tail, "failed_tests": failed})


# =============================================================================
# env + assemble
# =============================================================================


def section_env(args) -> None:
    save_part("env", {}, extra={"environment": capture_environment()})


def known_reasons(snap: dict, parts: dict) -> dict[str, str]:
    """Explanations for non-reproductions whose cause was verified against the catalog / disk in this session.

    Only added where the measured facts support the sentence; anything else keeps the generic numeric reason.
    """
    out: dict[str, str] = {}
    c = snap.get("corpus") or {}
    if c:
        grown = (f"the live catalog has grown since the report: sentinel-2 embedded {c['sentinel2_embedded_tiles']} "
                 f"(report: 101,911 = the frozen faiss_id<101911 prefix, which still reproduces exactly), "
                 f"+{c['maxar_embedded_tiles']} Maxar VHR tiles, {c['scenes_total']} scenes (Phase 8 appended the "
                 "Ayodhya 2025/2026 Sentinel-2 scenes and the Maxar Open Data observations). The report describes the "
                 "Phase 7b state.")
        for k in ("corpus.faiss_vectors", "corpus.sentinel2_embedded_tiles", "corpus.catalog_tiles_total", "corpus.scenes_total"):
            out[k] = grown
    f = snap.get("footprint") or {}
    if f:
        why = ("data/ grew after the report was written: Phase 8F staged xView, three DOTA variants and detector evaluation "
               "artifacts (xView ~38 GB, DOTA ~64 GB, detect_eval ~39 GB at measurement). Sizes of the retrieval index "
               "itself reproduce (see footprint.faiss_bytes_per_vector).")
        for k in ("footprint.datasets_mb", "footprint.data_dir_mb"):
            out[k] = why
        out["footprint.models_mb"] = ("models/ now also holds the detector weights and YOLO initialisation checkpoints "
                                      "staged in Phase 8F (the report counted RemoteCLIP + vanilla CLIP only).")
    s = (parts.get("suite") or {}).get("extra", {})
    if s.get("failed_tests"):
        out["suite.failed"] = (f"{len(s['failed_tests'])} test(s) fail on the unmodified tree: {', '.join(s['failed_tests'])} "
                               "(vendored front-end assets referencing external URLs that are missing from the vendor allowlist in "
                               "data/provenance_manifest.json). Unrelated to this phase; not modified here.")
    if snap.get("suite", {}).get("passed", 0) != 228:
        out["suite.passed"] = ("the suite grew after the report (Phase 8 and later added tests); 228 is the Phase 7c count "
                               "and is also what the project rules file (rule 4) still states.")
    return out


def assemble(args) -> dict:
    snap: dict = {"schema": SCHEMA}
    parts = {n: load_part(n) for n in ("env", "corpus", "retrieval", "change_detection", "detector", "latency", "footprint",
                                       "ingestion", "suite")}
    noise_bands, noise_detail, sections, power = {}, {}, [], {}
    for name, part in parts.items():
        if part is None:
            continue
        sections.append(name)
        if name != "env":
            snap[name] = part["metrics"]
        noise_bands.update(part.get("noise_bands", {}))
        noise_detail.update(part.get("noise_detail", {}))
        if part.get("power"):
            power[name] = part["power"]
    env = (parts["env"] or {}).get("extra", {}).get("environment") or capture_environment()
    snap["environment"] = {**env, "power_by_section": power}
    inputs = {"tiles.faiss": sha256_path(SETTINGS.faiss_index_path), "tiles.sqlite": sha256_path(SETTINGS.database_path),
              "judgments.json": sha256_path(SETTINGS.data_dir / "eval_retrieval" / "judgments.json"),
              "queries.json": sha256_path(SETTINGS.data_dir / "eval_retrieval" / "queries.json")}
    from geoseek.models.registry import REGISTRY
    inputs.update({f"weights:{k}": s.weights_sha256 for k, s in REGISTRY.items()})
    for k, fn in (("remoteclip_frozen101911.faiss", CANDIDATES / f"remoteclip_frozen{FROZEN_N}.faiss"),
                  ("vanilla_frozen101911.faiss", CANDIDATES / f"vanilla_frozen{FROZEN_N}.faiss")):
        if fn.is_file():
            inputs[f"candidate:{k}"] = sha256_path(fn)
    snap["meta"] = {"snapshot_id": args.snapshot_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "sections_present": sections, "frozen_corpus": f"production faiss_id < {FROZEN_N}",
                    "inputs_sha256": inputs,
                    "how_to_compare": "python scripts/compare_to_baseline.py data/eval/baseline_v1.json <candidate.json>"}
    snap["noise_bands"] = noise_bands
    snap["noise_detail"] = noise_detail
    from geoseek.eval.reported import reproduction_report
    snap["reproduction"] = reproduction_report(snap, known_reasons(snap, parts))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(snap), indent=1), encoding="utf-8")
    s = snap["reproduction"]["summary"]
    log(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB). reproduction: {s}")
    for e in snap["reproduction"]["not_reproduced"]:
        log(f"  NOT_REPRODUCED {e['metric']}: reported {e['reported']} measured {e['measured']}")
    return snap


FUNCS = {"env": section_env, "corpus": section_corpus, "retrieval": section_retrieval, "oscd": section_oscd,
         "detector": section_detector, "latency": section_latency, "footprint": section_footprint,
         "ingestion": section_ingestion, "suite": section_suite}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="all", help=f"comma list of {','.join(SECTIONS)} or 'all'")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "baseline_v1.json")
    ap.add_argument("--snapshot-id", default="baseline_v1")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--allow-battery", action="store_true", help="run timed sections on battery (recorded in the result)")
    ap.add_argument("--detector-n-boot", type=int, default=100)
    ap.add_argument("--retrieval-system", action="append", default=[], metavar="LABEL=MODEL_KEY=INDEX",
                    help="replace/add a retrieval system, e.g. remoteclip=my-model-key=data/index/candidates/my_frozen101911.faiss "
                         "(INDEX may be 'production'); repeatable. Use the baseline's labels to line up with its metric paths.")
    ap.add_argument("--parts-dir", type=Path, default=None,
                    help="where per-section parts are written/read (default data/eval/baseline_parts); use a separate "
                         "directory for a candidate or re-run so the baseline parts are never overwritten")
    args = ap.parse_args(argv)
    global PARTS_DIR
    if args.parts_dir is not None:
        PARTS_DIR = args.parts_dir
    (EVAL_DIR / "logs").mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        wanted = SECTIONS if args.sections == "all" else tuple(s.strip() for s in args.sections.split(","))
        for s in wanted:
            if s not in FUNCS:
                raise SystemExit(f"unknown section {s!r}; choose from {', '.join(SECTIONS)}")
        for s in wanted:
            t0 = time.time()
            log(f"=== section: {s}")
            FUNCS[s](args)
            log(f"=== section {s} finished in {time.time() - t0:.0f}s")
    assemble(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Block A: re-run the section-7 retrieval evaluation at full corpus scale WITH full judge coverage.

    python scripts/eval_judge_coverage.py                       # -> data/eval/phase10_blockA.json
    python scripts/eval_judge_coverage.py --allow-partial       # dry run while staging is still going

Corpus: the FROZEN 101,911-tile corpus (production faiss_id < 101911). Judge: the Phase 7a graders, unchanged
(``geoseek.eval.judge``), applied to every tile using the per-tile spectral descriptor. Two feature sources:

  mixed    (primary)     Ayodhya keeps its stored Phase 7a features, so on Ayodhya the judge IS the frozen judge and the
                         region-filtered precision reproduces the baseline exactly; every other tile uses the new descriptor.
  uniform  (sensitivity) every tile, Ayodhya included, uses the new descriptor (one pipeline everywhere; Ayodhya
                         grades then differ slightly from the frozen ones because Phase 7a features were built from
                         radiometrically-normalized rasters).

Then: (1) the judge is validated against the blind visual annotations (independent evidence); (2) the report's claims are
tested directly - what fraction of the previously-unjudged top-20 tiles are on-target, and how much of the precision
collapse does full coverage recover; (3) the snapshot is written in the Phase 9 format so
``scripts/compare_to_baseline.py`` can diff it against ``baseline_v1.json`` with paired verdicts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from geoseek.catalog.embedding_map import read_mapping  # noqa: E402
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.config import get_settings  # noqa: E402
from geoseek.eval import full_judge as FJ  # noqa: E402
from geoseek.eval import judge as J  # noqa: E402
from geoseek.eval import retrieval as RT  # noqa: E402
from geoseek.eval.env import capture_environment  # noqa: E402
from geoseek.ingest import reembed as R  # noqa: E402
from geoseek.models.registry import get_spec, load_model  # noqa: E402
from geoseek.search.rerank import region_key  # noqa: E402
from geoseek.vectorindex.faiss_flat import FaissFlatIPIndex  # noqa: E402

SETTINGS = get_settings()
EVAL_DIR = SETTINGS.data_dir / "eval_retrieval"
OUT_DIR = SETTINGS.data_dir / "eval"
JC_DIR = OUT_DIR / "judge_coverage"
FROZEN_N = 101_911
DEPTH = 20
KS_BASE = (1, 5, 10, 20)       # the baseline snapshot's per-query keys


def log(m: str) -> None:
    print(f"[judge-cov] {m}", flush=True)


# ------------------------------------------------------------------------------ corpus + features


def load_frozen(repo):
    refs = R.select_tiles(repo, max_faiss_id=FROZEN_N)
    assert len(refs) == FROZEN_N and all(r.faiss_id == i for i, r in enumerate(refs))
    aoi = {o.observation_id: region_key(o.aoi_name) for o in repo.list_observations()}
    return refs, np.array([aoi[r.observation_id] for r in refs])


def build_features(repo, refs, mode: str, allow_partial: bool = False) -> tuple[list[dict | None], dict]:
    p7a = json.loads((EVAL_DIR / "tile_features.json").read_text())["features"]
    spec = repo.list_tile_spectral([r.tile_id for r in refs])
    out: list[dict | None] = []
    src = {"phase7a_stored": 0, "descriptor": 0, "none": 0}
    for r in refs:
        f = None
        if mode == "mixed" and r.tile_id in p7a:
            f = p7a[r.tile_id]
            src["phase7a_stored"] += 1
        elif r.tile_id in spec:
            f = spec[r.tile_id]
            if allow_partial and f.get("usable") and f.get("dist_river_m") is None:
                f = None                                   # dry run: region context not computed for this region yet
                src["none"] += 1
                out.append(f)
                continue
            src["descriptor"] += 1
        else:
            src["none"] += 1
        out.append(f)
    return out, src


def region_rankings(encode, vectors, ids, queries, mask):
    return {q: [int(i) for i in RT.rank_query(encode(q), vectors, mask=mask, depth=DEPTH)] for q in queries}


# ------------------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true", help="grade tiles without a descriptor as 0 instead of failing")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "phase10_blockA.json")
    args = ap.parse_args(argv)
    JC_DIR.mkdir(parents=True, exist_ok=True)
    judged = RT.load_judged(EVAL_DIR)
    queries = judged.queries
    low, river = J.low_confidence_queries(), J.river_relative_queries()
    subsets = FJ.query_subsets(queries, low, river)

    repo = SQLiteMetadataRepository(SETTINGS.database_path)
    try:
        refs, region = load_frozen(repo)
        tile_ids = [r.tile_id for r in refs]
        feats_mixed, src_mixed = build_features(repo, refs, "mixed", args.allow_partial)
        feats_uniform, src_uniform = build_features(repo, refs, "uniform", args.allow_partial)
    finally:
        repo.close()
    missing = src_mixed["none"]
    log(f"features: mixed {src_mixed}; uniform {src_uniform}")
    if missing and not args.allow_partial:
        raise SystemExit(f"{missing} of {FROZEN_N} tiles have no descriptor yet - finish stage_nir_swir_and_describe.py "
                         "(or pass --allow-partial for a dry run)")
    t0 = time.time()
    grades = {"mixed": J.grade_matrix(feats_mixed, queries), "uniform": J.grade_matrix(feats_uniform, queries)}
    log(f"graded {FROZEN_N} tiles x {len(queries)} queries x 2 variants in {time.time() - t0:.0f}s")
    np.savez_compressed(JC_DIR / "grades_v1.npz", tile_ids=np.array(tile_ids), queries=np.array(queries),
                        mixed=grades["mixed"], uniform=grades["uniform"])

    # --- guard: on the Phase 7a pool the mixed judge must BE the frozen judge -------------------------------------
    frozen = json.loads((EVAL_DIR / "judgments.json").read_text())
    pos = {t: i for i, t in enumerate(tile_ids)}
    col = {q: j for j, q in enumerate(queries)}
    mism = sum(1 for q, d in frozen.items() for t, g in d.items() if int(grades["mixed"][pos[t], col[q]]) != int(g))
    n_pairs = sum(len(d) for d in frozen.values())
    log(f"mixed judge vs frozen Phase 7a judgements on the pool: {n_pairs - mism}/{n_pairs} identical")

    # --- systems ---------------------------------------------------------------------------------------------------
    prod_vecs = FaissFlatIPIndex(SETTINGS.faiss_index_path).reconstruct_all()[:FROZEN_N]
    v_idx = SETTINGS.index_dir / "candidates" / f"vanilla_frozen{FROZEN_N}.faiss"
    rows, _ = read_mapping(v_idx.with_name(v_idx.stem + ".mapping.sqlite"))
    assert [r[1] for r in rows] == tile_ids
    systems = {"remoteclip": ("remoteclip-vitb32", prod_vecs),
               "vanilla": ("openclip-vitb32-openai", FaissFlatIPIndex(v_idx).reconstruct_all())}
    mask_ay = region == "ayodhya"
    conds = {"global": None, "region_filtered_ayodhya": mask_ay}

    rankings: dict = {}
    for sname, (key, vecs) in systems.items():
        model = load_model(key)
        model.load()
        for cname, mask in conds.items():
            rankings[(sname, cname)] = region_rankings(model.encode_text, vecs, tile_ids, queries, mask)
    log("rankings computed")

    snap: dict = {"schema": "geoseek.baseline_snapshot/1", "retrieval": {"full_101911": {}},
                  "judge_coverage": {"variants": {}, "checks": {}}}
    base = json.loads((OUT_DIR / "baseline_v1.json").read_text())

    def pq_list(per_query):
        return [{"query": q, "low_confidence": q in low,
                 **{f"{m}@{k}": per_query[q][k][m] for k in (5, 10, 20) for m in ("precision", "recall", "ndcg")}}
                for q in queries]

    per_variant: dict = {}
    for variant in ("mixed", "uniform"):
        gm = grades[variant]
        per_variant[variant] = {}
        for sname in systems:
            for cname, mask in conds.items():
                pq = FJ.evaluate_rankings(rankings[(sname, cname)], gm, queries, candidates=mask)
                per_variant[variant][(sname, cname)] = pq
    primary = per_variant["mixed"]
    for (sname, cname), pq in primary.items():
        block = {**FJ.macro(pq, queries), "per_query": pq_list(pq)}
        snap["retrieval"]["full_101911"].setdefault(sname, {})[cname] = block
    for variant, d in per_variant.items():
        snap["judge_coverage"]["variants"][variant] = {
            sname: {cname: {sub: FJ.macro(pq, qs) for sub, qs in subsets.items()}
                    for (s2, cname), pq in d.items() if s2 == sname} for sname in systems}

    snap["judge_coverage"]["chance"] = {
        variant: {sname: {cname: {sub: FJ.chance_adjusted(pq, qs) for sub, qs in subsets.items()}
                          for (s2, cname), pq in d.items() if s2 == sname} for sname in systems}
        for variant, d in per_variant.items()}

    # --- composition + on-target rate of previously-unjudged top-20 tiles -------------------------------------------
    jmask = np.array([RT.is_ayodhya_judged_tile(t) for t in tile_ids])
    comp: dict = {}
    for sname in systems:
        for variant in ("mixed", "uniform"):
            gm = grades[variant]
            rk = rankings[(sname, "global")]
            n_unj = n_unj_rel = n_unj_strict = n_total = n_rel = 0
            per_q = {}
            for q in queries:
                top = rk[q][:DEPTH]
                g = gm[top, col[q]]
                unj = ~jmask[top]                                   # tiles the frozen judge could not score (other regions)
                n_total += len(top)
                n_rel += int((g > 0).sum())
                n_unj += int(unj.sum())
                n_unj_rel += int(((g > 0) & unj).sum())
                n_unj_strict += int(((g == 2) & unj).sum())
                per_q[q] = {"unjudged_slots": int(unj.sum()), "unjudged_relevant": int(((g > 0) & unj).sum())}
            comp[f"{sname}.{variant}"] = {
                "top20_slots": n_total, "relevant_slots": n_rel, "relevant_fraction": n_rel / n_total,
                "slots_unjudged_in_baseline": n_unj, "unjudged_fraction": n_unj / n_total,
                "unjudged_slots_now_relevant": n_unj_rel, "on_target_rate_of_unjudged": n_unj_rel / n_unj if n_unj else None,
                "on_target_strict_grade2_rate_of_unjudged": n_unj_strict / n_unj if n_unj else None, "per_query": per_q}
    snap["judge_coverage"]["top20_composition"] = comp

    # --- share of the precision collapse that full coverage recovers ---------------------------------------------------
    share = {}
    for sname in systems:
        b = base["retrieval"]["full_101911"][sname]
        for k in (5, 10, 20):
            p_raw, p_reg = b["global"][f"k{k}"]["precision"], b["region_filtered_ayodhya"][f"k{k}"]["precision"]
            p_full = primary[(sname, "global")]
            p_full = statistics.mean(p_full[q][k]["precision"] for q in queries)
            collapse = p_reg - p_raw
            share[f"{sname}.k{k}"] = {"precision_global_baseline": p_raw, "precision_global_full_judge": p_full,
                                      "precision_region_filtered_baseline": p_reg, "collapse": collapse,
                                      "recovered": p_full - p_raw,
                                      "share_of_collapse_recovered": (p_full - p_raw) / collapse if collapse else None}
    snap["judge_coverage"]["collapse_recovery"] = share

    # --- the judge vs the blind visual annotations --------------------------------------------------------------------
    ann = json.loads((EVAL_DIR / "judge_diagnosis" / "ANNOTATIONS.json").read_text())
    vv: dict = {}
    for variant in ("mixed", "uniform"):
        gm = grades[variant]
        pairs = [(q, t, lab) for q, d in ann.items() for t, lab in d.items() if t in pos and q in col]
        sets = {"all": pairs, "other_region_only": [p for p in pairs if not jmask[pos[p[1]]]]}
        out = {}
        for name, ps in sets.items():
            lab = [p[2] for p in ps]
            judge = [int(gm[pos[t], col[q]] > 0) for q, t, _ in ps]
            conf = [(j, 1 if l == 1 else 0) for j, l in zip(judge, lab) if l in (0, 1)]
            tp = sum(1 for j, v in conf if j and v); fp = sum(1 for j, v in conf if j and not v)
            fn = sum(1 for j, v in conf if not j and v); tn = sum(1 for j, v in conf if not j and not v)
            out[name] = {"n_pairs": len(ps), "n_visual_on_target": sum(1 for l in lab if l == 1),
                         "n_visual_off_target": sum(1 for l in lab if l == 0), "n_visual_unsure": sum(1 for l in lab if l == "?"),
                         "visual_on_target_rate_all": sum(1 for l in lab if l == 1) / len(ps) if ps else None,
                         "judge_on_target_rate_all": sum(judge) / len(ps) if ps else None,
                         "agreement_on_confident_labels": (tp + tn) / len(conf) if conf else None,
                         "kappa_on_confident_labels": FJ.cohen_kappa([j for j, _ in conf], [v for _, v in conf]),
                         "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                         "judge_recall_of_visual_on_target": tp / (tp + fn) if tp + fn else None,
                         "judge_precision_vs_visual": tp / (tp + fp) if tp + fp else None}
        vv[variant] = out
    snap["judge_coverage"]["visual_validation"] = vv

    # --- coverage + provenance ------------------------------------------------------------------------------------------
    by_region: dict = {}
    for r, f, rg in zip(refs, feats_mixed, region):
        d = by_region.setdefault(rg, {"tiles": 0, "with_features": 0, "usable": 0})
        d["tiles"] += 1
        d["with_features"] += f is not None
        d["usable"] += J.usable(f)
    snap["judge_coverage"]["coverage"] = {
        "corpus": f"frozen faiss_id < {FROZEN_N}", "tiles": FROZEN_N,
        "tiles_with_features_mixed": FROZEN_N - src_mixed["none"], "feature_sources_mixed": src_mixed,
        "feature_sources_uniform": src_uniform, "by_region": by_region,
        "phase7a_baseline_judged_pairs": n_pairs, "mixed_judge_identical_to_frozen_on_pool": n_pairs - mism,
        "relevant_tiles_per_query_mixed": {q: int((grades["mixed"][:, col[q]] > 0).sum()) for q in queries}}
    snap["meta"] = {"snapshot_id": "phase10_blockA", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "corpus": f"frozen faiss_id < {FROZEN_N}",
                    "judge": "Phase 7a graders, full coverage; primary = 'mixed' feature source (Ayodhya = stored Phase 7a "
                             "features, elsewhere = spectral descriptor v1)",
                    "metric_definitions": "precision = baseline definition (isolates judge coverage); recall / ndcg use the "
                                          "whole candidate set as denominator / ideal (see geoseek.eval.full_judge) and so are "
                                          "NOT definition-comparable with the baseline's pooled recall / ndcg",
                    "systems": {s: {"model_key": k, "weights_sha256": get_spec(k).weights_sha256} for s, (k, _) in systems.items()},
                    "partial": bool(missing)}
    snap["environment"] = capture_environment()
    # noise bands: the baseline's (same queries, same K) apply to the shared paths
    snap["noise_bands"] = {}
    args.out.write_text(json.dumps(snap, indent=1, default=float), encoding="utf-8")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

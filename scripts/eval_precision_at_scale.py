"""Phase 7b follow-up: is the Tier-3 precision drop real, and what recovers it?

Tier 3 reported recall@20 of the originally-relevant set falling to 10.6% with
a 90.6% "distractor" fraction at 100k tiles. This script separates two things
that number conflates:

  1. JUDGE COVERAGE. The Phase 7a relevance judgements are physical facts about
     3,267 specific Ayodhya tiles (NDVI/NDWI/NDBI/SCL + an Ayodhya river mask).
     At 101,911 tiles the top-K is dominated by tiles from 8 other regions that
     were never judged - and never CAN be with the frozen judge (no NIR/SWIR
     staged for them; 5 of 16 queries are river-relative to Ayodhya's river
     specifically). Scoring those unjudged tiles as "irrelevant" is what drives
     the apparent collapse. We report the top-K composition (judged-relevant /
     judged-0 / unjudged-Ayodhya / unjudged-other-region) so the reader sees
     how much of the "precision loss" is unjudged rather than judged-bad.

  2. THE CORRECTED MEASUREMENT. Restrict retrieval to the Ayodhya sub-corpus -
     the domain where the independent judge is valid (and how an analyst
     actually searches: a sector, not the whole archive). This is a real
     precision/recall/NDCG number again. We chose *restrict the evaluation
     domain* over *recalibrate the judge per region* because the latter needs
     NIR/SWIR re-staging for 8 regions AND the river-relative criteria are
     physically Ayodhya-specific - out of scope for a follow-up. Stated plainly
     in PHASE7B_FOLLOWUP.md.

Then it measures three precision improvements, each alone and combined, all
scored against the SAME frozen judgements:

  a) region metadata pre-filter  (== the corrected measurement)
  b) similarity-score thresholding (threshold chosen on a VALIDATION set - the
     Phase 7a 3,267-corpus rankings - never on the at-scale rankings)
  c) near-duplicate suppression

Writes data/eval_retrieval/precision_at_scale.json and prints the tables.

Usage:  python scripts/eval_precision_at_scale.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import time

import numpy as np

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine
from geoseek.search.rerank import (apply_score_threshold, best_threshold_by_f1,
                                   filter_by_region, region_key, suppress_near_duplicates)

EVAL_DIR = get_settings().data_dir / "eval_retrieval"
KS = (5, 10, 20)
AYODHYA_JUDGED_OBS = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)
TAU_GRID = [round(x, 4) for x in np.arange(0.150, 0.361, 0.005)]
TAU_DUP_PRIMARY = 0.97
TAU_DUP_SENSITIVITY = (0.95, 0.96, 0.97, 0.98, 0.99)


def _is_ayodhya_judged_tile(tid: str) -> bool:
    return any(tid.startswith(o) for o in AYODHYA_JUDGED_OBS)


def dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, judg, k) -> float:
    gains = [judg.get(t, 0) for t in ranked_ids[:k]]
    ideal = sorted(judg.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def score_condition(per_query_ranked: dict[str, list[tuple[str, float]]],
                    judgments: dict[str, dict[str, int]], k: int) -> dict:
    """Macro Recall/Precision/NDCG@k + judged-only precision + mean set size."""
    rec, prec, ndcg, prec_judged, setsize = [], [], [], [], []
    for q, ranked in per_query_ranked.items():
        judg = {t: int(v) for t, v in judgments[q].items()}
        total_rel = sum(1 for v in judg.values() if v > 0)
        topk = ranked[:k]
        ids = [t for t, _ in topk]
        hits = sum(1 for t in ids if judg.get(t, 0) > 0)
        rec.append(hits / total_rel if total_rel else 0.0)
        prec.append(hits / k)                                   # padded-to-k precision (Phase 7a convention)
        n_judged = sum(1 for t in ids if t in judg)
        prec_judged.append(hits / n_judged if n_judged else float("nan"))
        ndcg.append(ndcg_at_k(ids, judg, k))
        setsize.append(len(topk))
    valid_pj = [x for x in prec_judged if not math.isnan(x)]
    return {
        "recall": round(st.mean(rec), 4),
        "precision": round(st.mean(prec), 4),
        "ndcg": round(st.mean(ndcg), 4),
        "precision_over_judged_only": round(st.mean(valid_pj), 4) if valid_pj else None,
        "mean_result_set_size": round(st.mean(setsize), 2),
    }


def topk_composition(per_query_ranked, judgments, k) -> dict:
    """Where do the top-k tiles come from, averaged over queries."""
    buckets = {"judged_relevant": [], "judged_zero": [], "unjudged_ayodhya": [], "unjudged_other_region": []}
    for q, ranked in per_query_ranked.items():
        judg = judgments[q]
        cnt = dict.fromkeys(buckets, 0)
        for tid, _ in ranked[:k]:
            if tid in judg:
                cnt["judged_relevant" if judg[tid] > 0 else "judged_zero"] += 1
            elif _is_ayodhya_judged_tile(tid):
                cnt["unjudged_ayodhya"] += 1
            else:
                cnt["unjudged_other_region"] += 1
        for b in buckets:
            buckets[b].append(cnt[b] / k)
    return {b: round(st.mean(v), 4) for b, v in buckets.items()}


def main() -> None:
    queries = json.loads((EVAL_DIR / "queries.json").read_text())
    judgments = json.loads((EVAL_DIR / "judgments.json").read_text())
    pools = json.loads((EVAL_DIR / "pools.json").read_text())
    phase7a = json.loads((EVAL_DIR / "report.json").read_text())

    eng = SearchEngine()
    n_vectors = eng.count()
    print(f"[prec@scale] index: {n_vectors} vectors  (judge calibrated at 3,267 Ayodhya tiles)")

    # ---- region map: faiss_id -> region, tile_id -> region -------------------
    aoi_by_obs = {o.observation_id: region_key(o.aoi_name) for o in eng.repo.list_observations()}
    region_of_tile: dict[str, str] = {}
    for fid, row in eng._rows.items():
        region_of_tile[row["tile_id"]] = aoi_by_obs.get(row["observation_id"], "unknown")
    id_to_tile = {fid: row["tile_id"] for fid, row in eng._rows.items()}

    def region_of(tid: str) -> str:
        return region_of_tile.get(tid, "unknown")

    # ---- full ranking per query (direct index scan, like _rank_and_filter) --
    full_ranked: dict[str, list[tuple[str, float]]] = {}
    t0 = time.time()
    for q in queries:
        vec = eng._encode_text(q)
        scores, ids = eng.vector_index.search(vec, n_vectors)
        rk = [(id_to_tile[int(i)], float(s)) for i, s in zip(ids, scores)
              if int(i) >= 0 and int(i) in id_to_tile]
        full_ranked[q] = rk
    print(f"[prec@scale] ranked {len(queries)} queries x {n_vectors} vectors in {time.time()-t0:.1f}s")

    # ---- vector cache for near-dup suppression (lazy: fetch on first use) ---
    tile_row = {row["tile_id"]: fid for fid, row in eng._rows.items()}
    vec_cache: dict[str, np.ndarray] = {}

    def vec_of(tid: str) -> np.ndarray:
        v = vec_cache.get(tid)
        if v is None:
            v = eng.vector_index.get_vector(tile_row[tid])
            vec_cache[tid] = v
        return v

    # ---- VALIDATION: choose the score threshold on the Phase 7a 3,267 rankings
    val_rankings = []
    for q in queries:
        rc_ranked = pools[q]["remoteclip_ranked"]
        rc_score = pools[q]["remoteclip_score"]
        ranked = [(t, float(rc_score[t])) for t in rc_ranked if t in rc_score]
        val_rankings.append((ranked, {t: int(v) for t, v in judgments[q].items()}))
    tau_star, tau_curve = best_threshold_by_f1(val_rankings, TAU_GRID, k=20)
    print(f"[prec@scale] threshold chosen on validation (Phase 7a rankings): tau* = {tau_star}")

    # ---- near-dup sensitivity on the global top-K --------------------------
    dedup_sens = {}
    for td in TAU_DUP_SENSITIVITY:
        total_sup = 0
        for q in queries:
            _, sup = suppress_near_duplicates(full_ranked[q][:100], vec_of, tau_dup=td)
            total_sup += len(sup)
        dedup_sens[td] = total_sup
    print(f"[prec@scale] near-dup suppressed in global top-100 (summed over queries): {dedup_sens}")

    # ---- build each condition's per-query ranked list ----------------------
    AY = {"ayodhya"}

    def cond_global(q):
        return full_ranked[q]

    def cond_region(q):
        return filter_by_region(full_ranked[q], region_of, AY)

    def cond_threshold(base):
        return lambda q: apply_score_threshold(base(q), tau_star)

    def cond_dedup(base):
        def f(q):
            kept, _ = suppress_near_duplicates(base(q)[:200], vec_of, tau_dup=TAU_DUP_PRIMARY)
            return kept
        return f

    conditions = {
        "1_global_baseline": cond_global,
        "2a_region_filter": cond_region,
        "2b_score_threshold": cond_threshold(cond_global),
        "2c_near_dup_suppression": cond_dedup(cond_global),
        "3_region+threshold": cond_threshold(cond_region),
        "3_region+dedup": cond_dedup(cond_region),
        "3_region+threshold+dedup": lambda q: cond_dedup(cond_threshold(cond_region))(q),
    }

    results: dict = {}
    for name, fn in conditions.items():
        per_q = {q: fn(q) for q in queries}
        results[name] = {
            "by_k": {k: score_condition(per_q, judgments, k) for k in KS},
            "topk_composition_k20": topk_composition(per_q, judgments, 20),
        }
        # suppression / threshold bookkeeping
        if "dedup" in name or name.endswith("suppression"):
            sup_total = sum(len(suppress_near_duplicates(
                (cond_region(q) if "region" in name else cond_global(q))[:200], vec_of,
                tau_dup=TAU_DUP_PRIMARY)[1]) for q in queries)
            results[name]["near_dups_suppressed_total"] = sup_total

    # ---- assemble report --------------------------------------------------
    p7a = phase7a["macro_averaged_metrics"]["remoteclip"]
    report = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_vectors": n_vectors,
        "n_vectors_at_original_judgement": 3267,
        "k_values": list(KS),
        "part1_diagnosis": {
            "question": "Is the Tier-3 precision drop real retrieval failure, or judge coverage?",
            "tier3_original_numbers": {
                "recall_at_20_of_relevant": 0.106, "distractor_frac_at_20": 0.906,
                "source": "data/eval_retrieval/retrieval_at_scale_tier3.json",
            },
            "global_top20_composition": results["1_global_baseline"]["topk_composition_k20"],
            "reading": (
                "Of the global top-20, the 'judged_zero' fraction is the only part that is retrieval "
                "putting a KNOWN-bad Ayodhya tile high; 'unjudged_other_region' is the bulk and is "
                "unjudged, not judged-bad. The corrected measurement (2a) restricts to the judgeable "
                "Ayodhya sub-corpus."
            ),
            "correction_chosen": "restrict evaluation domain to Ayodhya (judge is valid there); "
                                 "not per-region recalibration (needs NIR/SWIR re-staging for 8 regions "
                                 "+ river criteria are Ayodhya-specific).",
        },
        "phase7a_reference_remoteclip": {str(k): p7a[str(k)] for k in KS},
        "validation": {
            "threshold_selection": "macro-F1@20 over Phase 7a 3,267-corpus RemoteCLIP rankings "
                                   "(disjoint from the at-scale rankings under test)",
            "tau_star": tau_star,
            "tau_curve": tau_curve,
            "near_dup_tau_primary": TAU_DUP_PRIMARY,
            "near_dup_sensitivity_suppressed_in_global_top100": dedup_sens,
        },
        "conditions": results,
    }
    (EVAL_DIR / "precision_at_scale.json").write_text(json.dumps(report, indent=2))

    # ---- print ----------------------------------------------------------
    def row(name, d):
        return (f"{name:32s} " + "  ".join(
            f"K{k}: R{d['by_k'][k]['recall']:.3f} P{d['by_k'][k]['precision']:.3f} "
            f"Pj{(d['by_k'][k]['precision_over_judged_only'] or float('nan')):.3f} "
            f"N{d['by_k'][k]['ndcg']:.3f} |sz{d['by_k'][k]['mean_result_set_size']:.1f}"
            for k in KS))

    print("\n" + "=" * 120)
    print("PRECISION AT SCALE  (R=recall  P=precision@K padded  Pj=precision over judged-only  "
          "N=NDCG  sz=mean result-set size)")
    print("=" * 120)
    print(f"{'Phase 7a (3,267 corpus)':32s} " + "  ".join(
        f"K{k}: R{p7a[str(k)]['recall']:.3f} P{p7a[str(k)]['precision']:.3f} "
        f"Pj{'  -  '} N{p7a[str(k)]['ndcg']:.3f} |sz{float(k):.1f}" for k in KS))
    print("-" * 120)
    for name, d in results.items():
        print(row(name, d))
    print("\ntop-20 composition (global baseline):", results["1_global_baseline"]["topk_composition_k20"])
    print("top-20 composition (region filter)  :", results["2a_region_filter"]["topk_composition_k20"])
    print(f"\nwrote {EVAL_DIR / 'precision_at_scale.json'}")
    eng.close()


if __name__ == "__main__":
    main()

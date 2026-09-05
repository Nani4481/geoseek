"""Phase 7a Step B (PS 2.3): retrieval evaluation - scoring phase.

Reads data/eval_retrieval/{pools,judgments,judgments_rationale,latency,queries}.json
and computes Recall@K, Precision@K and NDCG@K for K in {1,5,10,20} for
RemoteCLIP and the vanilla CLIP control on the IDENTICAL judgements, plus the
query-latency distribution (cold + warm, median/p95/p99).

Writes data/eval_retrieval/report.json, records the methodology into the
provenance manifest (key ``retrieval_evaluation``), and prints the acceptance
tables.

Usage:  python scripts/eval_retrieval_score.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from geoseek.config import get_settings
from geoseek.staging.manifest import record_analysis_section

KS = (1, 5, 10, 20)
OUT_DIR_NAME = "eval_retrieval"


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], judgments: dict[str, int], k: int) -> float:
    gains = [judgments.get(tid, 0) for tid in ranked_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked_ids, judgments, k, total_relevant) -> float:
    if total_relevant == 0:
        return 0.0
    hit = sum(1 for tid in ranked_ids[:k] if judgments.get(tid, 0) > 0)
    return hit / total_relevant


def precision_at_k(ranked_ids, judgments, k) -> float:
    return sum(1 for tid in ranked_ids[:k] if judgments.get(tid, 0) > 0) / k


def latency_stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "n": n,
        "median_ms": round(statistics.median(xs), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "min_ms": round(xs[0], 2),
        "max_ms": round(xs[-1], 2),
    }


def _macro(system_scores, systems, ks):
    macro = {}
    for system in systems:
        macro[system] = {}
        for k in ks:
            macro[system][k] = {
                m: round(statistics.mean(system_scores[system][k][m]), 4)
                for m in ("recall", "precision", "ndcg")
            }
    return macro


def main() -> None:
    out_dir = get_settings().data_dir / OUT_DIR_NAME
    pools = json.loads((out_dir / "pools.json").read_text())
    queries = json.loads((out_dir / "queries.json").read_text())
    latency = json.loads((out_dir / "latency.json").read_text())
    rationale = json.loads((out_dir / "judgments_rationale.json").read_text())
    low_conf = set(rationale.get("low_confidence_queries", []))

    judgments_path = out_dir / "judgments.json"
    if not judgments_path.is_file():
        raise SystemExit("judgments.json not found - run scripts/eval_retrieval_judge.py first.")
    judgments_all = json.loads(judgments_path.read_text())

    def fresh_scores(qs):
        return {s: {k: {"recall": [], "precision": [], "ndcg": []} for k in KS}
                for s in ("remoteclip", "vanilla")}

    all_scores = fresh_scores(queries)
    hi_scores = fresh_scores([q for q in queries if q not in low_conf])
    per_query_rows = []

    for query in queries:
        pool = pools[query]
        judg_raw = judgments_all[query]
        missing = [tid for tid, v in judg_raw.items() if v is None]
        if missing:
            raise SystemExit(f"query {query!r} has {len(missing)} unjudged pool tiles")
        judgments = {tid: int(v) for tid, v in judg_raw.items()}
        total_relevant = sum(1 for v in judgments.values() if v > 0)

        row = {
            "query": query,
            "low_confidence": query in low_conf,
            "pool_size": len(judgments),
            "total_relevant_in_pool": total_relevant,
            "grade2_in_pool": sum(1 for v in judgments.values() if v == 2),
            "relevant_from_random_only": sum(
                1 for tid in pool["random_sample_tile_ids"]
                if judgments.get(tid, 0) > 0
                and tid not in set(pool["remoteclip_ranked"]) | set(pool["vanilla_ranked"])
            ),
        }
        for system, ranked_key in (("remoteclip", "remoteclip_ranked"), ("vanilla", "vanilla_ranked")):
            ranked = pool[ranked_key]
            for k in KS:
                r = recall_at_k(ranked, judgments, k, total_relevant)
                p = precision_at_k(ranked, judgments, k)
                nd = ndcg_at_k(ranked, judgments, k)
                for bucket in (all_scores,) + ((hi_scores,) if query not in low_conf else ()):
                    bucket[system][k]["recall"].append(r)
                    bucket[system][k]["precision"].append(p)
                    bucket[system][k]["ndcg"].append(nd)
                row[f"{system}_recall@{k}"] = round(r, 4)
                row[f"{system}_precision@{k}"] = round(p, 4)
                row[f"{system}_ndcg@{k}"] = round(nd, 4)
        per_query_rows.append(row)

    macro_all = _macro(all_scores, ("remoteclip", "vanilla"), KS)
    macro_hi = _macro(hi_scores, ("remoteclip", "vanilla"), KS)
    cold_stats = latency_stats(latency["cold_ms"])
    warm_stats = latency_stats(latency["warm_ms"])

    pool_comp = {
        "queries": len(queries),
        "mean_pool_size": round(statistics.mean(len(judgments_all[q]) for q in queries), 1),
        "mean_rc_va_overlap": round(statistics.mean(
            pools[q]["pool_composition"]["n_overlap_rc_va"] for q in queries), 2),
        "random_sample_per_query": pools[queries[0]]["pool_composition"]["n_random"],
        "total_relevant_only_found_via_random": sum(r["relevant_from_random_only"] for r in per_query_rows),
    }

    report = {
        "num_queries": len(queries),
        "queries": queries,
        "low_confidence_queries": sorted(low_conf),
        "k_values": list(KS),
        "macro_averaged_metrics": macro_all,
        "macro_averaged_metrics_excluding_low_confidence": macro_hi,
        "pool_composition": pool_comp,
        "per_query": per_query_rows,
        "latency_ms": {"cold": cold_stats, "warm": warm_stats, "method": latency.get("method")},
        "methodology": rationale["methodology"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # ---- provenance manifest -------------------------------------------------
    record_analysis_section("retrieval_evaluation", {
        "problem_statement": "PS 2.3 - semantic retrieval quality",
        "corpus_tiles": 3267,
        "num_queries": len(queries),
        "queries_include_ps_example_phrasings": [
            "newly built structures near a river", "settlement along a riverbank",
        ],
        "k_values": list(KS),
        "systems": {
            "remoteclip": "production SearchEngine (RemoteCLIP ViT-B/32, FAISS IP)",
            "vanilla_clip": "OpenCLIP ViT-B/32 laion2b, force_quick_gelu=True",
        },
        "judgement_signal": rationale["methodology"]["judge_signal"],
        "circularity_avoidance": rationale["methodology"]["circularity_avoidance"],
        "pool_definition": rationale["methodology"]["pool_definition"],
        "pool_sampling": rationale["methodology"]["pool_sampling"],
        "limitations": rationale["methodology"]["limitations"],
        "artifacts": {
            "queries": "data/eval_retrieval/queries.json",
            "pools": "data/eval_retrieval/pools.json",
            "tile_features": "data/eval_retrieval/tile_features.json",
            "judgments": "data/eval_retrieval/judgments.json",
            "judgments_rationale": "data/eval_retrieval/judgments_rationale.json",
            "report": "data/eval_retrieval/report.json",
            "latency": "data/eval_retrieval/latency.json",
            "contact_sheets": "data/eval_retrieval/contact_sheets/",
        },
        "headline": {
            "remoteclip_ndcg@10": macro_all["remoteclip"][10]["ndcg"],
            "vanilla_ndcg@10": macro_all["vanilla"][10]["ndcg"],
            "remoteclip_recall@20": macro_all["remoteclip"][20]["recall"],
            "vanilla_recall@20": macro_all["vanilla"][20]["recall"],
            "query_latency_warm_median_ms": warm_stats["median_ms"],
            "query_latency_warm_p99_ms": warm_stats["p99_ms"],
            "query_latency_cold_p99_ms": cold_stats["p99_ms"],
        },
    })

    # ---- print ------------------------------------------------------------
    def _table(macro, title):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        hdr = (f"{'K':>3} | {'RC R@K':>8} {'RC P@K':>8} {'RC NDCG':>8} | "
               f"{'VA R@K':>8} {'VA P@K':>8} {'VA NDCG':>8} | {'dNDCG':>7}")
        print(hdr)
        print("-" * len(hdr))
        for k in KS:
            rc, va = macro["remoteclip"][k], macro["vanilla"][k]
            print(f"{k:>3} | {rc['recall']:>8.4f} {rc['precision']:>8.4f} {rc['ndcg']:>8.4f} | "
                  f"{va['recall']:>8.4f} {va['precision']:>8.4f} {va['ndcg']:>8.4f} | "
                  f"{rc['ndcg'] - va['ndcg']:>+7.4f}")

    _table(macro_all, f"RETRIEVAL EVALUATION - all {len(queries)} queries, macro-averaged "
                      f"(RC=RemoteCLIP, VA=vanilla CLIP)")
    _table(macro_hi, f"... excluding {len(low_conf)} low-confidence road/bridge queries "
                     f"({len(queries) - len(low_conf)} queries)")

    print(f"\nPool composition: {pool_comp}")
    print(f"\nQuery latency (search_text, k=20, ms):")
    print(f"  cold  (fresh process, no pre-warm) : n={cold_stats['n']} median={cold_stats['median_ms']} "
          f"p95={cold_stats['p95_ms']} p99={cold_stats['p99_ms']} max={cold_stats['max_ms']}")
    print(f"  warm  (pre-warmed, steady state)   : n={warm_stats['n']} median={warm_stats['median_ms']} "
          f"p95={warm_stats['p95_ms']} p99={warm_stats['p99_ms']} max={warm_stats['max_ms']}")
    print(f"\nFull report -> {out_dir / 'report.json'};  methodology -> manifest key 'retrieval_evaluation'")


if __name__ == "__main__":
    main()

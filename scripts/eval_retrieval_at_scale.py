"""Phase 7b Tier 3: does retrieval quality hold as the corpus grows?

Reuses the Phase 7a retrieval evaluation's FROZEN, independently-judged ground
truth (data/eval_retrieval/{queries,judgments,pools}.json - see
scripts/eval_retrieval_judge.py) completely unchanged: those judgements are
physical facts about specific tiles' NDVI/NDWI/NDBI/SCL content, and remain
true regardless of how many more tiles from other AOIs now sit in the index.

What changes at scale is what gets re-measured: at 3267 tiles, RemoteCLIP's
top-20 for a query was drawn from a small, Ayodhya-only pool. At ~100k tiles,
the SAME query's top-20 is drawn from the full, much larger, geographically
diverse corpus - competing against ~30x more distractors. This script:

  1. Re-runs engine.search_text(query, k) against the CURRENT (grown) index
     for the same 16 queries, at k=20/100/500.
  2. Reports RECALL of the original judged-relevant set at each k (of the
     tiles graded >0 for this query in Phase 7a, how many still surface?) -
     the direct, honest answer to "do more distractors push out the true
     positives", using zero new judging assumptions.
  3. Reports the DISTRACTOR INTRUSION RATE: what fraction of the new top-20
     are tiles from regions outside the original 3-date Ayodhya stack.

NOT attempted here: judging the relevance of brand-new (non-Ayodhya) tiles
that now appear in the top-K. The Phase 7a judge's region-agnostic criteria
(built_frac/ndvi/edge_density - dense_urban, agri, bare, residential, cropland
boundaries, irrigated, paved/dirt road) need only NDVI/NDWI/NDBI, but the
river-relative criteria (water body, river+sandbars, bridge, riverside
construction/vegetation) are computed against an NDWI-derived mask of the
Ayodhya river specifically - meaningless for a Dehradun or Kerala tile. More
fundamentally, NDVI/NDWI/NDBI require the NIR/SWIR bands, which Phase 7b
deliberately did NOT stage for the new regions (out of scope for a search/
index scalability test - see PHASE7B.md). So new tiles are treated as
"unjudged", not "irrelevant"; this script measures recall of known-good
tiles and distractor pressure, not a from-scratch precision@K over new
regions - a disclosed, deliberate scope boundary, not a hidden gap.

Usage: python scripts/eval_retrieval_at_scale.py --tier tier3
"""

from __future__ import annotations

import argparse
import json
import time

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine

EVAL_DIR = get_settings().data_dir / "eval_retrieval"
K_LEVELS = (20, 100, 500)
ORIGINAL_AYODHYA_OBS = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)


def _is_ayodhya_tile(tile_id: str) -> bool:
    return any(tile_id.startswith(obs) for obs in ORIGINAL_AYODHYA_OBS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    args = ap.parse_args()

    queries = json.loads((EVAL_DIR / "queries.json").read_text())
    judgments = json.loads((EVAL_DIR / "judgments.json").read_text())
    original_pools = json.loads((EVAL_DIR / "pools.json").read_text())

    eng = SearchEngine()
    n_vectors = eng.count()
    print(f"[scale-eval] current index: {n_vectors} vectors "
          f"(Phase 7a judged this ground truth at 3267)")

    per_query = []
    for q in queries:
        original_relevant = {tid for tid, g in judgments[q].items() if g > 0}
        original_top20 = original_pools[q]["remoteclip_ranked"]

        row = {"query": q, "n_original_relevant": len(original_relevant),
               "original_top20_still_valid_ground_truth": True}

        for k in K_LEVELS:
            results, latency_ms = eng.search_text(q, k=k)
            new_top_ids = [r.tile_id for r in results]
            found = original_relevant & set(new_top_ids)
            n_distractor = sum(1 for tid in new_top_ids if not _is_ayodhya_tile(tid))
            row[f"recall_at_{k}"] = round(len(found) / len(original_relevant), 4) if original_relevant else None
            row[f"n_found_at_{k}"] = len(found)
            row[f"distractor_frac_at_{k}"] = round(n_distractor / len(new_top_ids), 4) if new_top_ids else None
            row[f"latency_ms_at_{k}"] = round(latency_ms, 2)
            if k == 20:
                row["top20_rank_overlap_with_original"] = len(set(new_top_ids) & set(original_top20))

        per_query.append(row)
        print(f"[scale-eval] {q!r}: recall@20={row['recall_at_20']} "
              f"distractor_frac@20={row['distractor_frac_at_20']} "
              f"recall@500={row['recall_at_500']}")

    eng.close()

    macro = {}
    for k in K_LEVELS:
        recalls = [r[f"recall_at_{k}"] for r in per_query if r[f"recall_at_{k}"] is not None]
        distractors = [r[f"distractor_frac_at_{k}"] for r in per_query]
        macro[f"mean_recall_at_{k}"] = round(sum(recalls) / len(recalls), 4) if recalls else None
        macro[f"mean_distractor_frac_at_{k}"] = round(sum(distractors) / len(distractors), 4)

    report = {
        "tier": args.tier, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_vectors_at_measurement": n_vectors,
        "n_vectors_at_original_judgement": 3267,
        "methodology": (
            "Ground truth (judgments.json) is the FROZEN Phase 7a independent spectral "
            "judging, unchanged. Recall@K = fraction of the originally-judged-relevant tiles "
            "for a query that still appear in the top-K of a FRESH search_text() call against "
            "the current, grown index. Distractor_frac@K = fraction of the new top-K that are "
            "NOT from the original 3-date Ayodhya stack. New (non-Ayodhya) tiles are not judged "
            "for relevance here (needs NIR/SWIR indices not staged for the new regions - a "
            "disclosed scope boundary, not an oversight)."
        ),
        "macro": macro,
        "per_query": per_query,
    }
    out_path = EVAL_DIR / f"retrieval_at_scale_{args.tier}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[scale-eval] macro: {json.dumps(macro, indent=2)}")
    print(f"[scale-eval] wrote {out_path}")


if __name__ == "__main__":
    main()

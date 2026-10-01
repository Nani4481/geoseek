"""Retrieval evaluation against the FROZEN Phase 7a judgements, re-run live for any embedding system.

Everything here is a pure function over (text encoder, corpus vectors, tile ids): the same code
scores RemoteCLIP, the vanilla CLIP control, or any candidate model, so a candidate snapshot is
produced exactly like the baseline one. Rankings are computed fresh (exact inner product over the
unit-norm vectors == the FlatIP index); nothing is read from the stored ``pools.json`` rankings
except as an explicit comparison diagnostic.

Metric formulas are those of ``scripts/eval_retrieval_score.py`` / ``eval_precision_at_scale.py``
(tests pin the equivalence):  Recall@K = hits / (relevant judged for the query),
Precision@K = hits / K (padded), NDCG@K with the ideal ordering taken from the query's judgements;
macro-averaged over queries with ``statistics.mean`` (exact, so the result is independent of the
order the queries are visited in).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

KS = (1, 5, 10, 20)
DEPTH = max(KS)
AYODHYA_JUDGED_OBS = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)
METRICS = ("recall", "precision", "ndcg")


def is_ayodhya_judged_tile(tile_id: str) -> bool:
    return any(tile_id.startswith(o) for o in AYODHYA_JUDGED_OBS)


def dcg(gains: Sequence[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    gains = [judgments.get(t, 0) for t in ranked_ids[:k]]
    idcg = dcg(sorted(judgments.values(), reverse=True)[:k])
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    total = sum(1 for v in judgments.values() if v > 0)
    return sum(1 for t in ranked_ids[:k] if judgments.get(t, 0) > 0) / total if total else 0.0


def precision_at_k(ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int) -> float:
    return sum(1 for t in ranked_ids[:k] if judgments.get(t, 0) > 0) / k


@dataclass(frozen=True)
class Judged:
    queries: list[str]
    judgments: dict[str, dict[str, int]]
    low_confidence: frozenset[str]


def load_judged(eval_dir: Path) -> Judged:
    queries = json.loads((eval_dir / "queries.json").read_text(encoding="utf-8"))
    raw = json.loads((eval_dir / "judgments.json").read_text(encoding="utf-8"))
    rationale = json.loads((eval_dir / "judgments_rationale.json").read_text(encoding="utf-8"))
    judgments = {}
    for q in queries:
        if any(v is None for v in raw[q].values()):
            raise ValueError(f"query {q!r} has unjudged pool tiles")
        judgments[q] = {t: int(v) for t, v in raw[q].items()}
    return Judged(list(queries), judgments, frozenset(rationale.get("low_confidence_queries", [])))


def rank_indices(scores: np.ndarray, depth: int | None = None) -> np.ndarray:
    """Indices by descending score, ties broken by ascending index (deterministic)."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:depth] if depth else order


def rank_query(query_vec: np.ndarray, vectors: np.ndarray, *, mask: np.ndarray | None = None,
               depth: int = DEPTH) -> np.ndarray:
    """Top-``depth`` corpus row indices for one query, optionally restricted to ``mask`` rows."""
    scores = vectors @ np.asarray(query_vec, dtype=np.float32)
    if mask is None:
        return rank_indices(scores, depth)
    cand = np.flatnonzero(mask)
    return cand[rank_indices(scores[cand], depth)]


def macro(per_query: Mapping[str, Mapping[int, Mapping[str, float]]], queries: Sequence[str]) -> dict:
    return {f"k{k}": {m: statistics.mean(per_query[q][k][m] for q in queries) for m in METRICS} for k in KS}


def topk_composition(ranked: Mapping[str, Sequence[str]], judged: Judged, k: int = DEPTH) -> dict:
    """Where the top-k comes from, averaged over queries (judged relevant / judged 0 / unjudged Ayodhya / unjudged elsewhere)."""
    keys = ("judged_relevant", "judged_zero", "unjudged_ayodhya", "unjudged_other_region")
    acc = {b: [] for b in keys}
    for q in judged.queries:
        cnt = dict.fromkeys(keys, 0)
        for t in ranked[q][:k]:
            j = judged.judgments[q]
            if t in j:
                cnt["judged_relevant" if j[t] > 0 else "judged_zero"] += 1
            elif is_ayodhya_judged_tile(t):
                cnt["unjudged_ayodhya"] += 1
            else:
                cnt["unjudged_other_region"] += 1
        for b in keys:
            acc[b].append(cnt[b] / k)
    return {b: statistics.mean(v) for b, v in acc.items()}


def evaluate_system(encode_text: Callable[[str], np.ndarray], vectors: np.ndarray, tile_ids: Sequence[str],
                    judged: Judged, *, order: Sequence[str] | None = None, mask: np.ndarray | None = None) -> dict:
    """Run every query (in ``order``) through ``encode_text`` and rank the corpus; score vs the frozen judgements.

    Returns ``{"ranked": {q: [tile_id,...]}, "scores": {q: {k: {metric: v}}}}`` (``ranked`` has DEPTH entries).
    """
    ranked: dict[str, list[str]] = {}
    scores: dict[str, dict[int, dict[str, float]]] = {}
    for q in (order if order is not None else judged.queries):
        idx = rank_query(encode_text(q), vectors, mask=mask)
        ids = [tile_ids[i] for i in idx]
        ranked[q] = ids
        j = judged.judgments[q]
        scores[q] = {k: {"recall": recall_at_k(ids, j, k), "precision": precision_at_k(ids, j, k),
                         "ndcg": ndcg_at_k(ids, j, k)} for k in KS}
    return {"ranked": ranked, "scores": scores}


def summarize_system(result: Mapping, judged: Judged) -> dict:
    """The snapshot block for one system on one corpus condition."""
    hi = [q for q in judged.queries if q not in judged.low_confidence]
    per_query = [{"query": q, "low_confidence": q in judged.low_confidence,
                  **{f"{m}@{k}": result["scores"][q][k][m] for k in KS for m in METRICS}} for q in judged.queries]
    return {
        "all": macro(result["scores"], judged.queries),
        "excluding_low_confidence": macro(result["scores"], hi),
        "n_queries": len(judged.queries),
        "n_queries_excluding_low_confidence": len(hi),
        "topk_composition_k20": topk_composition(result["ranked"], judged),
        "per_query": per_query,                       # a list: carried for bootstrap/inspection, never diffed
    }


def per_query_vector(result: Mapping, judged: Judged, k: int, metric: str, *, exclude_low: bool = False) -> list[float]:
    qs = [q for q in judged.queries if not (exclude_low and q in judged.low_confidence)]
    return [result["scores"][q][k][metric] for q in qs]

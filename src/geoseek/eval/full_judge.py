"""Retrieval metrics when EVERY tile of the candidate set has a grade (full judge coverage).

Phase 7a judged only a pool per query (top-20 of both systems + 15 random tiles), so Recall and the NDCG ideal were
relative to that pool. With a grade for every tile the definitions become the textbook ones, and they are NOT the same
numbers as the pooled ones - so this module says exactly what each metric means:

* ``precision``  = hits / K, hits = tiles in the top-K with grade > 0. SAME definition as the baseline; the only thing
  that changes under full coverage is that no relevant tile is scored as irrelevant merely because it was never judged.
  This is the number that isolates the judge-coverage effect.
* ``ndcg``       = DCG@K / ideal DCG@K, where the ideal is the best K grades in the whole CANDIDATE SET (the pooled
  baseline took the ideal from the ~55-tile pool, a much weaker bound).
* ``recall``     = hits / (number of relevant tiles in the candidate set). Over a 100k corpus a query can have tens of
  thousands of relevant tiles, so this is bounded by K / n_relevant and is tiny by construction.
* ``recall_capped`` = hits / min(K, n_relevant): "of the best achievable K, how many did it return" - the readable form
  of recall at this scale.
* ``precision_strict`` = share of the top-K with grade 2 (clear match).
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

import numpy as np

KS = (5, 10, 20)
METRICS = ("precision", "recall", "recall_capped", "ndcg", "precision_strict")


def _dcg(gains: Sequence[float]) -> float:
    return sum(float(g) / math.log2(i + 2) for i, g in enumerate(gains))


def query_metrics(ranked: Sequence[int], grades: np.ndarray, k: int, candidates: np.ndarray | None = None) -> dict:
    """Metrics of one query's ranking. ``grades`` is the full-corpus grade vector for the query; ``candidates`` an
    optional boolean mask of the rows the ranking was allowed to draw from (region filter etc.): the ideal and the
    recall denominator are taken over exactly that set."""
    top = np.asarray(ranked[:k], dtype=int)
    gains = grades[top]
    pool = grades if candidates is None else grades[candidates]
    n_rel = int((pool > 0).sum())
    hits = int((gains > 0).sum())
    ideal = np.sort(pool)[::-1][:k]
    idcg = _dcg(ideal)
    return {
        "precision": hits / k,
        "recall": hits / n_rel if n_rel else 0.0,
        "recall_capped": hits / min(k, n_rel) if n_rel else 0.0,
        "ndcg": _dcg(gains) / idcg if idcg > 0 else 0.0,
        "precision_strict": int((gains == 2).sum()) / k,
        "n_relevant_in_candidates": n_rel,
        "n_candidates": int(pool.size),
    }


def evaluate_rankings(ranked_by_query: Mapping[str, Sequence[int]], grade_matrix: np.ndarray, queries: Sequence[str],
                      candidates: np.ndarray | None = None, ks: Sequence[int] = KS) -> dict[str, dict[int, dict]]:
    """{query: {k: metrics}} for rankings given as row indices into ``grade_matrix`` (columns = ``queries``)."""
    col = {q: j for j, q in enumerate(queries)}
    return {q: {k: query_metrics(ranked_by_query[q], grade_matrix[:, col[q]], k, candidates) for k in ks} for q in queries}


def macro(per_query: Mapping[str, Mapping[int, Mapping[str, float]]], queries: Sequence[str],
          ks: Sequence[int] = KS) -> dict[str, dict[str, float]]:
    """Macro-average over ``queries`` with an exact mean (independent of query order)."""
    return {f"k{k}": {m: statistics.mean(per_query[q][k][m] for q in queries) for m in METRICS} for k in ks}


def query_subsets(queries: Sequence[str], low_confidence: frozenset[str], river_relative: frozenset[str]) -> dict[str, list[str]]:
    """The four reporting subsets. ``core`` = region-agnostic and not low-confidence: no river-relative grader, no
    sub-pixel road/bridge grader - the queries whose judgement does not depend on the weakest assumptions."""
    return {
        "all": list(queries),
        "excluding_low_confidence": [q for q in queries if q not in low_confidence],
        "region_agnostic": [q for q in queries if q not in river_relative],
        "core": [q for q in queries if q not in river_relative and q not in low_confidence],
    }


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two binary raters."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def chance_adjusted(per_query: Mapping[str, Mapping[int, Mapping[str, float]]], queries: Sequence[str],
                    ks: Sequence[int] = KS) -> dict[str, dict[str, float]]:
    """Precision against what a RANDOM ranking of the same candidate set would score.

    ``prevalence`` = share of the candidate set graded relevant, macro-averaged over ``queries``; a random top-K has
    expected precision equal to it. ``lift`` = macro precision / macro prevalence. Needed because candidate sets differ
    (the whole corpus vs one region) and so do their base rates - raw precision is not comparable across them.
    """
    prev = statistics.mean(per_query[q][ks[0]]["n_relevant_in_candidates"] / per_query[q][ks[0]]["n_candidates"] for q in queries)
    out = {"prevalence": prev}
    for k in ks:
        p = statistics.mean(per_query[q][k]["precision"] for q in queries)
        out[f"k{k}"] = {"precision": p, "lift_over_chance": p / prev if prev else float("nan")}
    return out

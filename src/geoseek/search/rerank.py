"""Post-ranking result-set refinements (Phase 7b follow-up, precision work).

Pure, dependency-light helpers that take a already-ranked list of
``(tile_id, score)`` pairs (RemoteCLIP cosine, descending) and return a
refined list. They are deliberately outside :class:`SearchEngine` so the
retrieval evaluation can measure each one in isolation and in combination;
the engine can adopt whichever the evaluation shows to help.

Three independent refinements:

* :func:`filter_by_region` - metadata pre-filter: keep only tiles in a set of
  allowed regions. Architecturally this is "filter cheap first" - narrow the
  candidate set to the analyst's sector before semantic ranking. For an exact
  flat index, pre-filtering the candidates and post-filtering the full scan
  give the identical top-K, so the evaluation applies it as a post-filter.
* :func:`apply_score_threshold` - drop tail matches whose similarity is below
  an absolute cutoff instead of always padding the result set to K.
* :func:`suppress_near_duplicates` - collapse tiles that are near-identical in
  embedding space (same place, adjacent tile, or same place another date) so
  they do not consume several of the K slots.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

RankedItem = tuple[str, float]

# aoi_name suffixes that distinguish observations of the same physical region
_AOI_SUFFIXES = ("_scaled_82km", "-82km", "_diverse", "_scaled")


def region_key(aoi_name: str | None) -> str:
    """Normalise an observation ``aoi_name`` to its base region slug.

    ``dehradun_43RGP_diverse`` / ``dehradun_44RKU_diverse`` -> ``dehradun``;
    ``ayodhya_44RPQ_scaled_82km`` / ``ayodhya-82km`` -> ``ayodhya``.
    """
    if not aoi_name:
        return "unknown"
    s = aoi_name
    for suf in _AOI_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    # strip a trailing MGRS tile token like "_43RGP"
    parts = s.split("_")
    if len(parts) > 1 and len(parts[-1]) == 5 and parts[-1][:2].isdigit() and parts[-1][2:].isalpha():
        parts = parts[:-1]
    return "_".join(parts) or "unknown"


def filter_by_region(ranked: Sequence[RankedItem], region_of: Callable[[str], str],
                     allowed: Iterable[str]) -> list[RankedItem]:
    """Keep only ranked items whose tile's region is in ``allowed`` (order preserved)."""
    allow = set(allowed)
    return [(tid, sc) for tid, sc in ranked if region_of(tid) in allow]


def apply_score_threshold(ranked: Sequence[RankedItem], tau: float) -> list[RankedItem]:
    """Keep the rank-order prefix whose score is >= ``tau``.

    Scores are descending, so this is a prefix cut; a lone dip below ``tau``
    still terminates the list (matches "stop when matches get weak").
    """
    out: list[RankedItem] = []
    for tid, sc in ranked:
        if sc < tau:
            break
        out.append((tid, sc))
    return out


def suppress_near_duplicates(ranked: Sequence[RankedItem], vec_of: Callable[[str], np.ndarray],
                             tau_dup: float = 0.97) -> tuple[list[RankedItem], list[str]]:
    """Greedy near-duplicate collapse in rank order.

    A tile is dropped if its cosine similarity to any already-kept tile is
    >= ``tau_dup`` (vectors are unit-norm, so cosine == dot product). Returns
    ``(kept, suppressed_tile_ids)``.
    """
    kept: list[RankedItem] = []
    kept_vecs: list[np.ndarray] = []
    suppressed: list[str] = []
    for tid, sc in ranked:
        v = np.asarray(vec_of(tid), dtype=np.float32)
        if any(float(np.dot(v, kv)) >= tau_dup for kv in kept_vecs):
            suppressed.append(tid)
            continue
        kept.append((tid, sc))
        kept_vecs.append(v)
    return kept, suppressed


def best_threshold_by_f1(val_rankings: Sequence[tuple[Sequence[RankedItem], dict]],
                         taus: Sequence[float], k: int) -> tuple[float, dict]:
    """Pick the score threshold maximising macro-F1@k on a validation set.

    ``val_rankings`` is a list of ``(ranked_items, judgments)`` where
    ``judgments`` maps tile_id -> graded relevance (>0 == relevant). The
    threshold is chosen without ever seeing the test rankings.
    """
    best_tau, best_f1, curve = taus[0], -1.0, {}
    for tau in taus:
        precs, recs = [], []
        for ranked, judg in val_rankings:
            total_rel = sum(1 for g in judg.values() if g and g > 0)
            cut = apply_score_threshold(ranked, tau)[:k]
            hits = sum(1 for tid, _ in cut if judg.get(tid, 0) and judg[tid] > 0)
            precs.append(hits / len(cut) if cut else 0.0)
            recs.append(hits / total_rel if total_rel else 0.0)
        p, r = float(np.mean(precs)), float(np.mean(recs))
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        curve[round(float(tau), 4)] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau, curve

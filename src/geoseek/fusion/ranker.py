"""Phase 5 Step D - unified analyst-queue ranking.

One score per change candidate, combining:

  * **change confidence**  - the Step C/D confidence, which already aggregates the
    trained detector, temporal persistence, image quality, spectral agreement
    and Sentinel-1 corroboration.
  * **significance**       - Step A's ``area_term^0.6 * anomaly_term^0.4`` (a
    large, spectrally-strong change is worth more of an analyst's time).
  * **semantic relevance** - cosine similarity of the candidate's later-date
    tile embedding to a RemoteCLIP text query, rescaled to [0, 1]. Only when a
    query is active.

Formula (weighted **geometric mean** - matches the confidence engine, so no
single axis can carry the queue on its own):

    query active   : score = ( C^3.0 * S^1.5 * Q^2.0 ) ^ (1 / 6.5)
    no query       : score = ( C^3.0 * S^1.5 )         ^ (1 / 4.5)

Weights, and why:

  w_confidence = 3.0  the primary axis. A queue must lead with detections the
                      evidence supports; this term already rolls up model /
                      persistence / quality / spectral / SAR.
  w_significance = 1.5 re-orders within a confidence band (a 30 ha tank over a
                      0.5 ha one) but < w_confidence and geometric so a big
                      low-confidence blob can't jump the queue.
  w_semantic = 2.0    when the analyst states an intent, relevance to it is
                      nearly as important as raw change confidence - but not
                      more: a perfect-match tile with weak change evidence is
                      not a finding.

The semantic term uses the existing search seams (``EmbeddingModel`` +
``VectorIndex`` + ``MetadataRepository``) - no new model, no SAR (SAR is never
embedded).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

FUSION_WEIGHTS = {"confidence": 3.0, "significance": 1.5, "semantic": 2.0}

# RemoteCLIP image<->text cosine similarities for a good match sit ~0.20-0.32;
# rescale that band to [0, 1] for the fusion term.
_SEM_LO, _SEM_HI = 0.15, 0.32
_EPS = 1e-3


def _rescale_semantic(cos_sim: float) -> float:
    return float(min(1.0, max(0.0, (cos_sim - _SEM_LO) / (_SEM_HI - _SEM_LO))))


def fusion_score(confidence: float, significance: float,
                 semantic: float | None = None, weights: dict | None = None) -> float:
    """Weighted geometric mean. ``semantic=None`` drops that axis (no query)."""
    w = dict(weights or FUSION_WEIGHTS)
    terms = [(max(confidence, _EPS), w["confidence"]), (max(significance, _EPS), w["significance"])]
    if semantic is not None:
        terms.append((max(semantic, _EPS), w["semantic"]))
    wsum = sum(wt for _, wt in terms)
    return float(np.exp(sum(wt * np.log(v) for v, wt in terms) / wsum))


@dataclass
class RankedCandidate:
    candidate_id: str
    fusion_score: float
    confidence: float
    significance: float
    semantic: float | None
    change_type: str
    centroid_lonlat: tuple[float, float]
    area_m2: float
    tile_id: str | None = None
    semantic_cos: float | None = None
    breakdown: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "fusion_score": round(self.fusion_score, 4),
                "confidence": round(self.confidence, 4), "significance": round(self.significance, 4),
                "semantic": None if self.semantic is None else round(self.semantic, 4),
                "semantic_cos": None if self.semantic_cos is None else round(self.semantic_cos, 4),
                "change_type": self.change_type, "area_m2": round(self.area_m2, 1),
                "centroid_lonlat": [round(x, 6) for x in self.centroid_lonlat],
                "tile_id": self.tile_id, "breakdown": self.breakdown}


class FusionRanker:
    """Fuses change evidence with (optional) RemoteCLIP semantic relevance.

    ``search_engine`` is a :class:`geoseek.search.engine.SearchEngine` (composes
    the repo / vector-index / embedding-model seams). Pass ``None`` to rank
    without a semantic term.
    """

    def __init__(self, search_engine=None, *, weights: dict | None = None):
        self.engine = search_engine
        self.weights = dict(weights or FUSION_WEIGHTS)
        self._by_faiss: dict[int, np.ndarray] = {}

    # -- semantic -------------------------------------------------------

    def _tile_for_point(self, lon: float, lat: float, later_obs: str):
        """The embedded later-date tile whose footprint contains the point."""
        if self.engine is None:
            return None
        eps = 0.002
        recs = self.engine.repo.query_tiles(bbox=(lon - eps, lat - eps, lon + eps, lat + eps))
        cand = [r for r in recs if r.observation_id == later_obs and r.faiss_id is not None]
        if not cand:
            cand = [r for r in recs if r.faiss_id is not None]
        return cand[0] if cand else None

    def _semantic(self, lon, lat, later_obs, query_vec) -> tuple[float | None, float | None, str | None]:
        rec = self._tile_for_point(lon, lat, later_obs)
        if rec is None or query_vec is None:
            return None, None, (rec.tile_id if rec else None)
        vec = self.engine.vector_index.get_vector(int(rec.faiss_id))
        cos = float(np.dot(vec / (np.linalg.norm(vec) + 1e-9),
                           query_vec / (np.linalg.norm(query_vec) + 1e-9)))
        return _rescale_semantic(cos), cos, rec.tile_id

    # -- rank ---------------------------------------------------------

    def rank(self, candidates: list[dict], *, query: str | None = None) -> tuple[list[RankedCandidate], dict]:
        """``candidates``: dicts with candidate_id, confidence, significance,
        change_type, centroid_lonlat, area_m2, later_obs. Returns
        (ranked list, meta)."""
        query_vec = None
        t0 = time.time()
        if query and self.engine is not None:
            query_vec = self.engine.embedding_model.encode_text(query)
        out: list[RankedCandidate] = []
        for c in candidates:
            lon, lat = c["centroid_lonlat"]
            sem = sem_cos = tile_id = None
            if query_vec is not None:
                sem, sem_cos, tile_id = self._semantic(lon, lat, c.get("later_obs"), query_vec)
            score = fusion_score(c["confidence"], c["significance"],
                                 semantic=sem, weights=self.weights)
            bd = [f"confidence {c['confidence']:.2f} (w{self.weights['confidence']:.0f})",
                  f"significance {c['significance']:.2f} (w{self.weights['significance']:.1f})"]
            if query:
                bd.append(f"semantic {('%.2f' % sem) if sem is not None else 'n/a'} "
                          f"(cos {('%.3f' % sem_cos) if sem_cos is not None else 'n/a'}, "
                          f"w{self.weights['semantic']:.0f})")
            bd.append(f"=> fusion {score:.3f}")
            out.append(RankedCandidate(
                candidate_id=c["candidate_id"], fusion_score=score, confidence=c["confidence"],
                significance=c["significance"], semantic=sem, semantic_cos=sem_cos,
                change_type=c.get("change_type", "?"), centroid_lonlat=(lon, lat),
                area_m2=c.get("area_m2", 0.0), tile_id=tile_id, breakdown=bd,
                source={k: c.get(k) for k in ("pair", "persistence", "earliest_supported")}))
        out.sort(key=lambda r: r.fusion_score, reverse=True)
        meta = {"query": query, "weights": self.weights, "n": len(out),
                "formula": ("(C^3.0 * S^1.5 * Q^2.0)^(1/6.5)  [query]  |  "
                            "(C^3.0 * S^1.5)^(1/4.5)  [no query]"),
                "semantic_rescale": f"cos in [{_SEM_LO}, {_SEM_HI}] -> [0, 1]",
                "elapsed_s": round(time.time() - t0, 3)}
        return out, meta

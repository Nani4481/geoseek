"""Phase 4 Step D - the confidence engine.

One calibrated confidence in [0, 1] per surviving candidate, plus a
human-readable evidence breakdown. The raw model probability is **never**
exposed as the confidence - it is one of six terms.

Terms and why they are weighted as they are
-------------------------------------------
  model         2.0   the trained detector's own evidence. Primary, but a
                      single detector on out-of-domain (L1C->L2A) imagery is
                      not sufficient on its own.
  persistence   2.0   a change confirmed by independent *later* observations is
                      the strongest guard against a one-shot false alarm; a
                      transient signal is almost always noise here.
  spectral      1.5   NDVI/NDBI/NDWI deltas independently corroborating the
                      assigned change TYPE - physical evidence the CNN did not use
                      in the same way.
  quality       1.5   cloud / low valid-pixel fraction makes any detection on
                      that footprint untrustworthy regardless of the model.
  registration  1.0   ~0.15 px for this scene - near-ideal, so it rarely moves
                      the score, but it must be able to sink a mis-registered pair.
  radiometric   1.0   reliability of the Phase 3a relative normalization over
                      the footprint - guards a haze-driven artefact.

Combination: a **weighted geometric mean**
    confidence = exp( sum_i w_i * ln(clip(q_i, eps)) / sum_i w_i )
Multiplicative, so any single collapsing term (a cloud, a transient signal, no
spectral support) pulls the whole confidence down - a candidate cannot be
"high confidence" while failing on one axis, which an average would allow.

Pure functions - no rasters/torch/network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

WEIGHTS: dict[str, float] = {
    "model": 2.0,
    "persistence": 2.0,
    "spectral": 1.5,
    "quality": 1.5,
    "registration": 1.0,
    "radiometric": 1.0,
}

_EPS = 1e-3

# model-probability rescale: the operating point is 0.80, so map [0.50, 0.97]
# -> [0, 1] (a mean footprint prob of exactly the threshold is "weak", not
# "certain"). Below 0.50 -> floor; at/above 0.97 -> 1.
_MODEL_LO, _MODEL_HI = 0.50, 0.97


@dataclass
class EvidenceTerm:
    name: str
    value: float          # normalized contribution in [0, 1]
    weight: float
    raw: str              # human-readable ("registration: 0.16 px")

    def as_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4), "weight": self.weight, "raw": self.raw}


@dataclass
class ConfidenceReport:
    confidence: float
    terms: list[EvidenceTerm]
    method: str = "weighted geometric mean of 6 evidence terms"
    breakdown: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"confidence": round(self.confidence, 4), "method": self.method,
                "terms": [t.as_dict() for t in self.terms], "breakdown": self.breakdown}


def _rescale_model_prob(p: float) -> float:
    return float(min(1.0, max(0.0, (p - _MODEL_LO) / (_MODEL_HI - _MODEL_LO))))


def _quality_term(bad_scl_fraction: float, valid_fraction: float, max_bad: float = 0.05) -> float:
    return float(max(0.0, min(1.0, min(valid_fraction, 1.0 - bad_scl_fraction / max_bad))))


def _registration_term(residual_px: float, trust: float = 0.30, cap: float = 0.50,
                       floor: float = 0.35) -> float:
    if residual_px <= trust:
        return 1.0
    if residual_px >= cap:
        return floor
    frac = (residual_px - trust) / (cap - trust)
    return float(1.0 - frac * (1.0 - floor))


def compute_confidence(
    *,
    candidate_id: str,
    model_prob: float,                 # mean model probability over the candidate footprint
    bad_scl_fraction: float,           # max over both dates
    valid_fraction: float,
    coreg_residual_px: float,
    radiometric_reliability: float,    # [0,1] - 1 = index bands confidently normalized here
    persistence: str,
    persistence_confidence: float,     # from TemporalPersistenceAnalyzer
    spectral_agreement: float,         # [0,1] - how strongly the index deltas back the assigned type
    change_type: str,
    suppression_downweight: float = 1.0,   # product of Step A 'downweight' rules
    weights: dict[str, float] | None = None,
) -> ConfidenceReport:
    w = dict(WEIGHTS if weights is None else weights)

    q_model = _rescale_model_prob(model_prob)
    q_quality = _quality_term(bad_scl_fraction, valid_fraction)
    q_reg = _registration_term(coreg_residual_px)
    q_radio = float(max(0.0, min(1.0, radiometric_reliability)))
    q_persist = float(max(0.0, min(1.0, persistence_confidence)))
    q_spectral = float(max(0.0, min(1.0, spectral_agreement)))

    terms = [
        EvidenceTerm("model", q_model, w["model"],
                     f"model: mean p {model_prob:.2f} over footprint (rescaled {q_model:.2f})"),
        EvidenceTerm("persistence", q_persist, w["persistence"],
                     f"persistence: {persistence} (confidence {persistence_confidence:.2f})"),
        EvidenceTerm("spectral", q_spectral, w["spectral"],
                     f"spectral: index deltas {'strongly' if q_spectral >= 0.7 else 'weakly'} "
                     f"support '{change_type}' ({q_spectral:.2f})"),
        EvidenceTerm("quality", q_quality, w["quality"],
                     f"quality: bad-SCL {bad_scl_fraction*100:.1f}%, valid {valid_fraction*100:.1f}% "
                     f"({q_quality:.2f})"),
        EvidenceTerm("registration", q_reg, w["registration"],
                     f"registration: {coreg_residual_px:.2f} px ({q_reg:.2f})"),
        EvidenceTerm("radiometric", q_radio, w["radiometric"],
                     f"radiometric: normalization reliability {q_radio:.2f}"),
    ]

    wsum = sum(t.weight for t in terms)
    log_mean = sum(t.weight * math.log(max(t.value, _EPS)) for t in terms) / wsum
    conf = math.exp(log_mean)
    conf *= float(max(0.0, min(1.0, suppression_downweight)))   # fold in Step A down-weights

    from geoseek.temporal.persistence import PERSISTENCE_PENALTY
    pen = PERSISTENCE_PENALTY.get(persistence, 1.0)
    conf *= pen

    breakdown = [t.raw for t in terms]
    if suppression_downweight < 0.999:
        breakdown.append(f"suppression down-weight applied: x{suppression_downweight:.2f}")
    if pen < 1.0:
        breakdown.append(f"temporal-support penalty ({persistence}): x{pen:.2f}")
    breakdown.append(f"=> confidence {conf:.2f}")

    return ConfidenceReport(confidence=float(max(0.0, min(1.0, conf))), terms=terms, breakdown=breakdown)


def spectral_agreement_for(change_type: str, ndvi_anomaly: float, ndbi_anomaly: float,
                           ndwi_anomaly: float) -> float:
    """How strongly the index evidence backs the assigned type, in [0, 1].

    Scales with how far past its decision threshold the driving anomaly sits;
    'other' gets a fixed low-moderate value (no positive spectral evidence).
    """
    from geoseek.change.classify import NDBI_UP, NDVI_DOWN, NDWI_CHANGE

    def _ramp(x: float, lo: float, hi: float) -> float:
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))

    if change_type in ("water_gain", "water_loss"):
        return 0.4 + 0.6 * _ramp(abs(ndwi_anomaly), NDWI_CHANGE, NDWI_CHANGE + 0.25)
    if change_type == "construction":
        return 0.35 + 0.65 * _ramp(ndbi_anomaly, NDBI_UP, NDBI_UP + 0.20)
    if change_type == "road":
        return 0.3 + 0.5 * _ramp(ndbi_anomaly, 0.0, NDBI_UP + 0.15)
    if change_type == "clearance":
        return 0.35 + 0.65 * _ramp(-ndvi_anomaly, -NDVI_DOWN, -NDVI_DOWN + 0.20)
    return 0.35  # "other" - detected but not spectrally explained

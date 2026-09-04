"""Phase 4 Step B - rule-based change typing (PS 2.2.2).

We have **no change-type labels**, so typing is deliberately a small set of
transparent rules over the spectral-index deltas already computed in Phase 3a
(NDVI/NDWI/NDBI on the relatively-normalized reflectance) plus the candidate's
shape. No unlabelled ML, no clustering-and-guessing.

    construction / new built-up : NDBI up, NDVI not rising, compact-ish
    clearance / vegetation loss : NDVI down (beyond the seasonal trend), NDBI ~flat
    water gain / water loss     : |NDWI delta| large
    road development            : elongated component + NDBI up
    other / unclassified        : explicitly allowed - not forced into a bucket

All deltas are measured **against the scene-wide seasonal delta** (the
"anomaly"), because the 2019 drought -> 2024 green shift moves NDVI/NDBI
everywhere; a *structural* change is one that departs from that trend. Every
classification returns the anomalies that justified it.

Pure functions - no rasters/torch/network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CHANGE_TYPES = ("construction", "clearance", "water_gain", "water_loss", "road", "other")

# --- thresholds (justified; overridable via ClassifyConfig) ---------------

# NDBI anomaly (candidate NDBI delta minus the scene seasonal NDBI delta) at
# which new impervious surface is credible. Over the AOI the scene NDBI drifts
# ~-0.27 with the greening; construction sites in S2 NDBI typically read
# +0.05..+0.25 relative to their prior bare/veg state, so +0.05 above the
# seasonal drift is a conservative floor.
NDBI_UP = 0.05
# NDVI anomaly below which vegetation loss is credible *beyond* the seasonal
# greening. Scene NDVI drifts ~+0.35; genuine clearance shows NDVI flat or
# falling, i.e. an anomaly of -0.08 or more negative.
NDVI_DOWN = -0.08
# NDVI anomaly ceiling for "construction": if the plot is greening *faster*
# than the scene it is not being built on.
NDVI_FLAT_MAX = 0.05
# NDWI *anomaly* (candidate NDWI delta minus the scene seasonal NDWI delta) for
# a water-extent change. NDWI moves scene-wide with the greening (NIR up ->
# NDWI down), so raw NDWI deltas are dominated by phenology; only a departure
# of >=0.15 from that trend, WITH some absolute movement, is a real wet/dry
# transition.
NDWI_CHANGE = 0.15
NDWI_MIN_ABS = 0.05
# elongation (major/minor axis of the pixel cloud) for "road": a linear
# feature. 3.0 excludes blocky plots while keeping ribbon-like clearings/works.
ROAD_ELONGATION = 3.0
# roads are narrow, so their NDBI anomaly is diluted by mixed pixels - use a
# gentler floor when the shape is clearly linear.
ROAD_NDBI_UP = 0.03


@dataclass(frozen=True)
class ClassifyConfig:
    ndbi_up: float = NDBI_UP
    ndvi_down: float = NDVI_DOWN
    ndvi_flat_max: float = NDVI_FLAT_MAX
    ndwi_change: float = NDWI_CHANGE
    ndwi_min_abs: float = NDWI_MIN_ABS
    road_elongation: float = ROAD_ELONGATION
    road_ndbi_up: float = ROAD_NDBI_UP


@dataclass(frozen=True)
class CandidateSpectra:
    """Index deltas (later - earlier) + scene seasonal deltas + shape for one candidate."""

    candidate_id: str
    d_ndvi: float
    d_ndbi: float
    d_ndwi: float
    scene_d_ndvi: float
    scene_d_ndbi: float
    scene_d_ndwi: float
    area_px: int
    elongation: float          # major/minor axis ratio of the pixel cloud (>=1)
    fill_ratio: float          # component area / bbox area (compactness proxy)

    @property
    def ndvi_anomaly(self) -> float:
        return self.d_ndvi - self.scene_d_ndvi

    @property
    def ndbi_anomaly(self) -> float:
        return self.d_ndbi - self.scene_d_ndbi

    @property
    def ndwi_anomaly(self) -> float:
        return self.d_ndwi - self.scene_d_ndwi


@dataclass
class Classification:
    candidate_id: str
    change_type: str
    rule: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"candidate_id": self.candidate_id, "change_type": self.change_type,
                "rule": self.rule, "detail": self.detail, "evidence": self.evidence}


def classify_candidate(s: CandidateSpectra, cfg: ClassifyConfig | None = None) -> Classification:
    cfg = cfg or ClassifyConfig()
    nv, nb, nw = s.ndvi_anomaly, s.ndbi_anomaly, s.ndwi_anomaly
    ev = {"d_ndvi": round(s.d_ndvi, 4), "d_ndbi": round(s.d_ndbi, 4), "d_ndwi": round(s.d_ndwi, 4),
          "ndvi_anomaly": round(nv, 4), "ndbi_anomaly": round(nb, 4), "ndwi_anomaly": round(nw, 4),
          "scene_d_ndvi": round(s.scene_d_ndvi, 4), "scene_d_ndbi": round(s.scene_d_ndbi, 4),
          "scene_d_ndwi": round(s.scene_d_ndwi, 4),
          "elongation": round(s.elongation, 2), "fill_ratio": round(s.fill_ratio, 2)}

    # 1. water first - a strong NDWI departure from the seasonal trend, with real movement.
    #    water_gain: NDWI up beyond the trend AND actually wetter.
    #    water_loss: NDWI down beyond the trend AND actually drier AND NOT explained by
    #    vegetation growth (greening also drives NDWI down as NIR rises).
    if nw >= cfg.ndwi_change and s.d_ndwi >= cfg.ndwi_min_abs:
        return Classification(s.candidate_id, "water_gain", "ndwi_anomaly_up",
                              f"NDWI anomaly {nw:+.3f} (>= {cfg.ndwi_change}), raw delta "
                              f"{s.d_ndwi:+.3f} -> water gain", ev)
    if (nw <= -cfg.ndwi_change and s.d_ndwi <= -cfg.ndwi_min_abs
            and nv <= cfg.ndvi_flat_max):
        return Classification(s.candidate_id, "water_loss", "ndwi_anomaly_down_not_greening",
                              f"NDWI anomaly {nw:+.3f} (<= -{cfg.ndwi_change}), raw delta {s.d_ndwi:+.3f}, "
                              f"NDVI anomaly {nv:+.3f} (<= {cfg.ndvi_flat_max}, not greening) -> water loss",
                              ev)

    # 2. road - linear shape + a (gentle) built-up anomaly
    if s.elongation >= cfg.road_elongation and nb >= cfg.road_ndbi_up:
        return Classification(s.candidate_id, "road", "linear_morphology_plus_ndbi",
                              f"elongation {s.elongation:.1f} (>= {cfg.road_elongation}) with NDBI anomaly "
                              f"{nb:+.3f} (>= {cfg.road_ndbi_up}) -> road / linear works", ev)

    # 3. construction - NDBI anomaly up, NDVI not greening faster than the scene
    if nb >= cfg.ndbi_up and nv <= cfg.ndvi_flat_max:
        return Classification(s.candidate_id, "construction", "ndbi_up_ndvi_flat_or_down",
                              f"NDBI anomaly {nb:+.3f} (>= {cfg.ndbi_up}) with NDVI anomaly {nv:+.3f} "
                              f"(<= {cfg.ndvi_flat_max}) -> new built-up", ev)

    # 4. clearance - NDVI anomaly well below the greening trend, NDBI ~flat
    if nv <= cfg.ndvi_down and abs(nb) < cfg.ndbi_up:
        return Classification(s.candidate_id, "clearance", "ndvi_down_ndbi_flat",
                              f"NDVI anomaly {nv:+.3f} (<= {cfg.ndvi_down}) with NDBI anomaly {nb:+.3f} "
                              f"(|.| < {cfg.ndbi_up}) -> vegetation / land clearance", ev)

    # 5. explicitly unclassified
    return Classification(s.candidate_id, "other", "no_rule_matched",
                          f"NDVI anomaly {nv:+.3f}, NDBI anomaly {nb:+.3f}, NDWI delta {s.d_ndwi:+.3f}, "
                          f"elongation {s.elongation:.1f} -> no rule matched; left unclassified", ev)


def class_distribution(classifications: list[Classification]) -> dict:
    dist = {t: 0 for t in CHANGE_TYPES}
    for c in classifications:
        dist[c.change_type] = dist.get(c.change_type, 0) + 1
    return dist

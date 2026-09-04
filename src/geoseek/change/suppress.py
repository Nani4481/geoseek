"""Phase 4 Step A - false-alarm suppression (PS 2.2.3).

A stage between the raw change-model output and the reported candidates. Five
gates are applied **in a fixed order**; each records what it checked and its
verdict, so every suppressed candidate carries a full trace of *why*.

    1. quality        cloud / shadow / snow / saturated SCL on EITHER date, or
                      too few valid pixels                       -> SUPPRESS
    2. registration   measured co-registration residual of the pair; beyond a
                      trust threshold                            -> DOWN-WEIGHT
    3. radiometric    reliability of the Phase 3a relative normalization over
                      the footprint; low-confidence blocks       -> DOWN-WEIGHT
    4. phenology      change explained by seasonal vegetation shift alone
                      (NDVI tracks the scene-wide greening/senescence, NDBI
                      stable, no water change)                   -> SUPPRESS
    5. morphology     connected-component area below a minimum   -> SUPPRESS

Rules 1/4/5 reject; rules 2/3 attach a multiplicative down-weight consumed by
:mod:`geoseek.change.confidence`. Pure functions on plain numbers - no rasters,
no torch, no network - so the whole stage is unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# thresholds (documented + justified; overridable via SuppressionConfig)
# --------------------------------------------------------------------------

# Rule 1 - quality. SCL bad classes are cloud (8,9), thin cirrus (10), shadow
# (3), snow (11), saturated/defective (1), no-data (0) - the same set
# geoseek.ingest.quality.BAD_SCL_CLASSES uses. A candidate is a real surface
# change only if BOTH dates see the ground: >5% bad on either date, or <80%
# jointly-valid pixels, means the "change" may just be a cloud edge.
MAX_BAD_SCL_FRACTION = 0.05
MIN_VALID_FRACTION = 0.80

# Rule 2 - registration. Sentinel-2 same-MGRS-tile multitemporal registration
# spec is <=0.5 px; geoseek.change.coregister only *corrects* above 0.5 px.
# Below 0.30 px a 1-pixel change footprint is spatially trustworthy; from
# 0.30 -> 0.50 px, edge pixels of a small candidate become unreliable, so the
# candidate is linearly down-weighted; at/above 0.50 px (uncorrected) it is
# heavily down-weighted (the pair should have been resampled).
REGISTRATION_TRUST_PX = 0.30
REGISTRATION_MAX_PX = 0.50
REGISTRATION_FLOOR_WEIGHT = 0.35

# Rule 3 - radiometric. The Phase 3a normalization is reliable where the
# index-driving bands (B03/B04/B08/B11) are well correlated between dates and
# the per-block offset was not collapsed to "no confident correction". Below
# this inter-date correlation, or over a normalization surface that was
# entirely shrunk/deadbanded, an index delta may be residual haze, not ground
# change -> down-weight.
RADIOMETRIC_MIN_CORR = 0.60
RADIOMETRIC_LOWCONF_WEIGHT = 0.6

# Rule 4 - phenology. The 2019 scene is a drought March (scene NDVI median
# ~0.33), 2024 is green/wet (~0.68): a scene-wide shift in ALL THREE indices
# (NDVI up with the greening, NDBI down, NDWI down as NIR rises) affects
# *everything*. So "seasonal" is judged against the *scene-wide* seasonal
# delta of each index, not the raw delta. A candidate whose NDVI, NDBI AND
# NDWI deltas all stay within these bands of their scene trends is explained
# by phenology alone -> suppress. A structural change breaks from the trend on
# at least one axis (construction: NDBI jumps well above the seasonal drift;
# clearance: NDVI drops well below the greening trend; water: NDWI departs).
PHENO_NDVI_ANOMALY = 0.10
PHENO_NDBI_ANOMALY = 0.06
PHENO_NDWI_ANOMALY = 0.08

# Rule 5 - morphology. 1 px = 100 m^2 at Sentinel-2's 10 m GSD. A 2x2-pixel
# blob (400 m^2) is one S2 resolution element - below the scale at which a
# change can be told apart from co-registration jitter (0.15-0.29 px here) +
# mixed-pixel noise over a multi-year interval. 10 px (1,000 m^2, 0.1 ha) is
# the smallest footprint at which a discrete built structure, a cleared plot
# or a road segment is confidently attributable; erring toward precision per
# PS 2.2.3 and keeping the surviving list analyst-reviewable.
MORPH_MIN_AREA_PX = 10
PIXEL_AREA_M2 = 100.0

RULES = ("quality", "registration", "radiometric", "phenology", "morphology")


@dataclass(frozen=True)
class SuppressionConfig:
    max_bad_scl_fraction: float = MAX_BAD_SCL_FRACTION
    min_valid_fraction: float = MIN_VALID_FRACTION
    registration_trust_px: float = REGISTRATION_TRUST_PX
    registration_max_px: float = REGISTRATION_MAX_PX
    radiometric_min_corr: float = RADIOMETRIC_MIN_CORR
    pheno_ndvi_anomaly: float = PHENO_NDVI_ANOMALY
    pheno_ndbi_anomaly: float = PHENO_NDBI_ANOMALY
    pheno_ndwi_anomaly: float = PHENO_NDWI_ANOMALY
    morph_min_area_px: int = MORPH_MIN_AREA_PX


@dataclass(frozen=True)
class CandidateFeatures:
    """Everything a suppression rule needs about one raw candidate."""

    candidate_id: str
    area_px: int
    bad_scl_fraction_earlier: float
    bad_scl_fraction_later: float
    valid_fraction: float
    d_ndvi: float          # later - earlier, mean over the footprint (on normalized-reflectance indices)
    d_ndbi: float
    d_ndwi: float


@dataclass(frozen=True)
class PairSuppressionContext:
    """Per-pair facts shared by every candidate of that pair."""

    pair_id: str
    coreg_residual_px: float
    coreg_corrected: bool
    scene_d_ndvi: float                 # scene-wide seasonal NDVI delta (robust, over the whole AOI)
    scene_d_ndbi: float
    scene_d_ndwi: float
    radiometric_index_band_min_corr: float
    radiometric_low_confidence: bool


@dataclass
class RuleOutcome:
    rule: str
    verdict: str                       # "pass" | "suppress" | "downweight"
    weight: float = 1.0                # multiplicative down-weight in (0, 1]
    detail: str = ""
    values: dict = field(default_factory=dict)


@dataclass
class SuppressionTrace:
    candidate_id: str
    outcomes: list[RuleOutcome]
    suppressed: bool
    suppressed_by: str | None
    combined_downweight: float

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "suppressed": self.suppressed,
            "suppressed_by": self.suppressed_by,
            "combined_downweight": round(self.combined_downweight, 4),
            "trace": [{"rule": o.rule, "verdict": o.verdict, "weight": round(o.weight, 3),
                       "detail": o.detail, **({"values": o.values} if o.values else {})}
                      for o in self.outcomes],
        }


# --------------------------------------------------------------------------
# individual gates
# --------------------------------------------------------------------------


def gate_quality(f: CandidateFeatures, cfg: SuppressionConfig) -> RuleOutcome:
    bad = max(f.bad_scl_fraction_earlier, f.bad_scl_fraction_later)
    ok = bad <= cfg.max_bad_scl_fraction and f.valid_fraction >= cfg.min_valid_fraction
    return RuleOutcome(
        rule="quality",
        verdict="pass" if ok else "suppress",
        detail=(f"bad-SCL max {bad*100:.1f}% (<= {cfg.max_bad_scl_fraction*100:.0f}%), "
                f"valid {f.valid_fraction*100:.1f}% (>= {cfg.min_valid_fraction*100:.0f}%)"),
        values={"bad_scl_fraction_earlier": round(f.bad_scl_fraction_earlier, 4),
                "bad_scl_fraction_later": round(f.bad_scl_fraction_later, 4),
                "valid_fraction": round(f.valid_fraction, 4)},
    )


def gate_registration(ctx: PairSuppressionContext, cfg: SuppressionConfig) -> RuleOutcome:
    r = ctx.coreg_residual_px
    if r <= cfg.registration_trust_px:
        w, verdict = 1.0, "pass"
    elif r >= cfg.registration_max_px:
        w, verdict = REGISTRATION_FLOOR_WEIGHT, "downweight"
    else:
        frac = (r - cfg.registration_trust_px) / (cfg.registration_max_px - cfg.registration_trust_px)
        w, verdict = 1.0 - frac * (1.0 - REGISTRATION_FLOOR_WEIGHT), "downweight"
    return RuleOutcome(
        rule="registration", verdict=verdict, weight=w,
        detail=(f"pair co-registration residual {r:.2f} px"
                + (" (corrected)" if ctx.coreg_corrected else "")
                + (f"; trust <= {cfg.registration_trust_px} px" if verdict == "pass"
                   else f"; > {cfg.registration_trust_px} px -> weight {w:.2f}")),
        values={"coreg_residual_px": round(r, 4), "coreg_corrected": ctx.coreg_corrected},
    )


def gate_radiometric(ctx: PairSuppressionContext, cfg: SuppressionConfig) -> RuleOutcome:
    low = ctx.radiometric_low_confidence or ctx.radiometric_index_band_min_corr < cfg.radiometric_min_corr
    return RuleOutcome(
        rule="radiometric",
        verdict="downweight" if low else "pass",
        weight=RADIOMETRIC_LOWCONF_WEIGHT if low else 1.0,
        detail=(f"index-band min inter-date corr {ctx.radiometric_index_band_min_corr:.2f} "
                f"(>= {cfg.radiometric_min_corr} ok); "
                + ("normalization LOW confidence over this footprint -> down-weight"
                   if low else "normalization reliable")),
        values={"index_band_min_corr": round(ctx.radiometric_index_band_min_corr, 3),
                "low_confidence": bool(low)},
    )


def gate_phenology(f: CandidateFeatures, ctx: PairSuppressionContext,
                   cfg: SuppressionConfig) -> RuleOutcome:
    ndvi_anom = f.d_ndvi - ctx.scene_d_ndvi
    ndbi_anom = f.d_ndbi - ctx.scene_d_ndbi
    ndwi_anom = f.d_ndwi - ctx.scene_d_ndwi
    seasonal = (abs(ndvi_anom) <= cfg.pheno_ndvi_anomaly
                and abs(ndbi_anom) <= cfg.pheno_ndbi_anomaly
                and abs(ndwi_anom) <= cfg.pheno_ndwi_anomaly)
    return RuleOutcome(
        rule="phenology",
        verdict="suppress" if seasonal else "pass",
        detail=(f"NDVI {f.d_ndvi:+.3f} (scene {ctx.scene_d_ndvi:+.3f}, anom {ndvi_anom:+.3f}), "
                f"NDBI {f.d_ndbi:+.3f} (scene {ctx.scene_d_ndbi:+.3f}, anom {ndbi_anom:+.3f}), "
                f"NDWI {f.d_ndwi:+.3f} (scene {ctx.scene_d_ndwi:+.3f}, anom {ndwi_anom:+.3f})"
                + (" -> all three track the seasonal trend; phenology alone" if seasonal
                   else " -> breaks the seasonal trend on >=1 axis (structural)")),
        values={"d_ndvi": round(f.d_ndvi, 4), "d_ndbi": round(f.d_ndbi, 4),
                "d_ndwi": round(f.d_ndwi, 4), "ndvi_anomaly": round(ndvi_anom, 4),
                "ndbi_anomaly": round(ndbi_anom, 4), "ndwi_anomaly": round(ndwi_anom, 4)},
    )


def gate_morphology(f: CandidateFeatures, cfg: SuppressionConfig) -> RuleOutcome:
    ok = f.area_px >= cfg.morph_min_area_px
    return RuleOutcome(
        rule="morphology",
        verdict="pass" if ok else "suppress",
        detail=(f"area {f.area_px} px = {f.area_px * PIXEL_AREA_M2:.0f} m^2 "
                f"({'>=' if ok else '<'} {cfg.morph_min_area_px} px / "
                f"{cfg.morph_min_area_px * PIXEL_AREA_M2:.0f} m^2 minimum)"),
        values={"area_px": int(f.area_px), "area_m2": round(f.area_px * PIXEL_AREA_M2, 1)},
    )


# --------------------------------------------------------------------------
# orchestrated per-candidate suppression
# --------------------------------------------------------------------------


def suppress_candidate(f: CandidateFeatures, ctx: PairSuppressionContext,
                       cfg: SuppressionConfig | None = None) -> SuppressionTrace:
    """Run all five gates in order; return the full trace + the net verdict.

    A candidate is *suppressed* by the first rule that returns ``suppress``
    (all five are still evaluated for the trace). ``combined_downweight`` is the
    product of the ``downweight`` rules' weights, to be folded into the
    confidence score of a surviving candidate.
    """
    cfg = cfg or SuppressionConfig()
    outcomes = [
        gate_quality(f, cfg),
        gate_registration(ctx, cfg),
        gate_radiometric(ctx, cfg),
        gate_phenology(f, ctx, cfg),
        gate_morphology(f, cfg),
    ]
    suppressed_by = next((o.rule for o in outcomes if o.verdict == "suppress"), None)
    downweight = 1.0
    for o in outcomes:
        if o.verdict == "downweight":
            downweight *= o.weight
    return SuppressionTrace(
        candidate_id=f.candidate_id,
        outcomes=outcomes,
        suppressed=suppressed_by is not None,
        suppressed_by=suppressed_by,
        combined_downweight=downweight,
    )


def summarize_suppression(traces: list[SuppressionTrace]) -> dict:
    """Counts for the Step E report: total, survived, and per-rule suppressions."""
    by_rule = {r: 0 for r in RULES}
    for t in traces:
        if t.suppressed_by:
            by_rule[t.suppressed_by] += 1
    survived = sum(1 for t in traces if not t.suppressed)
    return {
        "raw_candidates": len(traces),
        "suppressed": len(traces) - survived,
        "survived": survived,
        "suppressed_by_rule": by_rule,
        "downweighted_survivors": sum(1 for t in traces
                                      if not t.suppressed and t.combined_downweight < 0.999),
    }

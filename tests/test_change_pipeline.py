"""Phase 4 - false-alarm suppression, change typing, temporal persistence, confidence.

All hermetic: pure functions on synthetic numbers + a tiny in-memory catalog for
the persistence analyzer. No rasters, no torch, no network.
"""

from __future__ import annotations

import math

import pytest

from geoseek.change.classify import (
    CandidateSpectra,
    ClassifyConfig,
    class_distribution,
    classify_candidate,
)
from geoseek.change.confidence import (
    WEIGHTS,
    compute_confidence,
    spectral_agreement_for,
    _rescale_model_prob,
)
from geoseek.change.suppress import (
    CandidateFeatures,
    PairSuppressionContext,
    SuppressionConfig,
    gate_morphology,
    gate_phenology,
    gate_quality,
    gate_radiometric,
    gate_registration,
    suppress_candidate,
    summarize_suppression,
)


# --------------------------------------------------------------------------
# Step A - suppression
# --------------------------------------------------------------------------

_CFG = SuppressionConfig()


def _feat(**kw):
    d = dict(candidate_id="c", area_px=200, bad_scl_fraction_earlier=0.0, bad_scl_fraction_later=0.0,
             valid_fraction=1.0, d_ndvi=0.0, d_ndbi=0.0, d_ndwi=0.0)
    d.update(kw)
    return CandidateFeatures(**d)


def _ctx(**kw):
    d = dict(pair_id="p", coreg_residual_px=0.15, coreg_corrected=False,
             scene_d_ndvi=0.35, scene_d_ndbi=-0.27, scene_d_ndwi=-0.18,
             radiometric_index_band_min_corr=0.85, radiometric_low_confidence=False)
    d.update(kw)
    return PairSuppressionContext(**d)


def test_quality_gate_passes_clean_and_suppresses_cloud_or_low_valid():
    assert gate_quality(_feat(), _CFG).verdict == "pass"
    assert gate_quality(_feat(bad_scl_fraction_later=0.20), _CFG).verdict == "suppress"
    assert gate_quality(_feat(bad_scl_fraction_earlier=0.30), _CFG).verdict == "suppress"
    assert gate_quality(_feat(valid_fraction=0.5), _CFG).verdict == "suppress"


def test_registration_gate_scales_between_trust_and_cap():
    assert gate_registration(_ctx(coreg_residual_px=0.15), _CFG).weight == 1.0
    o = gate_registration(_ctx(coreg_residual_px=0.40), _CFG)
    assert o.verdict == "downweight" and 0.35 < o.weight < 1.0
    assert gate_registration(_ctx(coreg_residual_px=0.60), _CFG).weight == pytest.approx(0.35)


def test_radiometric_gate_downweights_low_correlation_or_flagged():
    assert gate_radiometric(_ctx(radiometric_index_band_min_corr=0.85), _CFG).verdict == "pass"
    assert gate_radiometric(_ctx(radiometric_index_band_min_corr=0.40), _CFG).verdict == "downweight"
    assert gate_radiometric(_ctx(radiometric_low_confidence=True), _CFG).verdict == "downweight"


def test_phenology_gate_suppresses_seasonal_but_keeps_structural():
    # follows the scene greening trend on all three indices -> seasonal
    seasonal = gate_phenology(_feat(d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=-0.18), _ctx(), _CFG)
    assert seasonal.verdict == "suppress"
    # tracks NDVI/NDBI but NDWI departs the trend -> not purely seasonal
    ndwi_break = gate_phenology(_feat(d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=0.15), _ctx(), _CFG)
    assert ndwi_break.verdict == "pass"
    # NDBI jumps well above the seasonal drift -> structural (construction-like)
    built = gate_phenology(_feat(d_ndvi=0.05, d_ndbi=0.10), _ctx(), _CFG)
    assert built.verdict == "pass"
    # NDVI collapses far below the greening trend -> structural (clearance-like)
    cleared = gate_phenology(_feat(d_ndvi=-0.10, d_ndbi=-0.25), _ctx(), _CFG)
    assert cleared.verdict == "pass"
    # a real water change is not "seasonal" even if NDVI/NDBI track the trend
    water = gate_phenology(_feat(d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=0.30), _ctx(), _CFG)
    assert water.verdict == "pass"


def test_morphology_gate_drops_specks():
    assert _CFG.morph_min_area_px == 50            # Phase 5 Step A: 0.5 ha minimum reported change
    assert gate_morphology(_feat(area_px=3), _CFG).verdict == "suppress"
    assert gate_morphology(_feat(area_px=26), _CFG).verdict == "suppress"   # the Phase 4 speck
    assert gate_morphology(_feat(area_px=60), _CFG).verdict == "pass"
    assert gate_morphology(_feat(area_px=_CFG.morph_min_area_px), _CFG).verdict == "pass"
    assert gate_morphology(_feat(area_px=393), _CFG).verdict == "pass"      # the small real tank


def test_suppress_candidate_reports_first_rule_in_order():
    # cloudy AND tiny -> 'quality' fires first (rule 1 before rule 5)
    tr = suppress_candidate(_feat(area_px=2, bad_scl_fraction_later=0.5), _ctx(), _CFG)
    assert tr.suppressed and tr.suppressed_by == "quality"
    assert [o.rule for o in tr.outcomes] == ["quality", "registration", "radiometric",
                                             "phenology", "morphology"]


def test_suppress_candidate_combines_downweights_for_survivor():
    tr = suppress_candidate(
        _feat(d_ndvi=0.05, d_ndbi=0.12),  # structural -> survives phenology
        _ctx(coreg_residual_px=0.45, radiometric_low_confidence=True), _CFG)
    assert not tr.suppressed
    reg_w = next(o.weight for o in tr.outcomes if o.rule == "registration")
    rad_w = next(o.weight for o in tr.outcomes if o.rule == "radiometric")
    assert tr.combined_downweight == pytest.approx(reg_w * rad_w)
    assert tr.combined_downweight < 1.0


def test_summarize_suppression_counts_per_rule():
    traces = [
        suppress_candidate(_feat(area_px=2), _ctx(), _CFG),                              # morphology
        suppress_candidate(_feat(bad_scl_fraction_later=0.5), _ctx(), _CFG),             # quality
        suppress_candidate(_feat(d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=-0.18), _ctx(), _CFG),  # phenology
        suppress_candidate(_feat(d_ndvi=0.0, d_ndbi=0.15), _ctx(), _CFG),               # survives
    ]
    s = summarize_suppression(traces)
    assert s["raw_candidates"] == 4 and s["survived"] == 1
    assert s["suppressed_by_rule"]["morphology"] == 1
    assert s["suppressed_by_rule"]["quality"] == 1
    assert s["suppressed_by_rule"]["phenology"] == 1


# --------------------------------------------------------------------------
# Step B - change typing
# --------------------------------------------------------------------------

_SCENE = dict(scene_d_ndvi=0.35, scene_d_ndbi=-0.27, scene_d_ndwi=-0.18)


def _spec(**kw):
    # defaults track the scene trend on every index (all anomalies ~0)
    d = dict(candidate_id="c", d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=-0.18, area_px=100,
             elongation=1.2, fill_ratio=0.6, **_SCENE)
    d.update(kw)
    return CandidateSpectra(**d)


def test_typing_water_gain_and_loss_from_ndwi_anomaly():
    # scene NDWI trend is -0.18; a genuine wet gain departs it upward
    assert classify_candidate(_spec(d_ndwi=0.20)).change_type == "water_gain"
    # and a genuine loss departs it downward AND is not explained by greening
    assert classify_candidate(_spec(d_ndwi=-0.45, d_ndvi=0.0)).change_type == "water_loss"
    # NDWI drop that merely follows the seasonal trend is NOT water loss
    assert classify_candidate(_spec(d_ndwi=-0.20)).change_type != "water_loss"
    # NDWI drop driven by heavy greening is NOT water loss
    assert classify_candidate(_spec(d_ndwi=-0.45, d_ndvi=0.55)).change_type != "water_loss"


def test_typing_construction_ndbi_up_ndvi_not_greening():
    # NDBI anomaly = -0.10 - (-0.27) = +0.17 ; NDVI anomaly = 0.30 - 0.35 = -0.05
    c = classify_candidate(_spec(d_ndbi=-0.10, d_ndvi=0.30))
    assert c.change_type == "construction"
    assert c.evidence["ndbi_anomaly"] == pytest.approx(0.17, abs=1e-6)


def test_typing_clearance_ndvi_below_trend_ndbi_flat():
    # NDVI anomaly = 0.20 - 0.35 = -0.15 (<= -0.08) ; NDBI anomaly ~ 0
    c = classify_candidate(_spec(d_ndvi=0.20, d_ndbi=-0.27))
    assert c.change_type == "clearance"


def test_typing_road_needs_linear_shape():
    blocky = classify_candidate(_spec(d_ndbi=-0.20, elongation=1.5))   # NDBI anom +0.07
    linear = classify_candidate(_spec(d_ndbi=-0.20, elongation=5.0))
    assert blocky.change_type == "construction"      # same spectra, compact -> construction
    assert linear.change_type == "road"              # elongated -> road


def test_typing_other_when_nothing_matches():
    c = classify_candidate(_spec(d_ndvi=0.35, d_ndbi=-0.27, d_ndwi=0.0))  # tracks the scene exactly
    assert c.change_type == "other" and c.rule == "no_rule_matched"


def test_typing_uses_anomaly_not_raw_delta():
    # raw NDVI delta is very negative, but it exactly matches the scene trend -> NOT clearance
    c = classify_candidate(_spec(d_ndvi=0.35, d_ndbi=-0.27))
    assert c.change_type != "clearance"


def test_class_distribution_covers_all_types():
    cs = [classify_candidate(_spec(d_ndwi=0.2)), classify_candidate(_spec(d_ndbi=-0.05, d_ndvi=0.30))]
    d = class_distribution(cs)
    assert set(d) == {"construction", "clearance", "water_gain", "water_loss", "road", "other"}
    assert d["water_gain"] == 1 and d["construction"] == 1


# --------------------------------------------------------------------------
# Step C - temporal persistence
# --------------------------------------------------------------------------


def _persistence_repo(tmp_path):
    from geoseek.catalog.entities import Collection, Observation, Scene
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository

    repo = SQLiteMetadataRepository(tmp_path / "cat.sqlite")
    repo.register_collection(Collection(collection_id="sentinel-2-l2a", sensor="MSI",
                                        platform="Sentinel-2", bands=("B04", "B08"), native_gsd_m=10.0))
    aoi = "POLYGON ((82.0 26.5, 82.6 26.5, 82.6 27.1, 82.0 27.1, 82.0 26.5))"
    for oid, date, plat in (("O2019_scaled", "2019-03-30", "Sentinel-2B"),
                            ("O2021_scaled", "2021-03-04", "Sentinel-2A"),
                            ("O2024_scaled", "2024-03-08", "Sentinel-2A")):
        repo.register_scene(Scene(scene_id=oid.replace("_scaled", ""), collection_id="sentinel-2-l2a",
                                  platform=plat, acquired_at=date, footprint_wkt_4326=aoi))
        repo.register_observation(Observation(
            observation_id=oid, scene_id=oid.replace("_scaled", ""), acquired_at=date,
            footprint_wkt_4326=aoi, aoi_name="ayodhya", dataset_dir=oid,
            quality_summary={"n_tiles": 100, "cloud_fraction_mean": 0.0, "cloud_fraction_max": 0.0,
                             "n_clear_tiles_cf_le_0p05": 100},
            coregistration={"reference_scene": "O2024_scaled", "median_magnitude_px": 0.15}))
    return repo


def _analyzer(tmp_path):
    from geoseek.temporal.matcher import TemporalObservationMatcher
    from geoseek.temporal.persistence import TemporalPersistenceAnalyzer

    return TemporalPersistenceAnalyzer(TemporalObservationMatcher(_persistence_repo(tmp_path)))


def _lookup(mapping):
    """mapping: {(earlier_id, later_id): (changed, prob)}; default (False, 0.1)."""
    return lambda e, l, lon, lat: mapping.get((e, l), (False, 0.1))


def test_persistent_change_high_confidence(tmp_path):
    a = _analyzer(tmp_path)
    lk = _lookup({("O2019_scaled", "O2021_scaled"): (True, 0.9),
                  ("O2019_scaled", "O2024_scaled"): (True, 0.9),
                  ("O2021_scaled", "O2024_scaled"): (False, 0.1)})
    tr = a.trajectory_for_location(82.3, 26.8, lk)
    assert tr.persistence == "persistent"
    assert tr.persistence_confidence == pytest.approx(0.95)
    es = tr.earliest_supported
    assert es["window"] == ["2019-03-30", "2021-03-04"]
    assert es["supporting_observations"] == ["O2019_scaled", "O2021_scaled"]
    assert "earlier than the earliest usable observation (2019-03-30)" in es["caveat"]


def test_transient_change_is_downweighted(tmp_path):
    a = _analyzer(tmp_path)
    lk = _lookup({("O2019_scaled", "O2021_scaled"): (True, 0.85)})   # span pair NOT changed
    tr = a.trajectory_for_location(82.3, 26.8, lk)
    assert tr.persistence == "transient"
    assert tr.persistence_confidence < 0.3


def test_recent_change(tmp_path):
    a = _analyzer(tmp_path)
    lk = _lookup({("O2021_scaled", "O2024_scaled"): (True, 0.9),
                  ("O2019_scaled", "O2024_scaled"): (True, 0.9)})
    tr = a.trajectory_for_location(82.3, 26.8, lk)
    assert tr.persistence == "recent"
    assert tr.earliest_supported["window"] == ["2021-03-04", "2024-03-08"]


def test_progressive_change(tmp_path):
    a = _analyzer(tmp_path)
    lk = _lookup({("O2019_scaled", "O2021_scaled"): (True, 0.9),
                  ("O2021_scaled", "O2024_scaled"): (True, 0.9),
                  ("O2019_scaled", "O2024_scaled"): (True, 0.9)})
    tr = a.trajectory_for_location(82.3, 26.8, lk)
    assert tr.persistence == "progressive"


def test_trajectory_report_renders(tmp_path):
    a = _analyzer(tmp_path)
    lk = _lookup({("O2019_scaled", "O2021_scaled"): (True, 0.9),
                  ("O2019_scaled", "O2024_scaled"): (True, 0.9),
                  ("O2021_scaled", "O2024_scaled"): (False, 0.1)})
    txt = a.trajectory_for_location(82.3, 26.8, lk).format_report()
    assert "PERSISTENT" in txt and "earliest supported change" in txt and "caveat" in txt


# --------------------------------------------------------------------------
# Step D - confidence engine
# --------------------------------------------------------------------------


def _conf(**kw):
    d = dict(candidate_id="c", model_prob=0.9, bad_scl_fraction=0.0, valid_fraction=1.0,
             coreg_residual_px=0.15, radiometric_reliability=1.0, persistence="persistent",
             persistence_confidence=0.95, spectral_agreement=0.9, change_type="construction")
    d.update(kw)
    return compute_confidence(**d)


def test_all_evidence_strong_gives_high_confidence():
    r = _conf()
    assert r.confidence > 0.75
    assert any("persistence" in b for b in r.breakdown)
    assert any("=> confidence" in b for b in r.breakdown)


def test_single_weak_term_collapses_confidence_geometric_mean():
    strong = _conf().confidence
    transient = _conf(persistence="transient", persistence_confidence=0.20).confidence
    cloudy = _conf(bad_scl_fraction=0.045, valid_fraction=0.6).confidence
    assert strong > 0.8
    assert transient < 0.4 < strong           # temporally-unsupported cannot score "medium"
    assert cloudy < strong - 0.15             # a marginal-quality footprint is dragged down
    # an arithmetic mean of the same terms would keep transient > 0.8 - prove it is multiplicative
    assert transient < strong - 0.4


def test_confidence_is_not_the_raw_model_probability():
    r = _conf(model_prob=0.83)
    assert abs(r.confidence - 0.83) > 0.02
    model_term = next(t for t in r.terms if t.name == "model")
    assert model_term.value == pytest.approx(_rescale_model_prob(0.83))


def test_model_prob_rescale_endpoints():
    assert _rescale_model_prob(0.50) == pytest.approx(0.0)
    assert _rescale_model_prob(0.99) == pytest.approx(1.0)
    assert 0.5 < _rescale_model_prob(0.80) < 0.75


def test_suppression_downweight_folds_into_confidence():
    full = _conf().confidence
    dw = _conf(suppression_downweight=0.5).confidence
    assert dw == pytest.approx(full * 0.5, rel=1e-6)


def test_spectral_agreement_scales_with_evidence():
    weak = spectral_agreement_for("construction", ndvi_anomaly=-0.02, ndbi_anomaly=0.05, ndwi_anomaly=0.0)
    strong = spectral_agreement_for("construction", ndvi_anomaly=-0.15, ndbi_anomaly=0.30, ndwi_anomaly=0.0)
    assert strong > weak
    assert spectral_agreement_for("other", 0.0, 0.0, 0.0) == pytest.approx(0.35)
    wg_weak = spectral_agreement_for("water_gain", 0.0, 0.0, ndwi_anomaly=0.15)
    wg_strong = spectral_agreement_for("water_gain", 0.0, 0.0, ndwi_anomaly=0.45)
    assert wg_strong > wg_weak


def test_weights_documented_for_every_term():
    r = _conf()
    assert {t.name for t in r.terms} == set(WEIGHTS)


# --------------------------------------------------------------------------
# Phase 5 Step A - ranking by confidence AND significance, + diversity
# --------------------------------------------------------------------------

from geoseek.change.analyze import Candidate, diversify, rank_score, significance


def _cand(cid, *, area_px, conf, ctype="water_gain", lon=82.2, lat=26.8, anom=0.5):
    c = Candidate(candidate_id=cid, pair_id="p", earlier_obs="e", later_obs="l", label=1,
                  area_px=area_px, bbox=(0, 0, 1, 1), centroid_rc=(0.0, 0.0),
                  centroid_lonlat=(lon, lat), mean_prob=conf, max_prob=conf,
                  elongation=1.0, fill_ratio=1.0)
    c.confidence = conf
    c.classification = {"change_type": ctype,
                        "evidence": {"ndvi_anomaly": 0.0, "ndbi_anomaly": 0.0, "ndwi_anomaly": anom}}
    return c


def test_significance_rises_with_area_and_anomaly():
    small = _cand("s", area_px=60, conf=0.9, anom=0.1)
    big = _cand("b", area_px=3000, conf=0.9, anom=0.1)
    strong = _cand("x", area_px=60, conf=0.9, anom=0.6)
    assert significance(big) > significance(small)
    assert significance(strong) > significance(small)
    assert 0.0 < significance(small) <= 1.0


def test_rank_score_confidence_still_leads():
    big_weak = _cand("bw", area_px=5000, conf=0.55, anom=0.6)      # huge, low confidence
    small_strong = _cand("ss", area_px=200, conf=0.92, anom=0.5)   # 0.2 ha, high confidence
    assert rank_score(small_strong) > rank_score(big_weak)


def test_rank_score_significance_breaks_ties():
    tank = _cand("tank", area_px=1476, conf=0.90, anom=1.05)       # the real 14.8 ha tank
    speck = _cand("speck", area_px=23, conf=0.90, anom=0.30)       # the Phase 4 speck (would be gone anyway)
    assert rank_score(tank) > rank_score(speck)


def test_diversify_caps_per_type_and_spreads_locations():
    ranked = [
        _cand("w1", area_px=2000, conf=0.95, ctype="water_gain", lon=82.10, lat=26.60),
        _cand("w2", area_px=1900, conf=0.94, ctype="water_gain", lon=82.1005, lat=26.6003),  # ~40 m from w1
        _cand("w3", area_px=1800, conf=0.93, ctype="water_gain", lon=82.30, lat=26.80),
        _cand("w4", area_px=1700, conf=0.92, ctype="water_gain", lon=82.50, lat=26.50),
        _cand("w5", area_px=1600, conf=0.91, ctype="water_gain", lon=82.70, lat=27.00),
        _cand("c1", area_px=1500, conf=0.90, ctype="construction", lon=82.20, lat=26.75),
        _cand("r1", area_px=1400, conf=0.89, ctype="road", lon=82.25, lat=26.70),
    ]
    out = diversify(ranked, n=5, per_type_cap=3, min_sep_m=1500.0)
    ids = [c.candidate_id for c in out]
    assert "w2" not in ids                       # near-duplicate of w1 (same type, < 1.5 km)
    types = [c.classification["change_type"] for c in out]
    assert types.count("water_gain") == 3        # per-type cap
    assert "construction" in types and "road" in types   # other types surface
    # a request the caps cannot fill back-fills the near-duplicate rather than returning short
    filled = diversify(ranked, n=6, per_type_cap=3, min_sep_m=1500.0)
    assert len(filled) == 6 and "w2" in [c.candidate_id for c in filled]


def test_diversify_keeps_full_list_orthogonal():
    ranked = [_cand(f"w{i}", area_px=1000, conf=0.9 - i * 0.01, lon=82.0 + i, lat=26.5)
              for i in range(8)]
    out = diversify(ranked, n=3, per_type_cap=3, min_sep_m=1500.0)
    assert len(out) == 3 and len(ranked) == 8     # diversify does not mutate / shrink the source

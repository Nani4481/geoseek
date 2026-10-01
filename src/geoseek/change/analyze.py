"""Phase 4 Step E - the analyst-grade change pipeline over Ayodhya.

Ties the trained FC-Siam-diff model to the four Phase 4 stages:

    model probability raster (per pair)
      -> raw candidates            (connected components >= threshold)
      -> Step A  geoseek.change.suppress     (5 ordered gates + trace)
      -> Step B  geoseek.change.classify     (rule-based typing)
      -> Step C  geoseek.temporal.persistence (trajectory + earliest supported change)
      -> Step D  geoseek.change.confidence   (one calibrated score + evidence)
      -> report + [<every staged date> | overlay] panels

    python -m geoseek.change.analyze [--pairs <comma list, default: every
                                     consecutive pair across every ingested
                                     date + the full span>]
                                     [--top 10] [--panels 5] [--refresh] [--no-panels]

Offline. Reads only from ``data/datasets/`` + the staged model. The heavy
per-pair probability rasters are cached under ``data/change_model/`` and reused
unless ``--refresh``. Originally 3 dates (2019-03-30 / 2021-03-04 / 2024-03-08);
Phase 8 Step A extended the stack to 5 (added 2025-03-08 / 2026-03-08) - every
date/pair-dependent piece of this module (:func:`discover_date_to_obs`,
:func:`build_pair_keys`, :func:`pair_context`) discovers the current stack from
the catalog + manifest rather than hardcoding it, so a 6th date needs no code
change here, only staging + ingest + an alignment run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyproj
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from geoseek.change.classify import CandidateSpectra, classify_candidate, class_distribution
from geoseek.change.confidence import compute_confidence, spectral_agreement_for
from geoseek.change.indices import compute_indices
from geoseek.change.normalize import REFLECTANCE_SCALE, RadiometricNormalization
from geoseek.change.suppress import (
    CandidateFeatures,
    MORPH_MIN_AREA_PX,
    PairSuppressionContext,
    SuppressionConfig,
    suppress_candidate,
    summarize_suppression,
    PIXEL_AREA_M2,
)
from geoseek.config import get_settings
from geoseek.ingest.embed import make_true_color_uint8
from geoseek.ingest.quality import BAD_SCL_CLASSES
from geoseek.staging.manifest import load_manifest, record_analysis_section
from geoseek.temporal.matcher import TemporalObservationMatcher
from geoseek.temporal.persistence import TemporalPersistenceAnalyzer

warnings.filterwarnings("ignore", message=".*no geotransform.*")

OUT_DIR = get_settings().data_dir / "change_model"
AOI_POINT = (82.1998, 26.7922)   # Ayodhya (Ram Janmabhoomi) - the AOI anchor
INDEX_BANDS = ("B04", "B03", "B02", "B08", "B11")
SCL_WATER = 6
S2_COLLECTION = "sentinel-2-l2a"   # the change pipeline works one collection at a time

# The original Phase 3.5/4 stack, kept as a fallback so this module stays
# importable (and every unit test that only needs light symbols like
# ``Candidate``/``diversify`` keeps working) without a live catalog - e.g. a
# fresh checkout before staging has run. Once the catalog has observations at
# the AOI, :func:`discover_date_to_obs` supersedes this with whatever dates
# are actually ingested (3, 5, or more - Phase 8 Step A added 2025 + 2026).
_STATIC_DATE_TO_OBS = {
    "2019": "S2B_44RPQ_20190330_1_L2A_scaled",
    "2021": "S2A_44RPQ_20210304_1_L2A_scaled",
    "2024": "S2A_44RPQ_20240308_0_L2A_scaled",
}


def discover_date_to_obs(repo=None) -> dict[str, str]:
    """``{year_label: observation_id}`` for every Sentinel-2 observation at the
    Ayodhya AOI, live from the catalog, ordered by acquisition date (dict
    insertion order == time order, relied on by :func:`build_pair_keys`).

    Falls back to :data:`_STATIC_DATE_TO_OBS` if the catalog is unavailable or
    has fewer than 3 AOI observations - keeps this module safely importable
    without a live DB (unit tests importing light symbols) and gives an
    unambiguous answer when a scene has been staged but not yet ingested.
    """
    owns_repo = repo is None
    try:
        if owns_repo:
            from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
            repo = SQLiteMetadataRepository(get_settings().database_path)
        obs = repo.list_observations(location=AOI_POINT, collection=S2_COLLECTION)
        obs.sort(key=lambda o: o.acquired_at)
        if len(obs) < 3:
            return dict(_STATIC_DATE_TO_OBS)
        out: dict[str, str] = {}
        for o in obs:
            year = o.acquired_at[:4]
            key = year if year not in out else o.acquired_at[:7]  # guard a same-year collision
            out[key] = o.observation_id
        return out
    except Exception:
        return dict(_STATIC_DATE_TO_OBS)
    finally:
        if owns_repo and repo is not None:
            repo.close()


DATE_TO_OBS = discover_date_to_obs()
MAX_COMPONENTS = 20000            # process the largest N; the rest fold into the morphology count
_HUGE_BBOX_PX = 4_000_000         # subsample bbox reads / geometry above this area (memory guard)
BAD_SCL_ARR = np.array(sorted(BAD_SCL_CLASSES))


# ==========================================================================
# normalized spectral indices per observation (only 2019/2024 have Phase 3a
# indices on disk natively; every other date is normalized onto the 2024
# reference from its own alignment record before indices are computed)
# ==========================================================================


def _reference_obs() -> str | None:
    return DATE_TO_OBS.get("2024")


def _alignment_record_for(obs_id: str, m: dict) -> dict | None:
    """``{"coregistration", "radiometric_normalization"}`` for ``obs_id`` vs the
    fixed 2024 reference, wherever it was written - the original Phase 3a
    top-level sections for 2019, ``third_date_alignment`` for 2021, or Phase
    8's ``additional_dates_alignment`` (keyed by observation id) for any later
    date. ``None`` for the reference observation itself (2024) or an
    observation with no recorded alignment."""
    if obs_id == _reference_obs():
        return None
    if obs_id == DATE_TO_OBS.get("2019"):
        return {"coregistration": m.get("coregistration", {}),
                "radiometric_normalization": m.get("radiometric_normalization", {})}
    if obs_id == DATE_TO_OBS.get("2021"):
        return m.get("third_date_alignment")
    return (m.get("additional_dates_alignment") or {}).get(obs_id)


def _norm_for_obs(obs_id: str) -> RadiometricNormalization:
    """Scene-wide additive offsets for ``obs_id`` from its recorded alignment-vs-reference."""
    rec = _alignment_record_for(obs_id, load_manifest())
    if rec is None:
        raise RuntimeError(
            f"no alignment record for {obs_id} - stage + align it first "
            f"(scripts/align_third_date.py or scripts/align_additional_dates.py)"
        )
    rn = rec["radiometric_normalization"]
    per_band = {b: {"gain": 1.0, "offset": float(c["offset_dn"]), "corr": c.get("inter_date_corr"),
                    "n": c.get("n", 0), "rmse_dn": c.get("rmse_dn", 0.0), "method": "additive"}
                for b, c in rn["per_band_dn"].items()}
    return RadiometricNormalization(
        reference_scene=rn["reference_scene"], subject_scene=rn["subject_scene"],
        reference_date=rn["reference_date"], subject_date=rn["subject_date"],
        per_band=per_band, n_pif=rn.get("n_pseudo_invariant_pixels", 0),
        method="pif_additive_offset_scene_wide",
    )


def ensure_indices(obs_id: str) -> dict[str, Path]:
    """Return {NDVI,NDWI,NDBI: path}. 2019/2024 already written by Phase 3a; compute any other date if missing."""
    d = get_settings().datasets_dir / obs_id
    paths = {k: d / f"{k}.tif" for k in ("NDVI", "NDWI", "NDBI")}
    if all(p.is_file() for p in paths.values()):
        return paths

    print(f"  [indices] computing normalized NDVI/NDWI/NDBI for {obs_id} (not staged by Phase 3a) ...")
    norm = None if obs_id in (DATE_TO_OBS.get("2019"), _reference_obs()) else _norm_for_obs(obs_id)
    bands = {}
    with rasterio.open(d / "B04.tif") as ref:
        prof = {"driver": "GTiff", "height": ref.height, "width": ref.width, "count": 1,
                "dtype": "float32", "crs": ref.crs, "transform": ref.transform,
                "nodata": float("nan"), "compress": "deflate", "predictor": 3,
                "tiled": True, "blockxsize": 256, "blockysize": 256}
    for b in ("B04", "B03", "B08", "B11"):
        with rasterio.open(d / f"{b}.tif") as ds:
            dn = ds.read(1)
        refl = (norm.apply(dn, b).astype(np.float32) if norm is not None else dn.astype(np.float32))
        refl = refl / REFLECTANCE_SCALE
        refl[dn == 0] = np.nan
        bands[b] = refl
    idx = compute_indices(bands)
    for k, arr in idx.items():
        with rasterio.open(paths[k], "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)
    key = "third_date_indices" if obs_id == DATE_TO_OBS.get("2021") else "additional_date_indices"
    section = {
        "observation_id": obs_id,
        "note": (f"NDVI/NDWI/NDBI on {obs_id} reflectance mapped into the 2024 reference frame with its "
                 "recorded scene-wide PIF additive offsets. Same formulas as spectral_indices."),
        "normalization": f"scene-wide additive per-band offset ({obs_id} -> 2024)",
        "paths": {k: str(v) for k, v in paths.items()},
    }
    if key == "additional_date_indices":
        existing = dict(load_manifest().get(key, {}))
        existing[obs_id] = section
        record_analysis_section(key, existing)
    else:
        record_analysis_section(key, section)
    return paths


# ==========================================================================
# probability rasters
# ==========================================================================


def probability_raster(model, pair, *, refresh: bool) -> "object":
    from geoseek.change.models import ChangeProbabilityRaster  # noqa: F401

    tag = f"{pair.earlier.observation_id}__to__{pair.later.observation_id}"
    out = OUT_DIR / f"prob_{tag}.tif"
    if out.is_file() and not refresh:
        print(f"  [prob] reuse {out.name}")
        with rasterio.open(out) as ds:
            prob = ds.read(1).astype(np.float32)
            tr, crs = ds.transform, ds.crs
        from geoseek.change.models import ChangeProbabilityRaster as CPR
        return CPR(prob=prob, valid=(prob > 0), transform=tr, crs=crs,
                   earlier_observation_id=pair.earlier.observation_id,
                   later_observation_id=pair.later.observation_id,
                   threshold=float(model.threshold), path=str(out))
    print(f"  [prob] running FC-Siam-diff over {tag} (this is the slow step) ...")
    t0 = time.time()
    pr = model.infer_probability_raster(pair, out_path=out, progress_every=200)
    print(f"  [prob] done in {time.time()-t0:.0f}s -> {out.name}")
    return pr


# ==========================================================================
# candidate extraction
# ==========================================================================


@dataclass
class Candidate:
    candidate_id: str
    pair_id: str
    earlier_obs: str
    later_obs: str
    label: int
    area_px: int
    bbox: tuple[int, int, int, int]     # r0, c0, r1, c1  (r1/c1 exclusive)
    centroid_rc: tuple[float, float]
    centroid_lonlat: tuple[float, float]
    mean_prob: float
    max_prob: float
    elongation: float
    fill_ratio: float
    # filled downstream
    d_ndvi: float = 0.0
    d_ndbi: float = 0.0
    d_ndwi: float = 0.0
    bad_scl_earlier: float = 0.0
    bad_scl_later: float = 0.0
    valid_fraction: float = 1.0
    suppression: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    trajectory: dict = field(default_factory=dict)
    confidence: float = 0.0
    confidence_breakdown: list = field(default_factory=list)
    significance: float = 0.0
    queue_score: float = 0.0
    sar: dict = field(default_factory=dict)
    terrain: dict = field(default_factory=dict)


def _elongation_fill(mask_local: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask_local)
    n = ys.size
    fill = n / float(mask_local.shape[0] * mask_local.shape[1])
    if n < 3:
        return 1.0, fill
    cov = np.cov(np.vstack([ys.astype(np.float64), xs.astype(np.float64)]))
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-6, None)
    return float(np.sqrt(ev[-1] / ev[0])), fill


def label_change(pr):
    """8-connected connected-component labels of (prob >= threshold) & valid."""
    binary = (pr.prob >= pr.threshold) & pr.valid
    labels, n = ndi.label(binary, structure=np.ones((3, 3), bool))
    return binary, labels, n


def extract_candidates(pr, pair, labels, n) -> tuple[list[Candidate], int]:
    """Returns (candidates, n_dropped_tiny).

    Components below MORPH_MIN_AREA_PX are attributed directly to the morphology
    rule (rule 5) without the other four checks and counted in n_dropped_tiny -
    they cannot be a reported change at any confidence, and skipping their
    per-bbox raster reads is a large speed-up. Components beyond MAX_COMPONENTS
    (pathological only) are folded in the same way. The raw candidate total
    stays honest: len(candidates) + n_dropped_tiny."""
    if n == 0:
        return [], 0
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    ranked = [int(l) for l in np.argsort(areas)[::-1] if areas[l] > 0]
    above = [l for l in ranked if int(areas[l]) >= MORPH_MIN_AREA_PX]
    keep = above[:MAX_COMPONENTS]
    n_dropped_tiny = len(ranked) - len(keep)
    slices = ndi.find_objects(labels)
    to_wgs = pyproj.Transformer.from_crs(pr.crs, "EPSG:4326", always_xy=True)
    tr = pr.transform

    cands: list[Candidate] = []
    for li in keep:
        sl = slices[li - 1]
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        area = int(areas[li])
        # huge components: subsample the bbox for geometry to keep coord arrays small
        k = 4 if (r1 - r0) * (c1 - c0) > _HUGE_BBOX_PX else 1
        sub = (labels[sl] == li)[::k, ::k]
        pr_sub = pr.prob[sl][::k, ::k][sub]
        ys, xs = np.nonzero(sub)
        cr, cc = float(ys.mean() * k + r0), float(xs.mean() * k + c0)
        ux, uy = tr * (cc + 0.5, cr + 0.5)
        lon, lat = to_wgs.transform(ux, uy)
        elong, fill = _elongation_fill(sub)
        cands.append(Candidate(
            candidate_id=f"{pair.earlier.acquired_at[:4]}_{pair.later.acquired_at[:4]}_{li:06d}",
            pair_id=f"{pair.earlier.observation_id}->{pair.later.observation_id}",
            earlier_obs=pair.earlier.observation_id, later_obs=pair.later.observation_id,
            label=li, area_px=area, bbox=(r0, c0, r1, c1), centroid_rc=(cr, cc),
            centroid_lonlat=(float(lon), float(lat)),
            mean_prob=float(pr_sub.mean()) if pr_sub.size else 0.0,
            max_prob=float(pr_sub.max()) if pr_sub.size else 0.0,
            elongation=elong, fill_ratio=fill))
    return cands, n_dropped_tiny


# ==========================================================================
# per-candidate features (index deltas + SCL quality) - windowed, memory-frugal
# ==========================================================================


def _win(ds, bbox, step=1):
    r0, c0, r1, c1 = bbox
    a = ds.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
    return a[::step, ::step]


def attach_features(cands: list[Candidate], pair, idx_paths: dict, labels: np.ndarray) -> None:
    """Fill index deltas + SCL quality per candidate via per-bbox windowed reads (no full rasters)."""
    e, l = pair.earlier.observation_id, pair.later.observation_id
    e_dir = get_settings().datasets_dir / e
    l_dir = get_settings().datasets_dir / l
    ei = {k: rasterio.open(idx_paths[e][k]) for k in ("NDVI", "NDBI", "NDWI")}
    li = {k: rasterio.open(idx_paths[l][k]) for k in ("NDVI", "NDBI", "NDWI")}
    es, ls_ = rasterio.open(e_dir / "SCL.tif"), rasterio.open(l_dir / "SCL.tif")
    eb, lb = rasterio.open(e_dir / "B08.tif"), rasterio.open(l_dir / "B08.tif")
    try:
        for c in cands:
            r0, c0, r1, c1 = c.bbox
            step = 4 if (r1 - r0) * (c1 - c0) > _HUGE_BBOX_PX else 1
            sub = (labels[r0:r1, c0:c1] == c.label)[::step, ::step]
            for name, attr in (("NDVI", "d_ndvi"), ("NDBI", "d_ndbi"), ("NDWI", "d_ndwi")):
                w = _win(li[name], c.bbox, step) - _win(ei[name], c.bbox, step)
                m = sub & np.isfinite(w)
                setattr(c, attr, float(w[m].mean()) if m.any() else 0.0)
            se, sl = _win(es, c.bbox, step), _win(ls_, c.bbox, step)
            c.bad_scl_earlier = float(np.isin(se[sub], BAD_SCL_ARR).mean()) if sub.any() else 1.0
            c.bad_scl_later = float(np.isin(sl[sub], BAD_SCL_ARR).mean()) if sub.any() else 1.0
            vb = (_win(eb, c.bbox, step) > 0) & (_win(lb, c.bbox, step) > 0)
            c.valid_fraction = float((sub & vb).sum() / max(sub.sum(), 1))
    finally:
        for ds in list(ei.values()) + list(li.values()) + [es, ls_, eb, lb]:
            ds.close()


# ==========================================================================
# per-pair context (seasonal deltas, co-registration, radiometric reliability)
# ==========================================================================


def scene_seasonal_deltas(e_idx: dict, l_idx: dict, *, n_windows: int = 600,
                          win: int = 64) -> tuple[float, float, float]:
    """Robust scene-wide NDVI / NDBI / NDWI delta (later - earlier) from random windowed samples."""
    with rasterio.open(e_idx["NDVI"]) as ds:
        H, W = ds.height, ds.width
    rng = np.random.default_rng(0)
    boxes = [(int(rng.integers(0, H - win)), int(rng.integers(0, W - win))) for _ in range(n_windows)]
    out = []
    for name in ("NDVI", "NDBI", "NDWI"):
        diffs = []
        with rasterio.open(e_idx[name]) as de, rasterio.open(l_idx[name]) as dl:
            for (r0, c0) in boxes:
                w = rasterio.windows.Window(c0, r0, win, win)
                d = dl.read(1, window=w) - de.read(1, window=w)
                diffs.append(d[np.isfinite(d)])
        v = np.concatenate(diffs) if diffs else np.array([0.0])
        out.append(float(np.median(v)) if v.size else 0.0)
    return out[0], out[1], out[2]


def _radiometric_corrs_for(obs_id: str, m: dict) -> list[float]:
    """Inter-date correlation of the index-driving bands for ``obs_id`` vs the 2024
    reference - [1.0] (perfect self-agreement) for the reference itself."""
    if obs_id == _reference_obs():
        return [1.0]
    rec = _alignment_record_for(obs_id, m)
    rn = (rec or {}).get("radiometric_normalization", {}).get("per_band_dn", {})
    corrs = [rn[b]["inter_date_corr"] for b in ("B03", "B04", "B08", "B11")
             if rn.get(b, {}).get("inter_date_corr") is not None]
    return corrs or [1.0]


def pair_context(pair, seq, e_idx, l_idx) -> PairSuppressionContext:
    m = load_manifest()
    earlier, later = pair.earlier.observation_id, pair.later.observation_id
    # co-registration residual for this pair
    resid, corrected = _coreg_residual(m, earlier, later)
    # radiometric: min inter-date correlation among the index-driving bands, over
    # whichever endpoint(s) are not themselves the 2024 reference (both, for a pair
    # like 2019-2021 that doesn't touch the reference at all - the worse of the two
    # own normalization-vs-reference records governs a transitive comparison).
    non_ref = [o for o in (earlier, later) if o != _reference_obs()] or [earlier]
    corrs = min((_radiometric_corrs_for(o, m) for o in non_ref), key=lambda c: min(c))
    low = False
    sd_ndvi, sd_ndbi, sd_ndwi = scene_seasonal_deltas(e_idx, l_idx)
    return PairSuppressionContext(
        pair_id=f"{earlier}->{later}",
        coreg_residual_px=float(resid), coreg_corrected=bool(corrected),
        scene_d_ndvi=sd_ndvi, scene_d_ndbi=sd_ndbi, scene_d_ndwi=sd_ndwi,
        radiometric_index_band_min_corr=float(min(corrs)),
        radiometric_low_confidence=bool(low),
    )


def radiometric_reliability(min_corr: float) -> float:
    """Smoothly map the index-driving bands' worst inter-date correlation to a [0.5, 1] reliability.

    ~0.69 for the 2019<->2024 pair (B03/B04 only ~0.7 correlated raw), ~0.89 for the 2021 pairs.
    """
    return float(max(0.5, min(1.0, (min_corr - 0.55) / 0.35)))


def _spectra_for(c: "Candidate", ctx: PairSuppressionContext) -> CandidateSpectra:
    return CandidateSpectra(
        candidate_id=c.candidate_id, d_ndvi=c.d_ndvi, d_ndbi=c.d_ndbi, d_ndwi=c.d_ndwi,
        scene_d_ndvi=ctx.scene_d_ndvi, scene_d_ndbi=ctx.scene_d_ndbi, scene_d_ndwi=ctx.scene_d_ndwi,
        area_px=c.area_px, elongation=c.elongation, fill_ratio=c.fill_ratio)


def _coreg_vs_reference(obs_id: str, m: dict) -> dict:
    """The recorded ``coregistration`` dict for ``obs_id`` vs the fixed 2024 reference."""
    if obs_id == _reference_obs():
        return {}
    return (_alignment_record_for(obs_id, m) or {}).get("coregistration", {})


def _coreg_residual(m: dict, earlier: str, later: str) -> tuple[float, bool]:
    """Residual px + corrected flag for a pair, from the recorded co-registration sections.

    Every date is registered against the SAME fixed 2024 reference, so a pair
    where one end IS the reference reads that one record directly; a pair
    where NEITHER end is the reference (e.g. 2019-2021, or Phase 8's
    2025-2026) is transitive - the two residuals to the common reference add.
    With exactly the original 3 dates this reproduces the Phase 3.5/4 logic
    byte-for-byte."""
    ref = _reference_obs()
    if earlier == ref and later == ref:
        return 0.0, False
    if earlier == ref or later == ref:
        other = later if earlier == ref else earlier
        c = _coreg_vs_reference(other, m)
        return float(c.get("median_magnitude_px", 0.0)), bool(c.get("correction_applied", False))
    ce, cl = _coreg_vs_reference(earlier, m), _coreg_vs_reference(later, m)
    return (float(ce.get("median_magnitude_px", 0.0)) + float(cl.get("median_magnitude_px", 0.0))), False


# ==========================================================================
# temporal persistence lookup (samples the 3 cached probability rasters)
# ==========================================================================


class ProbLookup:
    """Samples the on-disk per-pair probability GeoTIFFs (memory-frugal - no full rasters held)."""

    def __init__(self, prob_paths: dict[tuple[str, str], str], threshold: float, crs):
        self.paths = prob_paths
        self.threshold = threshold
        self._ds = {k: rasterio.open(p) for k, p in prob_paths.items()}
        self.to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def close(self):
        for ds in self._ds.values():
            ds.close()

    def __call__(self, earlier_obs: str, later_obs: str, lon: float, lat: float):
        ds = self._ds.get((earlier_obs, later_obs))
        if ds is None:
            return False, 0.0
        ux, uy = self.to_utm.transform(lon, lat)
        col, row = (~ds.transform) * (ux, uy)
        r, c = int(round(row)), int(round(col))
        if not (0 <= r < ds.height and 0 <= c < ds.width):
            return False, 0.0
        r0, c0 = max(r - 1, 0), max(c - 1, 0)
        win = ds.read(1, window=rasterio.windows.Window(c0, r0, min(3, ds.width - c0),
                                                        min(3, ds.height - r0)))
        v = float(win[win > 0].mean()) if (win > 0).any() else 0.0
        return v >= self.threshold, v


# ==========================================================================
# panels
# ==========================================================================


def _crop_true_color(obs_id: str, bbox, margin_frac: float, min_px: int) -> tuple[np.ndarray, tuple]:
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0, c1 - c0
    mh, mw = max(int(h * margin_frac), (min_px - h) // 2, 8), max(int(w * margin_frac), (min_px - w) // 2, 8)
    d = get_settings().datasets_dir / obs_id
    with rasterio.open(d / "B04.tif") as ds:
        H, W = ds.height, ds.width
    R0, C0 = max(r0 - mh, 0), max(c0 - mw, 0)
    R1, C1 = min(r1 + mh, H), min(c1 + mw, W)
    win = rasterio.windows.Window(C0, R0, C1 - C0, R1 - R0)
    bands = {}
    for b in ("B04", "B03", "B02"):
        with rasterio.open(d / f"{b}.tif") as ds:
            bands[b] = ds.read(1, window=win)
    rgb = make_true_color_uint8(bands, nodata=0)
    return rgb, (R0, C0, R1, C1)


def save_panel(cand: Candidate, span_prob_path: str, threshold: float, out: Path,
               dates: tuple[str, ...] | None = None) -> Path:
    dates = dates or tuple(DATE_TO_OBS)   # every staged date, in acquisition order (3 originally, 5+ from Phase 8)
    tiles = []
    win = None
    for date in dates:
        rgb, win = _crop_true_color(DATE_TO_OBS[date], cand.bbox, margin_frac=1.5, min_px=280)
        tiles.append((date, rgb))
    R0, C0, R1, C1 = win
    with rasterio.open(span_prob_path) as ds:
        pw = ds.read(1, window=rasterio.windows.Window(C0, R0, C1 - C0, R1 - R0))
    change_win = pw >= threshold
    lab, _ = ndi.label(change_win, structure=np.ones((3, 3), bool))
    cr, cc = cand.centroid_rc
    lr, lc = int(round(cr - R0)), int(round(cc - C0))
    this_label = lab[lr, lc] if (0 <= lr < lab.shape[0] and 0 <= lc < lab.shape[1]) else 0
    comp = (lab == this_label) if this_label else np.zeros_like(change_win)
    overlay = tiles[-1][1].copy()
    ov = overlay.astype(np.int16)
    other = change_win & ~comp
    ov[other] = (ov[other] * 0.5 + np.array([255, 230, 0]) * 0.5).astype(np.int16)
    overlay = np.clip(ov, 0, 255).astype(np.uint8)
    edge = comp ^ ndi.binary_erosion(comp)
    overlay[edge] = (230, 30, 30)
    tiles.append((f"change overlay (red=this candidate, yellow=other change {dates[0]}->{dates[-1]})", overlay))

    h, w = tiles[0][1].shape[:2]
    pad, top, bot = 8, 24, 20
    W = w * len(tiles) + pad * (len(tiles) + 1)
    Hh = h + top + bot
    canvas = Image.new("RGB", (W, Hh), "white")
    dr = ImageDraw.Draw(canvas)
    for i, (label, im) in enumerate(tiles):
        x = pad + i * (w + pad)
        canvas.paste(Image.fromarray(im), (x, top))
        dr.text((x, 6), label if len(label) < 40 else label[:39], fill=(15, 15, 15))
    lon, lat = cand.centroid_lonlat
    dr.text((pad, Hh - bot + 3),
            f"{cand.candidate_id}  ({lon:.5f}, {lat:.5f})  {cand.classification.get('change_type','?')}  "
            f"conf {cand.confidence:.2f}  area {cand.area_px} px ({cand.area_px*PIXEL_AREA_M2:.0f} m^2)",
            fill=(90, 90, 90))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ==========================================================================
# ranking + diversity (Phase 5 Step A)
# ==========================================================================

# significance references: A_MIN = the smallest reported change (= MORPH_MIN_AREA_PX),
# A_REF = a "large" change (30 ha), ANOM_REF = a strong index departure.
_A_MIN, _A_REF, _ANOM_REF = float(MORPH_MIN_AREA_PX), 3000.0, 0.5
_CONF_EXP, _SIG_EXP = 0.65, 0.35        # confidence leads the queue; significance re-orders within a band
_AREA_EXP, _ANOM_EXP = 0.6, 0.4         # inside significance, footprint > anomaly magnitude


def _clip01(x: float, lo: float = 0.10) -> float:
    return float(max(lo, min(1.0, x)))


def significance(c: "Candidate") -> float:
    """area_term^0.6 * anomaly_term^0.4, each in [0.10, 1.0]."""
    area_term = _clip01((np.log10(max(c.area_px, 1)) - np.log10(_A_MIN))
                        / (np.log10(_A_REF) - np.log10(_A_MIN)))
    ev = c.classification.get("evidence", {})
    max_anom = max(abs(ev.get("ndvi_anomaly", 0.0)), abs(ev.get("ndbi_anomaly", 0.0)),
                   abs(ev.get("ndwi_anomaly", 0.0)))
    anom_term = _clip01(max_anom / _ANOM_REF)
    return float(area_term ** _AREA_EXP * anom_term ** _ANOM_EXP)


def rank_score(c: "Candidate") -> float:
    """Analyst-queue score = confidence^0.65 * significance^0.35 (weighted geometric mean).

    Geometric mean (matching the confidence engine) so no candidate tops the queue on
    one axis alone: a huge low-confidence blob and a tiny high-confidence speck both
    sink. Confidence is weighted 0.65 vs significance 0.35 - the queue must lead with
    trustworthy detections; significance (log-area 0.6, max |index anomaly| 0.4) only
    re-orders within a confidence band so a 30 ha tank outranks a 0.5 ha marginal
    change, while a 0.55-confidence change never overtakes a 0.90-confidence one.
    """
    return float(_clip01(c.confidence, 1e-3) ** _CONF_EXP * significance(c) ** _SIG_EXP)


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    from math import asin, cos, radians, sin, sqrt

    lon1, lat1, lon2, lat2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * asin(sqrt(h))


def diversify(ranked: list["Candidate"], *, n: int = 10, per_type_cap: int = 3,
              min_sep_m: float = 1500.0) -> list["Candidate"]:
    """MMR-lite: from the rank_score-sorted list, take the headline top-N so distinct
    change TYPES and distinct LOCATIONS surface. At most ``per_type_cap`` of any one
    type, and two same-type picks must be >= ``min_sep_m`` apart (near-duplicates of
    one feature are dropped; a road next to a construction site is kept - different
    type). Back-fills from the ranked list if the caps leave < N."""
    picked: list = []
    for c in ranked:
        t = c.classification.get("change_type", "other")
        same = [p for p in picked if p.classification.get("change_type", "other") == t]
        if len(same) >= per_type_cap:
            continue
        if any(_haversine_m(c.centroid_lonlat, p.centroid_lonlat) < min_sep_m for p in same):
            continue
        picked.append(c)
        if len(picked) >= n:
            return picked
    for c in ranked:                       # rare back-fill if caps were too strict
        if c not in picked:
            picked.append(c)
            if len(picked) >= n:
                break
    return picked


# ==========================================================================
# orchestration
# ==========================================================================


def _build_repo():
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    return SQLiteMetadataRepository(get_settings().database_path)


def _component_mask(span_prob_path: str, bbox, centroid_rc, span_pair) -> np.ndarray:
    """Reconstruct a candidate's connected-component mask within its bbox from the
    on-disk span probability raster (memory-frugal - no full-scene labels held)."""
    r0, c0, r1, c1 = bbox
    with rasterio.open(span_prob_path) as ds:
        pw = ds.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
        thr = 0.80
    lab, _ = ndi.label(pw >= thr, structure=np.ones((3, 3), bool))
    lr, lc = int(round(centroid_rc[0] - r0)), int(round(centroid_rc[1] - c0))
    lr = min(max(lr, 0), lab.shape[0] - 1)
    lc = min(max(lc, 0), lab.shape[1] - 1)
    tl = lab[lr, lc]
    return (lab == tl) if tl else (pw >= thr)


# --------------------------------------------------------------------------
# Step C report: SAR cross-sensor validation on water candidates + cloud penetration
# --------------------------------------------------------------------------


def _sar_corroboration_report(span_survivors: list[Candidate], per_pair: dict, sar_pair) -> dict:
    if sar_pair is None:
        return {"available": False,
                "note": "no Sentinel-1 staged - SAR corroboration term was neutral for every candidate"}
    wg = [c for c in span_survivors if c.classification.get("change_type") == "water_gain" and c.sar]
    wg_cov = [c for c in wg if c.sar.get("available")]
    # expected: open-water gain -> VV backscatter DROP (specular). Agreement = VV anomaly <= -1 dB.
    agree = [c for c in wg_cov
             if (c.sar["vv_median_db"] - c.sar["scene_dvv_db"]) <= -1.0]
    rate = (len(agree) / len(wg_cov)) if wg_cov else None
    print(f"\n[Step C] SAR cross-sensor validation on water_gain candidates:")
    print(f"    {len(wg)} water_gain survivors; {len(wg_cov)} with usable co-located SAR "
          f"({len(wg) - len(wg_cov)} outside the S1 swath -> neutral)")
    if rate is not None:
        print(f"    show the expected VV backscatter DROP (<= -1 dB vs the scene trend): "
              f"{len(agree)}/{len(wg_cov)} = {rate*100:.0f}%")
        med = float(np.median([c.sar['vv_median_db'] - c.sar['scene_dvv_db'] for c in wg_cov]))
        print(f"    median VV anomaly over water_gain candidates: {med:+.1f} dB "
              f"(scene VV trend {sar_pair.scene_dvv_db:+.1f} dB)")
    # cloud-penetration value: optical candidates quality-suppressed for cloud but with usable SAR
    cloud_supp = sum(v["suppression"]["suppressed_by_rule"].get("quality", 0) for v in per_pair.values())
    print(f"\n[Step C] cloud-penetration value:")
    print(f"    optical components quality-suppressed (cloud/shadow/etc) across all pairs: {cloud_supp}")
    print(f"    the three Ayodhya S2 dates are ~cloud-free (bad-SCL << 1%), so the cloud-penetration "
          f"benefit of SAR is real but not demonstrable on this AOI - stated as a limitation.")
    return {
        "available": True, "s1_pair": [sar_pair.s1_earlier_obs, sar_pair.s1_later_obs],
        "aoi_coverage_fraction": round(sar_pair.coverage_fraction, 3),
        "speckle_filter": "adaptive Lee 7x7 (ENL 4.4), intensity domain, before the dB ratio",
        "scene_db_trend": {"vv": round(sar_pair.scene_dvv_db, 2), "vh": round(sar_pair.scene_dvh_db, 2)},
        "water_gain_validation": {
            "n_water_gain_survivors": len(wg), "n_with_usable_sar": len(wg_cov),
            "expected_signature": "VV backscatter DROP (specular reflection off open water)",
            "agreement_threshold_db": -1.0, "n_agree": len(agree),
            "agreement_rate": None if rate is None else round(rate, 3),
            "median_vv_anomaly_db": None if not wg_cov else
                round(float(np.median([c.sar['vv_median_db'] - c.sar['scene_dvv_db'] for c in wg_cov])), 2),
        },
        "cloud_penetration": {
            "optical_components_quality_suppressed_all_pairs": int(cloud_supp),
            "finding": ("the 2019 / 2021 / 2024 Sentinel-2 dates are near cloud-free (bad-SCL << 1%), "
                        "so SAR's all-weather value is real but cannot be demonstrated on this AOI"),
        },
    }


def _fuse_with_query(span_survivors: list[Candidate], span_pair, query: str, top: int) -> dict:
    from geoseek.fusion.ranker import FusionRanker
    from geoseek.search.engine import SearchEngine

    print(f"\n--- Step D: fusion re-ranking with text query: {query!r} ---")
    eng = SearchEngine()
    try:
        ranker = FusionRanker(eng)
        cand_dicts = [{
            "candidate_id": c.candidate_id, "confidence": c.confidence, "significance": c.significance,
            "change_type": c.classification.get("change_type"),
            "centroid_lonlat": c.centroid_lonlat, "area_m2": c.area_px * PIXEL_AREA_M2,
            "later_obs": span_pair.later.observation_id, "pair": c.pair_id,
            "persistence": c.trajectory.get("persistence"),
            "earliest_supported": c.trajectory.get("earliest_supported_change", {}).get("window"),
        } for c in span_survivors]
        ranked, meta = ranker.rank(cand_dicts, query=query)
    finally:
        eng.close()
    print(f"  formula: {meta['formula']}")
    print(f"  weights: {meta['weights']}   ({meta['n']} candidates, {meta['elapsed_s']}s)")
    print(f"  {'#':>3} {'candidate':>18} {'type':>12} {'conf':>5} {'sig':>5} {'sem':>5} {'fusion':>6}")
    for i, r in enumerate(ranked[:top], 1):
        print(f"  {i:>3} {r.candidate_id:>18} {r.change_type:>12} {r.confidence:5.2f} "
              f"{r.significance:5.2f} {('%.2f' % r.semantic) if r.semantic is not None else '  - ':>5} "
              f"{r.fusion_score:6.3f}")
    w = ranked[0]
    print(f"\n  worked example (top by fusion for {query!r}): {w.candidate_id}  [{w.change_type}]")
    for line in w.breakdown:
        print(f"       {line}")
    print(f"       {w.area_m2:.0f} m^2 @ ({w.centroid_lonlat[0]:.5f}, {w.centroid_lonlat[1]:.5f})  "
          f"tile {w.tile_id}")
    return {"query": query, **meta,
            "ranked_top": [r.as_dict() for r in ranked[:top]],
            "worked_example": ranked[0].as_dict()}


def build_pair_keys(date_to_obs: dict[str, str]) -> dict[str, tuple[str, str]]:
    """``{"<a>-<b>": (a, b)}`` for every consecutive pair (in acquisition order,
    per :func:`discover_date_to_obs`'s dict-insertion-order guarantee) plus the
    full first->last span. With exactly the original 3 dates this reproduces
    the Phase 3.5/4 pair set byte-for-byte: ``{"2019-2021", "2021-2024",
    "2019-2024"}``. Phase 8's 5 dates add ``2024-2025``, ``2025-2026`` and
    ``2019-2026`` (the new full span) automatically; ``2024-2026`` (skipping
    2025) is added explicitly below so the newest interval can be isolated
    from the 2024-2025 / 2025-2026 consecutive pairs."""
    years = list(date_to_obs)
    pairs: dict[str, tuple[str, str]] = {}
    for a, b in zip(years, years[1:]):
        pairs[f"{a}-{b}"] = (a, b)
    if len(years) >= 2:
        pairs[f"{years[0]}-{years[-1]}"] = (years[0], years[-1])
    return pairs


PAIR_KEYS = build_pair_keys(DATE_TO_OBS)
if "2024" in DATE_TO_OBS and "2026" in DATE_TO_OBS:
    # Phase 8 Step A: isolate changes unique to the newest interval from those
    # already visible in the 2024-2025 / 2025-2026 consecutive steps.
    PAIR_KEYS.setdefault("2024-2026", ("2024", "2026"))


def run(pair_names: list[str], *, top: int = 10, n_panels: int = 5, refresh: bool = False,
        make_panels: bool = True, query: str | None = None) -> dict:
    from geoseek.change.models import FCSiamDiffChangeModel

    settings = get_settings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("geoseek Phase 4 - analyst-grade change pipeline over Ayodhya")
    print("=" * 78)

    repo = _build_repo()
    matcher = TemporalObservationMatcher(repo)
    seq = matcher.match(location=AOI_POINT, all_pairs=True, collection=S2_COLLECTION)
    obs_by_id = {o.observation_id: o for o in seq.observations}
    print(f"  observations: {[o.acquired_at for o in seq.observations]}")

    # normalized indices for every observation we will touch
    idx_paths = {oid: {k: p for k, p in ensure_indices(oid).items()} for oid in obs_by_id}

    model = FCSiamDiffChangeModel(device=settings.device)
    model.load()
    print(f"  model: FC-Siam-diff {model._loaded.model.num_parameters():,} params, "
          f"threshold {model.threshold:.2f} (frozen precision-favouring operating point)")

    # resolve requested pairs to matcher ObservationPairs
    pairs = {}
    for name in pair_names:
        de, dl = PAIR_KEYS[name]
        oe, ol = DATE_TO_OBS[de], DATE_TO_OBS[dl]
        p = next((pp for pp in seq.pairs
                  if pp.earlier.observation_id == oe and pp.later.observation_id == ol), None)
        if p is None:
            raise RuntimeError(f"matcher has no pair {oe} -> {ol}")
        pairs[name] = p

    import gc

    prob_paths: dict[tuple[str, str], str] = {}
    per_pair: dict[str, dict] = {}
    all_candidates: dict[str, list[Candidate]] = {}
    raster_crs = None

    for name, pair in pairs.items():
        print(f"\n--- pair {name}  ({pair.earlier.acquired_at} -> {pair.later.acquired_at})  "
              f"comparable={pair.comparable} ---")
        pr = probability_raster(model, pair, refresh=refresh)
        key = (pair.earlier.observation_id, pair.later.observation_id)
        prob_paths[key] = pr.path
        raster_crs = pr.crs

        binary, labels, ncomp = label_change(pr)
        cands, n_dropped_tiny = extract_candidates(pr, pair, labels, ncomp)
        raw_area = sum(c.area_px for c in cands)
        valid_px = int(pr.valid.sum())
        print(f"  raw candidates: {len(cands) + n_dropped_tiny} components  changed px {raw_area:,} = "
              f"{100*raw_area/max(valid_px,1):.2f}% of valid AOI"
              + (f"  ({n_dropped_tiny} sub-min-area folded into morphology)" if n_dropped_tiny else ""))
        del pr, binary
        gc.collect()
        attach_features(cands, pair, idx_paths, labels)
        del labels
        gc.collect()

        ctx = pair_context(pair, seq, idx_paths[pair.earlier.observation_id],
                           idx_paths[pair.later.observation_id])
        print(f"  scene seasonal delta: NDVI {ctx.scene_d_ndvi:+.3f}  NDBI {ctx.scene_d_ndbi:+.3f}  "
              f"NDWI {ctx.scene_d_ndwi:+.3f}  | co-reg residual {ctx.coreg_residual_px:.2f}px  "
              f"| index-band min corr {ctx.radiometric_index_band_min_corr:.2f} "
              f"(reliability {radiometric_reliability(ctx.radiometric_index_band_min_corr):.2f})")

        cfg = SuppressionConfig()
        traces = []
        for c in cands:
            f = CandidateFeatures(
                candidate_id=c.candidate_id, area_px=c.area_px,
                bad_scl_fraction_earlier=c.bad_scl_earlier, bad_scl_fraction_later=c.bad_scl_later,
                valid_fraction=c.valid_fraction, d_ndvi=c.d_ndvi, d_ndbi=c.d_ndbi, d_ndwi=c.d_ndwi)
            tr = suppress_candidate(f, ctx, cfg)
            c.suppression = tr.as_dict()
            traces.append(tr)
        supp = summarize_suppression(traces)
        if n_dropped_tiny:
            supp["raw_candidates"] += n_dropped_tiny
            supp["suppressed"] += n_dropped_tiny
            supp["suppressed_by_rule"]["morphology"] += n_dropped_tiny
        print(f"  suppression: {supp['raw_candidates']} -> {supp['survived']} survived  "
              f"({supp['suppressed']} suppressed)")
        for rule, n in supp["suppressed_by_rule"].items():
            if n:
                print(f"      - {rule:12s} {n}")

        survivors = [c for c, tr in zip(cands, traces) if not tr.suppressed]
        classifications = []
        for c in survivors:
            cl = classify_candidate(_spectra_for(c, ctx))
            c.classification = cl.as_dict()
            classifications.append(cl)
        dist = class_distribution(classifications)
        print(f"  class distribution (survivors): {dist}")

        per_pair[name] = {"comparable": pair.comparable, "context": {
            "scene_d_ndvi": round(ctx.scene_d_ndvi, 4), "scene_d_ndbi": round(ctx.scene_d_ndbi, 4),
            "scene_d_ndwi": round(ctx.scene_d_ndwi, 4),
            "coreg_residual_px": round(ctx.coreg_residual_px, 4),
            "radiometric_index_band_min_corr": round(ctx.radiometric_index_band_min_corr, 3),
            "radiometric_reliability": round(radiometric_reliability(
                ctx.radiometric_index_band_min_corr), 3)},
            "suppression": supp, "class_distribution": dist,
            "survivors": len(survivors)}
        all_candidates[name] = cands

    # -------- Step C: trajectories + Step D: confidence, on the span pair --------
    _years = list(DATE_TO_OBS)
    full_span_name = f"{_years[0]}-{_years[-1]}"
    span_name = full_span_name if full_span_name in pairs else pair_names[-1]
    span_pair = pairs[span_name]
    span_survivors = [c for c in all_candidates[span_name]
                      if not c.suppression.get("suppressed", True)]

    analyzer = TemporalPersistenceAnalyzer(matcher, change_threshold=model.threshold,
                                           collection=S2_COLLECTION)
    lookup = ProbLookup(prob_paths, model.threshold, raster_crs)
    span_ctx = pair_context(span_pair, seq, idx_paths[span_pair.earlier.observation_id],
                            idx_paths[span_pair.later.observation_id])
    span_prob_path = prob_paths[(span_pair.earlier.observation_id, span_pair.later.observation_id)]

    # Step C (cont.): Sentinel-1 SAR corroboration (weight only; unavailable = neutral)
    from geoseek.sar.evidence import SarCorroborator, sar_factor
    sar_corr = SarCorroborator()
    sar_pair = (sar_corr.for_pair(span_pair.earlier.observation_id, span_pair.later.observation_id)
                if sar_corr.available else None)
    if sar_pair:
        print(f"  SAR: {sar_pair.notes[0]}  coverage {sar_pair.coverage_fraction*100:.0f}%  "
              f"{sar_pair.notes[1]}")
    else:
        print("  SAR: no Sentinel-1 staged for this pair - corroboration term neutral for all candidates")

    # Step B: terrain context (elevation/slope/aspect/distance-to-water/distance-
    # to-built-up), sampled at each candidate's centroid - additive evidence, no
    # effect on confidence/ranking. Unavailable = an empty terrain dict, not a
    # pipeline failure (mirrors the SAR "unstaged is fine" degradation above).
    from geoseek.terrain.dem import terrain_staged
    from geoseek.terrain.features import TerrainSampler

    terrain_sampler = TerrainSampler() if terrain_staged() else None
    if terrain_sampler is None:
        print("  terrain: DEM not staged - run `python -m geoseek.staging.download_dem` "
              "+ `python -m geoseek.terrain.build` - terrain fields left empty")

    print(f"\n--- Step C/D on {span_name}: {len(span_survivors)} survivors ---")
    for c in span_survivors:
        lon, lat = c.centroid_lonlat
        traj = analyzer.trajectory_for_location(lon, lat, lookup)
        c.trajectory = traj.as_dict()
        cl = c.classification
        ev = cl.get("evidence", {})
        nv_an, nb_an, nw_an = (ev.get("ndvi_anomaly", 0.0), ev.get("ndbi_anomaly", 0.0),
                               ev.get("ndwi_anomaly", 0.0))
        sa = spectral_agreement_for(cl.get("change_type", "other"), nv_an, nb_an, nw_an)

        sar_f, sar_d = 1.0, None
        if sar_pair is not None:
            cmask = _component_mask(span_prob_path, c.bbox, c.centroid_rc, span_pair)
            look = sar_pair.lookup(c.bbox, cmask)
            sar_f, sar_d = sar_factor(cl.get("change_type", "other"),
                                      look["vv_median_db"], look["vh_median_db"],
                                      scene_dvv_db=sar_pair.scene_dvv_db,
                                      scene_dvh_db=sar_pair.scene_dvh_db, available=look["available"])
            c.sar = {**look, "factor": round(sar_f, 3), "verdict": sar_d,
                     "scene_dvv_db": round(sar_pair.scene_dvv_db, 2),
                     "scene_dvh_db": round(sar_pair.scene_dvh_db, 2)}

        rep = compute_confidence(
            candidate_id=c.candidate_id, model_prob=c.mean_prob,
            bad_scl_fraction=max(c.bad_scl_earlier, c.bad_scl_later), valid_fraction=c.valid_fraction,
            coreg_residual_px=span_ctx.coreg_residual_px,
            radiometric_reliability=radiometric_reliability(span_ctx.radiometric_index_band_min_corr),
            persistence=traj.persistence, persistence_confidence=traj.persistence_confidence,
            spectral_agreement=sa, change_type=cl.get("change_type", "other"),
            suppression_downweight=c.suppression.get("combined_downweight", 1.0),
            sar_corroboration=sar_f, sar_detail=sar_d)
        c.confidence = rep.confidence
        c.confidence_breakdown = rep.breakdown
        c.significance = round(significance(c), 4)
        c.queue_score = round(rank_score(c), 4)
        if terrain_sampler is not None:
            c.terrain = terrain_sampler.sample_rc(*c.centroid_rc).as_dict()

    if terrain_sampler is not None:
        terrain_sampler.close()

    # full ranked queue (confidence AND significance), then a diversified headline top-N
    span_survivors.sort(key=lambda c: c.queue_score, reverse=True)
    diverse_top = diversify(span_survivors, n=top)

    sar_summary = _sar_corroboration_report(span_survivors, per_pair, sar_pair)

    # -------- Step D: optional fusion re-ranking with a text query --------
    fusion_block = None
    if query:
        fusion_block = _fuse_with_query(span_survivors, span_pair, query, top)

    # -------- report --------
    _print_report(pair_names, per_pair, span_name, span_survivors, diverse_top, top)
    example_traj = None
    if diverse_top:
        example_traj = analyzer.trajectory_for_location(*diverse_top[0].centroid_lonlat, lookup)
        print("\n=== EXAMPLE FULL TEMPORAL TRAJECTORY (top-ranked location) ===")
        print(example_traj.format_report())

    panels = []
    if make_panels and diverse_top:
        print(f"\n--- saving {min(n_panels, len(diverse_top))} panels (diversified top-N) ---")
        for c in diverse_top[:n_panels]:
            p = save_panel(c, span_prob_path, model.threshold,
                           OUT_DIR / f"ayodhya_change_{c.candidate_id}.png")
            panels.append(str(p))
            print(f"  {p.name}")

    lookup.close()

    report = _assemble_report(pair_names, per_pair, span_name, span_survivors, diverse_top, top,
                              panels, model)
    report["example_trajectory"] = example_traj.as_dict() if example_traj else None
    report["sar_corroboration"] = sar_summary
    if fusion_block is not None:
        report["fusion"] = fusion_block

    # full ranked queue -> CSV (JSON keeps only the head) + a full-detail sidecar
    # (every survivor's evidence/suppression/trajectory/sar) for the analyst UI:
    # the change report itself stays slim, the UI reads the sidecar for detail.
    detail_rows = [_candidate_row(c, i, full=True) for i, c in enumerate(span_survivors, 1)]
    detail_path = OUT_DIR / "ayodhya_change_ranked_detail.json"
    detail_path.write_text(json.dumps(detail_rows, indent=1), encoding="utf-8")
    report["full_ranked_detail_json"] = str(detail_path)
    record_analysis_section("ayodhya_change_pipeline", report)
    print(f"  full-detail sidecar -> {detail_path.name} ({len(span_survivors)} candidates)")

    # Step C: evaluate every standing watch area against the freshly-computed
    # candidate set (the earliest point at which "new candidates" from this run
    # exist at all) and record any newly-matching one as a notification.
    from geoseek.watch.evaluator import evaluate_and_notify

    fired = evaluate_and_notify(repo, detail_rows, observation_id=span_pair.later.observation_id)
    if fired:
        print(f"\n--- Step C: watch areas ---")
        for n in fired:
            print(f"  watch {n.watch_id}: {len(n.candidate_ids)} new matching candidate(s) "
                  f"-> notification {n.notification_id}")
    report["watch_notifications_fired"] = [n.as_dict() for n in fired]
    repo.close()

    import csv as _csv
    with open(OUT_DIR / "ayodhya_change_ranked.csv", "w", newline="", encoding="utf-8") as fh:
        wr = _csv.writer(fh)
        wr.writerow(["rank", "candidate_id", "lon", "lat", "change_type", "area_m2",
                     "confidence", "significance", "queue_score", "persistence", "earliest_supported"])
        for i, c in enumerate(span_survivors, 1):
            lon, lat = c.centroid_lonlat
            wr.writerow([i, c.candidate_id, round(lon, 6), round(lat, 6),
                         c.classification.get("change_type"), round(c.area_px * PIXEL_AREA_M2, 1),
                         round(c.confidence, 4), c.significance, c.queue_score,
                         c.trajectory.get("persistence"),
                         c.trajectory.get("earliest_supported_change", {}).get("window")])

    (OUT_DIR / "ayodhya_change_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    record_analysis_section("ayodhya_change_pipeline", report)
    print(f"\n  report -> {OUT_DIR/'ayodhya_change_report.json'}  +  manifest 'ayodhya_change_pipeline'")
    return report


def _print_report(pair_names, per_pair, span_name, span_survivors, diverse_top, top):
    print("\n" + "=" * 78)
    print("PHASE 4/5 REPORT - Ayodhya")
    print("=" * 78)
    print("\n[1] Candidates before/after suppression, per rule:")
    for name in pair_names:
        s = per_pair[name]["suppression"]
        print(f"  {name}: raw {s['raw_candidates']:>6}  ->  survived {s['survived']:>5}   "
              f"(suppressed {s['suppressed']}; " +
              ", ".join(f"{k} {v}" for k, v in s["suppressed_by_rule"].items() if v) + ")")
    print("\n[2] Change-type distribution (survivors):")
    for name in pair_names:
        print(f"  {name}: {per_pair[name]['class_distribution']}")
    print(f"\n[3] Diversified top {top} candidates (span pair {span_name}) - ranked by "
          f"queue_score = confidence^0.65 * significance^0.35;")
    print(f"    diversified: <=3 per type, same-type picks >=1.5 km apart. Full ranked list "
          f"({len(span_survivors)}) in the JSON.")
    hdr = (f"  {'#':>3} {'candidate':>18} {'lon':>9} {'lat':>8} {'type':>12} {'conf':>5} "
           f"{'sig':>5} {'queue':>6} {'area_m2':>9}  earliest")
    print(hdr)
    for i, c in enumerate(diverse_top[:top], 1):
        lon, lat = c.centroid_lonlat
        win = c.trajectory.get("earliest_supported_change", {}).get("window")
        print(f"  {i:>3} {c.candidate_id:>18} {lon:9.5f} {lat:8.5f} "
              f"{c.classification.get('change_type','?'):>12} {c.confidence:5.2f} "
              f"{c.significance:5.2f} {c.queue_score:6.3f} {c.area_px*PIXEL_AREA_M2:9.0f}  {win}")
    print("\n[3b] Evidence breakdown for the diversified top candidates:")
    for i, c in enumerate(diverse_top[:min(top, 10)], 1):
        print(f"  #{i} {c.candidate_id}  [{c.classification.get('change_type','?')}]  conf {c.confidence:.2f}")
        for line in c.confidence_breakdown:
            print(f"       {line}")
        cl_ev = c.classification.get("evidence", {})
        print(f"       index deltas: NDVI {cl_ev.get('d_ndvi'):+.3f} (anom {cl_ev.get('ndvi_anomaly'):+.3f})  "
              f"NDBI {cl_ev.get('d_ndbi'):+.3f} (anom {cl_ev.get('ndbi_anomaly'):+.3f})  "
              f"NDWI {cl_ev.get('d_ndwi'):+.3f}")
        es = c.trajectory.get("earliest_supported_change", {})
        print(f"       earliest supported: {es.get('window')}  via {es.get('supporting_observations')}")
        print(f"       caveat: {es.get('caveat')}")


def _candidate_row(c, i=None, full=False):
    row = {"candidate_id": c.candidate_id, "pair": c.pair_id,
           "centroid_lonlat": [round(x, 6) for x in c.centroid_lonlat],
           "area_px": c.area_px, "area_m2": round(c.area_px * PIXEL_AREA_M2, 1),
           "change_type": c.classification.get("change_type"),
           "confidence": round(c.confidence, 4), "significance": round(c.significance, 4),
           "queue_score": round(c.queue_score, 4),
           "persistence": c.trajectory.get("persistence"),
           "earliest_supported": c.trajectory.get("earliest_supported_change", {}).get("window")}
    if i is not None:
        row = {"rank": i, **row}
    if full:
        row.update({"bbox_rc": list(c.bbox), "label": c.label,
                    "centroid_rc": [round(x, 2) for x in c.centroid_rc],
                    "mean_model_prob": round(c.mean_prob, 4),
                    "classification": c.classification,
                    "confidence_breakdown": c.confidence_breakdown,
                    "suppression": c.suppression, "trajectory": c.trajectory,
                    "sar": c.sar, "terrain": c.terrain})
    return row


def _assemble_report(pair_names, per_pair, span_name, span_survivors, diverse_top, top, panels, model):
    return {
        "aoi": "Ayodhya, Uttar Pradesh (82km scaled AOI, MGRS 44RPQ)",
        "observations": list(DATE_TO_OBS.values()),
        "model": {"name": "FCSiamDiff", "checkpoint": str(model.checkpoint_path),
                  "threshold": float(model.threshold),
                  "weights_sha256": model._loaded.weights_sha256},
        "pairs": {name: per_pair[name] for name in pair_names},
        "span_pair": span_name,
        "ranking": {
            "queue_score": "confidence^0.65 * significance^0.35  (weighted geometric mean)",
            "significance": "area_term^0.6 * anomaly_term^0.4, each clipped to [0.10, 1.0]",
            "area_term": f"(log10(area_px) - log10({_A_MIN:.0f})) / (log10({_A_REF:.0f}) - log10({_A_MIN:.0f}))",
            "anomaly_term": f"max(|NDVI/NDBI/NDWI anomaly|) / {_ANOM_REF}",
            "diversity": "headline top-N: <=3 candidates per change_type; same-type picks >= 1.5 km apart",
        },
        "diverse_top_candidates": [_candidate_row(c, i, full=True)
                                   for i, c in enumerate(diverse_top[:top], 1)],
        "top_candidates": [_candidate_row(c, i, full=True)      # alias: the reported headline list
                           for i, c in enumerate(diverse_top[:top], 1)],
        "full_ranked_head": [_candidate_row(c, i) for i, c in enumerate(span_survivors[:100], 1)],
        "full_ranked_total": len(span_survivors),
        "full_ranked_csv": str(OUT_DIR / "ayodhya_change_ranked.csv"),
        "panels": panels,
        "domain_gap_statement": (
            "The FC-Siam-diff weights were trained on OSCD, which is Sentinel-2 L1C "
            "top-of-atmosphere imagery in same-season pairs; Ayodhya here is L2A "
            "bottom-of-atmosphere across a drought March (2019) and a green March (2024). "
            "This is an out-of-distribution transfer: expect the effective precision/recall "
            "to be materially worse than the 56% F1 measured on the OSCD held-out split. "
            "The suppression stage (esp. the phenology gate) removes the bulk of the "
            "seasonal false positives that the domain gap produces; the numbers above are "
            "reported as observed, not tuned to flatter the result."),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-change-analyze", description=__doc__)
    p.add_argument("--pairs", default=",".join(PAIR_KEYS),
                   help=f"comma list from {list(PAIR_KEYS)} (default: every consecutive pair "
                        "across every ingested date + the full span)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--panels", type=int, default=5)
    p.add_argument("--refresh", action="store_true", help="recompute probability rasters")
    p.add_argument("--no-panels", action="store_true")
    p.add_argument("--query", default=None,
                   help="Step D: re-rank candidates by a fused change + RemoteCLIP semantic score, "
                        "e.g. --query 'new construction near a river'")
    args = p.parse_args(argv)
    names = [x.strip() for x in args.pairs.split(",") if x.strip()]
    bad = [n for n in names if n not in PAIR_KEYS]
    if bad:
        p.error(f"unknown pair(s) {bad}; choose from {list(PAIR_KEYS)}")
    run(names, top=args.top, n_panels=args.panels, refresh=args.refresh,
        make_panels=not args.no_panels, query=args.query)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[change.analyze] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

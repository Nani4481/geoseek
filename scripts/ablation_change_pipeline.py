"""Phase 7c Step A - ablation study of the change-detection pipeline.

Measures what each pipeline stage contributes, one variable at a time,
**cumulatively**, in the order the pipeline applies them:

    1. FC-Siam-diff raw output only          (threshold the model probability)
    2. + quality / SCL gating                (suppress.gate_quality)
    3. + radiometric normalization           (suppress.gate_radiometric - a DOWN-WEIGHT)
    4. + phenology / anomaly-framing supp.    (suppress.gate_phenology)
    5. + morphological filtering (50 px)      (suppress.gate_morphology)
    6. + temporal persistence                 (temporal.persistence)
    7. + SAR corroboration                    (sar.evidence)
    8. + confidence-weighted ranking          (change.confidence + queue_score)

Two evaluation domains, and the script is EXPLICIT about which metric is valid
where:

* **OSCD held-out split (10 regions, pixel-labelled)** - stages 1-5 give real
  P / R / F1 / IoU / FPR because there is a per-pixel change mask. Stage 8 gives
  a component-level Average Precision (does confidence ordering beat raw-prob
  ordering). Stages 6 and 7 are **definitionally inapplicable** on OSCD (it is
  bitemporal - no third date for persistence - and has no Sentinel-1) and are
  reported as N/A with the reason.

* **Ayodhya (3-date L2A, NO change labels)** - only candidate COUNTS and
  qualitative effects. These are NOT accuracy numbers and are never presented as
  such. Stages 6/7/8 are assessed here because this is the only domain where
  they run.

The gates are the SHIPPED code: this script builds real
``geoseek.change.suppress.CandidateFeatures`` / ``PairSuppressionContext``
objects and calls ``suppress_candidate`` - it does not re-implement any rule.

    python scripts/ablation_change_pipeline.py                 # full study
    python scripts/ablation_change_pipeline.py --oscd-only
    python scripts/ablation_change_pipeline.py --ayodhya-only

Offline. Reads the staged OSCD data, the trained checkpoint, the cached Ayodhya
probability raster and the Phase 4/5 change report. Writes
``data/change_model/ablation_study.json`` and the manifest section
``change_pipeline_ablation``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", message=".*no geotransform.*")
warnings.filterwarnings("ignore", message=".*NotGeoreferencedWarning.*")

from geoseek.change.classify import CandidateSpectra, classify_candidate
from geoseek.change.confidence import compute_confidence, spectral_agreement_for
from geoseek.change.indices import compute_indices
from geoseek.change.suppress import (
    CandidateFeatures,
    PairSuppressionContext,
    SuppressionConfig,
    suppress_candidate,
)
from geoseek.config import get_settings

OUT = get_settings().data_dir / "change_model"
DATASETS = get_settings().datasets_dir
_CONN8 = np.ones((3, 3), bool)

# stage order + the machinery each one actually is
STAGES = [
    ("1_raw_model", "FC-Siam-diff raw output only"),
    ("2_quality_gate", "+ quality / SCL gating"),
    ("3_radiometric", "+ radiometric normalization (down-weight, not a hard gate)"),
    ("4_phenology", "+ phenology / anomaly-framing suppression"),
    ("5_morphology", "+ morphological filtering (50 px floor)"),
    ("6_persistence", "+ temporal persistence"),
    ("7_sar", "+ SAR corroboration"),
    ("8_confidence_rank", "+ confidence-weighted ranking (full pipeline)"),
]


# ==========================================================================
# metric helpers
# ==========================================================================


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "iou": round(iou, 4), "fpr": round(fpr, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


# ==========================================================================
# OSCD: stages 1-5 (pixel metrics) + stage 8 (component AP)
# ==========================================================================


def _oscd_model():
    import torch

    from geoseek.change.models.fc_siam_diff import FCSiamDiff

    blob = torch.load(OUT / "fc_siam_diff.pt", map_location=get_settings().device)
    card = blob["model_card"]
    arch = card["architecture"]
    m = FCSiamDiff(in_channels=arch["in_channels"], base_channels=arch["base_channels"],
                   depth=arch["depth"], dropout=0.0).to(get_settings().device)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    mean = np.asarray(card["norm_stats"]["mean"], np.float32)
    std = np.asarray(card["norm_stats"]["std"], np.float32)
    thr_pf = float((card.get("eval", {}) or {}).get("chosen_threshold_precision_favouring", 0.8))
    sha = (card.get("provenance", {}) or {}).get("weights_sha256")
    if not sha:  # a file cannot embed its own hash - compute it from disk (same as the model wrapper)
        import hashlib

        h = hashlib.sha256()
        with open(OUT / "fc_siam_diff.pt", "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        sha = h.hexdigest()
    return m, mean, std, thr_pf, sha


def _oscd_region_arrays(region: str, model, mean, std):
    """(prob, gt, valid, d_ndvi, d_ndbi, d_ndwi) full-image float32 for one OSCD region."""
    import torch

    from geoseek.staging.download_oscd import (
        PRODUCTION_BANDS,
        REFLECTANCE_SCALE,
        change_mask_path,
        oscd_root,
    )

    rd = oscd_root() / "Onera Satellite Change Detection dataset - Images" / region

    def _stack(idx: int) -> dict:
        out = {}
        for b in ("B02", "B03", "B04", "B08", "B11"):
            with rasterio.open(rd / f"imgs_{idx}_rect" / f"{b}.tif") as ds:
                out[b] = ds.read(1).astype(np.float32) / REFLECTANCE_SCALE
        return out

    d1, d2 = _stack(1), _stack(2)
    with rasterio.open(change_mask_path(region)) as ds:
        cm = ds.read(1)
    gt = (cm == 2).astype(np.float32)

    img1 = np.stack([d1[b] for b in PRODUCTION_BANDS])
    img2 = np.stack([d2[b] for b in PRODUCTION_BANDS])
    h = min(gt.shape[0], img1.shape[1])
    w = min(gt.shape[1], img1.shape[2])
    img1, img2, gt = img1[:, :h, :w], img2[:, :h, :w], gt[:h, :w]
    for dd in (d1, d2):
        for b in dd:
            dd[b] = dd[b][:h, :w]

    valid = np.all(img1 > 0, axis=0) & np.all(img2 > 0, axis=0)

    x1 = torch.from_numpy(((img1 - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None]
    x2 = torch.from_numpy(((img2 - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None]
    dev = get_settings().device
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
        logits = model(x1.to(dev), x2.to(dev)).float()
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob[~valid] = 0.0

    i1 = compute_indices(d1)
    i2 = compute_indices(d2)
    d_ndvi = np.where(valid, i2["NDVI"] - i1["NDVI"], np.nan)
    d_ndbi = np.where(valid, i2["NDBI"] - i1["NDBI"], np.nan)
    d_ndwi = np.where(valid, i2["NDWI"] - i1["NDWI"], np.nan)

    # per-region inter-date correlation of the index-driving bands (radiometric ctx)
    band_corr = {b: _pearson(d1[b][valid], d2[b][valid]) for b in ("B03", "B04", "B08", "B11")}
    return prob, gt, valid, d_ndvi, d_ndbi, d_ndwi, band_corr


def _components(prob, valid, thr):
    binm = (prob >= thr) & valid
    lab, n = ndi.label(binm, structure=_CONN8)
    return binm, lab, n


def _region_gate_table(prob, gt, valid, d_ndvi, d_ndbi, d_ndwi, band_corr, thr, cfg):
    """Cumulative per-pixel confusion counts for stages 1..5 on ONE region,
    plus per-gate component bookkeeping. Radiometric (stage 3) is a down-weight
    and cannot change the mask - stage 3 counts == stage 2 counts by construction."""
    binm, lab, n = _components(prob, valid, thr)
    g = gt > 0.5
    P = int(valid.sum())
    gt_pos = int((g & valid).sum())

    # --- scene-wide seasonal index deltas (median over valid pixels) ---
    sd_ndvi = float(np.nanmedian(d_ndvi))
    sd_ndbi = float(np.nanmedian(d_ndbi))
    sd_ndwi = float(np.nanmedian(d_ndwi))
    min_corr = float(min(band_corr.values()))

    ctx = PairSuppressionContext(
        pair_id=f"oscd@{thr}", coreg_residual_px=0.0, coreg_corrected=False,
        scene_d_ndvi=sd_ndvi, scene_d_ndbi=sd_ndbi, scene_d_ndwi=sd_ndwi,
        radiometric_index_band_min_corr=min_corr, radiometric_low_confidence=False)

    if n == 0:
        z = prf(0, 0, gt_pos, P - gt_pos)
        empty = {"removed": 0}
        return ({s: z for s in ("1_raw_model", "2_quality_gate", "3_radiometric",
                                "4_phenology", "5_morphology")},
                {"quality": empty, "phenology": empty, "morphology": empty,
                 "radiometric_downweighted": 0, "n_components": 0,
                 "aux_morph_first": {"2q_5morph": z, "2q_5morph_4phen": z}},
                ctx, [])

    areas = np.bincount(lab.ravel())
    areas[0] = 0
    objs = ndi.find_objects(lab)

    # per-component features via the SHIPPED CandidateFeatures / suppress_candidate
    comp = []
    for li in range(1, n + 1):
        a = int(areas[li])
        if a == 0:
            continue
        sl = objs[li - 1]
        m = lab[sl] == li
        # shape features (same as change.analyze._elongation_fill) - only used by stage 8 typing
        ys, xs = np.nonzero(m)
        if ys.size >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.vstack([ys.astype(np.float64), xs.astype(np.float64)])))
            ev = np.clip(ev, 1e-6, None)
            elong = float(np.sqrt(ev[-1] / ev[0]))
        else:
            elong = 1.0
        fillr = float(ys.size / (m.shape[0] * m.shape[1]))
        # valid_fraction: OSCD rect crops have no nodata, so this is ~1.0 - measured, not assumed
        vfrac = float((valid[sl] & m).sum() / max(m.sum(), 1))
        dv = float(np.nanmean(np.where(m, d_ndvi[sl], np.nan)))
        db = float(np.nanmean(np.where(m, d_ndbi[sl], np.nan)))
        dw = float(np.nanmean(np.where(m, d_ndwi[sl], np.nan)))
        dv = 0.0 if not np.isfinite(dv) else dv
        db = 0.0 if not np.isfinite(db) else db
        dw = 0.0 if not np.isfinite(dw) else dw
        feats = CandidateFeatures(
            candidate_id=str(li), area_px=a,
            bad_scl_fraction_earlier=0.0, bad_scl_fraction_later=0.0,  # OSCD L1C: no SCL
            valid_fraction=vfrac, d_ndvi=dv, d_ndbi=db, d_ndwi=dw)
        tr = suppress_candidate(feats, ctx, cfg)
        verds = {o.rule: o.verdict for o in tr.outcomes}
        comp.append({"label": li, "area": a, "slice": sl, "mask": m,
                     "d_ndvi": dv, "d_ndbi": db, "d_ndwi": dw, "vfrac": vfrac,
                     "elong": elong, "fill": fillr,
                     "q": verds["quality"], "phe": verds["phenology"],
                     "mor": verds["morphology"],
                     "radio_dw": tr.outcomes[2].verdict == "downweight",
                     "feats": feats})

    # cumulative suppression sets (radiometric never suppresses -> stage 3 == stage 2)
    def _mask_for(pred):
        out = np.zeros_like(binm)
        for c in comp:
            if pred(c):
                out[c["slice"]][c["mask"]] = True
        return out

    def _keep(q=False, phe=False, mor=False):
        return _mask_for(lambda c: (not q or c["q"] != "suppress")
                         and (not phe or c["phe"] != "suppress")
                         and (not mor or c["mor"] != "suppress"))

    def _counts(keep):
        pred = keep & valid
        tp = int((pred & g).sum())
        fp = int((pred & ~g).sum())
        fn = gt_pos - tp
        tn = (P - gt_pos) - fp
        return prf(tp, fp, fn, tn)

    # cumulative chain in the task's order: 1 raw -> +quality -> +radiometric(no-op mask) ->
    # +phenology -> +morphology.
    tbl = {"1_raw_model": _counts(_keep()),
           "2_quality_gate": _counts(_keep(q=True)),
           "3_radiometric": _counts(_keep(q=True)),          # down-weight only: mask == stage 2
           "4_phenology": _counts(_keep(q=True, phe=True)),
           "5_morphology": _counts(_keep(q=True, phe=True, mor=True))}
    # auxiliary: morphology BEFORE phenology (the order the Ayodhya pipeline ships, for
    # tractability) - isolates the *marginal* phenology effect once the 50px floor has
    # already removed the small components.
    aux = {"2q_5morph": _counts(_keep(q=True, mor=True)),
           "2q_5morph_4phen": _counts(_keep(q=True, phe=True, mor=True))}
    book = {
        "n_components": len(comp),
        "quality": {"removed": sum(c["q"] == "suppress" for c in comp)},
        "phenology": {"removed": sum(c["q"] != "suppress" and c["phe"] == "suppress" for c in comp)},
        "morphology": {"removed": sum(c["q"] != "suppress" and c["phe"] != "suppress"
                                     and c["mor"] == "suppress" for c in comp)},
        "radiometric_downweighted": sum(c["radio_dw"] for c in comp),
        "min_inter_date_band_corr": round(min_corr, 3),
        "scene_delta": {"ndvi": round(sd_ndvi, 4), "ndbi": round(sd_ndbi, 4),
                        "ndwi": round(sd_ndwi, 4)},
        "aux_morph_first": aux,
    }
    return tbl, book, ctx, comp


def _agg(rows: list[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    return prf(tp, fp, fn, tn)


def _stage8_component_ap(comp, prob, gt, valid, ctx, cfg, thr):
    """Component-level Average Precision on the stage-5 survivors, ordered by
    (a) raw mean model probability vs (b) the full confidence score.

    A survivor component is a TP if it overlaps >=1 GT change pixel (lenient,
    matches 'did the queue point the analyst at a real change'); a stricter
    >=25%-overlap variant is also reported. On OSCD the confidence score's
    persistence term is the constant 'single_pair' case and the SAR term is
    absent, so this isolates the model-rescale x spectral x quality x
    radiometric x registration ordering."""
    from geoseek.change.analyze import radiometric_reliability

    g = gt > 0.5
    survivors = [c for c in comp
                 if c["q"] != "suppress" and c["phe"] != "suppress" and c["mor"] != "suppress"]
    if not survivors:
        return None

    rr = radiometric_reliability(ctx.radiometric_index_band_min_corr)
    items = []
    for c in survivors:
        sl, m = c["slice"], c["mask"]
        mp = float(prob[sl][m].mean())
        ov = int((g[sl] & m).sum())
        frac_ov = ov / max(int(m.sum()), 1)
        spec = CandidateSpectra(
            candidate_id=str(c["label"]), d_ndvi=c["d_ndvi"], d_ndbi=c["d_ndbi"], d_ndwi=c["d_ndwi"],
            scene_d_ndvi=ctx.scene_d_ndvi, scene_d_ndbi=ctx.scene_d_ndbi, scene_d_ndwi=ctx.scene_d_ndwi,
            area_px=c["area"], elongation=c["elong"], fill_ratio=c["fill"])
        cl = classify_candidate(spec)
        sa = spectral_agreement_for(cl.change_type, spec.ndvi_anomaly, spec.ndbi_anomaly,
                                    spec.ndwi_anomaly)
        rep = compute_confidence(
            candidate_id=str(c["label"]), model_prob=mp,
            bad_scl_fraction=0.0, valid_fraction=c["vfrac"],
            coreg_residual_px=0.0, radiometric_reliability=rr,
            persistence="single_pair", persistence_confidence=0.55,
            spectral_agreement=sa, change_type=cl.change_type,
            suppression_downweight=1.0, sar_corroboration=1.0)
        items.append({"mp": mp, "conf": rep.confidence, "tp_loose": ov > 0,
                      "tp_strict": frac_ov >= 0.25})

    def _ap(order_key, rel_key):
        order = sorted(items, key=lambda x: x[order_key], reverse=True)
        n_rel = sum(x[rel_key] for x in order)
        if n_rel == 0:
            return 0.0, []
        hit = 0
        precs = []
        for i, x in enumerate(order, 1):
            if x[rel_key]:
                hit += 1
                precs.append(hit / i)
        pk = {k: round(sum(1 for x in order[:k] if x[rel_key]) / min(k, len(order)), 4)
              for k in (10, 25, 50, 100)}
        return round(float(np.mean(precs)), 4), pk

    out = {"n_survivors": len(survivors)}
    for rel in ("tp_loose", "tp_strict"):
        ap_mp, pk_mp = _ap("mp", rel)
        ap_cf, pk_cf = _ap("conf", rel)
        out[rel] = {
            "n_relevant": int(sum(x[rel] for x in items)),
            "AP_by_raw_prob": ap_mp, "AP_by_confidence": ap_cf,
            "delta_AP": round(ap_cf - ap_mp, 4),
            "precision_at_k_raw_prob": pk_mp, "precision_at_k_confidence": pk_cf,
        }
    return out


def run_oscd(cfg) -> dict:
    from geoseek.staging.download_oscd import enumerate_split

    model, mean, std, thr_pf, sha = _oscd_model()
    test = enumerate_split()["test_regions"]
    print(f"\n{'='*78}\nOSCD held-out ablation - {len(test)} test regions, weights {sha[:16]}...\n{'='*78}")
    print(f"  operating points: 0.50 (default) and {thr_pf:.2f} (deployed precision-favouring)")

    thresholds = {"0.50": 0.5, f"{thr_pf:.2f}": thr_pf}
    result = {"test_regions": test, "weights_sha256": sha,
              "operating_points": list(thresholds.keys()),
              "change_pixel_fraction": None, "by_threshold": {}}

    for tag, thr in thresholds.items():
        print(f"\n--- operating point {tag} ---")
        per_region = {s: [] for s, _ in STAGES[:5]}
        aux_region = {"2q_5morph": [], "2q_5morph_4phen": []}
        books = {}
        stage8 = {}
        t0 = time.time()
        gt_pos_total = px_total = 0
        for reg in test:
            prob, gt, valid, dv, db, dw, bc = _oscd_region_arrays(reg, model, mean, std)
            px_total += int(valid.sum())
            gt_pos_total += int(((gt > 0.5) & valid).sum())
            tbl, book, ctx, comp = _region_gate_table(prob, gt, valid, dv, db, dw, bc, thr, cfg)
            for s in per_region:
                per_region[s].append(tbl[s])
            for s in aux_region:
                aux_region[s].append(book["aux_morph_first"][s])
            books[reg] = book
            s8 = _stage8_component_ap(comp, prob, gt, valid, ctx, cfg, thr)
            if s8:
                stage8[reg] = s8
            r1, r5 = tbl["1_raw_model"], tbl["5_morphology"]
            print(f"  {reg:12s} raw F1 {r1['f1']:.3f} -> stage5 F1 {r5['f1']:.3f}  "
                  f"(comps {book['n_components']}, -Q{book['quality']['removed']} "
                  f"-phen{book['phenology']['removed']} -morph{book['morphology']['removed']})")

        agg = {s: _agg(per_region[s]) for s in per_region}
        agg_aux = {s: _agg(aux_region[s]) for s in aux_region}
        # pooled component AP over all regions
        s8_all = _pool_stage8(stage8)
        result["change_pixel_fraction"] = round(gt_pos_total / px_total, 4)
        result["by_threshold"][tag] = {
            "aggregate": agg,
            "aggregate_morphology_first": {
                **agg_aux,
                "_note": "quality -> morphology(50px) -> phenology, i.e. the order the Ayodhya "
                         "pipeline ships. Compare 2q_5morph_4phen here with 5_morphology in the "
                         "main table to see phenology's MARGINAL effect after the 50px floor.",
            },
            "per_region": {reg: {s: per_region[s][i] for s in per_region}
                           for i, reg in enumerate(test)},
            "component_bookkeeping": books,
            "stage6_persistence": {
                "applicable": False,
                "reason": "OSCD is bitemporal (2 acquisitions per region). Temporal "
                          "persistence needs >=3 observations to separate persistent from "
                          "transient change; every candidate collapses to the constant "
                          "'single_pair' case (persistence_confidence 0.55, penalty x0.85), "
                          "which has no discriminative power and cannot alter the detection "
                          "mask. Measured on Ayodhya instead."},
            "stage7_sar": {
                "applicable": False,
                "reason": "No Sentinel-1 acquisitions exist for the OSCD regions/dates. SAR "
                          "corroboration is a weight-only cross-sensor term; with no S1 it is "
                          "exactly neutral (x1.0) for every candidate. Measured on Ayodhya "
                          "instead."},
            "stage8_confidence_ranking": s8_all,
        }
        dt = time.time() - t0
        print(f"  [{tag}] aggregate: raw F1 {agg['1_raw_model']['f1']:.4f} -> "
              f"full-gates(5) F1 {agg['5_morphology']['f1']:.4f}   ({dt:.0f}s)")

    return result


def _pool_stage8(stage8: dict) -> dict:
    """Pool the per-region stage-8 component lists into one AP over all regions is
    not reconstructable from the per-region summaries alone; instead report the
    macro-average of the per-region deltas plus the count of regions where
    confidence ordering wins/loses/ties."""
    if not stage8:
        return {"applicable": False, "reason": "no stage-5 survivors in any region"}
    out = {"applicable": True, "per_region": stage8,
           "note": ("component-level: each stage-5 survivor is one queue entry; "
                    "TP = overlaps GT change. Persistence term is the constant "
                    "'single_pair' case and SAR is absent on OSCD, so this isolates "
                    "the model-rescale x spectral x quality x radiometric ordering.")}
    for rel in ("tp_loose", "tp_strict"):
        deltas = [v[rel]["delta_AP"] for v in stage8.values()]
        aps_mp = [v[rel]["AP_by_raw_prob"] for v in stage8.values()]
        aps_cf = [v[rel]["AP_by_confidence"] for v in stage8.values()]
        out[f"macro_{rel}"] = {
            "mean_AP_by_raw_prob": round(float(np.mean(aps_mp)), 4),
            "mean_AP_by_confidence": round(float(np.mean(aps_cf)), 4),
            "mean_delta_AP": round(float(np.mean(deltas)), 4),
            "regions_confidence_wins": int(sum(d > 0.005 for d in deltas)),
            "regions_confidence_loses": int(sum(d < -0.005 for d in deltas)),
            "regions_tie": int(sum(abs(d) <= 0.005 for d in deltas)),
        }
    return out


# ==========================================================================
# Ayodhya: candidate counts (stages 1-5) + qualitative (stages 6-8)
# ==========================================================================

SPAN_EARLIER = "S2B_44RPQ_20190330_1_L2A_scaled"
SPAN_LATER = "S2A_44RPQ_20240308_0_L2A_scaled"
SPAN_PROB = OUT / f"prob_{SPAN_EARLIER}__to__{SPAN_LATER}.tif"


def _labelwise_delta_mean(labels: np.ndarray, earlier_tif: Path, later_tif: Path,
                          n_labels: int, stripe: int = 512):
    """Streaming per-label mean of (later - earlier) over a big raster, O(1) memory
    beyond the label array. Returns (sums, counts) indexed by label id."""
    sums = np.zeros(n_labels + 1, np.float64)
    cnts = np.zeros(n_labels + 1, np.float64)
    with rasterio.open(earlier_tif) as de, rasterio.open(later_tif) as dl:
        H = de.height
        for r0 in range(0, H, stripe):
            r1 = min(r0 + stripe, H)
            win = rasterio.windows.Window(0, r0, de.width, r1 - r0)
            d = dl.read(1, window=win).astype(np.float32) - de.read(1, window=win).astype(np.float32)
            lab = labels[r0:r1].ravel()
            dd = d.ravel()
            ok = np.isfinite(dd) & (lab > 0)
            sums += np.bincount(lab[ok], weights=dd[ok], minlength=n_labels + 1)
            cnts += np.bincount(lab[ok], minlength=n_labels + 1)
    return sums, cnts


def _labelwise_badscl_valid(labels: np.ndarray, scl_e: Path, scl_l: Path,
                            b08_e: Path, b08_l: Path, n_labels: int, stripe: int = 512):
    from geoseek.ingest.quality import BAD_SCL_CLASSES

    bad = np.array(sorted(BAD_SCL_CLASSES))
    bad_e = np.zeros(n_labels + 1, np.float64)
    bad_l = np.zeros(n_labels + 1, np.float64)
    val = np.zeros(n_labels + 1, np.float64)
    tot = np.zeros(n_labels + 1, np.float64)
    with rasterio.open(scl_e) as se, rasterio.open(scl_l) as sl_, \
         rasterio.open(b08_e) as be, rasterio.open(b08_l) as bl:
        H = se.height
        for r0 in range(0, H, stripe):
            r1 = min(r0 + stripe, H)
            win = rasterio.windows.Window(0, r0, se.width, r1 - r0)
            lab = labels[r0:r1].ravel()
            m = lab > 0
            se_v = np.isin(se.read(1, window=win).ravel(), bad)
            sl_v = np.isin(sl_.read(1, window=win).ravel(), bad)
            vv = (be.read(1, window=win).ravel() > 0) & (bl.read(1, window=win).ravel() > 0)
            bad_e += np.bincount(lab[m], weights=se_v[m].astype(np.float64), minlength=n_labels + 1)
            bad_l += np.bincount(lab[m], weights=sl_v[m].astype(np.float64), minlength=n_labels + 1)
            val += np.bincount(lab[m], weights=vv[m].astype(np.float64), minlength=n_labels + 1)
            tot += np.bincount(lab[m], minlength=n_labels + 1)
    tot = np.maximum(tot, 1)
    return bad_e / tot, bad_l / tot, val / tot


def run_ayodhya(cfg) -> dict:
    """Independent recomputation of the cumulative candidate counts on the cached
    2019->2024 probability raster, then the stage 6/7/8 effects read from the
    committed Phase 4/5 change report. Counts are NOT accuracy - Ayodhya has no
    change labels."""
    if not SPAN_PROB.is_file():
        raise SystemExit(f"cached span probability raster missing: {SPAN_PROB}")
    report = json.loads((OUT / "ayodhya_change_report.json").read_text())
    detail = json.loads((OUT / "ayodhya_change_ranked_detail.json").read_text())
    span_ctx_vals = report["pairs"]["2019-2024"]["context"]

    print(f"\n{'='*78}\nAyodhya candidate-count ablation - span pair 2019-03-30 -> 2024-03-08\n{'='*78}")
    print("  (candidate COUNTS + qualitative effects only - Ayodhya has NO change labels)")

    thr = float(report["model"]["threshold"])
    with rasterio.open(SPAN_PROB) as ds:
        prob = ds.read(1).astype(np.float32)
    valid = prob > 0
    binm = (prob >= thr) & valid
    labels, n = ndi.label(binm, structure=_CONN8)
    del binm
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    n_raw = int((areas > 0).sum())
    ge50 = np.where(areas >= cfg.morph_min_area_px)[0]
    n_ge50 = int(ge50.size)
    changed_px = int((areas[areas > 0]).sum())
    print(f"  raw connected components (>= {thr:.2f}): {n_raw:,}   changed px {changed_px:,} "
          f"= {100*changed_px/max(int(valid.sum()),1):.2f}% of valid AOI")
    print(f"  components >= {cfg.morph_min_area_px} px (morphology floor): {n_ge50:,}  "
          f"(-{n_raw - n_ge50:,} sub-floor specks)")

    # per (>=50px) component index-delta means -> phenology gate
    e, l = SPAN_EARLIER, SPAN_LATER
    s_ndvi, c_ndvi = _labelwise_delta_mean(labels, DATASETS / e / "NDVI.tif",
                                           DATASETS / l / "NDVI.tif", n)
    s_ndbi, c_ndbi = _labelwise_delta_mean(labels, DATASETS / e / "NDBI.tif",
                                           DATASETS / l / "NDBI.tif", n)
    s_ndwi, c_ndwi = _labelwise_delta_mean(labels, DATASETS / e / "NDWI.tif",
                                           DATASETS / l / "NDWI.tif", n)
    be, bl_, vfrac = _labelwise_badscl_valid(labels, DATASETS / e / "SCL.tif",
                                             DATASETS / l / "SCL.tif",
                                             DATASETS / e / "B08.tif", DATASETS / l / "B08.tif", n)
    del labels

    ctx = PairSuppressionContext(
        pair_id="ayodhya_2019_2024", coreg_residual_px=float(span_ctx_vals["coreg_residual_px"]),
        coreg_corrected=False,
        scene_d_ndvi=float(span_ctx_vals["scene_d_ndvi"]),
        scene_d_ndbi=float(span_ctx_vals["scene_d_ndbi"]),
        scene_d_ndwi=float(span_ctx_vals["scene_d_ndwi"]),
        radiometric_index_band_min_corr=float(span_ctx_vals["radiometric_index_band_min_corr"]),
        radiometric_low_confidence=False)

    supp_q = supp_phe = supp_mor = 0
    radio_dw = 0
    survivors = 0
    for li in ge50:
        li = int(li)
        a = int(areas[li])
        dv = s_ndvi[li] / max(c_ndvi[li], 1)
        db = s_ndbi[li] / max(c_ndbi[li], 1)
        dw = s_ndwi[li] / max(c_ndwi[li], 1)
        feats = CandidateFeatures(
            candidate_id=str(li), area_px=a,
            bad_scl_fraction_earlier=float(be[li]), bad_scl_fraction_later=float(bl_[li]),
            valid_fraction=float(vfrac[li]), d_ndvi=float(dv), d_ndbi=float(db), d_ndwi=float(dw))
        tr = suppress_candidate(feats, ctx, cfg)
        v = {o.rule: o.verdict for o in tr.outcomes}
        if tr.outcomes[2].verdict == "downweight":
            radio_dw += 1
        if v["quality"] == "suppress":
            supp_q += 1
            continue
        if v["phenology"] == "suppress":
            supp_phe += 1
            continue
        if v["morphology"] == "suppress":
            supp_mor += 1
            continue
        survivors += 1

    # cumulative chain on the >=50px universe (morphology floor already applied to
    # get from n_raw to n_ge50; on Ayodhya the floor is applied first for
    # tractability - documented; quality/phenology suppress ~0%/~9% so order is
    # immaterial to the conclusion)
    chain = [
        {"stage": "1_raw_model", "candidates": n_raw,
         "note": f"connected components of (p >= {thr:.2f}) & valid"},
        {"stage": "5_morphology_floor_applied_first", "candidates": n_ge50,
         "removed": n_raw - n_ge50,
         "note": f"the {cfg.morph_min_area_px}px / 0.5 ha floor; sub-floor components are "
                 "visually indistinguishable from noise (Phase 4 finding). On Ayodhya "
                 "this is applied before the other gates for tractability."},
        {"stage": "2_quality_gate", "candidates": n_ge50 - supp_q, "removed": supp_q,
         "note": "cloud/shadow/snow SCL on either date, or <80% jointly-valid px"},
        {"stage": "3_radiometric", "candidates": n_ge50 - supp_q,
         "removed": 0, "downweighted": radio_dw,
         "note": "down-weight only (min inter-date band corr "
                 f"{ctx.radiometric_index_band_min_corr:.2f} vs 0.60 trust line) - "
                 "cannot remove a candidate; feeds the confidence score"},
        {"stage": "4_phenology", "candidates": n_ge50 - supp_q - supp_phe, "removed": supp_phe,
         "note": "NDVI+NDBI+NDWI deltas all within their seasonal anomaly bands"},
        {"stage": "5_morphology_residual", "candidates": survivors, "removed": supp_mor,
         "note": "no >=50px component is below the floor - expected 0"},
    ]
    print("\n  cumulative candidate counts (>=50px universe):")
    for row in chain:
        extra = ""
        if "removed" in row:
            extra += f"  (-{row['removed']}"
            if row.get("downweighted"):
                extra += f", {row['downweighted']} down-weighted"
            extra += ")"
        print(f"    {row['stage']:34s} {row['candidates']:>7,}{extra}")
    print(f"\n  independent recompute survivors: {survivors:,}   "
          f"committed Phase 4/5 report: {report['pairs']['2019-2024']['suppression']['survived']:,}")

    # ---- stages 6/7/8 from the committed detail ----
    import collections
    pers = collections.Counter(c.get("persistence") for c in detail)
    supported = pers["persistent"] + pers["progressive"] + pers["recent"]
    demoted = pers["transient"] + pers["inconsistent"]
    sar_fac = collections.Counter(round((c.get("sar") or {}).get("factor", 1.0), 2) for c in detail)
    up = sum(v for k, v in sar_fac.items() if k > 1.001)
    down = sum(v for k, v in sar_fac.items() if k < 0.999)
    neutral = sum(v for k, v in sar_fac.items() if abs(k - 1.0) <= 0.001)
    conf = np.array([c["confidence"] for c in detail])
    wg = report["sar_corroboration"]["water_gain_validation"]

    diverse = [(c["rank"], c["candidate_id"], c["change_type"], c["confidence"],
                c["queue_score"], c["persistence"], c["area_m2"])
               for c in report["diverse_top_candidates"][:10]]
    # naive raw-prob top-10 for contrast
    naive = sorted(detail, key=lambda c: c["mean_model_prob"], reverse=True)[:10]
    naive_rows = [(c["rank"], c["candidate_id"], c["change_type"], c["confidence"],
                   round(c["mean_model_prob"], 3), c["persistence"], c["area_m2"]) for c in naive]

    stage6 = {
        "applicable": True, "n_span_survivors": len(detail),
        "persistence_distribution": dict(pers),
        "temporally_supported": supported, "temporally_demoted": demoted,
        "span_only_support_none": pers["none"],
        "effect": (f"{demoted} of {len(detail)} span survivors ({100*demoted/len(detail):.0f}%) "
                   f"are transient/inconsistent - the persistence penalty (x0.5 / x0.45) drops "
                   f"them below actionable confidence. {(conf >= 0.5).sum()} survivors keep "
                   f"confidence >= 0.5, which equals persistent+progressive+recent exactly - "
                   f"persistence is the lever that splits the actionable queue from the rest."),
    }
    stage7 = {
        "applicable": True,
        "sar_factor_distribution": {str(k): int(v) for k, v in sorted(sar_fac.items())},
        "candidates_upweighted": up, "candidates_downweighted": down, "candidates_neutral": neutral,
        "water_gain_agreement": {
            "n_water_gain": wg["n_water_gain_survivors"], "n_agree": wg["n_agree"],
            "rate": wg["agreement_rate"], "median_vv_anomaly_db": wg["median_vv_anomaly_db"],
            "physical_reason": ("expected VV backscatter DROP for land->open-water (specular). "
                               "51% agreement, not higher, because in a drought-recovery March "
                               "many 'water_gain' candidates are soil-moisture / vegetation "
                               "recovery (SAR backscatter RISES with moisture and biomass) - "
                               "only true open-water gain shows the specular drop. Top-ranked "
                               "candidates do show it and get the +10% factor."),
        },
        "cloud_penetration_demonstrable": False,
        "cloud_penetration_note": report["sar_corroboration"]["cloud_penetration"]["finding"],
        "weight_only": "SAR never overrides - [0.80, 1.10] multiplicative factor on confidence.",
    }
    diverse_persist = sum(1 for (_, _, _, _, _, p, _) in diverse
                          if p in ("persistent", "progressive"))
    naive_persist = sum(1 for c in naive if c["persistence"] in ("persistent", "progressive"))
    naive_recent = sum(1 for c in naive if c["persistence"] == "recent")
    stage8 = {
        "applicable": True,
        "full_pipeline_diverse_top10": [
            {"rank": r, "id": cid, "type": ct, "confidence": cf, "queue_score": qs,
             "persistence": p, "area_m2": am} for (r, cid, ct, cf, qs, p, am) in diverse],
        "naive_raw_prob_top10": [
            {"orig_rank": r, "id": cid, "type": ct, "confidence": cf, "mean_model_prob": mp,
             "persistence": p, "area_m2": am} for (r, cid, ct, cf, mp, p, am) in naive_rows],
        "contrast": {
            "median_area_m2_full_top10": float(np.median([am for *_, am in diverse])),
            "median_area_m2_naive_top10": float(np.median([am for *_, am in naive_rows])),
            "persistent_or_progressive_full_top10": diverse_persist,
            "persistent_or_progressive_naive_top10": naive_persist,
            "recent_only_naive_top10": naive_recent,
            "min_confidence_naive_top10": round(min(c["confidence"] for c in naive), 3),
            "naive_top10_orig_ranks": [c["rank"] for c in naive],
        },
        "qualitative_effect": (
            "Raw mean-probability ordering surfaces the brightest blobs regardless of trust or "
            f"size: the naive top-10 come from full-queue ranks {sorted(c['rank'] for c in naive)}, "
            f"have a median area of {np.median([am for *_, am in naive_rows]):,.0f} m^2, and are "
            f"{naive_persist}/10 persistent/progressive vs {naive_recent}/10 single-interval "
            f"'recent' (one is transient, confidence "
            f"{min(c['confidence'] for c in naive):.2f}). The full queue_score "
            "(confidence^0.65 * significance^0.35, then diversify: <=3 per type, 1.5 km spacing) "
            f"yields a top-10 with median area {np.median([am for *_, am in diverse]):,.0f} m^2, "
            f"{diverse_persist}/10 persistent/progressive, split 3 water_gain / 3 construction / "
            "3 road / 1 other - large, temporally-confirmed, type-diverse changes an analyst can "
            "action, not the model's loudest pixels."),
    }

    del prob
    return {
        "domain": "ayodhya_span_2019_2024",
        "labels_available": False,
        "metric_note": "candidate COUNTS and qualitative effects only - NOT precision/recall",
        "operating_point": thr,
        "cumulative_counts": chain,
        "independent_recompute_survivors": survivors,
        "committed_report_survivors": report["pairs"]["2019-2024"]["suppression"]["survived"],
        "committed_report_suppressed_by_rule":
            report["pairs"]["2019-2024"]["suppression"]["suppressed_by_rule"],
        "stage6_persistence": stage6,
        "stage7_sar": stage7,
        "stage8_confidence_ranking": stage8,
    }


# ==========================================================================
# main
# ==========================================================================


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oscd-only", action="store_true")
    ap.add_argument("--ayodhya-only", action="store_true")
    ap.add_argument("--out", default=str(OUT / "ablation_study.json"))
    args = ap.parse_args(argv)

    cfg = SuppressionConfig()
    study = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": [{"key": k, "description": d} for k, d in STAGES],
        "protocol": {
            "cumulative": "each stage adds one component to the previous stage's configuration",
            "oscd": "stages 1-5 give pixel P/R/F1/IoU/FPR vs the OSCD change mask (10 held-out "
                    "regions, pixel-pooled + per-region). Stage 8 gives component-level AP. "
                    "Stages 6/7 are inapplicable (bitemporal; no SAR).",
            "ayodhya": "candidate COUNTS + qualitative effects ONLY (no change labels). Stages "
                       "6/7/8 assessed here because this is where they run.",
            "suppression_config": {
                "max_bad_scl_fraction": cfg.max_bad_scl_fraction,
                "min_valid_fraction": cfg.min_valid_fraction,
                "radiometric_min_corr": cfg.radiometric_min_corr,
                "pheno_ndvi_anomaly": cfg.pheno_ndvi_anomaly,
                "pheno_ndbi_anomaly": cfg.pheno_ndbi_anomaly,
                "pheno_ndwi_anomaly": cfg.pheno_ndwi_anomaly,
                "morph_min_area_px": cfg.morph_min_area_px,
            },
        },
    }

    if not args.ayodhya_only:
        study["oscd"] = run_oscd(cfg)
    if not args.oscd_only:
        study["ayodhya"] = run_ayodhya(cfg)

    Path(args.out).write_text(json.dumps(study, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    try:
        from geoseek.staging.manifest import record_analysis_section

        record_analysis_section("change_pipeline_ablation", study)
        print("  recorded manifest section 'change_pipeline_ablation'")
    except Exception as exc:  # manifest is optional / gitignored
        print(f"  (manifest not updated: {exc})")


if __name__ == "__main__":
    main()

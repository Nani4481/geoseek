"""Phase 7a Step B (PS 2.3): constructed relevance judgements for the retrieval
evaluation - the INDEPENDENT step.

Circularity, and how it is avoided
----------------------------------
RemoteCLIP is the system under test. Judging its results with RemoteCLIP (or
with the vanilla CLIP control, or any learned embedding) makes the evaluation
circular and drives every metric to ~1.0 by construction. So the judgements
here come from a signal that neither retrieval model can see: per-tile
physical/spectral criteria computed by ``scripts/eval_retrieval_features.py``
from the Sentinel-2 NDVI / NDWI / NDBI index rasters, the SCL validity layer,
and an NDWI-derived AOI river/water mask. The retrieval models embed only the
true-colour RGB; they never receive NDWI, NDBI, SCL, or the river mask.

These are CONSTRUCTED judgements, not expert ground truth. Their limitations
are written into ``judgments_rationale.json`` and repeated in the report.

Pooling
-------
For each query the judged pool is the union of:
  * RemoteCLIP's top-20,
  * vanilla CLIP's top-20,
  * a fixed random sample of the full 3267-tile corpus (seed in pools.json).
Pooling both systems symmetrically plus a random draw keeps either model from
being advantaged by pool bias, and lets Recall@K see relevant tiles that
neither system ranked highly. Every pooled tile is judged, so no unjudged item
can enter a metric at any reported K (max K = pool depth = 20).

Grades: 2 = clearly matches the query's physical definition, 1 = partial /
marginal match, 0 = does not match.

Usage:  python scripts/eval_retrieval_judge.py
        (needs tile_features.json + pools.json in data/eval_retrieval/)
"""

from __future__ import annotations

import json
from pathlib import Path

from geoseek.config import get_settings

OUT_DIR_NAME = "eval_retrieval"

# Queries whose physical criterion is weak at 10 m GSD (a road/bridge is
# 1-2 px wide in a 256 px tile). Their judgements are kept but flagged, and the
# report shows the metric table with and without them.
LOW_CONFIDENCE_QUERIES = {
    "a paved road",
    "a dirt track or unpaved road",
    "a bridge crossing a river",
}


# --------------------------------------------------------------------------
# per-query graders: (features) -> (grade 0/1/2, criterion string)
# thresholds are calibrated against the corpus-wide feature distribution
# (see PHASE7.md); this AOI is dominated by peak-rabi cropland, so "vegetation"
# is the mode and true bare ground / wide open water are both rare.
# --------------------------------------------------------------------------

def _g_water_body(f):
    wf, dw = f["water_frac"], f["dist_water_m"]
    if wf >= 0.10:
        return 2, f"water_frac {wf:.2f} >= 0.10 (substantial open water in tile)"
    if wf >= 0.03 or (dw == 0.0 and wf >= 0.015):
        return 1, f"water_frac {wf:.2f} / touches water (dist_water_m {dw:.0f})"
    return 0, f"water_frac {wf:.2f} < 0.03 and not on water"


def _g_river_sandbars(f):
    dr, wf, bf = f["dist_river_m"], f["water_frac"], f["bare_frac"]
    if dr <= 200 and wf >= 0.05 and bf >= 0.04:
        return 2, f"on river (dist {dr:.0f} m), water_frac {wf:.2f}, exposed sediment bare_frac {bf:.2f}"
    if dr <= 400 and wf >= 0.02:
        return 1, f"near river channel (dist {dr:.0f} m), water_frac {wf:.2f}"
    return 0, f"not on the river channel (dist_river_m {dr:.0f}, water_frac {wf:.2f})"


def _g_bridge(f):
    dr, wf, bld, ed = f["dist_river_m"], f["water_frac"], f["built_frac"], f["edge_density"]
    if dr <= 100 and wf >= 0.03 and bld >= 0.20 and ed >= 0.18:
        return 2, f"structure signature over the river (dist {dr:.0f} m, built {bld:.2f}, edges {ed:.2f})"
    if dr <= 250 and (bld >= 0.20 or ed >= 0.22):
        return 1, f"built/linear feature adjacent to river (dist {dr:.0f} m, built {bld:.2f}, edges {ed:.2f})"
    return 0, f"no bridging structure at the river (dist_river_m {dr:.0f})"


def _g_riverside_built(f):
    dr, bld, ndvi = f["dist_river_m"], f["built_frac"], f["ndvi_p50"]
    if dr <= 600 and bld >= 0.28 and ndvi <= 0.50:
        return 2, f"built-up on the riverbank (dist {dr:.0f} m, built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"
    if (dr <= 1500 and bld >= 0.25 and ndvi <= 0.55) or (dr <= 600 and bld >= 0.20):
        return 1, f"partial built-up near the river (dist {dr:.0f} m, built_frac {bld:.2f})"
    return 0, f"not built-up near a river (dist_river_m {dr:.0f}, built_frac {bld:.2f})"


def _g_dense_urban(f):
    bld, ndvi, ed = f["built_frac"], f["ndvi_p50"], f["edge_density"]
    if bld >= 0.45 and ndvi <= 0.32 and ed >= 0.18:
        return 2, f"dense built-up (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f}, edges {ed:.2f})"
    if bld >= 0.32 and ndvi <= 0.42:
        return 1, f"moderate built-up (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"
    return 0, f"not densely built (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"


def _g_residential(f):
    bld, ndvi, ed = f["built_frac"], f["ndvi_p50"], f["edge_density"]
    if bld >= 0.35 and 0.20 <= ndvi <= 0.48 and ed >= 0.15:
        return 2, f"built + interstitial greenery (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"
    if bld >= 0.27 and ndvi <= 0.52:
        return 1, f"partial residential mix (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"
    return 0, f"not a residential built pattern (built_frac {bld:.2f}, ndvi_p50 {ndvi:.2f})"


def _g_agri(f):
    ndvi, ed, wf, bld = f["ndvi_p50"], f["edge_density"], f["water_frac"], f["built_frac"]
    if 0.35 <= ndvi <= 0.75 and ed >= 0.15 and wf <= 0.03 and bld <= 0.25:
        return 2, f"vegetated + field texture (ndvi_p50 {ndvi:.2f}, edges {ed:.2f}, built {bld:.2f})"
    if ndvi >= 0.30 and ed >= 0.10 and bld <= 0.30:
        return 1, f"vegetated, weaker field texture (ndvi_p50 {ndvi:.2f}, edges {ed:.2f})"
    return 0, f"not cropland (ndvi_p50 {ndvi:.2f}, edges {ed:.2f}, built {bld:.2f})"


def _g_cropland_boundaries(f):
    ndvi, ed, bld = f["ndvi_p50"], f["edge_density"], f["built_frac"]
    if 0.35 <= ndvi <= 0.78 and ed >= 0.20 and bld <= 0.22:
        return 2, f"strong field-boundary texture (ndvi_p50 {ndvi:.2f}, edge_density {ed:.2f})"
    if ndvi >= 0.30 and ed >= 0.14 and bld <= 0.28:
        return 1, f"some field-boundary texture (ndvi_p50 {ndvi:.2f}, edge_density {ed:.2f})"
    return 0, f"no visible field boundaries (ndvi_p50 {ndvi:.2f}, edge_density {ed:.2f})"


def _g_irrigated(f):
    ndvi, ndwi90, bld, ed = f["ndvi_p50"], f["ndwi_p90"], f["built_frac"], f["edge_density"]
    if ndvi >= 0.40 and ndwi90 >= -0.28 and bld <= 0.22 and ed >= 0.12:
        return 2, f"green + moist fields (ndvi_p50 {ndvi:.2f}, ndwi_p90 {ndwi90:.2f})"
    if ndvi >= 0.35 and ndwi90 >= -0.34:
        return 1, f"green fields, drier signal (ndvi_p50 {ndvi:.2f}, ndwi_p90 {ndwi90:.2f})"
    return 0, f"not irrigated farmland (ndvi_p50 {ndvi:.2f}, ndwi_p90 {ndwi90:.2f})"


def _g_bare(f):
    bare, ndvi, wf, ed = f["bare_frac"], f["ndvi_p50"], f["water_frac"], f["edge_density"]
    if bare >= 0.40 and ndvi <= 0.25 and wf <= 0.02 and ed <= 0.16:
        return 2, f"open bare ground (bare_frac {bare:.2f}, ndvi_p50 {ndvi:.2f}, edges {ed:.2f})"
    if bare >= 0.22 and ndvi <= 0.35 and wf <= 0.03:
        return 1, f"partly bare (bare_frac {bare:.2f}, ndvi_p50 {ndvi:.2f})"
    return 0, f"not bare ground (bare_frac {bare:.2f}, ndvi_p50 {ndvi:.2f})"


def _g_riverbank_veg(f):
    dr, dvf, ndvi = f["dist_river_m"], f["dense_veg_frac"], f["ndvi_p50"]
    if dr <= 400 and dvf >= 0.55 and ndvi >= 0.55:
        return 2, f"dense canopy on the riverbank (dist {dr:.0f} m, dense_veg_frac {dvf:.2f})"
    if (dr <= 1200 and dvf >= 0.50) or (dr <= 400 and ndvi >= 0.45):
        return 1, f"vegetation near the river (dist {dr:.0f} m, dense_veg_frac {dvf:.2f})"
    return 0, f"not riverbank vegetation (dist_river_m {dr:.0f}, dense_veg_frac {dvf:.2f})"


def _g_paved_road(f):
    ed, ndvi10, bld = f["edge_density"], f["ndvi_p10"], f["built_frac"]
    if ed >= 0.28 and ndvi10 <= 0.15 and 0.18 <= bld <= 0.55:
        return 2, f"high-contrast low-veg linear feature near built (edges {ed:.2f}, ndvi_p10 {ndvi10:.2f})"
    if ed >= 0.22 and ndvi10 <= 0.22 and bld >= 0.15:
        return 1, f"possible paved corridor (edges {ed:.2f}, ndvi_p10 {ndvi10:.2f})"
    return 0, f"no paved-road signature (edges {ed:.2f}, ndvi_p10 {ndvi10:.2f}, built {bld:.2f})"


def _g_dirt_road(f):
    ed, ndvi10, bare = f["edge_density"], f["ndvi_p10"], f["bare_frac"]
    if ed >= 0.26 and ndvi10 <= 0.18 and bare >= 0.10:
        return 2, f"exposed-soil linear feature (edges {ed:.2f}, ndvi_p10 {ndvi10:.2f}, bare_frac {bare:.2f})"
    if ed >= 0.20 and ndvi10 <= 0.25 and bare >= 0.05:
        return 1, f"possible unpaved track (edges {ed:.2f}, bare_frac {bare:.2f})"
    return 0, f"no unpaved-track signature (edges {ed:.2f}, bare_frac {bare:.2f})"


GRADERS = {
    "newly built structures near a river": _g_riverside_built,
    "settlement along a riverbank": _g_riverside_built,
    "a river with sandbars": _g_river_sandbars,
    "dense urban buildings": _g_dense_urban,
    "agricultural fields": _g_agri,
    "open bare ground": _g_bare,
    "a water body": _g_water_body,
    "cropland with visible field boundaries": _g_cropland_boundaries,
    "a paved road": _g_paved_road,
    "riverside construction": _g_riverside_built,
    "a bridge crossing a river": _g_bridge,
    "dense vegetation along a riverbank": _g_riverbank_veg,
    "an urban residential neighborhood": _g_residential,
    "irrigated farmland": _g_irrigated,
    "a braided river channel with sandbars": _g_river_sandbars,
    "a dirt track or unpaved road": _g_dirt_road,
}


def main() -> None:
    out_dir: Path = get_settings().data_dir / OUT_DIR_NAME
    feats = json.loads((out_dir / "tile_features.json").read_text())
    features = feats["features"]
    pools = json.loads((out_dir / "pools.json").read_text())
    queries = json.loads((out_dir / "queries.json").read_text())

    missing_graders = [q for q in queries if q not in GRADERS]
    if missing_graders:
        raise SystemExit(f"no grader for: {missing_graders}")

    judgments: dict[str, dict[str, int]] = {}
    rationale_per_query: dict[str, dict[str, dict]] = {}
    summary_rows = []

    for q in queries:
        grader = GRADERS[q]
        pool_ids = pools[q]["pool_tile_ids"]
        j: dict[str, int] = {}
        rat: dict[str, dict] = {}
        for tid in pool_ids:
            f = features.get(tid)
            if f is None or not f.get("usable", False):
                j[tid] = 0
                rat[tid] = {"grade": 0, "criterion": "tile not usable (insufficient valid pixels)"}
                continue
            grade, why = grader(f)
            j[tid] = grade
            rat[tid] = {
                "grade": grade,
                "criterion": why,
                "source_pools": [
                    name for name, key in (("remoteclip", "remoteclip_ranked"),
                                            ("vanilla", "vanilla_ranked"),
                                            ("random", "random_sample_tile_ids"))
                    if tid in set(pools[q].get(key, []))
                ],
            }
        judgments[q] = j
        rationale_per_query[q] = rat
        n_rel = sum(1 for v in j.values() if v > 0)
        n_g2 = sum(1 for v in j.values() if v == 2)
        summary_rows.append((q, len(j), n_rel, n_g2, q in LOW_CONFIDENCE_QUERIES))

    (out_dir / "judgments.json").write_text(json.dumps(judgments, indent=2))

    rationale = {
        "methodology": {
            "judge_signal": "per-tile NDVI/NDWI/NDBI/SCL spectral criteria + NDWI-derived river mask "
                            "(from scripts/eval_retrieval_features.py)",
            "judge_is_the_system_under_test": False,
            "circularity_avoidance": "The retrieval models (RemoteCLIP, vanilla CLIP) embed only the "
                                     "true-colour RGB. The judge uses NDWI/NDBI/SCL and a river mask, "
                                     "which neither model receives. No embedding, learned or otherwise, "
                                     "is used to assign any grade.",
            "grades": {"2": "clear physical match", "1": "partial/marginal", "0": "no match"},
            "pool_definition": "per query: union(RemoteCLIP top-20, vanilla CLIP top-20, "
                               "fixed random sample of the 3267-tile corpus)",
            "pool_sampling": "both systems pooled symmetrically to the max reported K (20); a random "
                             "draw is added so neither model is advantaged by pool bias and so Recall@K "
                             "can see relevant tiles neither system ranked highly",
            "tie_and_ambiguity_handling": "graded 0/1/2 by fixed thresholds calibrated to the "
                                          "corpus-wide feature distribution; a tile that meets no "
                                          "positive rule is 0; ambiguous cases land at 1, not 2",
            "limitations": [
                "Constructed from spectral proxies, not photointerpretation or field data - not expert "
                "ground truth.",
                "Single-date: 'newly built' / 'construction' are judged as 'built-up near a river', "
                "not verified as recent; temporal novelty would need the change pipeline.",
                "Road / track / bridge criteria are weak at 10 m GSD (sub-pixel width) - these three "
                "queries are flagged low-confidence and the report shows metrics with and without them.",
                "NDBI in this AOI is only weakly positive over low-rise mixed built-up, so 'built' is a "
                "relative, texture-assisted threshold, not an absolute one.",
                "Thresholds are one fixed set; no sensitivity sweep was run.",
            ],
        },
        "low_confidence_queries": sorted(LOW_CONFIDENCE_QUERIES),
        "per_query": rationale_per_query,
    }
    (out_dir / "judgments_rationale.json").write_text(json.dumps(rationale, indent=2))

    print(f"{'query':44s} {'pool':>5} {'rel':>5} {'g2':>4}  lowconf")
    print("-" * 72)
    for q, pool, rel, g2, low in summary_rows:
        print(f"{q:44s} {pool:>5} {rel:>5} {g2:>4}  {'yes' if low else ''}")
    print(f"\n[judge] wrote judgments.json + judgments_rationale.json under {out_dir}")


if __name__ == "__main__":
    main()

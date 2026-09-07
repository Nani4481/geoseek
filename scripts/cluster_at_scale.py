"""Phase 7b follow-up: HDBSCAN clustering that actually completes at ~100k tiles.

Phase 7b Tier 3 established that full single-pass HDBSCAN over 101,911 raw
512-d embeddings does not finish in practical batch time (>60 min, killed).
This is the standard, disclosed fix: cluster a REPRESENTATIVE STRATIFIED
SAMPLE (default 20k, stratified across regions), then assign every remaining
tile to its nearest sample-cluster centroid (one dense cosine matmul).

Reports:
  * runtime (sample HDBSCAN + assignment), separately
  * cluster count / sizes / concept labels (full corpus)
  * region purity per cluster (the generalization signal)
  * agreement with the Tier-1 FULL clustering on the tiles common to both
    (ARI / AMI / homogeneity), i.e. does sampling recover the same structure

Artifacts under data/discovery/:
  cluster_at_scale_100k.json         full result + metrics
  tile_clusters_at_scale_100k.json   {tile_id: label} for every tile
  cluster_at_scale_region_heatmap.png

Usage:
  python scripts/cluster_at_scale.py [--sample-size 20000] [--min-cluster-size 40] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from geoseek.config import get_settings
from geoseek.discovery.cluster import CONCEPTS, cluster_sample_and_assign, load_all_vectors
from geoseek.search.engine import SearchEngine
from geoseek.search.rerank import region_key

OUT_DIR = get_settings().data_dir / "discovery"
TIER1_FULL = OUT_DIR / "tile_clusters_tier1_eom.json"


def _region_map(engine: SearchEngine) -> dict[str, str]:
    """tile_id -> base region slug, via observation.aoi_name."""
    aoi_by_obs: dict[str, str] = {}
    for obs in engine.repo.list_observations():
        aoi_by_obs[obs.observation_id] = region_key(obs.aoi_name)
    out: dict[str, str] = {}
    for rec in engine.repo.iter_tile_records():
        if rec.faiss_id is not None:
            out[rec.tile_id] = aoi_by_obs.get(rec.observation_id, "unknown")
    return out


def _agreement_with_tier1(tile_ids, labels) -> dict:
    if not TIER1_FULL.is_file():
        return {"available": False, "reason": f"{TIER1_FULL.name} not found"}
    try:
        from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                     homogeneity_completeness_v_measure)
    except Exception as e:  # pragma: no cover
        return {"available": False, "reason": f"sklearn.metrics import failed: {e}"}

    tier1 = json.loads(TIER1_FULL.read_text())
    t1_map = tier1.get("tile_cluster") or tier1.get("tile_clusters") or {}
    lab_now = dict(zip(tile_ids, [int(x) for x in labels]))
    common = [t for t in t1_map if t in lab_now]
    if not common:
        return {"available": False, "reason": "no common tiles"}
    a = np.array([int(t1_map[t]) for t in common])
    b = np.array([lab_now[t] for t in common])

    def _metrics(mask_desc, m):
        aa, bb = a[m], b[m]
        if len(aa) < 2 or len(set(aa.tolist())) < 2:
            return {"n": int(len(aa)), "note": "too few / single-class"}
        h, c, v = homogeneity_completeness_v_measure(aa, bb)
        return {"n": int(len(aa)),
                "adjusted_rand_index": round(float(adjusted_rand_score(aa, bb)), 4),
                "adjusted_mutual_info": round(float(adjusted_mutual_info_score(aa, bb)), 4),
                "homogeneity": round(float(h), 4), "completeness": round(float(c), 4),
                "v_measure": round(float(v), 4)}

    all_mask = np.ones(len(common), bool)
    non_noise = (a >= 0) & (b >= 0)
    # contingency (rows = Tier-1 cluster, cols = at-scale cluster), non-noise only
    cont: dict[str, dict[str, int]] = {}
    for ta, tb in zip(a[non_noise], b[non_noise]):
        cont.setdefault(f"tier1_c{ta}", {}).setdefault(f"scale_c{tb}", 0)
        cont[f"tier1_c{ta}"][f"scale_c{tb}"] += 1

    return {
        "available": True,
        "n_common_tiles": len(common),
        "all_tiles_incl_noise": _metrics("all", all_mask),
        "non_noise_both": _metrics("non_noise", non_noise),
        "tier1_noise_frac": round(float((a < 0).mean()), 4),
        "at_scale_noise_frac_on_common": round(float((b < 0).mean()), 4),
        "contingency_non_noise": cont,
    }


def _region_purity(tile_ids, labels, region_map) -> dict:
    by_c: dict[int, Counter] = {}
    for tid, lab in zip(tile_ids, labels):
        by_c.setdefault(int(lab), Counter())[region_map.get(tid, "unknown")] += 1
    out = {}
    for c, ctr in sorted(by_c.items()):
        tot = sum(ctr.values())
        top_region, top_n = ctr.most_common(1)[0]
        out[str(c)] = {"size": tot, "dominant_region": top_region,
                       "purity": round(top_n / tot, 4),
                       "regions": dict(ctr.most_common())}
    return out


def _region_heatmap(tile_ids, labels, region_map, centroid_ids, concepts, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = sorted(set(region_map.values()))
    cids = list(centroid_ids)
    M = np.zeros((len(cids), len(regions)))
    ridx = {r: i for i, r in enumerate(regions)}
    cidx = {c: i for i, c in enumerate(cids)}
    for tid, lab in zip(tile_ids, labels):
        if int(lab) in cidx:
            M[cidx[int(lab)], ridx[region_map.get(tid, "unknown")]] += 1
    Mn = M / (M.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(regions)), max(4, 0.5 * len(cids) + 1)))
    im = ax.imshow(Mn, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cids)))
    ax.set_yticklabels([f"c{c} (n={int(M[i].sum())})  {concepts.get(c, [['', 0]])[0][0][:26]}"
                        for i, c in enumerate(cids)], fontsize=8)
    ax.set_title("at-scale cluster → region composition (row-normalised)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction of cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=20_000)
    ap.add_argument("--min-cluster-size", type=int, default=40)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--abstain-below-sim", type=float, default=None,
                    help="optional: label tiles with best-centroid cosine below this as noise (-1)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("geoseek Phase 7b follow-up - clustering at 100k via sample + assign")
    print("=" * 78)
    eng = SearchEngine()
    try:
        recs = [r for r in eng.repo.iter_tile_records() if r.faiss_id is not None]
        recs.sort(key=lambda r: int(r.faiss_id))
        tile_ids = [r.tile_id for r in recs]
        region_map = _region_map(eng)
        groups = [region_map.get(t, "unknown") for t in tile_ids]
        print(f"  embedded tiles: {len(tile_ids)}   regions: {sorted(set(groups))}")
        print(f"  region sizes: {dict(Counter(groups).most_common())}")

        t0 = time.time()
        vectors = load_all_vectors(eng.vector_index)
        load_s = time.time() - t0
        print(f"  loaded {vectors.shape} vectors in {load_s:.1f}s")

        t0 = time.time()
        res = cluster_sample_and_assign(
            vectors, tile_ids, groups, eng.embedding_model,
            sample_size=args.sample_size, min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples, core_dist_n_jobs=args.n_jobs, seed=args.seed,
            abstain_below_sim=args.abstain_below_sim)
        total_s = time.time() - t0

        labels = res.labels
        print(f"\n  sample: {len(res.sample_idx)} tiles -> HDBSCAN -> "
              f"{res.sample_result.n_clusters} clusters, {res.sample_result.noise_count} noise")
        print(f"  full corpus: {res.n_clusters} clusters, {res.noise_count} noise "
              f"({100 * res.noise_count / len(labels):.1f}%)")
        print(f"  runtime: sample HDBSCAN + assignment = {total_s:.1f}s  (vector load {load_s:.1f}s)")
        print(f"  {'cluster':>8} {'full_size':>10} {'sample_size':>12}   concepts")
        for cid in sorted(res.sizes, key=lambda k: res.sizes[k], reverse=True):
            cc = ", ".join(f"{c} ({s:.2f})" for c, s in res.cluster_concepts[cid])
            print(f"  {cid:>8} {res.sizes[cid]:>10} {res.sample_result.sizes.get(cid, 0):>12}   {cc}")

        purity = _region_purity(tile_ids, labels, region_map)
        print("\n  region purity (full clusters):")
        for c, p in sorted(purity.items(), key=lambda kv: -kv[1]["size"]):
            if int(c) < 0:
                continue
            print(f"    c{c:>3} n={p['size']:>6}  {p['purity']*100:5.1f}% {p['dominant_region']}")

        agree = _agreement_with_tier1(tile_ids, labels)
        print(f"\n  agreement with Tier-1 full clustering (hard assignment): "
              f"{json.dumps(agree.get('non_noise_both', agree), indent=2)}")

        # abstention sweep: hard nearest-centroid labels every tile, so it
        # necessarily disagrees with Tier-1 exactly on the ~27% Tier-1 calls
        # noise. Re-score after abstaining low-confidence assignments, incl. the
        # threshold that reproduces Tier-1's own noise fraction.
        top_sim = res.assign_top_sim
        t1_noise = agree.get("tier1_noise_frac")
        sweep_taus = [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]
        if t1_noise:
            sweep_taus.append(round(float(np.quantile(top_sim, t1_noise)), 4))
        abstain_sweep = []
        for tau in sorted(set(sweep_taus)):
            lab_a = labels.copy()
            lab_a[top_sim < tau] = -1
            ag = _agreement_with_tier1(tile_ids, lab_a)
            abstain_sweep.append({
                "abstain_below_sim": tau,
                "noise_frac": round(float((lab_a < 0).mean()), 4),
                "ARI_all_tiles": ag["all_tiles_incl_noise"].get("adjusted_rand_index"),
                "AMI_all_tiles": ag["all_tiles_incl_noise"].get("adjusted_mutual_info"),
                "v_measure_all_tiles": ag["all_tiles_incl_noise"].get("v_measure"),
            })
        agree["abstention_sweep_vs_tier1"] = abstain_sweep
        print("  abstention sweep (all-tiles agreement vs Tier-1):")
        for s in abstain_sweep:
            print(f"    tau>={s['abstain_below_sim']}  noise {s['noise_frac']*100:4.1f}%  "
                  f"ARI {s['ARI_all_tiles']}  V {s['v_measure_all_tiles']}")

        heat = _region_heatmap(tile_ids, labels, region_map, res.centroid_ids,
                               res.cluster_concepts, OUT_DIR / "cluster_at_scale_region_heatmap.png")
        print(f"  region heatmap -> {heat}")

        payload = res.as_dict()
        payload.update({
            "runtime_seconds": {"vector_load": round(load_s, 1),
                                "sample_hdbscan_plus_assign": round(total_s, 1)},
            "region_purity": purity,
            "tier1_agreement": agree,
            "concepts_vocabulary": CONCEPTS,
            "region_sizes": dict(Counter(groups).most_common()),
            "assign_margin_pctl": {p: round(float(np.percentile(res.assign_margin, p)), 4)
                                   for p in (1, 5, 25, 50, 75)},
            "region_heatmap_png": heat,
        })
        (OUT_DIR / "cluster_at_scale_100k.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (OUT_DIR / "tile_clusters_at_scale_100k.json").write_text(
            json.dumps({t: int(l) for t, l in zip(tile_ids, labels.tolist())}), encoding="utf-8")
        print(f"\n  -> {OUT_DIR / 'cluster_at_scale_100k.json'}")
        print(f"  -> {OUT_DIR / 'tile_clusters_at_scale_100k.json'}")
    finally:
        eng.close()


if __name__ == "__main__":
    main()

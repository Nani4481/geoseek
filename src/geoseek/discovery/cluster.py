"""Offline HDBSCAN clustering over all tile embeddings (PS 2.2.4).

A **batch** job (``scripts/cluster_tiles.py``), never per query: cluster the
whole 512-d RemoteCLIP embedding set, label each cluster with its nearest text
concepts, and save a cluster map. The per-query interactive path is
:mod:`geoseek.discovery.knn`.

Vectors are unit-norm, so Euclidean distance is monotonic in cosine distance
(``||a-b||^2 = 2 - 2 cos``) and HDBSCAN's default euclidean metric behaves as a
cosine clusterer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# satellite-scene vocabulary used to name clusters (RemoteCLIP text tower)
CONCEPTS = [
    "a river with wide sandbars", "dense urban buildings and rooftops",
    "green irrigated agricultural cropland", "bare dry open ground",
    "a large building or institutional complex", "an active construction site with cleared earth",
    "an open water reservoir or pond", "a network of roads and streets",
    "trees and dense vegetation", "a sandy braided riverbed",
    "a small rural village settlement", "a quarry or excavation pit",
    "fallow ploughed fields", "a temple or religious complex",
    "flooded or waterlogged land",
]

DEFAULT_MIN_CLUSTER_SIZE = 40


@dataclass
class ClusterResult:
    labels: np.ndarray                 # (N,) HDBSCAN labels; -1 = noise
    n_clusters: int
    noise_count: int
    sizes: dict                        # {cluster_id: size}
    cluster_concepts: dict             # {cluster_id: [(concept, cos), ...] top-3}
    params: dict
    tile_ids: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n_clusters": self.n_clusters, "noise_count": self.noise_count,
                "n_tiles": int(len(self.labels)), "sizes": self.sizes,
                "cluster_concepts": {str(k): [[c, round(float(s), 4)] for c, s in v]
                                     for k, v in self.cluster_concepts.items()},
                "params": self.params}


def load_all_vectors(vector_index) -> np.ndarray:
    recon_all = getattr(vector_index, "reconstruct_all", None)
    if callable(recon_all):
        return recon_all()
    n = vector_index.count()
    v = np.zeros((n, 512), dtype=np.float32)
    for i in range(n):
        v[i] = vector_index.get_vector(i)
    return v


def cluster_embeddings(vectors: np.ndarray, tile_ids: list[str], embedding_model, *,
                       min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                       min_samples: int | None = None,
                       core_dist_n_jobs: int = 1,
                       cluster_selection_method: str = "eom",
                       cluster_vectors: np.ndarray | None = None,
                       n_concepts_per_cluster: int = 3) -> ClusterResult:
    """``cluster_vectors`` (optional): a separate, e.g. PCA-reduced, array used
    ONLY for the HDBSCAN distance computation (same row order/count as
    ``vectors``). Concept labeling always uses the original ``vectors`` - PCA
    axes are not meaningful in RemoteCLIP's text-embedding space, so cluster
    centroids for concept similarity must stay in the native 512-d space.
    ``cluster_selection_method``: HDBSCAN's 'eom' (default; tends toward a few
    large, stable clusters) or 'leaf' (selects further down the condensed
    tree - more, smaller, finer-grained clusters; see PS 2.2.4 quality sweep
    in PHASE8_POLISH.md for a comparison on this corpus)."""
    from hdbscan import HDBSCAN

    fit_vectors = cluster_vectors if cluster_vectors is not None else vectors

    # core_dist_n_jobs: parallelism for the core-distance computation only - a
    # pure performance knob (sklearn's KNN backend), not part of the HDBSCAN
    # algorithm itself; same inputs produce the same labels regardless of
    # this value. Default 1 preserves exact prior behavior. At Phase 7b's
    # ~100k tiles / 512-d, single-threaded core-distance computation in this
    # dimensionality is impractical (>45 min, still not finished) - see
    # PHASE7B.md.
    clf = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                  metric="euclidean", core_dist_n_jobs=core_dist_n_jobs,
                  cluster_selection_method=cluster_selection_method)
    labels = clf.fit_predict(fit_vectors.astype(np.float64))
    uniq = sorted(int(x) for x in set(labels) if x >= 0)
    sizes = {int(c): int((labels == c).sum()) for c in uniq}
    noise = int((labels == -1).sum())

    concept_vecs = np.stack([embedding_model.encode_text(c) for c in CONCEPTS])
    concept_vecs /= np.linalg.norm(concept_vecs, axis=1, keepdims=True) + 1e-9
    cluster_concepts: dict = {}
    for c in uniq:
        centroid = vectors[labels == c].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        sims = concept_vecs @ centroid
        top = np.argsort(sims)[::-1][:n_concepts_per_cluster]
        cluster_concepts[int(c)] = [(CONCEPTS[i], float(sims[i])) for i in top]

    return ClusterResult(
        labels=np.asarray(labels), n_clusters=len(uniq), noise_count=noise, sizes=sizes,
        cluster_concepts=cluster_concepts,
        params={"algorithm": "HDBSCAN", "metric": "euclidean (unit-norm -> cosine)",
                "min_cluster_size": min_cluster_size, "min_samples": min_samples,
                "cluster_selection_method": cluster_selection_method,
                "pca_dim": int(cluster_vectors.shape[1]) if cluster_vectors is not None else None,
                "n_concepts": len(CONCEPTS)},
        tile_ids=list(tile_ids))


# --------------------------------------------------------------------------
# sample-and-assign clustering for large corpora (PS 2.2.4 at scale)
# --------------------------------------------------------------------------
#
# Full HDBSCAN over ~100k raw 512-d embeddings does not finish in practical
# batch time (Phase 7b Tier 3: >60 min, killed - the MST / hierarchy-extraction
# step, which HDBSCAN does not parallelize, is the wall). The standard fix:
# cluster a representative SAMPLE, then assign every remaining tile to the
# nearest sample-cluster centroid. Runtime becomes O(sample HDBSCAN) +
# O(N x n_clusters) for the assignment (a single dense matmul on unit-norm
# vectors == cosine).


@dataclass
class SampleAssignResult:
    labels: np.ndarray                # (N,) label per tile in the FULL corpus; -1 = noise/abstained
    tile_ids: list                    # aligned with labels
    sample_idx: np.ndarray            # indices (into the full arrays) that were clustered directly
    sample_result: ClusterResult      # the HDBSCAN result on the sample
    centroids: np.ndarray             # (n_clusters, dim) unit-norm; row i == cluster id `centroid_ids[i]`
    centroid_ids: list                # cluster id for each centroid row
    assign_top_sim: np.ndarray        # (N,) cosine of each tile to its assigned centroid
    assign_margin: np.ndarray         # (N,) top1 - top2 cosine (assignment confidence)
    n_clusters: int
    sizes: dict                       # {cluster_id: size} over the FULL corpus
    noise_count: int                  # tiles with label -1 in the FULL corpus
    cluster_concepts: dict            # carried from sample_result
    params: dict

    def as_dict(self) -> dict:
        return {
            "method": "hdbscan-on-sample + nearest-centroid assignment",
            "n_tiles": int(len(self.labels)),
            "n_sample": int(len(self.sample_idx)),
            "n_clusters": self.n_clusters,
            "noise_count": self.noise_count,
            "sizes": self.sizes,
            "sample_sizes": self.sample_result.sizes,
            "sample_noise_count": self.sample_result.noise_count,
            "cluster_concepts": {str(k): [[c, round(float(s), 4)] for c, s in v]
                                 for k, v in self.cluster_concepts.items()},
            "assign_top_sim_pctl": {
                p: round(float(np.percentile(self.assign_top_sim, p)), 4)
                for p in (1, 5, 25, 50, 75)
            },
            "params": self.params,
        }


def stratified_sample_indices(groups: list[str], sample_size: int, *, seed: int = 0,
                              min_per_group: int = 200) -> np.ndarray:
    """Indices of a size-``sample_size`` sample, stratified by ``groups``.

    Proportional allocation, but every group with >= ``min_per_group`` members
    contributes at least ``min_per_group`` so small regions are not washed out;
    the remainder is filled proportionally. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    n_total = len(groups)
    sample_size = min(sample_size, n_total)

    alloc: dict[str, int] = {}
    for g in uniq:
        gsize = len(idx_by_group[g])
        alloc[g] = min(gsize, min_per_group) if gsize >= min_per_group else gsize
    base = sum(alloc.values())
    remaining = max(0, sample_size - base)
    # proportional split of the remainder over the still-available headroom
    headroom = {g: len(idx_by_group[g]) - alloc[g] for g in uniq}
    head_total = sum(headroom.values())
    if head_total > 0 and remaining > 0:
        for g in uniq:
            add = int(round(remaining * headroom[g] / head_total))
            alloc[g] = min(len(idx_by_group[g]), alloc[g] + add)

    picked: list[np.ndarray] = []
    for g in uniq:
        pool = idx_by_group[g]
        take = min(len(pool), alloc[g])
        picked.append(rng.choice(pool, size=take, replace=False))
    out = np.concatenate(picked) if picked else np.array([], dtype=int)
    out.sort()
    return out


def cluster_sample_and_assign(vectors: np.ndarray, tile_ids: list[str], groups: list[str],
                              embedding_model, *, sample_size: int = 20_000,
                              min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                              min_samples: int | None = None, core_dist_n_jobs: int = -1,
                              seed: int = 0, min_per_group: int = 200,
                              abstain_below_sim: float | None = None,
                              cluster_selection_method: str = "eom",
                              pca_dim: int | None = None) -> SampleAssignResult:
    """Cluster a stratified sample with HDBSCAN, assign the rest by nearest centroid.

    ``vectors`` must be unit-norm (they are, coming out of the embedding model /
    FAISS). ``groups`` is the stratification key per tile (here: region).
    ``abstain_below_sim`` (optional): tiles whose best centroid cosine is below
    this are labelled -1 instead of force-assigned. ``pca_dim`` (optional):
    fit PCA on the sample and run HDBSCAN in that reduced space (mitigates the
    curse of dimensionality at 512-d) - concept labeling and the final
    nearest-centroid assignment both still happen in the native unit-norm
    512-d space (PCA axes carry no meaning for RemoteCLIP's text tower, and
    centroids need to be comparable to the full corpus's own 512-d vectors).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n = len(vectors)
    sample_idx = stratified_sample_indices(groups, sample_size, seed=seed, min_per_group=min_per_group)
    s_vecs = vectors[sample_idx]
    s_ids = [tile_ids[i] for i in sample_idx]

    s_fit_vecs = None
    if pca_dim is not None:
        from sklearn.decomposition import PCA

        s_fit_vecs = PCA(n_components=pca_dim, random_state=seed).fit_transform(s_vecs)

    sample_result = cluster_embeddings(s_vecs, s_ids, embedding_model,
                                       min_cluster_size=min_cluster_size, min_samples=min_samples,
                                       core_dist_n_jobs=core_dist_n_jobs,
                                       cluster_selection_method=cluster_selection_method,
                                       cluster_vectors=s_fit_vecs)

    centroid_ids = sorted(sample_result.sizes)
    if not centroid_ids:
        raise RuntimeError("sample HDBSCAN produced no clusters - lower min_cluster_size or raise sample_size")
    cents = np.zeros((len(centroid_ids), vectors.shape[1]), dtype=np.float32)
    for i, cid in enumerate(centroid_ids):
        m = sample_result.labels == cid
        c = s_vecs[m].mean(axis=0)
        c /= np.linalg.norm(c) + 1e-9
        cents[i] = c

    # cosine of every corpus vector to every centroid (unit-norm -> dot product)
    sims = vectors @ cents.T                          # (N, n_clusters)
    order = np.argsort(-sims, axis=1)
    top1 = order[:, 0]
    top_sim = sims[np.arange(n), top1]
    second_sim = sims[np.arange(n), order[:, 1]] if len(centroid_ids) > 1 else np.zeros(n, np.float32)
    margin = top_sim - second_sim
    labels = np.array([centroid_ids[t] for t in top1], dtype=int)
    if abstain_below_sim is not None:
        labels[top_sim < abstain_below_sim] = -1

    sizes = {int(c): int((labels == c).sum()) for c in centroid_ids}
    noise = int((labels == -1).sum())
    params = {
        "algorithm": "HDBSCAN(sample) + nearest-centroid(rest)",
        "sample_size": int(len(sample_idx)), "corpus_size": int(n),
        "min_cluster_size": min_cluster_size, "min_samples": min_samples,
        "core_dist_n_jobs": core_dist_n_jobs, "seed": seed, "min_per_group": min_per_group,
        "abstain_below_sim": abstain_below_sim,
        "cluster_selection_method": cluster_selection_method, "pca_dim": pca_dim,
        "metric": "euclidean on unit-norm (== cosine); assignment by cosine to centroid",
    }
    return SampleAssignResult(
        labels=labels, tile_ids=list(tile_ids), sample_idx=sample_idx, sample_result=sample_result,
        centroids=cents, centroid_ids=centroid_ids, assign_top_sim=top_sim, assign_margin=margin,
        n_clusters=len(centroid_ids), sizes=sizes, noise_count=noise,
        cluster_concepts=sample_result.cluster_concepts, params=params)


# --------------------------------------------------------------------------
# geographic cluster map (one true lon/lat scatter, all regions at once) -
# this is what the app actually serves at /discovery/cluster-map.png.
# --------------------------------------------------------------------------


def disambiguate_cluster_labels(cluster_concepts: dict, region_purity: dict | None = None) -> dict:
    """When two clusters share the same top-1 concept (common once a mega-
    cluster is split into several still-similar-looking sub-clusters - they
    all score highest on the same one of only 15 fixed concept phrases), make
    the DISPLAYED label distinct by appending each cluster's dominant region
    (already computed for the region-purity report) - e.g. "bare dry open
    ground (kutch)" vs "bare dry open ground (jaisalmer)" - genuinely more
    informative than the bare concept string, not just a tie-breaker.
    Two clusters can still collide after the region suffix (e.g. two small
    open-water clusters both dominated by the same region) - a second pass
    appends the cluster id to any label still shared by more than one
    cluster, so the result is always fully distinct.
    Returns {cluster_id: display_label}; falls back to "<concept> (c<id>)"
    if no region info is available for a tied cluster."""
    top = {cid: concepts[0][0] for cid, concepts in cluster_concepts.items()}
    counts: dict[str, int] = {}
    for lbl in top.values():
        counts[lbl] = counts.get(lbl, 0) + 1
    out = {}
    for cid, lbl in top.items():
        if counts[lbl] <= 1:
            out[cid] = lbl
            continue
        region = None
        if region_purity is not None:
            info = region_purity.get(str(cid)) or region_purity.get(cid)
            if info:
                region = info.get("dominant_region")
        out[cid] = f"{lbl} ({region})" if region else f"{lbl} (c{cid})"

    final_counts: dict[str, int] = {}
    for lbl in out.values():
        final_counts[lbl] = final_counts.get(lbl, 0) + 1
    for cid, lbl in list(out.items()):
        if final_counts[lbl] > 1 and not lbl.endswith(f"(c{cid})"):
            out[cid] = f"{lbl} (c{cid})"
    return out


_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
]
_NOISE_RGB = (235, 235, 235)


def save_cluster_map(result: ClusterResult, tile_records: list, out_path, *, cell: int = 9) -> "object":
    """One grid panel per observation: cell (row, col) coloured by the tile's cluster.

    NOTE: this lays panels out side by side, so it does not scale past a
    handful of observations - at 77 (Phase 8) it produces a ~29000x500px
    strip that renders as an unreadable sliver at any normal display width.
    Kept for small/QA runs (``scripts/cluster_tiles.py``); the live app's
    Discovery view is served by :func:`save_geographic_cluster_map` instead,
    which plots every tile at its real lon/lat and scales with region count,
    not observation count."""
    from PIL import Image, ImageDraw, ImageFont

    by_obs: dict[str, list] = {}
    lab_by_tile = dict(zip(result.tile_ids, result.labels.tolist()))
    for rec in tile_records:
        by_obs.setdefault(rec.observation_id, []).append(rec)
    obs_ids = sorted(by_obs)

    def _rc(rec):
        # tile_id ends '..._rNNN_cNNN'
        r, c = rec.tile_id.rsplit("_r", 1)[1].split("_c")
        return int(r), int(c)

    grids = []
    for oid in obs_ids:
        recs = by_obs[oid]
        rcs = [_rc(r) for r in recs]
        maxr = max(r for r, _ in rcs) + 1
        maxc = max(c for _, c in rcs) + 1
        g = Image.new("RGB", (maxc * cell, maxr * cell), "white")
        px = g.load()
        for rec, (r, c) in zip(recs, rcs):
            lab = lab_by_tile.get(rec.tile_id, -1)
            col = _NOISE_RGB if lab < 0 else _PALETTE[lab % len(_PALETTE)]
            for dy in range(cell):
                for dx in range(cell):
                    px[c * cell + dx, r * cell + dy] = col
        grids.append((oid, g))

    pad, top, legrow = 10, 22, 20
    gw = max(g.width for _, g in grids)
    gh = max(g.height for _, g in grids)
    n_leg = result.n_clusters + 1
    H = top + gh + pad + legrow * ((n_leg + 2) // 3) + pad
    W = pad + len(grids) * (gw + pad)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.load_default(13)
    except TypeError:
        f = ImageFont.load_default()
    for i, (oid, g) in enumerate(grids):
        x = pad + i * (gw + pad)
        canvas.paste(g, (x, top))
        d.text((x, 4), oid.split("_")[2] if "_" in oid else oid, font=f, fill=(20, 20, 20))
    y0 = top + gh + pad
    for j in range(n_leg):
        cid = j if j < result.n_clusters else -1
        col = _NOISE_RGB if cid < 0 else _PALETTE[cid % len(_PALETTE)]
        cx = pad + (j % 3) * (W // 3)
        cy = y0 + (j // 3) * legrow
        d.rectangle([cx, cy, cx + 14, cy + 14], fill=col, outline=(120, 120, 120))
        if cid < 0:
            lbl = f"noise ({result.noise_count})"
        else:
            top1 = result.cluster_concepts[cid][0][0]
            lbl = f"c{cid} n={result.sizes[cid]}: {top1[:34]}"
        d.text((cx + 20, cy), lbl, font=f, fill=(20, 20, 20))
    out_path = str(out_path)
    canvas.save(out_path)
    return out_path


_MPL_PALETTE = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in _PALETTE]


def save_geographic_cluster_map(tile_ids: list[str], labels, lonlat: dict[str, tuple[float, float]],
                                cluster_concepts: dict, sizes: dict, out_path, *,
                                display_labels: dict | None = None, noise_count: int = 0,
                                figsize=(11, 9), dpi: int = 130) -> str:
    """A single true lon/lat scatter over every tile, coloured by cluster -
    legible at any corpus/observation count (unlike :func:`save_cluster_map`'s
    per-observation strip). One point per tile; regions that are geographically
    far apart (this project spans 9 AOIs across India) naturally separate into
    distinct clusters of points, with a legend naming each cluster id, its
    size/share, and its label (``display_labels`` - see
    :func:`disambiguate_cluster_labels` - or the bare top concept)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_arr = np.asarray(labels)
    total = len(labels_arr)
    lons = np.array([lonlat.get(t, (np.nan, np.nan))[0] for t in tile_ids])
    lats = np.array([lonlat.get(t, (np.nan, np.nan))[1] for t in tile_ids])
    valid = ~(np.isnan(lons) | np.isnan(lats))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    uniq = sorted(int(c) for c in sizes)
    noise_mask = valid & (labels_arr < 0)
    if noise_mask.any():
        ax.scatter(lons[noise_mask], lats[noise_mask], s=2, c="#d9dde3", label=None, linewidths=0)
    for cid in uniq:
        m = valid & (labels_arr == cid)
        if not m.any():
            continue
        lbl = (display_labels or {}).get(cid) or cluster_concepts.get(cid, [["?"]])[0][0]
        pct = 100 * sizes[cid] / total if total else 0.0
        ax.scatter(lons[m], lats[m], s=3, c=_MPL_PALETTE[cid % len(_MPL_PALETTE)], linewidths=0,
                  label=f"c{cid} n={sizes[cid]} ({pct:.1f}%): {lbl}")

    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Discovery clusters — {total:,} tiles, {len(uniq)} clusters"
                + (f", {noise_count:,} noise" if noise_count else ""))
    ax.set_aspect("equal", adjustable="datalim")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5,
                    markerscale=4, frameon=True, title="cluster (n, share of corpus): label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

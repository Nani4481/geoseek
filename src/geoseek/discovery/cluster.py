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
    n = vector_index.count()
    v = np.zeros((n, 512), dtype=np.float32)
    for i in range(n):
        v[i] = vector_index.get_vector(i)
    return v


def cluster_embeddings(vectors: np.ndarray, tile_ids: list[str], embedding_model, *,
                       min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                       min_samples: int | None = None) -> ClusterResult:
    from hdbscan import HDBSCAN

    clf = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                  metric="euclidean", core_dist_n_jobs=1)
    labels = clf.fit_predict(vectors.astype(np.float64))
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
        top = np.argsort(sims)[::-1][:3]
        cluster_concepts[int(c)] = [(CONCEPTS[i], float(sims[i])) for i in top]

    return ClusterResult(
        labels=np.asarray(labels), n_clusters=len(uniq), noise_count=noise, sizes=sizes,
        cluster_concepts=cluster_concepts,
        params={"algorithm": "HDBSCAN", "metric": "euclidean (unit-norm -> cosine)",
                "min_cluster_size": min_cluster_size, "min_samples": min_samples,
                "n_concepts": len(CONCEPTS)},
        tile_ids=list(tile_ids))


# --------------------------------------------------------------------------
# cluster map (spatial, one panel per observation)
# --------------------------------------------------------------------------

_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
]
_NOISE_RGB = (235, 235, 235)


def save_cluster_map(result: ClusterResult, tile_records: list, out_path, *, cell: int = 9) -> "object":
    """One grid panel per observation: cell (row, col) coloured by the tile's cluster."""
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

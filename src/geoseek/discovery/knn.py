"""KNN "find more like this" (PS 2.2.4) - a thin, latency-reporting wrapper.

The heavy lifting is :meth:`geoseek.search.engine.SearchEngine.search_image`
(RemoteCLIP vector already in the FAISS index -> brute-force IP scan -> ranked,
filtered tile records). This just packages the call for the discovery workflow
and surfaces the interactive latency.
"""

from __future__ import annotations

import time


def find_more_like_this(engine, *, tile_id: str | None = None, lon: float | None = None,
                        lat: float | None = None, k: int = 8, filters=None) -> dict:
    """Similar sites to a seed tile (by ``tile_id``) or a point (``lon``/``lat``).

    Returns ``{seed_tile_id, results:[{tile_id, score, obs, centroid}], latency_ms}``.
    """
    if tile_id is None:
        if lon is None or lat is None:
            raise ValueError("find_more_like_this needs tile_id or lon+lat")
        eps = 0.002
        t0 = time.time()
        recs = engine.repo.query_tiles(bbox=(lon - eps, lat - eps, lon + eps, lat + eps))
        embedded = [r for r in recs if r.faiss_id is not None]
        if not embedded:
            raise KeyError(f"no embedded tile at ({lon:.4f}, {lat:.4f})")
        tile_id = embedded[0].tile_id
        lookup_ms = (time.time() - t0) * 1000.0
    else:
        lookup_ms = 0.0

    results, latency_ms = engine.search_image(tile_id=tile_id, k=k + 1, filters=filters)
    results = [r for r in results if r.tile_id != tile_id][:k]     # drop the self-match
    return {
        "seed_tile_id": tile_id,
        "k": k,
        "results": [{"tile_id": r.tile_id, "score": round(float(r.score), 4),
                     "observation_id": r.scene_id, "acq_date": r.acq_date,
                     "centroid_lonlat": [round(r.lon, 6), round(r.lat, 6)]}
                    for r in results],
        "latency_ms": round(lookup_ms + latency_ms, 2),
        "point_lookup_ms": round(lookup_ms, 2),
    }

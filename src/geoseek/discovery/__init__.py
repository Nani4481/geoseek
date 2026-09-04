"""Phase 5 Step E: discovery (PS 2.2.4).

  * ``knn``     - "find more like this": nearest neighbours of a tile / candidate
    from the FAISS index. Interactive, fast (latency reported).
  * ``cluster`` - offline HDBSCAN over all tile embeddings, run as a batch job
    (``scripts/cluster_tiles.py``), not per query. Each cluster is labelled with
    its nearest RemoteCLIP text concepts; a cluster map is saved.
"""

from geoseek.discovery.knn import find_more_like_this

__all__ = ["find_more_like_this"]

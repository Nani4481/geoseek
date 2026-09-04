"""Phase 5 Step D: one ranked analyst queue.

``ranker.FusionRanker`` fuses the independent evidence axes - change
confidence, significance (area x index-anomaly), and (when a text query is
active) RemoteCLIP semantic relevance - into a single ordering, so an analyst
can ask "new construction near a river" and get change candidates ranked by
BOTH what changed and how well it matches the intent.
"""

from geoseek.fusion.ranker import FUSION_WEIGHTS, FusionRanker, fusion_score

__all__ = ["FusionRanker", "fusion_score", "FUSION_WEIGHTS"]

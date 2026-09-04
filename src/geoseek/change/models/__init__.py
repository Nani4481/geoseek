"""Phase 3b: a TRAINED change-detection model behind the ChangeDetectionModel seam.

``fc_siam_diff``        - :class:`FCSiamDiff`, the Siamese fully-convolutional
                         encoder/decoder (shared weights, feature differencing at
                         every skip level, U-Net decoder, per-pixel change logits).
``fc_siam_diff_model``  - :class:`FCSiamDiffChangeModel`, which wraps a trained
                         :class:`FCSiamDiff` checkpoint as a
                         :class:`geoseek.models.base.ChangeDetectionModel` and
                         consumes a :class:`geoseek.temporal.contract.ObservationPair`
                         with zero reshaping.
"""

from geoseek.change.models.fc_siam_diff import FCSiamDiff, count_parameters

__all__ = ["FCSiamDiff", "count_parameters", "FCSiamDiffChangeModel"]


def __getattr__(name: str):  # lazy: keep torch-only model import cheap, defer the raster/catalog deps
    if name == "FCSiamDiffChangeModel":
        from geoseek.change.models.fc_siam_diff_model import FCSiamDiffChangeModel

        return FCSiamDiffChangeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

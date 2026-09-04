"""SCL-based :class:`QualityEstimator`.

Wraps :func:`geoseek.ingest.quality.cloud_fraction` (Sentinel-2 Scene
Classification band) and adds a per-class fraction breakdown as flags.
"""

from __future__ import annotations

import numpy as np

from geoseek.ingest.quality import BAD_SCL_CLASSES, SCL_CLASS_NAMES, cloud_fraction
from geoseek.models.base import QualityEstimator, QualityReport

SCL_WATER = 6


class SclQualityEstimator(QualityEstimator):
    """Usability from the SCL band: cloud/shadow/snow/saturated/no-data -> unusable."""

    def estimate(self, bands: dict[str, np.ndarray]) -> QualityReport:
        if "SCL" not in bands:
            raise ValueError("SclQualityEstimator needs an 'SCL' band")
        scl = np.asarray(bands["SCL"])
        cf = cloud_fraction(scl)
        total = max(scl.size, 1)
        by_class = {
            SCL_CLASS_NAMES.get(int(c), str(int(c))): round(float((scl == c).sum()) / total, 6)
            for c in np.unique(scl)
        }
        flags = {
            "class_fractions": by_class,
            "bad_class_fraction": round(cf, 6),
            "water_fraction": round(float(np.isin(scl, [SCL_WATER]).sum()) / total, 6),
            "bad_classes": sorted(BAD_SCL_CLASSES),
        }
        return QualityReport(cloud_fraction=cf, usable_fraction=round(1.0 - cf, 6), flags=flags)

"""Per-tile cloud/quality scoring from the Sentinel-2 Scene Classification (SCL) band.

SCL class codes (ESA S2 L2A product spec):
    0  no_data
    1  saturated_or_defective
    2  dark_area_pixels
    3  cloud_shadows
    4  vegetation
    5  not_vegetated
    6  water
    7  unclassified
    8  cloud_medium_probability
    9  cloud_high_probability
    10 thin_cirrus
    11 snow
"""

from __future__ import annotations

import numpy as np

SCL_CLASS_NAMES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels",
    3: "cloud_shadows",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow",
}

# "bad" per spec: cloud, cloud-shadow, snow, saturated -> bad. no_data pixels
# carry no valid classification either, so they count as bad too.
BAD_SCL_CLASSES = frozenset({0, 1, 3, 8, 9, 10, 11})


def cloud_fraction(scl_tile: np.ndarray) -> float:
    """Fraction (0..1) of pixels in a bad SCL class (cloud/shadow/snow/saturated/no-data)."""
    if scl_tile.size == 0:
        return 1.0
    bad = np.isin(scl_tile, list(BAD_SCL_CLASSES))
    return float(bad.sum()) / float(scl_tile.size)

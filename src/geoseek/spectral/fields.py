"""Field definitions of the per-tile spectral descriptor - pure constants, shared by the catalog schema and the
descriptor computation so the two cannot drift apart.

Version 1. Every field is computed on the tile's VALID pixels (SCL classes 2,4,5,6,7,11 with finite indices; see
``geoseek.spectral.descriptor``). The set is a superset of what the Phase 7a retrieval judge reads
(``scripts/eval_retrieval_features.py``), plus std/mean and band-ratio moments.
"""

from __future__ import annotations

DESCRIPTOR_VERSION = 1

INDICES = ("ndvi", "ndwi", "ndbi")
INDEX_STATS = ("mean", "std", "p10", "p50", "p90")
RATIOS = ("sr_nir_red", "sr_swir_nir", "sr_green_red")        # B08/B04, B11/B08, B03/B04
RATIO_STATS = ("mean", "std")
FRACTIONS = ("water_frac", "veg_frac", "dense_veg_frac", "bare_frac", "built_frac", "edge_density")

# computed from the tile's own pixels
SPECTRAL_FIELDS: tuple[str, ...] = (
    *(f"{i}_{s}" for i in INDICES for s in INDEX_STATS),
    *FRACTIONS,
    *(f"{r}_{s}" for r in RATIOS for s in RATIO_STATS),
)
# derived from the region's water mask (needs every date of the region); NULL until that step has run
CONTEXT_FIELDS: tuple[str, ...] = ("dist_river_m", "dist_water_m")
HEADER_FIELDS: tuple[str, ...] = ("descriptor_version", "usable", "valid_frac", "n_valid")
ALL_FIELDS: tuple[str, ...] = HEADER_FIELDS + SPECTRAL_FIELDS + CONTEXT_FIELDS

# columns that get a B-tree index (range predicates used by the constrained-retrieval query builder)
INDEXED_FIELDS: tuple[str, ...] = ("usable", "ndvi_mean", "ndwi_mean", "ndbi_mean", "water_frac", "veg_frac",
                                   "bare_frac", "built_frac", "edge_density")

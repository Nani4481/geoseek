"""The Phase 7a relevance judge, applied to EVERY tile of a corpus (full judge coverage).

The 16 graders are the Phase 7a ones, loaded unchanged from ``scripts/eval_retrieval_judge.py`` (one source of
truth - there is no copy to drift): each is a pure function of per-tile spectral features and returns grade
0 / 1 / 2. Phase 7a could only judge the pooled Ayodhya tiles because only Ayodhya had NIR/SWIR; with the per-tile
spectral descriptor (``geoseek.spectral``) every Sentinel-2 tile can be graded.

Independence: the judge reads NDVI / NDWI / NDBI / SCL statistics and a water-mask distance - bands and products the
RGB-only retrieval models never receive. No embedding is used to assign any grade.

Two facts about generalizing it, stated rather than hidden:
  * the thresholds were calibrated on Ayodhya (cropland-dominated). Outside it they are applied as-is; whether they
    still mean what they meant is tested against the blind visual annotations (``scripts/eval_judge_coverage.py``);
  * the river-relative graders use ``dist_river_m`` = distance to the largest water component of the tile's REGION
    (``geoseek.spectral.context``) - a real river in Ayodhya, merely "the biggest water body" elsewhere. Results are
    therefore reported with and without the river-relative queries.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from geoseek.config import PROJECT_ROOT

JUDGE_SCRIPT = PROJECT_ROOT / "scripts" / "eval_retrieval_judge.py"
_CACHE: dict = {}


def _phase7a():
    if "mod" not in _CACHE:
        spec = importlib.util.spec_from_file_location("phase7a_judge", JUDGE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["mod"] = mod
    return _CACHE["mod"]


def graders() -> dict[str, Callable[[Mapping], tuple[int, str]]]:
    return dict(_phase7a().GRADERS)


def low_confidence_queries() -> frozenset[str]:
    return frozenset(_phase7a().LOW_CONFIDENCE_QUERIES)


def river_relative_queries() -> frozenset[str]:
    """Queries whose grader reads ``dist_river_m`` (identified from the grader functions, not a hand-kept list)."""
    m = _phase7a()
    river_fns = {m._g_river_sandbars, m._g_bridge, m._g_riverside_built, m._g_riverbank_veg}
    return frozenset(q for q, g in m.GRADERS.items() if g in river_fns)


# features the Phase 7a graders read
FEATURE_KEYS = ("water_frac", "dist_water_m", "dist_river_m", "bare_frac", "built_frac", "edge_density", "ndvi_p50",
                "ndvi_p10", "ndwi_p90", "dense_veg_frac")


def usable(f: Mapping | None) -> bool:
    return bool(f) and bool(f.get("usable"))


def grade_tile(f: Mapping | None, query: str) -> int:
    """Grade of one tile for one query; a tile with no features / not usable is 0 (as in Phase 7a)."""
    if not usable(f):
        return 0
    return int(graders()[query](f)[0])


def check_context_present(features: Sequence[Mapping | None], queries: Sequence[str]) -> None:
    """Refuse to grade river-relative queries on tiles whose river distance was never computed (silent zeros)."""
    if not (set(queries) & river_relative_queries()):
        return
    missing = sum(1 for f in features if usable(f) and (f.get("dist_river_m") is None or f.get("dist_water_m") is None))
    if missing:
        raise ValueError(f"{missing} usable tiles have no dist_river_m/dist_water_m - run the region context step first")


def grade_matrix(features: Sequence[Mapping | None], queries: Sequence[str]) -> np.ndarray:
    """int8 (n_tiles, n_queries) of grades 0/1/2 for every tile and query."""
    check_context_present(features, queries)
    fns = [graders()[q] for q in queries]
    out = np.zeros((len(features), len(queries)), dtype=np.int8)
    for i, f in enumerate(features):
        if not usable(f):
            continue
        for j, g in enumerate(fns):
            out[i, j] = g(f)[0]
    return out

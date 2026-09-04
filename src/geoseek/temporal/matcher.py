"""TemporalObservationMatcher: which observations of a location are comparable.

Given a point / bbox / AOI polygon, it pulls every observation there from the
catalog, orders them in time, and for each consecutive step reports a
:class:`geoseek.temporal.contract.PairComparability` verdict - ``comparable:
yes/no`` plus a per-criterion breakdown:

    spatial_overlap          footprint overlap fraction     (BLOCKING)
    temporal_separation      days between acquisitions
    sensor_compatibility     same sensor family
    collection_compatibility same source collection         (BLOCKING)
    resolution_compatibility same native GSD / tile grid    (BLOCKING)
    quality                  cloud / usable pixels both dates
    coregistration           registered? residual px?

The output :class:`ObservationSequence` is the exact object a Phase 3b
``ChangeDetectionModel.predict_change`` consumes - see
:mod:`geoseek.models.base`.

    python -m geoseek.temporal.matcher --lon 82.1998 --lat 26.7922
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shapely import wkt as shapely_wkt

from geoseek.catalog.entities import Observation
from geoseek.catalog.repository import MetadataRepository
from geoseek.temporal.contract import (
    ComparabilityCriterion,
    ObservationPair,
    ObservationSequence,
    PairComparability,
)

# thresholds
MIN_SPATIAL_OVERLAP = 0.30        # blocking below this
MIN_TEMPORAL_SEPARATION_DAYS = 5  # warn below (too close for meaningful change)
MAX_CLOUD_MEAN = 0.20
MAX_CLOUD_MAX = 0.60
COREG_RESIDUAL_PX_MAX = 0.5


class TemporalObservationMatcher:
    def __init__(self, repo: MetadataRepository):
        self.repo = repo
        self._collection_gsd: dict[str, float] = {}

    # -- helpers ------------------------------------------------------------

    def _gsd_for_observation(self, obs: Observation) -> float | None:
        scene = self.repo.get_scene(obs.scene_id)
        if scene is None:
            return None
        cid = scene.collection_id
        if cid not in self._collection_gsd:
            c = self.repo.get_collection(cid)
            self._collection_gsd[cid] = c.native_gsd_m if c else None
        return self._collection_gsd[cid]

    def _collection_for(self, obs: Observation) -> str | None:
        scene = self.repo.get_scene(obs.scene_id)
        return scene.collection_id if scene else None

    def _sensor_for(self, obs: Observation) -> str | None:
        cid = self._collection_for(obs)
        c = self.repo.get_collection(cid) if cid else None
        return c.sensor if c else None

    # -- per-criterion assessments --------------------------------------------

    @staticmethod
    def _spatial_overlap(a: Observation, b: Observation) -> ComparabilityCriterion:
        ga, gb = shapely_wkt.loads(a.footprint_wkt_4326), shapely_wkt.loads(b.footprint_wkt_4326)
        inter = ga.intersection(gb).area
        denom = min(ga.area, gb.area) or 1.0
        frac = inter / denom
        return ComparabilityCriterion(
            name="spatial_overlap", passed=frac >= MIN_SPATIAL_OVERLAP, blocking=True,
            value=round(frac, 4),
            detail=(f"{frac * 100:.1f}% of the smaller footprint overlaps"
                    + ("" if frac >= MIN_SPATIAL_OVERLAP
                       else f" (< {MIN_SPATIAL_OVERLAP * 100:.0f}% required)")),
        )

    @staticmethod
    def _temporal_separation(a: Observation, b: Observation) -> ComparabilityCriterion:
        from geoseek.temporal.contract import _days_between

        days = _days_between(a.acquired_at, b.acquired_at)
        too_close = days < MIN_TEMPORAL_SEPARATION_DAYS
        month_a, month_b = a.acquired_at[5:7], b.acquired_at[5:7]
        season = "same-season (phenology controlled)" if month_a == month_b else \
                 f"different months ({month_a} vs {month_b}) - expect phenology signal"
        return ComparabilityCriterion(
            name="temporal_separation", passed=not too_close, blocking=False, value=days,
            detail=f"{days} days apart ({days / 365.25:.1f} yr); {season}"
                   + ("  [WARNING: < 5 days, likely no real change]" if too_close else ""),
        )

    def _sensor_compat(self, a: Observation, b: Observation) -> ComparabilityCriterion:
        sa, sb = self._sensor_for(a), self._sensor_for(b)
        same = sa is not None and sa == sb
        return ComparabilityCriterion(
            name="sensor_compatibility", passed=same, blocking=False, value=f"{sa} / {sb}",
            detail=(f"both {sa}" if same else f"cross-sensor ({sa} vs {sb}) - harmonise carefully"),
        )

    def _collection_compat(self, a: Observation, b: Observation) -> ComparabilityCriterion:
        ca, cb = self._collection_for(a), self._collection_for(b)
        same = ca is not None and ca == cb
        return ComparabilityCriterion(
            name="collection_compatibility", passed=same, blocking=True, value=f"{ca} / {cb}",
            detail=(f"both {ca}" if same else f"different collections ({ca} vs {cb})"),
        )

    def _resolution_compat(self, a: Observation, b: Observation) -> ComparabilityCriterion:
        ga, gb = self._gsd_for_observation(a), self._gsd_for_observation(b)
        na = a.quality_summary.get("n_tiles")
        nb = b.quality_summary.get("n_tiles")
        gsd_same = ga is not None and ga == gb
        grid_same = na is not None and na == nb
        passed = gsd_same  # GSD mismatch is blocking; a tile-count difference is only a warning
        detail = f"native GSD {ga} m vs {gb} m"
        if gsd_same:
            detail = f"both {ga} m native GSD"
        detail += f"; {na} vs {nb} tiles" + ("" if grid_same else " (grids differ - clip mismatch)")
        return ComparabilityCriterion(
            name="resolution_compatibility", passed=passed, blocking=True,
            value=f"{ga}m/{gb}m", detail=detail,
        )

    @staticmethod
    def _quality(a: Observation, b: Observation) -> ComparabilityCriterion:
        def _q(o: Observation) -> tuple[float | None, float | None, int | None]:
            qs = o.quality_summary or {}
            return qs.get("cloud_fraction_mean"), qs.get("cloud_fraction_max"), qs.get("n_clear_tiles_cf_le_0p05")

        ma, xa, ca = _q(a)
        mb, xb, cb = _q(b)
        ok = all(v is not None for v in (ma, mb)) and ma <= MAX_CLOUD_MEAN and mb <= MAX_CLOUD_MEAN \
            and (xa is None or xa <= MAX_CLOUD_MAX) and (xb is None or xb <= MAX_CLOUD_MAX)
        return ComparabilityCriterion(
            name="quality", passed=bool(ok), blocking=False,
            value=round(max(ma or 0.0, mb or 0.0), 5),
            detail=(f"cloud mean {ma:.4f}/{mb:.4f}, max {xa}/{xb}, clear tiles {ca}/{cb}"
                    if ma is not None and mb is not None else "quality summary missing for a date"),
        )

    @staticmethod
    def _coregistration(a: Observation, b: Observation) -> ComparabilityCriterion:
        """a = earlier, b = later. Comparable co-registration if one is registered to the
        other, or both are registered to the same common reference."""
        ca, cb = a.coregistration or {}, b.coregistration or {}

        def _residual(c: dict) -> float | None:
            if "median_magnitude_px" in c:
                return float(c["median_magnitude_px"])
            if "residual_magnitude_px" in c:
                return float(c["residual_magnitude_px"])
            return None

        # direct: earlier registered onto later (or vice versa)
        for src, dst, other_id in ((ca, cb, b.observation_id), (cb, ca, a.observation_id)):
            if src.get("reference_scene") == other_id or src.get("reference_observation") == other_id:
                r = _residual(src)
                applied = src.get("correction_applied")
                passed = r is not None and r <= COREG_RESIDUAL_PX_MAX
                return ComparabilityCriterion(
                    name="coregistration", passed=passed or bool(applied), blocking=False,
                    value=f"residual {r} px" if r is not None else "registered",
                    detail=(f"registered pair: residual {r} px "
                            f"({'correction applied' if applied else 'within sub-pixel spec'})"),
                )

        # transitive: both registered to the same reference
        ra, rb = ca.get("reference_scene"), cb.get("reference_scene")
        if ra and ra == rb:
            res_a, res_b = _residual(ca) or 0.0, _residual(cb) or 0.0
            combined = res_a + res_b
            return ComparabilityCriterion(
                name="coregistration", passed=combined <= COREG_RESIDUAL_PX_MAX, blocking=False,
                value=f"~{combined:.2f} px via {ra}",
                detail=f"both co-registered to {ra} (residuals {res_a:.2f} + {res_b:.2f} px)",
            )
        return ComparabilityCriterion(
            name="coregistration", passed=False, blocking=False, value="unknown",
            detail="co-registration status unknown for this pair - measure before change detection",
        )

    def _assess(self, earlier: Observation, later: Observation) -> PairComparability:
        criteria = [
            self._spatial_overlap(earlier, later),
            self._temporal_separation(earlier, later),
            self._sensor_compat(earlier, later),
            self._collection_compat(earlier, later),
            self._resolution_compat(earlier, later),
            self._quality(earlier, later),
            self._coregistration(earlier, later),
        ]
        comparable = all(not (c.blocking and not c.passed) for c in criteria)
        return PairComparability(comparable=comparable, criteria=criteria)

    # -- entry point --------------------------------------------------------

    def match(
        self,
        *,
        location: tuple[float, float] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        aoi_wkt: str | None = None,
        all_pairs: bool = False,
        collection: str | None = None,
    ) -> ObservationSequence:
        """``collection`` restricts the sequence to one source collection - change
        detection never mixes collections (that would fail ``collection_compatibility``
        anyway), so a caller working on Sentinel-2 passes ``collection='sentinel-2-l2a'``
        to keep a co-located Sentinel-1 (or other) stack out of the sequence."""
        if location is None and bbox is None and aoi_wkt is None:
            raise ValueError("match() needs one of location, bbox or aoi_wkt")

        obs = self.repo.list_observations(location=location, bbox=bbox, collection=collection)
        if aoi_wkt is not None:
            poly = shapely_wkt.loads(aoi_wkt)
            obs = [o for o in obs if shapely_wkt.loads(o.footprint_wkt_4326).intersects(poly)]
        obs.sort(key=lambda o: o.acquired_at)

        notes: list[str] = []
        if location is not None and aoi_wkt is None and bbox is None:
            notes.append(f"query point ({location[0]:.5f}, {location[1]:.5f})")
        if len(obs) < 2:
            notes.append(f"only {len(obs)} observation(s) at this location - no pair to compare")
            return ObservationSequence(observations=obs, pairs=[], location=location,
                                       aoi_wkt=aoi_wkt, notes=notes)

        index_pairs = (
            [(i, j) for i in range(len(obs)) for j in range(i + 1, len(obs))]
            if all_pairs else [(i, i + 1) for i in range(len(obs) - 1)]
        )
        pairs = [ObservationPair(obs[i], obs[j], self._assess(obs[i], obs[j])) for i, j in index_pairs]

        n_ok = sum(p.comparable for p in pairs)
        notes.append(f"{n_ok}/{len(pairs)} pair(s) comparable")
        return ObservationSequence(observations=obs, pairs=pairs, location=location,
                                   aoi_wkt=aoi_wkt, notes=notes)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_repo():
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
    from geoseek.config import get_settings

    return SQLiteMetadataRepository(get_settings().index_dir / "tiles.sqlite")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="geoseek-temporal-matcher")
    p.add_argument("--lon", type=float, help="query longitude (EPSG:4326)")
    p.add_argument("--lat", type=float, help="query latitude (EPSG:4326)")
    p.add_argument("--bbox", type=str, help="west,south,east,north")
    p.add_argument("--all-pairs", action="store_true", help="assess every pair, not just consecutive")
    p.add_argument("--json", action="store_true", help="emit JSON instead of the text report")
    args = p.parse_args(argv)

    location = (args.lon, args.lat) if args.lon is not None and args.lat is not None else None
    bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
    if location is None and bbox is None:
        p.error("provide --lon/--lat or --bbox")

    repo = _build_repo()
    try:
        seq = TemporalObservationMatcher(repo).match(location=location, bbox=bbox, all_pairs=args.all_pairs)
    finally:
        repo.close()

    if args.json:
        print(json.dumps(seq.to_dict(), indent=2))
    else:
        print(seq.format_report())
    raise SystemExit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[temporal.matcher] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

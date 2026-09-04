"""The temporal-matching contract.

``TemporalObservationMatcher`` produces an :class:`ObservationSequence` for a
location: the time-ordered observations there, plus an :class:`ObservationPair`
for each consecutive step carrying a :class:`PairComparability` verdict
(``comparable: yes/no`` + reasons). A Phase 3b ``ChangeDetectionModel`` takes an
``ObservationPair`` straight from this module - nothing is reshaped at the
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geoseek.catalog.entities import Observation

# criteria a change-detection model must not run across if they fail
BLOCKING_CRITERIA = ("spatial_overlap", "collection_compatibility", "resolution_compatibility")


@dataclass(frozen=True)
class ComparabilityCriterion:
    """One assessed dimension of whether two observations can be compared."""

    name: str                       # "spatial_overlap" | "temporal_separation" | "sensor_compatibility"
    passed: bool                    # ...  | "resolution_compatibility" | "quality" | "coregistration"
    detail: str
    value: float | str | None = None
    blocking: bool = False          # a failed blocking criterion makes the pair not comparable

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "blocking": self.blocking,
                "value": self.value, "detail": self.detail}


@dataclass(frozen=True)
class PairComparability:
    """Verdict for one ordered observation pair."""

    comparable: bool
    criteria: list[ComparabilityCriterion] = field(default_factory=list)

    def _get(self, name: str) -> ComparabilityCriterion | None:
        return next((c for c in self.criteria if c.name == name), None)

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.criteria]

    @property
    def blocking_reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.criteria if c.blocking and not c.passed]

    @property
    def spatial_overlap_fraction(self) -> float | None:
        c = self._get("spatial_overlap")
        return float(c.value) if c and isinstance(c.value, (int, float)) else None

    @property
    def temporal_separation_days(self) -> int | None:
        c = self._get("temporal_separation")
        return int(c.value) if c and isinstance(c.value, (int, float)) else None

    @property
    def sensor_compatible(self) -> bool | None:
        c = self._get("sensor_compatibility")
        return c.passed if c else None

    @property
    def collection_compatible(self) -> bool | None:
        c = self._get("collection_compatibility")
        return c.passed if c else None

    @property
    def resolution_compatible(self) -> bool | None:
        c = self._get("resolution_compatibility")
        return c.passed if c else None

    @property
    def quality_ok(self) -> bool | None:
        c = self._get("quality")
        return c.passed if c else None

    @property
    def coregistration_status(self) -> str | None:
        c = self._get("coregistration")
        return str(c.value) if c else None

    def to_dict(self) -> dict:
        return {
            "comparable": self.comparable,
            "reasons": self.reasons,
            "blocking_reasons": self.blocking_reasons,
            "criteria": [c.as_dict() for c in self.criteria],
        }


@dataclass(frozen=True)
class ObservationPair:
    """An ordered pair of observations of the same location, with a comparability verdict.

    ``earlier`` / ``later`` are :class:`geoseek.catalog.entities.Observation`
    objects, so a change-detection model has the full provenance, the AOI
    footprint, and the radiometry / co-registration params recorded on each.
    """

    earlier: Observation
    later: Observation
    comparability: PairComparability

    @property
    def days_apart(self) -> int:
        return _days_between(self.earlier.acquired_at, self.later.acquired_at)

    @property
    def comparable(self) -> bool:
        return self.comparability.comparable

    def to_dict(self) -> dict:
        return {
            "earlier": self.earlier.observation_id,
            "later": self.later.observation_id,
            "earlier_date": self.earlier.acquired_at,
            "later_date": self.later.acquired_at,
            "days_apart": self.days_apart,
            "comparability": self.comparability.to_dict(),
        }


@dataclass(frozen=True)
class ObservationSequence:
    """The matcher's output for one location: ordered observations + consecutive pairs."""

    observations: list[Observation]
    pairs: list[ObservationPair]
    location: tuple[float, float] | None = None
    aoi_wkt: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def comparable_pairs(self) -> list[ObservationPair]:
        return [p for p in self.pairs if p.comparable]

    def to_dict(self) -> dict:
        return {
            "location": list(self.location) if self.location else None,
            "aoi_wkt": self.aoi_wkt,
            "observations": [
                {"observation_id": o.observation_id, "scene_id": o.scene_id,
                 "acquired_at": o.acquired_at, "aoi_name": o.aoi_name}
                for o in self.observations
            ],
            "pairs": [p.to_dict() for p in self.pairs],
            "notes": self.notes,
        }

    def format_report(self) -> str:
        lines: list[str] = []
        loc = f"({self.location[0]:.5f}, {self.location[1]:.5f})" if self.location else self.aoi_wkt
        lines.append(f"observation sequence for {loc}  -  {len(self.observations)} observations, "
                     f"{len(self.pairs)} pair(s) assessed")
        for i, o in enumerate(self.observations):
            lines.append(f"  [{i}] {o.acquired_at}  {o.observation_id}  (scene {o.scene_id})")
        for p in self.pairs:
            verdict = "COMPARABLE" if p.comparable else "NOT COMPARABLE"
            lines.append(f"\n  {p.earlier.acquired_at} -> {p.later.acquired_at}  [{verdict}]  "
                         f"({p.days_apart} days apart)")
            for c in p.comparability.criteria:
                mark = "ok  " if c.passed else ("BLOCK" if c.blocking else "warn ")
                lines.append(f"      {mark} {c.name:24s} {c.detail}")
        if self.notes:
            lines.append("\n  notes: " + "; ".join(self.notes))
        return "\n".join(lines)


def _days_between(a: str, b: str) -> int:
    from datetime import date

    def _d(s: str) -> date:
        return date.fromisoformat(s[:10])

    return abs((_d(b) - _d(a)).days)

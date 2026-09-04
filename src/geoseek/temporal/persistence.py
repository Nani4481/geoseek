"""Phase 4 Step C - temporal persistence + earliest-supported change (PS 2.2.2).

Uses all three Ayodhya observations (2019-03-30, 2021-03-04, 2024-03-08) via
:class:`geoseek.temporal.matcher.TemporalObservationMatcher`. For a location it
builds the change trajectory across the consecutive pairs, decides whether a
change is *persistent* (appears and stays) or *transient* (appears then
reverts - a likely false alarm or a temporary phenomenon), and reports the
earliest observation at which the change is supported by usable imagery -
never claiming a date earlier than the earliest usable observation.

The change status of a pair at a location is supplied by a caller-provided
lookup (the Phase 4 orchestrator samples its per-pair probability rasters), so
this module has no raster / torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from geoseek.temporal.contract import ObservationSequence
from geoseek.temporal.matcher import TemporalObservationMatcher

# probability at/above which a pair is considered "changed" at a location.
# Defaults to the trained model's frozen precision-favouring operating point.
DEFAULT_CHANGE_THRESHOLD = 0.80

PERSISTENCE_CONFIDENCE = {
    "persistent": 0.95,    # changed in an early pair, present in the span pair, stable since
    "progressive": 0.85,   # changing across every pair - ongoing works
    "recent": 0.75,        # only the latest pair changed (one supporting "after" observation)
    "transient": 0.20,     # appeared then reverted - likely a false alarm / temporary phenomenon
    "inconsistent": 0.15,  # consecutive pairs say "changed" but the span pair does not (contradiction)
    "single_pair": 0.55,   # only two observations available - cannot assess persistence
    "none": 0.0,
}

# a temporally unsupported candidate has its final confidence multiplied by this,
# on top of its (already low) persistence term - so a transient detection cannot
# score "medium" no matter how strong the model / spectral / quality evidence is.
PERSISTENCE_PENALTY = {"transient": 0.5, "inconsistent": 0.45, "single_pair": 0.85}

# ChangeLookup: (earlier_observation_id, later_observation_id, lon, lat) -> (changed, probability)
ChangeLookup = Callable[[str, str, float, float], "tuple[bool, float]"]


@dataclass
class PairChange:
    earlier_obs: str
    later_obs: str
    earlier_date: str
    later_date: str
    comparable: bool
    changed: bool
    probability: float

    def as_dict(self) -> dict:
        return {"earlier": self.earlier_obs, "later": self.later_obs,
                "window": [self.earlier_date, self.later_date], "comparable": self.comparable,
                "changed": self.changed, "probability": round(self.probability, 4)}


@dataclass
class ChangeTrajectory:
    location: tuple[float, float] | None
    consecutive: list[PairChange]         # time-ordered consecutive pairs
    span: PairChange | None               # first observation vs last observation
    persistence: str
    persistence_confidence: float
    earliest_supported: dict
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "location": list(self.location) if self.location else None,
            "consecutive_pairs": [p.as_dict() for p in self.consecutive],
            "span_pair": self.span.as_dict() if self.span else None,
            "persistence": self.persistence,
            "persistence_confidence": round(self.persistence_confidence, 3),
            "earliest_supported_change": self.earliest_supported,
            "notes": self.notes,
        }

    def format_report(self) -> str:
        loc = f"({self.location[0]:.5f}, {self.location[1]:.5f})" if self.location else "?"
        lines = [f"change trajectory @ {loc}",
                 f"  persistence: {self.persistence.upper()} "
                 f"(confidence {self.persistence_confidence:.2f})"]
        for p in self.consecutive:
            mark = "CHANGED" if p.changed else "stable "
            lines.append(f"    {p.earlier_date} -> {p.later_date}  [{mark}]  p={p.probability:.2f}"
                         + ("" if p.comparable else "  (pair not comparable)"))
        if self.span:
            mark = "CHANGED" if self.span.changed else "stable"
            lines.append(f"    span {self.span.earlier_date} -> {self.span.later_date}  [{mark}]  "
                         f"p={self.span.probability:.2f}")
        es = self.earliest_supported
        lines.append(f"  earliest supported change: {es.get('window')}")
        lines.append(f"    supporting observations: {es.get('supporting_observations')}")
        lines.append(f"    caveat: {es.get('caveat')}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


class TemporalPersistenceAnalyzer:
    def __init__(self, matcher: TemporalObservationMatcher,
                 *, change_threshold: float = DEFAULT_CHANGE_THRESHOLD):
        self.matcher = matcher
        self.change_threshold = change_threshold

    # -- public ----------------------------------------------------------

    def sequence_for(self, lon: float, lat: float) -> ObservationSequence:
        return self.matcher.match(location=(lon, lat), all_pairs=True)

    def trajectory_for_location(self, lon: float, lat: float, lookup: ChangeLookup) -> ChangeTrajectory:
        seq = self.sequence_for(lon, lat)
        obs = seq.observations
        notes: list[str] = []
        if len(obs) < 2:
            return ChangeTrajectory((lon, lat), [], None, "none", 0.0,
                                    self._earliest(None, [], obs),
                                    notes=[f"only {len(obs)} observation(s) here - no trajectory"])

        # consecutive pairs, time-ordered
        consec: list[PairChange] = []
        for a, b in zip(obs[:-1], obs[1:]):
            pc = next((p for p in seq.pairs
                       if p.earlier.observation_id == a.observation_id
                       and p.later.observation_id == b.observation_id), None)
            comparable = bool(pc.comparable) if pc is not None else False
            changed, prob = lookup(a.observation_id, b.observation_id, lon, lat)
            consec.append(PairChange(a.observation_id, b.observation_id, a.acquired_at, b.acquired_at,
                                     comparable, bool(changed) and comparable, float(prob)))

        # span pair: earliest vs latest
        span_pc = next((p for p in seq.pairs
                        if p.earlier.observation_id == obs[0].observation_id
                        and p.later.observation_id == obs[-1].observation_id), None)
        span_changed, span_prob = lookup(obs[0].observation_id, obs[-1].observation_id, lon, lat)
        span = PairChange(obs[0].observation_id, obs[-1].observation_id,
                          obs[0].acquired_at, obs[-1].acquired_at,
                          bool(span_pc.comparable) if span_pc else False,
                          bool(span_changed) and bool(span_pc.comparable if span_pc else False),
                          float(span_prob))

        persistence, notes = self._classify_persistence(consec, span)
        conf = PERSISTENCE_CONFIDENCE.get(persistence, 0.0)
        earliest = self._earliest(span, consec, obs)
        return ChangeTrajectory((lon, lat), consec, span, persistence, conf, earliest, notes)

    # -- persistence logic --------------------------------------------

    @staticmethod
    def _classify_persistence(consec: list[PairChange], span: PairChange) -> tuple[str, list[str]]:
        notes: list[str] = []
        flags = [p.changed for p in consec]
        n = len(consec)

        if n < 2:
            if n == 1:
                notes.append("only two observations available - persistence cannot be assessed")
                return ("single_pair" if flags and flags[0] else "none"), notes
            return "none", notes

        early_changed = flags[0]
        later_changed = any(flags[1:])
        all_later_stable = not any(flags[1:])

        if not any(flags) and not span.changed:
            return "none", notes
        if not span.changed and any(flags):
            if early_changed and later_changed:
                notes.append("both consecutive intervals flag change but the first-vs-last pair does "
                             "not - a genuine contradiction (registration / model noise)")
                return "inconsistent", notes
            notes.append("a consecutive pair flags change but the first-vs-last (span) pair does not "
                         "- the change reverted; transient (temporary phenomenon or false alarm)")
            return "transient", notes
        if early_changed and later_changed and span.changed:
            notes.append("change signal in every interval - consistent with ongoing works")
            return "progressive", notes
        if early_changed and all_later_stable and span.changed:
            notes.append(f"changed in the first interval, then stable across {n - 1} later "
                         f"observation(s), and still present at the last date")
            return "persistent", notes
        if not early_changed and later_changed and span.changed:
            notes.append("change appears only in the most recent interval")
            return "recent", notes
        if early_changed and not span.changed:
            notes.append("changed early then reverted by the last date - transient (temporary "
                         "phenomenon or false alarm)")
            return "transient", notes
        notes.append("mixed / weak temporal support")
        return "transient", notes

    # -- earliest supported change ----------------------------------

    def _earliest(self, span, consec, obs) -> dict:
        earliest_obs_date = obs[0].acquired_at if obs else None
        caveat = (f"cannot claim a change date earlier than the earliest usable observation "
                  f"({earliest_obs_date}); any change before that is unobservable with the current "
                  f"three-date stack.") if earliest_obs_date else "no usable observations."
        first_changed = next((p for p in consec if p.changed), None)
        if first_changed is not None:
            return {
                "window": [first_changed.earlier_date, first_changed.later_date],
                "supporting_observations": [first_changed.earlier_obs, first_changed.later_obs],
                "imagery_quality_note": "both bounding observations are usable (comparable pair).",
                "caveat": caveat,
            }
        if span is not None and span.changed:
            return {
                "window": [span.earlier_date, span.later_date],
                "supporting_observations": [span.earlier_obs, span.later_obs],
                "imagery_quality_note": ("only the first-vs-last comparison supports the change; no "
                                         "single consecutive interval isolates it."),
                "caveat": caveat,
            }
        return {"window": None, "supporting_observations": [],
                "imagery_quality_note": "no pair supports a change at this location.",
                "caveat": caveat}

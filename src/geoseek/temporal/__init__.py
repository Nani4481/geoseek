"""Temporal reasoning over observations.

:mod:`geoseek.temporal.contract` holds the data types that
:class:`geoseek.temporal.matcher.TemporalObservationMatcher` produces and that
a Phase 3b change-detection model consumes unchanged.
"""

from geoseek.temporal.contract import (
    ComparabilityCriterion,
    ObservationPair,
    ObservationSequence,
    PairComparability,
)

__all__ = [
    "ComparabilityCriterion",
    "PairComparability",
    "ObservationPair",
    "ObservationSequence",
]

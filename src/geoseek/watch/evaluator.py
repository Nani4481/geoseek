"""Phase 8 Step C - evaluate standing watch areas against change candidates.

A watch area is defined once (:class:`geoseek.catalog.entities.WatchArea`,
stored through the :class:`~geoseek.catalog.repository.MetadataRepository`
seam - the same pattern as the Phase 6 analyst-decisions audit table, but a
normal editable/deletable record rather than a log). It is evaluated here
against the candidate rows a change-pipeline run produces
(``geoseek.change.analyze.run`` calls :func:`evaluate_and_notify` once it has
written the full-detail candidate list) - **not** on a background timer or
file-watcher (this project has neither): re-running
``python -m geoseek.change.analyze`` after a new observation is ingested IS
the trigger, since that is the point at which "new candidates" for that
observation come into existence at all.

Matching is 4 independent, all-must-pass predicates:

  * spatial   - inside ``watch.polygon_wkt_4326`` if set, else inside
    ``watch.bbox``, else unrestricted (an intentionally-global watch).
  * change_types - membership in ``watch.change_types`` (empty = any type).
  * min_confidence - candidate confidence >= the threshold.
  * text_query - a plain **keyword** match (any query word appearing, case-
    insensitive) against the candidate's change type + classification
    rule/detail text. This is deliberately NOT semantic search (no embedding
    model is loaded here - watch evaluation must stay cheap and run on every
    pipeline pass): stated explicitly so it is never mistaken for the
    RemoteCLIP-backed query relevance :mod:`geoseek.fusion.ranker` uses
    elsewhere.

A watch area only ever gets a *new* notification for candidates it has not
already notified about (the union of every prior notification's
``candidate_ids`` for that watch, tracked via
:meth:`MetadataRepository.list_notifications`) - so re-running the pipeline
repeatedly does not re-fire on the same match.

Pure functions over plain dicts/entities - no rasters/torch/network.
"""

from __future__ import annotations

from geoseek.catalog.entities import WatchArea, WatchNotification
from geoseek.catalog.repository import MetadataRepository


def _spatial_match(watch: WatchArea, centroid_lonlat: tuple[float, float]) -> bool:
    lon, lat = centroid_lonlat
    if watch.polygon_wkt_4326:
        from shapely import wkt as shapely_wkt
        from shapely.geometry import Point

        return shapely_wkt.loads(watch.polygon_wkt_4326).contains(Point(lon, lat))
    if watch.bbox:
        w, s, e, n = watch.bbox
        return w <= lon <= e and s <= lat <= n
    return True   # no spatial filter defined - an intentionally AOI-unrestricted watch


def _type_match(watch: WatchArea, candidate: dict) -> bool:
    return not watch.change_types or candidate.get("change_type") in watch.change_types


def _confidence_match(watch: WatchArea, candidate: dict) -> bool:
    if watch.min_confidence is None:
        return True
    return float(candidate.get("confidence") or 0.0) >= float(watch.min_confidence)


def _text_match(watch: WatchArea, candidate: dict) -> bool:
    query = (watch.text_query or "").strip()
    if not query:
        return True
    terms = [t.lower() for t in query.split() if t.strip()]
    cl = candidate.get("classification") or {}
    blob = " ".join(str(x) for x in (candidate.get("change_type", ""), cl.get("rule", ""),
                                     cl.get("detail", ""))).lower()
    return any(t in blob for t in terms)


def candidate_matches(watch: WatchArea, candidate: dict) -> bool:
    lonlat = candidate.get("centroid_lonlat")
    if not lonlat:
        return False
    return (_spatial_match(watch, tuple(lonlat)) and _type_match(watch, candidate)
            and _confidence_match(watch, candidate) and _text_match(watch, candidate))


def evaluate_watch_area(watch: WatchArea, candidates: list[dict]) -> list[str]:
    """candidate_ids among ``candidates`` matching ``watch``'s filters (any
    match history is NOT considered here - see :func:`evaluate_and_notify`
    for the "new since last time" diff)."""
    return [c["candidate_id"] for c in candidates if candidate_matches(watch, c)]


def evaluate_and_notify(
    repo: MetadataRepository, candidates: list[dict], observation_id: str
) -> list[WatchNotification]:
    """Evaluate every active watch area against ``candidates``; record + return
    one :class:`WatchNotification` per watch area that has at least one
    NEWLY-matching candidate (never notified for that candidate_id before)."""
    fired: list[WatchNotification] = []
    for watch in repo.list_watch_areas(active_only=True):
        matching = set(evaluate_watch_area(watch, candidates))
        if not matching:
            continue
        already_notified: set[str] = set()
        for n in repo.list_notifications(watch_id=watch.watch_id):
            already_notified.update(n.candidate_ids)
        new_ids = sorted(matching - already_notified)
        if not new_ids:
            continue
        stored = repo.record_notification(WatchNotification(
            notification_id="", watch_id=watch.watch_id, observation_id=observation_id,
            candidate_ids=tuple(new_ids),
        ))
        fired.append(stored)
    return fired

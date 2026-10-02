"""Threat buffer rings: what is inside N metres of a point, from data the system already holds.

Read-only, computed per request (nothing stored). The feature layers that are intersected are exactly the ones the
archive has:

  * ``restricted_zone``  - the existing sensitive-area boxes (``analyst.service.RESTRICTED_ZONES``, the same ones that
                           produce the restricted-zone alerts)
  * ``change_candidate`` - every ranked change candidate's footprint polygon
  * ``detection``        - every stored object detection (oriented-box footprints in ``data/detections``)
  * ``watch_area``       - standing watch areas an analyst has defined

There is **no** road / building / infrastructure layer in the catalog, and none is invented: the response says which
layers were searched and how many features each holds (``sources``) so an empty ring is never mistaken for "checked
everything".

Geometry: every distance is measured in an azimuthal-equidistant projection centred on the query point, where the distance
from the centre to any point is the true geodesic distance, so "inside a ring" is an exact point-to-footprint distance
(0 when the footprint contains the centre), not a bbox or centroid approximation.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry import Point, Polygon, box
from shapely.ops import nearest_points

KINDS = ("restricted_zone", "change_candidate", "detection", "watch_area")
MIN_RADIUS_M, MAX_RADIUS_M, MAX_RINGS = 10.0, 200_000.0, 8
DEFAULT_RADII_M = (500.0, 1000.0, 2500.0, 5000.0)
DEFAULT_LIMIT = 150


def parse_radii(text: str | None) -> list[float]:
    """``"500,1000,2500"`` -> sorted unique radii in metres; ValueError (-> HTTP 400) on anything unusable."""
    if not text:
        return list(DEFAULT_RADII_M)
    try:
        vals = sorted({float(t) for t in text.split(",") if t.strip()})
    except ValueError:
        raise ValueError("radii_m must be comma-separated numbers (metres)")
    if not vals or len(vals) > MAX_RINGS:
        raise ValueError(f"give between 1 and {MAX_RINGS} ring radii")
    if vals[0] < MIN_RADIUS_M or vals[-1] > MAX_RADIUS_M:
        raise ValueError(f"ring radii must be between {MIN_RADIUS_M:g} m and {MAX_RADIUS_M:g} m")
    return vals


# --------------------------------------------------------------------------- feature layers


def _polygon_ll(geom: dict) -> Polygon | None:
    try:
        return Polygon(geom["coordinates"][0])
    except Exception:
        return None


def zone_features() -> list[dict]:
    from geoseek.analyst.service import RESTRICTED_ZONES

    return [{"kind": "restricted_zone", "id": f"zone:{z['name']}", "name": z["name"],
             "geom": box(z["min_lon"], z["min_lat"], z["max_lon"], z["max_lat"]),
             "detail": {"level": z["level"]}} for z in RESTRICTED_ZONES]


def candidate_features(svc) -> list[dict]:
    out = []
    for c in getattr(svc, "details", []):
        g = _polygon_ll(c.get("_geometry") or {})
        if g is None or g.is_empty:
            continue
        out.append({"kind": "change_candidate", "id": f"candidate:{c['candidate_id']}", "name": c["candidate_id"], "geom": g,
                    "detail": {"candidate_id": c["candidate_id"], "change_type": c.get("change_type"),
                               "confidence": c.get("confidence"), "area_m2": c.get("area_m2"),
                               "persistence": c.get("persistence")}})
    return out


@lru_cache(maxsize=8)
def _detection_file(path: str, mtime_ns: int) -> tuple:
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for i, f in enumerate(gj.get("features", [])):
        try:
            ring = f["geometry"]["coordinates"][0]
            p = f.get("properties", {})
            rows.append((i, ring, p.get("class"), p.get("score"), p.get("tile_id")))
        except Exception:
            continue
    return tuple(rows)


@lru_cache(maxsize=8)
def _detection_extra(path: str, mtime_ns: int) -> tuple:
    """(heading_deg, long_side_px) per feature, parallel to ``_detection_file`` rows (same order, same index)."""
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for f in gj.get("features", []):
        p = f.get("properties", {})
        out.append((p.get("heading_deg"), p.get("long_side_px")))
    return tuple(out)


def _box_dims_m(ring) -> tuple[float, float]:
    """(length, width) in metres of an oriented box from its first three corners, using the WGS84 metres-per-degree series at the
    box's latitude (boxes are metres across, so the local-plane error is millimetres)."""
    phi = math.radians(sum(p[1] for p in ring[:4]) / 4.0)
    ky = 111_132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    kx = 111_412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    a, b, c = ring[0], ring[1], ring[2]
    e1 = math.hypot((b[0] - a[0]) * kx, (b[1] - a[1]) * ky)
    e2 = math.hypot((c[0] - b[0]) * kx, (c[1] - b[1]) * ky)
    return max(e1, e2), min(e1, e2)


def detection_observation_ids(data_dir: Path) -> list[str]:
    root = Path(data_dir) / "detections"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "detections.geojson").is_file())


def detection_rows(data_dir: Path, observation_id: str) -> tuple:
    p = Path(data_dir) / "detections" / observation_id / "detections.geojson"
    if not p.is_file():
        raise FileNotFoundError(observation_id)
    return _detection_file(str(p), p.stat().st_mtime_ns)


def detection_points(data_dir: Path, observation_id: str) -> list[dict]:
    """One lon/lat point (footprint centroid) per stored detection, for drawing on a map and picking as a ring centre, with the
    oriented box's size (metres, from its stored footprint) and heading (as stored by the detector)."""
    out = []
    p = Path(data_dir) / "detections" / observation_id / "detections.geojson"
    extra = _detection_extra(str(p), p.stat().st_mtime_ns) if p.is_file() else ()
    for i, ring, cls, score, tile in detection_rows(data_dir, observation_id):
        xs, ys = [q[0] for q in ring[:-1]], [q[1] for q in ring[:-1]]
        length_m, width_m = _box_dims_m(ring) if len(ring) >= 4 else (None, None)
        heading, long_px = extra[i] if i < len(extra) else (None, None)
        out.append({"id": f"detection:{observation_id}:{i}", "lon": sum(xs) / len(xs), "lat": sum(ys) / len(ys),
                    "class": cls, "score": score, "tile_id": tile,
                    "length_m": None if length_m is None else round(length_m, 2), "width_m": None if width_m is None else round(width_m, 2),
                    "heading_deg": heading, "long_side_px": long_px})
    return out


def detection_features(data_dir: Path, bbox: tuple[float, float, float, float]) -> tuple[list[dict], int]:
    """Detections whose footprint touches ``bbox`` (a cheap pre-filter), plus the size of the whole layer."""
    w, s, e, n = bbox
    out, total = [], 0
    for obs in detection_observation_ids(data_dir):
        for i, ring, cls, score, tile in detection_rows(data_dir, obs):
            total += 1
            xs, ys = [p[0] for p in ring], [p[1] for p in ring]
            if max(xs) < w or min(xs) > e or max(ys) < s or min(ys) > n:       # cheap reject before any projection
                continue
            out.append({"kind": "detection", "id": f"detection:{obs}:{i}", "name": f"{cls} · {obs}", "geom": Polygon(ring),
                        "detail": {"class": cls, "score": score, "observation_id": obs, "tile_id": tile}})
    return out, total


def watch_features(svc) -> list[dict]:
    out = []
    try:
        areas = svc.list_watch_areas()
    except Exception:
        return out
    for a in areas:
        g = None
        if a.get("bbox"):
            g = box(*a["bbox"])
        elif a.get("polygon_wkt_4326"):
            from shapely import wkt

            try:
                g = wkt.loads(a["polygon_wkt_4326"])
            except Exception:
                g = None
        if g is not None and not g.is_empty:
            out.append({"kind": "watch_area", "id": f"watch:{a['watch_id']}", "name": a.get("name") or a["watch_id"], "geom": g,
                        "detail": {"active": bool(a.get("active", True)), "text_query": a.get("text_query") or None}})
    return out


# --------------------------------------------------------------------------- geometry


def _project(tr: pyproj.Transformer, g):
    """A lon/lat shapely geometry into the metre plane centred on the query point."""
    if g.geom_type == "Polygon":
        x, y = tr.transform(np.asarray(g.exterior.coords)[:, 0], np.asarray(g.exterior.coords)[:, 1])
        return Polygon(np.column_stack([x, y]))
    raise ValueError(g.geom_type)


def _deg_margin(lat: float, radius_m: float) -> tuple[float, float]:
    dlat = radius_m / 110_540.0 * 1.05
    dlon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.01)) * 1.05
    return dlon, dlat


def threat_rings(svc, lon: float, lat: float, radii_m: list[float], *, exclude: str | None = None,
                 kinds: tuple[str, ...] = KINDS, limit: int = DEFAULT_LIMIT) -> dict:
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("lon/lat out of range")
    radii = sorted(radii_m)
    rmax = radii[-1]
    dlon, dlat = _deg_margin(lat, rmax)
    qbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    inv = pyproj.Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)

    layers: dict[str, list[dict]] = {}
    layer_sizes: dict[str, int] = {}
    if "restricted_zone" in kinds:
        layers["restricted_zone"] = zone_features()
    if "change_candidate" in kinds:
        layers["change_candidate"] = candidate_features(svc)
    if "detection" in kinds:
        layers["detection"], layer_sizes["detection"] = detection_features(svc.settings.data_dir, qbox)
    if "watch_area" in kinds:
        layers["watch_area"] = watch_features(svc)

    centre = Point(0.0, 0.0)
    hits, sources = [], {}
    for kind, feats in layers.items():
        n_in_range = 0
        for f in feats:
            if f["id"] == exclude:
                continue
            w, s, e, n = f["geom"].bounds
            if e < qbox[0] or w > qbox[2] or n < qbox[1] or s > qbox[3]:
                continue
            g = _project(fwd, f["geom"])
            d = float(centre.distance(g))
            if d > rmax:
                continue
            n_in_range += 1
            np_x, np_y = nearest_points(centre, g)[1].coords[0]
            nlon, nlat = inv.transform(np_x, np_y)
            c = f["geom"].centroid
            ring_index = next(i for i, r in enumerate(radii) if d <= r)
            hits.append({"kind": kind, "id": f["id"], "name": f["name"], "distance_m": round(d, 1), "ring_index": ring_index,
                         "ring_m": radii[ring_index], "contains_centre": d == 0.0,
                         "lon": round(c.x, 6), "lat": round(c.y, 6), "nearest": [round(nlon, 6), round(nlat, 6)],
                         "detail": f["detail"]})
        sources[kind] = {"features_searched": layer_sizes.get(kind, len(feats)), "within_largest_ring": n_in_range}
    hits.sort(key=lambda h: (h["distance_m"], h["id"]))

    rings = []
    for i, r in enumerate(radii):
        counts = {k: sum(1 for h in hits if h["kind"] == k and h["ring_index"] <= i) for k in layers}
        rings.append({"radius_m": r, "counts": counts, "total": sum(counts.values())})
    return {
        "center": {"lon": lon, "lat": lat}, "radii_m": radii, "rings": rings,
        "features": hits[:limit], "truncated": len(hits) > limit, "n_features": len(hits),
        "sources": sources, "excluded": exclude,
        "notes": [
            "Distances are exact point-to-footprint geodesic distances (azimuthal-equidistant projection centred on the point); "
            "0 m means the footprint contains the centre.",
            "Restricted zones are the existing demonstration boxes in analyst/service.py, not an authoritative gazetteer.",
            "The catalog holds no road, building or infrastructure layer: only the layers listed under `sources` were searched.",
        ],
    }

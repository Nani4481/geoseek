"""Build the analyst UI's bundled, offline basemap layer.

Fetches Natural Earth 110m/50m source files (a one-time build-time step, same
category as staging Sentinel-2/Maxar imagery - never touched at runtime),
filters them down to South Asia + major world context, simplifies and rounds
coordinates, and writes a single small GeoJSON bundle that CoordMap loads from
this same FastAPI origin - no tile service, no CDN, no network call once the
app is running.

    python scripts/build_basemap.py

Output: src/geoseek/analyst/web/basemap.json (~370KB: coastline, India +
neighbouring country borders, India state borders, major rivers).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from geoseek.staging.manifest import build_record, load_manifest, write_manifest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "basemap_src"
OUT = ROOT / "src" / "geoseek" / "analyst" / "web" / "basemap.json"

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
SOURCES = [
    "ne_110m_admin_0_countries.geojson",
    "ne_50m_admin_1_states_provinces.geojson",
    "ne_110m_coastline.geojson",
    "ne_50m_rivers_lake_centerlines.geojson",
    "ne_10m_populated_places.geojson",
]

SOUTH_ASIA_COUNTRIES = {
    "India", "Pakistan", "Nepal", "Bhutan", "Bangladesh", "China", "Myanmar",
    "Sri Lanka", "Afghanistan",
}
SOUTH_ASIA_BBOX = (60.0, 5.0, 100.0, 40.0)  # w, s, e, n - used to keep regional rivers
ROUND = 4  # decimal degrees, ~11m at the equator - plenty for a background line


def _fetch_sources() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        dest = SRC_DIR / name
        if dest.is_file():
            continue
        print(f"fetching {name} ...")
        urllib.request.urlretrieve(f"{NE_BASE}/{name}", dest)


def _round_coords(geom):
    def _r(x, y, z=None):
        return (round(x, ROUND), round(y, ROUND))
    return shp_transform(_r, geom)


def _load(name: str) -> dict:
    return json.loads((SRC_DIR / name).read_text(encoding="utf-8"))


def _feature(name: str | None, geom) -> dict:
    props = {"name": name} if name else {}
    return {"type": "Feature", "properties": props, "geometry": mapping(geom)}


def build_countries() -> list[dict]:
    feats = []
    for f in _load("ne_110m_admin_0_countries.geojson")["features"]:
        name = f["properties"].get("ADMIN") or f["properties"].get("NAME")
        if name not in SOUTH_ASIA_COUNTRIES:
            continue
        geom = _round_coords(shape(f["geometry"]).simplify(0.01, preserve_topology=True))
        feats.append(_feature(name, geom))
    return feats


def build_states() -> list[dict]:
    feats = []
    for f in _load("ne_50m_admin_1_states_provinces.geojson")["features"]:
        if f["properties"].get("admin") != "India":
            continue
        geom = _round_coords(shape(f["geometry"]).simplify(0.008, preserve_topology=True))
        feats.append(_feature(f["properties"].get("name"), geom))
    return feats


def build_coastline() -> list[dict]:
    feats = []
    for f in _load("ne_110m_coastline.geojson")["features"]:
        geom = _round_coords(shape(f["geometry"]).simplify(0.02, preserve_topology=True))
        feats.append(_feature(None, geom))
    return feats


def _bbox_intersects(geom, bbox) -> bool:
    w, s, e, n = bbox
    gw, gs, ge, gn = geom.bounds
    return not (ge < w or gw > e or gn < s or gs > n)


def build_rivers() -> list[dict]:
    feats = []
    for f in _load("ne_50m_rivers_lake_centerlines.geojson")["features"]:
        scalerank = f["properties"].get("scalerank", 99) or 99
        geom0 = shape(f["geometry"])
        # major world rivers by Natural Earth's own ranking, plus anything
        # touching South Asia even if minor by that global ranking
        if scalerank > 3 and not _bbox_intersects(geom0, SOUTH_ASIA_BBOX):
            continue
        geom = _round_coords(geom0.simplify(0.01, preserve_topology=True))
        feats.append(_feature(f["properties"].get("name"), geom))
    return feats


def build_cities() -> list[dict]:
    # Indian cities/towns only, ranked by population so CoordMap can reveal
    # them progressively as the analyst zooms in (see MIN_ZOOM handling in
    # app.js) - a flat list would either clutter the full-India view or, if
    # filtered to just the biggest few, disappear at regional zoom.
    feats = []
    for f in _load("ne_10m_populated_places.geojson")["features"]:
        p = f["properties"]
        if p.get("ADM0NAME") != "India":
            continue
        pop = int(p.get("POP_MAX") or 0)
        if pop < 20000:
            continue
        geom = shape(f["geometry"])
        x, y = round(geom.x, 4), round(geom.y, 4)
        feats.append({
            "type": "Feature",
            "properties": {"name": p.get("NAME"), "pop": pop, "capital": bool(p.get("ADM0CAP"))},
            "geometry": {"type": "Point", "coordinates": [x, y]},
        })
    feats.sort(key=lambda f: -f["properties"]["pop"])
    return feats


def main() -> None:
    _fetch_sources()
    out = {
        "coastline": {"type": "FeatureCollection", "features": build_coastline()},
        "countries": {"type": "FeatureCollection", "features": build_countries()},
        "states": {"type": "FeatureCollection", "features": build_states()},
        "rivers": {"type": "FeatureCollection", "features": build_rivers()},
        "cities": {"type": "FeatureCollection", "features": build_cities()},
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    sizes = {k: len(v["features"]) for k, v in out.items()}
    kb = round(OUT.stat().st_size / 1024, 1)
    print(f"features: {sizes}")
    print(f"wrote {OUT} - {kb} KB")

    record = build_record(
        name="analyst_ui_basemap_vectors",
        source_url=f"{NE_BASE}/{{{','.join(SOURCES)}}}",
        local_path=OUT,
        license="Natural Earth (public domain, no attribution required) - https://www.naturalearthdata.com/about/terms-of-use/",
    )
    manifest = load_manifest()
    artifacts = [a for a in manifest.setdefault("artifacts", []) if a.get("name") != record.name]
    artifacts.append(asdict(record))
    manifest["artifacts"] = artifacts
    write_manifest(manifest)
    print(f"recorded provenance for {record.name} in the manifest")


if __name__ == "__main__":
    main()

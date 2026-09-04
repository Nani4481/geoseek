"""Phase 6 Step D - build a provenance-carrying GeoJSON/CSV export and validate it.

  python scripts/validate_export.py [--filtered]

Checks:
  1. structural  - FeatureCollection, every Feature has geometry + properties,
     coordinates are [lon, lat] within valid ranges, rings are closed;
  2. provenance  - every feature carries the required PS 2.2.5 fields
     (geometry EPSG:4326, change type, confidence, evidence summary, earliest
     supported change, source scene ids, acquisition dates, sensor,
     processing/model versions, weights SHA-256, git commit, analyst decision);
  3. GIS reader  - the file opens in GDAL/OGR (the standard reader): layer,
     SRS = WGS 84, geometry type, field schema, feature count, one geometry
     round-tripped through OGR back to WKT.
"""

from __future__ import annotations

import argparse
import json
import sys

REQUIRED_PROPS = [
    "candidate_id", "change_type", "confidence", "evidence_summary",
    "earliest_supported_change", "source_scene_ids", "acquisition_dates", "sensor",
    "weights_sha256", "git_commit", "pipeline_version", "model_version",
    "processing", "analyst_decision",
]
REQUIRED_PROCESSING = ["model", "model_threshold", "weights_sha256", "git_commit", "pipeline_version"]


def _structural(fc: dict) -> list[str]:
    errs: list[str] = []
    if fc.get("type") != "FeatureCollection":
        errs.append("top-level type is not FeatureCollection")
    feats = fc.get("features", [])
    if not feats:
        errs.append("no features")
    for i, f in enumerate(feats):
        tag = f.get("properties", {}).get("candidate_id", f"#{i}")
        if f.get("type") != "Feature":
            errs.append(f"{tag}: not a Feature")
        g = f.get("geometry") or {}
        if g.get("type") != "Polygon":
            errs.append(f"{tag}: geometry type {g.get('type')!r} (expected Polygon)")
            continue
        rings = g.get("coordinates") or []
        if not rings or len(rings[0]) < 4:
            errs.append(f"{tag}: ring has < 4 positions")
            continue
        ring = rings[0]
        if ring[0] != ring[-1]:
            errs.append(f"{tag}: ring not closed")
        for lon, lat in ring:
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                errs.append(f"{tag}: coordinate out of range ({lon}, {lat})")
                break
            if not (81.0 <= lon <= 84.0 and 25.0 <= lat <= 28.0):
                errs.append(f"{tag}: coordinate outside the Ayodhya AOI ({lon}, {lat})")
                break
    return errs


def _provenance(fc: dict) -> list[str]:
    errs: list[str] = []
    for f in fc.get("features", []):
        p = f.get("properties", {})
        tag = p.get("candidate_id", "?")
        for k in REQUIRED_PROPS:
            if k not in p:
                errs.append(f"{tag}: missing property {k!r}")
        proc = p.get("processing") or {}
        for k in REQUIRED_PROCESSING:
            if not proc.get(k) and proc.get(k) != 0:
                errs.append(f"{tag}: processing.{k} empty")
        if len(p.get("source_scene_ids") or []) != 2:
            errs.append(f"{tag}: expected 2 source_scene_ids")
        if len(p.get("acquisition_dates") or []) != 2 or not all(p.get("acquisition_dates") or [None]):
            errs.append(f"{tag}: acquisition_dates not two real dates ({p.get('acquisition_dates')})")
        if not (p.get("weights_sha256") or proc.get("weights_sha256")):
            errs.append(f"{tag}: no weights_sha256")
        wsha = p.get("weights_sha256") or proc.get("weights_sha256", "")
        if wsha and (len(wsha) != 64 or any(c not in "0123456789abcdef" for c in wsha)):
            errs.append(f"{tag}: weights_sha256 not a 64-hex digest")
        # analyst_decision may legitimately be null, but the KEY must be present (checked above)
    return errs


def _gis_reader(path: str) -> list[str]:
    from osgeo import ogr, osr

    ogr.UseExceptions()
    errs: list[str] = []
    ds = ogr.Open(path)
    if ds is None:
        return [f"GDAL/OGR could not open {path}"]
    drv = ds.GetDriver().GetName()
    print(f"   OGR driver          : {drv}")
    lyr = ds.GetLayer(0)
    n = lyr.GetFeatureCount()
    print(f"   layer / features    : {lyr.GetName()!r} / {n}")
    geom_type = ogr.GeometryTypeToName(lyr.GetGeomType())
    print(f"   geometry type       : {geom_type}")
    srs = lyr.GetSpatialRef()
    srs_name = srs.GetAttrValue("GEOGCS") if srs else None
    is4326 = bool(srs) and (srs.GetAuthorityCode(None) == "4326"
                            or (srs_name or "").upper().replace(" ", "") in ("WGS84", "WGS_1984"))
    print(f"   spatial ref         : {srs_name} (EPSG:4326 = {is4326})")
    if not is4326:
        errs.append(f"layer SRS is not WGS84/EPSG:4326 (got {srs_name})")
    fields = [lyr.GetLayerDefn().GetFieldDefn(i).GetName()
              for i in range(lyr.GetLayerDefn().GetFieldCount())]
    print(f"   field count         : {len(fields)}")
    for want in ("candidate_id", "change_type", "confidence", "weights_sha256", "git_commit",
                 "earliest_supported_change", "sensor", "model_version"):
        if not any(fld == want or fld.endswith("." + want) or fld.endswith("_" + want) for fld in fields):
            errs.append(f"OGR layer is missing attribute {want!r} (fields: {fields})")
    if drv != "GeoJSON":
        errs.append(f"expected the GeoJSON driver, got {drv}")
    if n <= 0:
        errs.append("OGR reports 0 features")
    f0 = lyr.GetNextFeature()
    if f0 is not None:
        g = f0.GetGeometryRef()
        print(f"   feature[0] cid      : {f0.GetField('candidate_id')}")
        print(f"   feature[0] geom WKT : {g.ExportToWkt()[:78]}...")
        print(f"   feature[0] area     : {g.GetArea():.3e} deg^2, valid={bool(g.IsValid())}")
        if not g.IsValid():
            errs.append("feature[0] geometry is not OGC-valid")
    ds = None
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", action="store_true",
                    help="export only construction candidates with confidence >= 0.9")
    args = ap.parse_args()

    from geoseek.analyst.service import AnalystService

    svc = AnalystService()
    # seed one decision so the export demonstrates a populated analyst_decision block
    try:
        svc.record_decision(svc.details[0]["candidate_id"], decision="confirm",
                            note="validation-run confirm", analyst="validator")
    except Exception:
        pass

    filters = {"change_type": "construction", "min_confidence": 0.9} if args.filtered else None
    res = svc.export(filters=filters, fmt="both")
    fc = res["geojson"]
    path = res["geojson_path"]
    print("=" * 78)
    print(f"exported {res['count']} features -> {path}")
    print(f"          + CSV               -> {res.get('csv_path')}")
    print("=" * 78)

    all_errs = []
    print("\n[1] structural validation")
    e = _structural(fc); all_errs += e
    print("   " + ("OK" if not e else "\n   ".join(e)))

    print("\n[2] provenance-completeness validation")
    e = _provenance(fc); all_errs += e
    print("   " + ("OK - every feature carries geometry(4326), change_type, confidence, "
                    "evidence_summary, earliest_supported_change, source_scene_ids, "
                    "acquisition_dates, sensor, processing{model,weights_sha256,git_commit,"
                    "pipeline_version}, analyst_decision" if not e else "\n   ".join(e)))

    print("\n[3] GDAL/OGR - loads in a standard GIS reader")
    e = _gis_reader(path); all_errs += e
    print("   " + ("OK" if not e else "\n   ".join(e)))

    print("\n" + "=" * 78)
    sample = fc["features"][0]
    print("SAMPLE FEATURE (properties):")
    print(json.dumps({**sample, "geometry": {"type": "Polygon", "coordinates": "...5 positions..."}},
                     indent=1)[:2600])
    print("=" * 78)
    print("ALL CHECKS PASSED" if not all_errs else f"!! {len(all_errs)} PROBLEM(S)")
    svc.close()
    raise SystemExit(0 if not all_errs else 2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise SystemExit(1)

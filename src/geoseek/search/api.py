"""FastAPI service for geoseek semantic search.

The SearchEngine (RemoteCLIP + FAISS index, both resident) is constructed
exactly ONCE at app startup and reused for every request - never reloaded
per query.

Run:
    uvicorn geoseek.search.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from geoseek.search.engine import SearchEngine, SearchFilters

_engine: SearchEngine | None = None
_analyst = None            # geoseek.analyst.service.AnalystService (built at startup)
_analyst_error: str | None = None

_WEB_DIR = Path(__file__).resolve().parents[1] / "analyst" / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _analyst, _analyst_error
    _engine = SearchEngine()  # loads RemoteCLIP + FAISS index ONCE, at startup
    try:
        from geoseek.analyst.service import AnalystService

        _analyst = AnalystService(engine=_engine)
        print(f"[analyst] service ready: {len(_analyst.details)} ranked candidates.")
    except Exception as e:  # the search API must still come up without a change report
        _analyst_error = f"{type(e).__name__}: {e}"
        print(f"[analyst] service unavailable: {_analyst_error}")
    yield
    _engine.close()


app = FastAPI(title="geoseek analyst interface", lifespan=lifespan)


def _get_engine() -> SearchEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="SearchEngine not ready")
    return _engine


def _parse_bbox(bbox: Optional[str]) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must be 'west,south,east,north'")
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox values must be numeric")
    return (west, south, east, north)


def _make_filters(
    bbox: Optional[str], date_start: Optional[str], date_end: Optional[str],
    sensor: Optional[str], max_cloud_fraction: Optional[float],
) -> SearchFilters:
    return SearchFilters(
        bbox=_parse_bbox(bbox), date_start=date_start, date_end=date_end,
        sensor=sensor, max_cloud_fraction=max_cloud_fraction,
    )


@app.get("/search/text")
def search_text(
    q: str = Query(..., description="Free-text query, e.g. 'a river with sandbars'"),
    k: int = Query(10, ge=1, le=200),
    bbox: Optional[str] = Query(None, description="west,south,east,north (EPSG:4326)"),
    date_start: Optional[str] = Query(None, description="YYYY-MM-DD, inclusive"),
    date_end: Optional[str] = Query(None, description="YYYY-MM-DD, inclusive"),
    sensor: Optional[str] = Query(None, description="e.g. 'Sentinel-2A'"),
    max_cloud_fraction: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    engine = _get_engine()
    filters = _make_filters(bbox, date_start, date_end, sensor, max_cloud_fraction)
    results, latency_ms = engine.search_text(q, k=k, filters=filters)
    return {
        "query": q,
        "k": k,
        "latency_ms": round(latency_ms, 3),
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


class ImageSearchRequest(BaseModel):
    tile_id: Optional[str] = None
    image_base64: Optional[str] = None
    k: int = 10
    bbox: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    sensor: Optional[str] = None
    max_cloud_fraction: Optional[float] = None


@app.post("/search/image")
def search_image(req: ImageSearchRequest):
    if (req.tile_id is None) == (req.image_base64 is None):
        raise HTTPException(status_code=400, detail="provide exactly one of tile_id or image_base64")

    engine = _get_engine()
    filters = _make_filters(req.bbox, req.date_start, req.date_end, req.sensor, req.max_cloud_fraction)

    image_rgb_uint8 = None
    if req.image_base64 is not None:
        from PIL import Image

        try:
            raw = base64.b64decode(req.image_base64)
            image_rgb_uint8 = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not decode image_base64: {e}")

    try:
        results, latency_ms = engine.search_image(
            tile_id=req.tile_id, image_rgb_uint8=image_rgb_uint8, k=req.k, filters=filters,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "tile_id": req.tile_id,
        "k": req.k,
        "latency_ms": round(latency_ms, 3),
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.get("/tile/{tile_id}/thumbnail")
def tile_thumbnail(tile_id: str):
    engine = _get_engine()
    try:
        png_bytes = engine.get_tile_thumbnail_png(tile_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=png_bytes, media_type="image/png")


@app.get("/health")
def health():
    engine = _get_engine()
    body = {"status": "ok", "vectors": engine.count(), "analyst": _analyst_error or "ready"}
    if _analyst is not None:
        body.update(_analyst.health())
    return body


# ==========================================================================
# Phase 6 - analyst interface
# ==========================================================================


def _get_analyst():
    if _analyst is None:
        raise HTTPException(status_code=503,
                            detail=f"analyst service unavailable ({_analyst_error})")
    return _analyst


@app.get("/stats")
def stats():
    return _get_analyst().stats()


@app.get("/candidates")
def list_candidates(
    bbox: Optional[str] = Query(None, description="west,south,east,north (EPSG:4326)"),
    date_start: Optional[str] = Query(None, description="YYYY-MM-DD; change-window overlap"),
    date_end: Optional[str] = Query(None, description="YYYY-MM-DD; change-window overlap"),
    change_type: Optional[str] = Query(None, description="water_gain|water_loss|construction|clearance|road|other"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    sensor: Optional[str] = Query(None, description="'sentinel-2' or 'sentinel-1' (has SAR corroboration)"),
    persistence: Optional[str] = Query(None, description="persistent|progressive|recent|transient|..."),
    decision: Optional[str] = Query(None, description="confirm|reject|undecided (current analyst verdict)"),
    sort: str = Query("queue_score", description="queue_score|confidence|significance|area_m2|rank; '-' prefix = ascending"),
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    return _get_analyst().list_candidates(
        bbox=_parse_bbox(bbox), date_start=date_start, date_end=date_end, change_type=change_type,
        min_confidence=min_confidence, sensor=sensor, persistence=persistence, decision=decision,
        sort=sort, limit=limit, offset=offset,
    )


@app.get("/candidates/{candidate_id}")
def candidate_detail(candidate_id: str):
    detail = _get_analyst().get_candidate(candidate_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")
    return detail


@app.get("/candidates/{candidate_id}/imagery")
def candidate_imagery(
    candidate_id: str,
    date: str = Query("2024", description="2019 | 2021 | 2024"),
    view: str = Query("rgb", description="rgb | overlay (change mask on that date)"),
):
    svc = _get_analyst()
    c = svc._by_id.get(candidate_id)
    if c is None:
        raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")
    from geoseek.analyst.imagery import render_candidate_imagery

    try:
        png = render_candidate_imagery(c, date=date, view=view, prob_raster_path=svc.prob_raster_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


class DecisionRequest(BaseModel):
    decision: str                      # "confirm" | "reject"
    note: str = ""
    analyst: str = ""


@app.post("/candidates/{candidate_id}/decision")
def candidate_decision(candidate_id: str, req: DecisionRequest):
    if req.decision not in ("confirm", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'confirm' or 'reject'")
    try:
        return _get_analyst().record_decision(
            candidate_id, decision=req.decision, note=req.note, analyst=req.analyst)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")


@app.get("/audit")
def audit(
    candidate_id: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100000),
    full: bool = Query(False, description="include the full evidence_snapshot per row"),
):
    return _get_analyst().audit(candidate_id=candidate_id, limit=limit, full=full)


@app.get("/audit/{decision_id}")
def audit_row(decision_id: str):
    d = _get_analyst().get_decision(decision_id)
    if d is None:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id!r}")
    return d


class ExportRequest(BaseModel):
    candidate_ids: Optional[list[str]] = None
    filters: Optional[dict] = None
    format: str = "geojson"            # "geojson" | "csv" | "both"


@app.post("/export")
def export(req: ExportRequest = Body(...)):
    if req.format not in ("geojson", "csv", "both"):
        raise HTTPException(status_code=400, detail="format must be geojson | csv | both")
    filters = dict(req.filters or {})
    if "bbox" in filters and isinstance(filters["bbox"], str):
        filters["bbox"] = _parse_bbox(filters["bbox"])
    return _get_analyst().export(
        candidate_ids=req.candidate_ids, filters=filters or None, fmt=req.format)


@app.get("/discovery/clusters")
def discovery_clusters():
    return _get_analyst().discovery_clusters()


@app.get("/discovery/cluster-map.png")
def discovery_cluster_map():
    svc = _get_analyst()
    d = svc.discovery_clusters()
    p = Path(d.get("cluster_map_png") or "")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="cluster map not generated")
    return FileResponse(p, media_type="image/png")


@app.get("/candidates/{candidate_id}/similar")
def candidate_similar(candidate_id: str, k: int = Query(8, ge=1, le=50)):
    try:
        return _get_analyst().similar_to_candidate(candidate_id, k=k)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no candidate {candidate_id!r}")


@app.get("/discovery/similar")
def discovery_similar(
    tile_id: Optional[str] = Query(None),
    lon: Optional[float] = Query(None),
    lat: Optional[float] = Query(None),
    k: int = Query(8, ge=1, le=50),
):
    engine = _get_engine()
    from geoseek.discovery.knn import find_more_like_this

    if not tile_id and (lon is None or lat is None):
        raise HTTPException(status_code=400, detail="provide tile_id or lon+lat")
    try:
        res = find_more_like_this(engine, tile_id=tile_id, lon=lon, lat=lat, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if _analyst is not None:
        res["seed_cluster"] = _analyst.cluster_for_tile(res.get("seed_tile_id", ""))
        for r in res.get("results", []):
            r["cluster"] = _analyst.cluster_for_tile(r["tile_id"])
    return res


# the offline single-page frontend - served by this same process, no CDN.
if _WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(_WEB_DIR), html=True), name="analyst-web")


@app.get("/")
def _root():
    return {"service": "geoseek analyst interface", "ui": "/app/", "docs": "/docs",
            "endpoints": ["/search/text", "/search/image", "/candidates", "/candidates/{id}",
                          "/candidates/{id}/imagery", "/candidates/{id}/decision", "/audit",
                          "/export", "/health", "/stats", "/discovery/clusters",
                          "/discovery/similar", "/candidates/{id}/similar"]}

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
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from geoseek.search.engine import SearchEngine, SearchFilters

_engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = SearchEngine()  # loads RemoteCLIP + FAISS index ONCE, at startup
    yield
    _engine.close()


app = FastAPI(title="geoseek semantic search", lifespan=lifespan)


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
    return {"status": "ok", "vectors": engine.count()}

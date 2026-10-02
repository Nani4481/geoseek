"""On-demand serving for the Object Detection tab: REAL stored oriented-box
detections drawn on the REAL Maxar tile they were found on. Nothing here
re-runs inference - every box shown is exactly what scripts/detect_maxar.py
already wrote to data/detections/<observation_id>/detections.geojson.

That file only carries each detection's EPSG:4326 footprint (a WKT polygon
turned into GeoJSON coordinates), not its tile-local pixel polygon - so
tile_detections() reprojects lon/lat back to tile pixels using the same
per-tile affine + CRS scripts/detect_maxar.py used to go the other way
(read_rgb_tile's TileGeoRef), rather than re-deriving anything from the model.
"""

from __future__ import annotations

import functools
import io
import json
from pathlib import Path

import numpy as np

from geoseek.config import get_settings
from geoseek.staging.download_maxar import COLLECTION_ID, MAXAR_TILE_SIZE

BAND_ORDER = ("R", "G", "B")


@functools.lru_cache(maxsize=1)
def _card() -> dict | None:
    p = get_settings().models_dir / "detector" / "geoseek_obb_v15_yolo26s.card.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def model_info() -> dict:
    """Classes, each one's operating point (conf/precision/recall/AP50/AP50_95,
    with chosen_on/protocol provenance) and low-precision caveats. Reuses
    YoloObbDetectionModel.info's caveat logic without loading the actual
    (heavy) ultralytics model - the constructor only reads the card."""
    from geoseek.models.yolo_obb import YoloObbDetectionModel

    weights = get_settings().models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt"
    m = YoloObbDetectionModel(weights)
    card = _card() or {}
    op = (card.get("operating_point") or {}).get("per_class") or {}
    return {
        "classes": list(m.class_names) if card.get("classes") else sorted(op),
        "operating_points": op,
        "caveats": m.info["caveats"],
        "weights_sha256": card.get("weights_sha256"),
        "architecture": card.get("architecture"),
    }


def _obs_by_id(repo, observation_id: str):
    for o in repo.list_observations(collection=COLLECTION_ID):
        if o.observation_id == observation_id:
            return o
    raise KeyError(observation_id)


def list_observations(repo) -> list[dict]:
    settings = get_settings()
    out = []
    coll = repo.get_collection(COLLECTION_ID)
    for o in repo.list_observations(collection=COLLECTION_ID):
        sp = settings.data_dir / "detections" / o.observation_id / "summary.json"
        if not sp.is_file():
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        scene = repo.get_scene(o.scene_id)
        out.append({
            "observation_id": o.observation_id, "aoi_name": o.aoi_name,
            "acquired_at": o.acquired_at, "platform": scene.platform if scene else None,
            "sensor": coll.sensor if coll else None, "native_gsd_m": coll.native_gsd_m if coll else None,
            "role": (o.metadata or {}).get("role"),
            "n_tiles": s.get("n_tiles"), "n_tiles_with_detections": s.get("n_tiles_with_detections"),
            "n_detections": s.get("n_detections"), "by_class": s.get("by_class"),
        })
    out.sort(key=lambda o: -(o["n_detections"] or 0))
    return out


def list_tiles_with_detections(observation_id: str) -> list[dict]:
    sp = get_settings().data_dir / "detections" / observation_id / "summary.json"
    if not sp.is_file():
        raise FileNotFoundError(observation_id)
    s = json.loads(sp.read_text(encoding="utf-8"))
    rows = [t for t in s.get("per_tile", []) if t.get("n_detections")]
    rows.sort(key=lambda t: -t["n_detections"])
    return rows


def _tile_geo(scene_dir: Path, row: int, col: int):
    import rasterio
    from rasterio.windows import Window, transform as window_transform

    col0, row0 = col * MAXAR_TILE_SIZE, row * MAXAR_TILE_SIZE
    with rasterio.open(scene_dir / "R.tif") as ref:
        w = min(MAXAR_TILE_SIZE, ref.width - col0)
        h = min(MAXAR_TILE_SIZE, ref.height - row0)
        win = Window(col0, row0, w, h)
        aff = window_transform(win, ref.transform)
        crs = ref.crs
    return aff, crs, w, h


def tile_detections(repo, observation_id: str, row: int, col: int) -> dict:
    import pyproj

    obs = _obs_by_id(repo, observation_id)
    scene_dir = Path(obs.dataset_dir)
    aff, crs, w, h = _tile_geo(scene_dir, row, col)
    gj_path = get_settings().data_dir / "detections" / observation_id / "detections.geojson"
    gj = json.loads(gj_path.read_text(encoding="utf-8"))
    tile_id = f"{observation_id}_r{row:03d}_c{col:03d}"
    tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    out = []
    for f in gj["features"]:
        p = f["properties"]
        if p.get("tile_id") != tile_id:
            continue
        ring = f["geometry"]["coordinates"][0]
        polygon_px = []
        for lon, lat in ring[:-1]:                 # drop the closing repeated vertex
            x, y = tr.transform(lon, lat)
            c, r = ~aff @ (x, y)                    # inverse affine: ground -> tile pixel
            polygon_px.append([round(c, 1), round(r, 1)])
        out.append({"class": p["class"], "score": p["score"], "polygon_px": polygon_px,
                    "long_side_px": p.get("long_side_px"), "heading_deg": p.get("heading_deg")})
    out.sort(key=lambda d: -d["score"])
    return {"tile_id": tile_id, "width": w, "height": h, "detections": out}


@functools.lru_cache(maxsize=64)
def _tile_png_cached(scene_dir: str, row: int, col: int) -> bytes:
    import rasterio
    from PIL import Image
    from rasterio.windows import Window

    col0, row0 = col * MAXAR_TILE_SIZE, row * MAXAR_TILE_SIZE
    bands = []
    for b in BAND_ORDER:
        with rasterio.open(Path(scene_dir) / f"{b}.tif") as ds:
            w = min(MAXAR_TILE_SIZE, ds.width - col0)
            h = min(MAXAR_TILE_SIZE, ds.height - row0)
            bands.append(ds.read(1, window=Window(col0, row0, w, h)))
    rgb = np.stack(bands, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def tile_image_png(repo, observation_id: str, row: int, col: int) -> bytes:
    obs = _obs_by_id(repo, observation_id)
    return _tile_png_cached(str(obs.dataset_dir), row, col)

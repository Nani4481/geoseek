"""On-demand BEFORE / AFTER / CHANGE-OVERLAY tiles for a change candidate.

Rendered from files already on disk - the staged Sentinel-2 observations and
the cached span probability raster under ``data/change_model/`` - so this path
is fully offline and needs no model. A small crop window (a few hundred pixels)
is read per call and the result is LRU-cached, keeping the interactive latency
well under the 1 s budget.

Coordinates: a candidate's ``bbox_rc`` is ``[r0, c0, r1, c1]`` in the 2024
reference grid, which every staged observation and the probability raster share
(Phase 3a co-registered them all to it).
"""

from __future__ import annotations

import functools
import io

import numpy as np

from geoseek.change.analyze import DATE_TO_OBS
from geoseek.config import get_settings
from geoseek.ingest.embed import make_true_color_uint8

RGB_BANDS = ("B04", "B03", "B02")
CHANGE_THRESHOLD = 0.80
VALID_DATES = tuple(DATE_TO_OBS)          # ("2019", "2021", "2024")
VALID_VIEWS = ("rgb", "overlay")


def _crop_true_color(obs_id: str, bbox, margin_frac: float, min_px: int):
    """True-colour uint8 crop around a pixel bbox, plus the window used
    (R0, C0, R1, C1) so the overlay can align to it."""
    import rasterio

    r0, c0, r1, c1 = bbox
    h, w = r1 - r0, c1 - c0
    mh = max(int(h * margin_frac), (min_px - h) // 2, 8)
    mw = max(int(w * margin_frac), (min_px - w) // 2, 8)
    d = get_settings().datasets_dir / obs_id
    with rasterio.open(d / "B04.tif") as ds:
        H, W = ds.height, ds.width
    R0, C0 = max(r0 - mh, 0), max(c0 - mw, 0)
    R1, C1 = min(r1 + mh, H), min(c1 + mw, W)
    win = rasterio.windows.Window(C0, R0, C1 - C0, R1 - R0)
    bands = {}
    for b in RGB_BANDS:
        with rasterio.open(d / f"{b}.tif") as ds:
            bands[b] = ds.read(1, window=win)
    rgb = make_true_color_uint8(bands, nodata=0)
    return rgb, (R0, C0, R1, C1)


def _overlay(rgb: np.ndarray, window, bbox, centroid_rc, prob_raster_path: str) -> np.ndarray:
    """Paint the 2019->2024 change mask onto ``rgb``: this candidate's connected
    component outlined red, every other changed pixel in the window tinted yellow."""
    import rasterio
    from scipy import ndimage as ndi

    R0, C0, R1, C1 = window
    with rasterio.open(prob_raster_path) as ds:
        pw = ds.read(1, window=rasterio.windows.Window(C0, R0, C1 - C0, R1 - R0))
    changed = pw >= CHANGE_THRESHOLD
    lab, _ = ndi.label(changed, structure=np.ones((3, 3), bool))
    lr = int(round(centroid_rc[0] - R0))
    lc = int(round(centroid_rc[1] - C0))
    lr = min(max(lr, 0), lab.shape[0] - 1)
    lc = min(max(lc, 0), lab.shape[1] - 1)
    this = lab[lr, lc]
    comp = (lab == this) if this else np.zeros_like(changed)

    out = rgb.astype(np.int16)
    other = changed & ~comp
    out[other] = (out[other] * 0.5 + np.array([255, 230, 0]) * 0.5).astype(np.int16)
    out = np.clip(out, 0, 255).astype(np.uint8)
    edge = comp ^ ndi.binary_erosion(comp)
    out[edge] = (230, 30, 30)
    return out


@functools.lru_cache(maxsize=256)
def _render_cached(candidate_id: str, date: str, view: str, bbox_key: tuple,
                   centroid_key: tuple, prob_raster_path: str) -> bytes:
    from PIL import Image

    obs_id = DATE_TO_OBS[date]
    rgb, window = _crop_true_color(obs_id, bbox_key, margin_frac=1.6, min_px=320)
    if view == "overlay":
        rgb = _overlay(rgb, window, bbox_key, centroid_key, prob_raster_path)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_candidate_imagery(candidate: dict, *, date: str, view: str, prob_raster_path: str) -> bytes:
    if date not in VALID_DATES:
        raise ValueError(f"date must be one of {VALID_DATES}, got {date!r}")
    if view not in VALID_VIEWS:
        raise ValueError(f"view must be one of {VALID_VIEWS}, got {view!r}")
    bbox_key = tuple(int(x) for x in candidate["bbox_rc"])
    centroid_key = tuple(float(x) for x in candidate["centroid_rc"])
    return _render_cached(candidate["candidate_id"], date, view, bbox_key, centroid_key,
                          str(prob_raster_path))

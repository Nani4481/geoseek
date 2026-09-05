"""Phase 7a Step B (PS 2.3): per-tile physical/spectral features for the
retrieval-evaluation judge.

This is the INDEPENDENT signal used to judge retrieval relevance. It never
touches RemoteCLIP, the vanilla CLIP control, or any learned embedding - it is
computed straight from the Sentinel-2 surface-reflectance products already on
disk:

  * the per-date NDVI / NDWI / NDBI index rasters
    (``data/datasets/<scene>/{NDVI,NDWI,NDBI}.tif``, produced in Phase 3a/4),
  * the Scene Classification Layer (``SCL.tif``) for pixel validity,
  * an AOI-wide river/water mask derived by thresholding NDWI (McFeeters
    NDWI > 0) and taking the union across all three dates, so the 2019 drought
    does not shrink the channel; the single largest connected component of that
    union is "the river" (the Saryu / Ghaghara), the whole union is
    "any open water".

For every one of the 3267 Sentinel-2 tiles it writes, to
``data/eval_retrieval/tile_features.json``:

  ndvi_p50/p10/p90, ndwi_p50/p90, ndbi_p50/p90,
  water_frac        - fraction of valid pixels with NDWI > 0
  veg_frac          - NDVI > 0.35
  dense_veg_frac    - NDVI > 0.55
  bare_frac         - NDVI < 0.20 and NDWI < 0 and NDBI > -0.15
  built_frac        - NDBI > -0.05 and NDVI < 0.30 and NDWI < 0 (built-up signature)
  edge_density      - fraction of valid pixels where |Sobel(NDVI)| > 0.10
                      (field boundaries / roads / building edges; low for
                      uniform water, bare ground, closed canopy)
  dist_river_m      - min distance from the tile to the river channel
  dist_water_m      - min distance from the tile to any open water
  valid_frac        - fraction of the tile's pixels that are valid

Usage:  python scripts/eval_retrieval_features.py
"""

from __future__ import annotations

import json

import numpy as np
import rasterio
from scipy import ndimage as ndi

from geoseek.config import get_settings
from geoseek.ingest.tiler import TILE_SIZE, parse_tile_row_col

S2_SCENES = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)
DECIMATE = 4          # water-mask resolution: 8384 -> 2096 px, ~40 m/px
NDWI_WATER = 0.0      # McFeeters NDWI water threshold
EDGE_NDVI_STEP = 0.10  # per-pixel NDVI gradient that counts as a boundary
OUT_DIR_NAME = "eval_retrieval"
# SCL classes that are usable ground (exclude nodata=0, defective=1, cloud
# shadow=3, cloud med/high=8/9, cirrus=10). Keep dark=2 (often water/shadowed
# built), veg=4, bare=5, water=6, unclassified=7, snow/ice=11.
_SCL_VALID = frozenset({2, 4, 5, 6, 7, 11})


def _valid_mask_from_scl(scl: np.ndarray) -> np.ndarray:
    return np.isin(scl, list(_SCL_VALID))


def build_water_masks(datasets_dir) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (dist_river_m_dec, dist_water_m_dec, meta) at 1/DECIMATE res."""
    h_dec = w_dec = 8384 // DECIMATE
    any_water = np.zeros((h_dec, w_dec), dtype=bool)
    per_date_px = {}
    for scene in S2_SCENES:
        with rasterio.open(datasets_dir / scene / "NDWI.tif") as ds:
            ndwi = ds.read(1, out_shape=(h_dec, w_dec))
        w = np.isfinite(ndwi) & (ndwi > NDWI_WATER)
        per_date_px[scene] = int(w.sum())
        any_water |= w

    # drop decimated specks, then take the largest connected component = the river
    any_water = ndi.binary_opening(any_water, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndi.label(any_water, structure=np.ones((3, 3)))
    if n == 0:
        raise SystemExit("no water pixels found in the AOI - check NDWI rasters")
    sizes = ndi.sum(any_water, labels, index=np.arange(1, n + 1))
    river_label = int(np.argmax(sizes)) + 1
    river = labels == river_label

    px_m = 10.0 * DECIMATE
    dist_river = ndi.distance_transform_edt(~river) * px_m
    dist_water = ndi.distance_transform_edt(~any_water) * px_m
    meta = {
        "decimate": DECIMATE,
        "px_m_decimated": px_m,
        "ndwi_water_threshold": NDWI_WATER,
        "per_date_water_px_decimated": per_date_px,
        "union_water_px_decimated": int(any_water.sum()),
        "river_component_px_decimated": int(river.sum()),
        "n_water_components": int(n),
        "river_is_largest_component": True,
    }
    return dist_river.astype(np.float32), dist_water.astype(np.float32), meta


def _sobel_mag(a: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(a, axis=1, mode="nearest")
    gy = ndi.sobel(a, axis=0, mode="nearest")
    # ndi.sobel sums a [1,2,1]*[-1,0,1] kernel -> divide by 8 for the gradient/px
    return np.hypot(gx, gy) / 8.0


def tile_features(ndvi, ndwi, ndbi, valid) -> dict:
    v = valid & np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi)
    n = int(v.sum())
    total = ndvi.size
    if n < 0.05 * total:
        return {"valid_frac": round(n / total, 4), "usable": False}

    dv, dw, db = ndvi[v], ndwi[v], ndbi[v]
    ndvi_filled = np.where(np.isfinite(ndvi), ndvi, np.nanmedian(dv))
    edges = _sobel_mag(ndvi_filled)
    edge_density = float(np.mean(edges[v] > EDGE_NDVI_STEP))

    return {
        "valid_frac": round(n / total, 4),
        "usable": True,
        "ndvi_p50": round(float(np.median(dv)), 4),
        "ndvi_p10": round(float(np.percentile(dv, 10)), 4),
        "ndvi_p90": round(float(np.percentile(dv, 90)), 4),
        "ndwi_p50": round(float(np.median(dw)), 4),
        "ndwi_p90": round(float(np.percentile(dw, 90)), 4),
        "ndbi_p50": round(float(np.median(db)), 4),
        "ndbi_p90": round(float(np.percentile(db, 90)), 4),
        "water_frac": round(float(np.mean(dw > NDWI_WATER)), 4),
        "veg_frac": round(float(np.mean(dv > 0.35)), 4),
        "dense_veg_frac": round(float(np.mean(dv > 0.55)), 4),
        "bare_frac": round(float(np.mean((dv < 0.20) & (dw < 0.0) & (db > -0.15))), 4),
        "built_frac": round(float(np.mean((db > -0.05) & (dv < 0.30) & (dw < 0.0))), 4),
        "edge_density": round(edge_density, 4),
    }


def main() -> None:
    settings = get_settings()
    out_dir = settings.data_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[feat] building AOI river / water masks (union of NDWI>0 across 3 dates) ...")
    dist_river_dec, dist_water_dec, mask_meta = build_water_masks(settings.datasets_dir)
    print(f"[feat]   {mask_meta}")

    features: dict[str, dict] = {}
    for scene in S2_SCENES:
        sd = settings.datasets_dir / scene
        print(f"[feat] {scene}: windowed per-tile features ...")
        with rasterio.open(sd / "NDVI.tif") as d_ndvi, \
             rasterio.open(sd / "NDWI.tif") as d_ndwi, \
             rasterio.open(sd / "NDBI.tif") as d_ndbi, \
             rasterio.open(sd / "SCL.tif") as d_scl:
            H, W = d_ndvi.height, d_ndvi.width
            ny, nx = -(-H // TILE_SIZE), -(-W // TILE_SIZE)
            for r in range(ny):
                for c in range(nx):
                    y0, x0 = r * TILE_SIZE, c * TILE_SIZE
                    y1, x1 = min(y0 + TILE_SIZE, H), min(x0 + TILE_SIZE, W)
                    win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                    ndvi = d_ndvi.read(1, window=win)
                    ndwi = d_ndwi.read(1, window=win)
                    ndbi = d_ndbi.read(1, window=win)
                    scl = d_scl.read(1, window=win)
                    valid = _valid_mask_from_scl(scl)
                    feat = tile_features(ndvi, ndwi, ndbi, valid)

                    dy0, dx0 = y0 // DECIMATE, x0 // DECIMATE
                    dy1, dx1 = max(dy0 + 1, y1 // DECIMATE), max(dx0 + 1, x1 // DECIMATE)
                    feat["dist_river_m"] = round(float(dist_river_dec[dy0:dy1, dx0:dx1].min()), 1)
                    feat["dist_water_m"] = round(float(dist_water_dec[dy0:dy1, dx0:dx1].min()), 1)

                    tile_id = f"{scene}_r{r:03d}_c{c:03d}"
                    features[tile_id] = feat

    payload = {
        "n_tiles": len(features),
        "params": {
            "tile_size_px": TILE_SIZE,
            "edge_ndvi_step": EDGE_NDVI_STEP,
            "scl_valid_classes": sorted(_SCL_VALID),
        },
        "water_mask": mask_meta,
        "features": features,
    }
    out_path = out_dir / "tile_features.json"
    out_path.write_text(json.dumps(payload, indent=1))
    usable = sum(1 for f in features.values() if f.get("usable"))
    print(f"[feat] wrote {out_path}  ({len(features)} tiles, {usable} usable)")
    # sanity peek
    ex = next(iter(features.values()))
    print(f"[feat] example feature keys: {sorted(ex)}")


if __name__ == "__main__":
    main()

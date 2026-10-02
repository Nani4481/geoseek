"""Real-data GeoTIFF fixtures for the browser raster views, plus the server-side (numpy / rasterio) reference numbers.

The files are windows cut from the archive's own staged Sentinel-2 L2A scenes (Ayodhya, EPSG:32644, 10 m), so the browser is exercised
on genuine imagery in a genuine projected CRS. The reference numbers are computed here with numpy and GDAL, independently of the
TypeScript that is being verified.

    python tests/raster_fixtures.py OUT_DIR      # writes the .tif files and OUT_DIR/reference.json

``build(out_dir)`` returns the same dict, for pytest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "data" / "datasets"
SCENES = {"2019-03-30": "S2B_44RPQ_20190330_1_L2A_scaled", "2024-03-08": "S2A_44RPQ_20240308_0_L2A_scaled"}
BANDS4 = ("B02", "B03", "B04", "B08")                    # blue, green, red, NIR
WIN = 800                                                # a window of 800 x 800 px = 8 km; <= the browser's native-decode budget, so no resampling
# window origins (col, row) inside the 8384 px scene: the second file overlaps the first by 600 x 600 px but is not the same grid
ORIGINS = {"2019-03-30": (3000, 3000), "2024-03-08": (3200, 3200)}


def available() -> bool:
    return all((DATASETS / s / "B04.tif").is_file() for s in SCENES.values())


def _write(path: Path, scene: str, bands: tuple[str, ...], origin: tuple[int, int], acquisition: str | None, descriptions: bool = False):
    import rasterio
    from rasterio.windows import Window

    c0, r0 = origin
    arrs = []
    for b in bands:
        with rasterio.open(DATASETS / scene / f"{b}.tif") as ds:
            win = Window(c0, r0, WIN, WIN)
            arrs.append(ds.read(1, window=win))
            transform, crs, nodata = ds.window_transform(win), ds.crs, ds.nodata
    with rasterio.open(path, "w", driver="GTiff", width=WIN, height=WIN, count=len(bands), dtype="uint16", crs=crs, transform=transform,
                       nodata=nodata, tiled=True, blockxsize=256, blockysize=256, compress="deflate") as out:
        for i, a in enumerate(arrs, 1):
            out.write(a, i)
            if descriptions:
                out.set_band_description(i, bands[i - 1])
        if acquisition:
            out.update_tags(ACQUISITION_DATE=f"{acquisition}T05:21:00Z")
    return arrs


def _stats(a: np.ndarray, nodata) -> dict:
    v = a.astype("float64").ravel()
    v = v[np.isfinite(v)]
    if nodata is not None:
        v = v[v != nodata]
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "p2": float(np.percentile(v, 2)), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "p98": float(np.percentile(v, 98))}


def _nd(a: np.ndarray, b: np.ndarray, nodata) -> np.ndarray:
    a, b = a.astype("float64"), b.astype("float64")
    ok = (a + b) != 0
    if nodata is not None:
        ok &= (a != nodata) & (b != nodata)
    out = np.full(a.shape, np.nan)
    out[ok] = (a[ok] - b[ok]) / (a[ok] + b[ok])
    return out


def build(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref: dict = {"files": {}}
    data = {}
    for date, scene in SCENES.items():
        p = out_dir / f"ayodhya_{date}_4band.tif"
        arrs = _write(p, scene, BANDS4, ORIGINS[date], date, descriptions=False)
        data[date] = arrs
        nodata = 0
        ref["files"][date] = {
            "path": str(p), "nodata": nodata,
            "bands": {b: _stats(a, nodata) for b, a in zip(BANDS4, arrs)},
            "indices": {"ndvi": _stats(_nd(arrs[3], arrs[2], nodata), None), "ndwi": _stats(_nd(arrs[1], arrs[3], nodata), None)},
            # a stretched-composite sample: the 2..98 percentile of each band, applied to B04/B03/B02, at a few pixels
            "composite_samples": [],
        }
        stre = [(np.percentile(a[a != nodata].astype("float64"), 2), np.percentile(a[a != nodata].astype("float64"), 98)) for a in arrs]
        for (r, c) in ((10, 10), (400, 400), (790, 100), (123, 456)):
            vals = [arrs[2][r, c], arrs[1][r, c], arrs[0][r, c]]
            idx = [2, 1, 0]
            if any(v == nodata for v in vals):
                continue
            rgb = [int(np.round(np.clip((float(v) - stre[i][0]) / (stre[i][1] - stre[i][0]), 0, 1) * 255)) for v, i in zip(vals, idx)]
            ref["files"][date]["composite_samples"].append({"row": r, "col": c, "rgb": rgb})
    # a 3-band and a 1-band file for the band-count defaults
    p3 = out_dir / "ayodhya_2024-03-08_3band_rgb.tif"
    _write(p3, SCENES["2024-03-08"], ("B04", "B03", "B02"), ORIGINS["2024-03-08"], None, descriptions=True)
    p1 = out_dir / "ayodhya_2019-03-30_1band_nir.tif"
    _write(p1, SCENES["2019-03-30"], ("B08",), ORIGINS["2019-03-30"], None)
    pz = out_dir / "ayodhya_zstd.tif"                                    # a ZSTD-compressed file: its decoder is WebAssembly, which the console's CSP refuses
    import rasterio

    with rasterio.open(p1) as src, rasterio.open(pz, "w", **{**src.profile, "compress": "zstd"}) as dst:
        dst.write(src.read())
    ref["files"]["zstd"] = {"path": str(pz)}
    ref["files"]["3band"] = {"path": str(p3)}
    ref["files"]["1band"] = {"path": str(p1)}
    ref["profile"] = profile_reference({d: ref["files"][d]["path"] for d in SCENES})
    ref["window_px"] = WIN
    ref["origins"] = {k: list(v) for k, v in ORIGINS.items()}
    (out_dir / "reference.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


# --------------------------------------------------------------------------- spectral profile reference (independent of the TypeScript)

R_MERC = 6378137.0
WORLD = 2 * np.pi * R_MERC


def _merc(lon, lat):
    return R_MERC * np.radians(lon), R_MERC * np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85.05112878, 85.05112878)) / 2))


def _tiles(box, z):
    n = 2 ** z
    res = WORLD / (256 * n)
    mx0, my0 = _merc(box[0], box[1])
    mx1, my1 = _merc(box[2], box[3])
    px = lambda mx: (mx + WORLD / 2) / res
    py = lambda my: (WORLD / 2 - my) / res
    x0, x1 = int(np.floor(px(mx0) / 256)), int(np.floor(px(mx1) / 256))
    y0, y1 = int(np.floor(py(my1) / 256)), int(np.floor(py(my0) / 256))
    return max(0, x0), max(0, y0), min(n - 1, x1), min(n - 1, y1), res


def profile_reference(paths: dict[str, str], max_tiles: int = 36) -> dict:
    """The spectral profile of two dated 4-band files over their shared ground, computed with numpy / pyproj / rasterio only:
    footprint overlap -> Web-Mercator grid -> exact nearest-pixel sampling -> per-band and NDVI / NDWI statistics inside the box."""
    import pyproj
    import rasterio

    infos = {}
    for date, path in paths.items():
        with rasterio.open(path) as ds:
            b = ds.bounds
            t = pyproj.Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            xs, ys = zip(*[t.transform(x, y) for x, y in ((b.left, b.bottom), (b.right, b.bottom), (b.right, b.top), (b.left, b.top))])
            infos[date] = {"box": (min(xs), min(ys), max(xs), max(ys)), "res": ds.res[0], "crs": ds.crs, "tf": ds.transform, "arr": ds.read().astype("float64"), "nodata": ds.nodata}
    boxes = [i["box"] for i in infos.values()]
    box = (max(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), min(b[3] for b in boxes))
    lat = (box[1] + box[3]) / 2
    finest = min(i["res"] for i in infos.values())
    want = int(np.ceil(np.log2(40075016.686 / (256 * (finest / np.cos(np.radians(lat)))))))
    top = max(6, min(17, want))
    z = 2
    for zz in range(top, 1, -1):
        x0, y0, x1, y1, _ = _tiles(box, zz)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles:
            z = zz
            break
    x0, y0, x1, y1, res = _tiles(box, z)
    w, h = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    gx0, gy0 = x0 * 256 * res - WORLD / 2, WORLD / 2 - y0 * 256 * res
    jj, ii = np.mgrid[0:h, 0:w]
    mx, my = gx0 + (ii + 0.5) * res, gy0 - (jj + 0.5) * res
    lon, la = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(mx, my)
    inbox = (lon >= box[0]) & (lon <= box[2]) & (la >= box[1]) & (la <= box[3])
    out = {"z": z, "grid": {"w": w, "h": h, "x0": gx0, "y0": gy0, "res": res}, "cells": int(inbox.sum()), "box": list(box), "files": {}}
    for date, i in infos.items():
        x, y = pyproj.Transformer.from_crs("EPSG:3857", i["crs"], always_xy=True).transform(mx, my)
        a, b_, c, d, e, f = i["tf"].a, i["tf"].b, i["tf"].c, i["tf"].d, i["tf"].e, i["tf"].f
        det = a * e - b_ * d
        col = np.floor((e * (x - c) - b_ * (y - f)) / det).astype(int)
        row = np.floor((-d * (x - c) + a * (y - f)) / det).astype(int)
        ok = (col >= 0) & (row >= 0) & (col < i["arr"].shape[2]) & (row < i["arr"].shape[1])
        bands = []
        for k in range(i["arr"].shape[0]):
            g = np.full((h, w), np.nan)
            g[ok] = i["arr"][k][row[ok], col[ok]]
            g[g == i["nodata"]] = np.nan
            bands.append(g)
        met = {name: _stats(g[inbox], None) for name, g in zip(("blue", "green", "red", "nir"), bands)}
        blue, green, red, nir = bands
        for name, num, den in (("ndvi", nir, red), ("ndwi", green, nir)):
            met[name] = _stats(((num - den) / (num + den))[inbox & np.isfinite(num) & np.isfinite(den) & ((num + den) != 0)], None)
        out["files"][date] = met
    return out


if __name__ == "__main__":
    if not available():
        sys.exit("staged Sentinel-2 scenes are not on this machine")
    r = build(Path(sys.argv[1]))
    print(json.dumps({k: v["path"] for k, v in r["files"].items()}))

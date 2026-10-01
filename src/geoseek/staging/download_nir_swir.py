"""Stage B08 (NIR, 10 m) and B11 (SWIR-1, 20 m) for Sentinel-2 scenes that already have B02/B03/B04/SCL on disk.

Staging module: the only kind of geoseek code allowed to touch the network. It adds files; it never rewrites an
existing band and never touches the catalog.

* **Same grid, by construction.** The new bands are cropped onto the existing ``B04.tif`` grid of the SAME scene
  directory (B08: integer-aligned window of the 10 m COG, no resampling - alignment is asserted; B11: bilinear 20 m ->
  10 m onto that grid, as for the Ayodhya extra bands). Tile geometry and tile ids come from the catalog and are
  unaffected.
* **Incremental and resumable.** A band that already exists on the correct grid is skipped (nothing is re-downloaded).
  Downloads go to ``<tmp_root>/<scene>/<band>.part`` and resume with an HTTP ``Range`` request after an interruption;
  a finished file is size-checked against ``Content-Length``. Outputs are written to a temp name and renamed, so a crash
  never leaves a half-written band that looks complete.
* **Provenance** (SHA256 of the source COG and of the written crop, source URL, bytes, resampling, licence) is
  returned per band; the caller records it in the manifest.

Source: Earth Search STAC v1 -> the public AWS Sentinel-2 L2A COG bucket (same as ``scripts/stage_diverse_aois.py``).
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

STAC_ROOT = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-l2a"
ASSET_KEY = {"B08": "nir", "B11": "swir16"}
RESAMPLING_NOTE = {"B08": "native 10 m grid: integer-aligned window crop, no resampling",
                   "B11": "20 m -> 10 m bilinear (WarpedVRT) onto the scene's B04 grid"}
DATA_LICENSE = "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)"
BANDS = ("B08", "B11")
_ALIGN_TOL = 1e-3


class StagingError(RuntimeError):
    pass


def fetch_item(scene_id: str, *, retries: int = 4, session: requests.Session | None = None) -> dict:
    """The STAC item whose id is the scene id (diverse scenes were staged from STAC item ids)."""
    http = session or requests
    url = f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items/{scene_id}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = http.get(url, timeout=30)
            if r.status_code == 404:
                raise StagingError(f"STAC has no item {scene_id!r}")
            r.raise_for_status()
            return r.json()
        except StagingError:
            raise
        except Exception as e:                    # network blip: back off and retry
            last = e
            time.sleep(2 ** attempt)
    raise StagingError(f"STAC item {scene_id} unreachable after {retries} attempts: {last}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def download_resumable(url: str, dest: Path, *, chunk: int = 4 << 20, retries: int = 6,
                       log: Callable[[str], None] | None = None) -> dict:
    """Download ``url`` to ``dest`` (final name), resuming a ``dest.part`` left by an interrupted run.

    Returns ``{"bytes", "sha256", "seconds", "resumed_from"}``. The file is renamed into place only when its size equals
    the server's ``Content-Length``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    t0, resumed_from = time.time(), 0
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have > total:                                    # corrupt/foreign partial: start over
            part.unlink()
            have = 0
        if have == total:
            break
        if attempt == 0:
            resumed_from = have
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    r.raise_for_status()
                if have and r.status_code == 200:           # server ignored Range: restart cleanly
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    for block in r.iter_content(chunk_size=chunk):
                        if block:
                            f.write(block)
        except (requests.RequestException, OSError) as e:
            if log:
                log(f"      download interrupted ({type(e).__name__}); retry {attempt + 1}/{retries} from byte {part.stat().st_size if part.exists() else 0}")
            time.sleep(min(30, 2 ** attempt))
    size = part.stat().st_size if part.exists() else 0
    if size != total:
        raise StagingError(f"{dest.name}: got {size} of {total} bytes after {retries} attempts (partial kept for resume)")
    part.replace(dest)
    return {"bytes": total, "sha256": sha256_file(dest), "seconds": time.time() - t0, "resumed_from": resumed_from}


def band_on_reference_grid(path: Path, ref_path: Path) -> bool:
    """True if ``path`` exists, opens, and sits exactly on ``ref_path``'s grid (shape + transform + CRS)."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as ds, rasterio.open(ref_path) as ref:
            return (ds.width, ds.height) == (ref.width, ref.height) and ds.transform.almost_equals(ref.transform) \
                and ds.crs == ref.crs
    except rasterio.errors.RasterioIOError:
        return False


def crop_to_reference(full_path: Path, ref_path: Path, band: str, out_path: Path) -> None:
    """Write ``out_path``: ``band`` cut from the full-tile COG onto the reference crop's exact grid."""
    with rasterio.open(ref_path) as ref, rasterio.open(full_path) as src:
        if src.crs != ref.crs:
            raise StagingError(f"{band}: CRS {src.crs} differs from the reference crop's {ref.crs}")
        if band == "B11":
            with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width, height=ref.height,
                           resampling=Resampling.bilinear) as vrt:
                data, nodata = vrt.read(1), vrt.nodata
        else:
            win = src.window(*ref.bounds)
            offs = (win.col_off, win.row_off, win.width, win.height)
            if any(abs(v - round(v)) > _ALIGN_TOL for v in offs):
                raise StagingError(f"{band}: crop window {offs} is not pixel-aligned to the source grid")
            col_off, row_off, w, h = (int(round(v)) for v in offs)
            if (w, h) != (ref.width, ref.height) or col_off < 0 or row_off < 0 \
                    or col_off + w > src.width or row_off + h > src.height:
                raise StagingError(f"{band}: reference crop ({ref.width}x{ref.height}) does not fit the source "
                                   f"({src.width}x{src.height}) at offset ({col_off},{row_off})")
            data, nodata = src.read(1, window=Window(col_off, row_off, w, h)), src.nodata
        profile = {"driver": "GTiff", "height": ref.height, "width": ref.width, "count": 1, "dtype": data.dtype,
                   "crs": ref.crs, "transform": ref.transform, "nodata": nodata, "compress": "deflate"}
    tmp = out_path.with_name(out_path.stem + ".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(out_path)


def ensure_bands(scene_id: str, scene_dir: Path, tmp_root: Path, *, log: Callable[[str], None] = print) -> dict:
    """Make sure ``B08.tif`` and ``B11.tif`` exist on the scene's grid. Returns per-band provenance (+ whether staged now)."""
    ref_path = scene_dir / "B04.tif"
    if not ref_path.is_file():
        raise StagingError(f"{scene_dir} has no B04.tif to define the grid")
    todo = [b for b in BANDS if not band_on_reference_grid(scene_dir / f"{b}.tif", ref_path)]
    out: dict = {}
    for b in BANDS:
        if b not in todo:
            out[b] = {"staged_now": False, "path": str(scene_dir / f"{b}.tif"),
                      "sha256": sha256_file(scene_dir / f"{b}.tif"), "bytes": (scene_dir / f"{b}.tif").stat().st_size}
    if not todo:
        return out
    item = fetch_item(scene_id)
    work = tmp_root / scene_id
    work.mkdir(parents=True, exist_ok=True)
    urls = {b: item["assets"][ASSET_KEY[b]]["href"] for b in todo}
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        futures = {b: pool.submit(download_resumable, urls[b], work / f"{b}_full.tif", log=log) for b in todo}
        downloads = {b: f.result() for b, f in futures.items()}
    for b in todo:
        dl = downloads[b]
        out_path = scene_dir / f"{b}.tif"
        crop_to_reference(work / f"{b}_full.tif", ref_path, b, out_path)
        (work / f"{b}_full.tif").unlink()
        out[b] = {"staged_now": True, "path": str(out_path), "sha256": sha256_file(out_path), "bytes": out_path.stat().st_size,
                  "source_url": urls[b], "source_bytes": dl["bytes"], "source_sha256": dl["sha256"],
                  "download_seconds": round(dl["seconds"], 1), "resumed_from_byte": dl["resumed_from"],
                  "resampling": RESAMPLING_NOTE[b], "license": DATA_LICENSE,
                  "stac_item": f"{STAC_ROOT}/collections/{STAC_COLLECTION}/items/{scene_id}",
                  "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        work.rmdir()
    except OSError:
        pass
    return out

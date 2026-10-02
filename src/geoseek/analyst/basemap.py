"""A local raster basemap for the console's maps, rendered from imagery the archive already holds.

Nothing is fetched and nothing is invented. A slippy-map tile (z/x/y, Web-Mercator) is assembled at request time from

  * ``sentinel-2`` (default): one acquisition per MGRS granule - the clearest one, i.e. lowest mean tile cloud fraction -
    so a map is a mosaic of real imagery and not a patchwork of dates. At z >= OVERVIEW_MAX_ZOOM + 1 it is assembled from
    the catalog's own true-colour tile thumbnails (the images the search cards show). At lower zooms (a view spanning
    hundreds of kilometres) that would mean thousands of thumbnails, so each granule is read once from the same band
    rasters at a reduced size and held in a byte-capped in-memory LRU; the tile is warped from that. Three reduced levels,
    each about as coarse as the map pixel it serves: "mid" 1/8 (z 11; a 4 x 4 grid of real pixels averaged per 8 x 8 block),
    "fine" 1/16 (z 8-10; 4 x 4 per 16 x 16) and "coarse" 1/64 (z <= 7, whole-archive views; 8 x 8 per 64 x 64). Only the
    sampled rows are inflated, so a first view stays quick.
    Same real pixels, only reduced; nothing is written to disk;
  * ``scene=<observation id>``: a staged Maxar scene (R/G/B COG bands), for the Detect map - the Sentinel-2 archive has no
    coverage where those scenes lie, and the scene itself is the imagery the detections were found on.

Where nothing is staged the tile is fully transparent, so the console's dark canvas shows through: unstaged areas stay dark
and are never filled. ``coverage`` states what the layer covers (dates, scenes, fraction of a view) so the caption is honest.

Read-only. Rendered tiles are LRU-cached in memory; the catalog is reached only through the MetadataRepository seam.
"""

from __future__ import annotations

import io
import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TILE_PX = 256
ORIGIN = 20037508.342789244          # half the Web-Mercator world, metres
S2_COLLECTION = "sentinel-2-l2a"
MAXAR_COLLECTION = "maxar-opendata"
S2_NATIVE_MAX_ZOOM = 14              # 10 m/px; deeper zooms are an honest upscale of the same pixels
MAXAR_NATIVE_MAX_ZOOM = 19
OVERVIEW_MAX_ZOOM = 11               # z <= this is warped from the per-granule overviews; above it, from the tile thumbnails
COARSE_MAX_ZOOM = 7                  # z <= this: coarse level; FINE_MAX_ZOOM >= z > this: fine level; above that, up to OVERVIEW_MAX_ZOOM: mid
FINE_MAX_ZOOM = 10
# level -> (linear reduction of a 10 m band, real pixels sampled per output-pixel edge). Measured mean absolute error of the
# reduced image against the full-area average of the same band: mid ~1.7% (k=4), fine ~2.1% (k=4), coarse ~1.7% (k=8) - see
# docs/FRONTEND_REACT.md. Resident per granule: mid ~6.4 MB, fine ~1.6 MB, coarse ~0.1 MB; all three levels for all 15 granules ~122 MB < the cap.
OVERVIEW_LEVELS = {"coarse": (64, 8), "fine": (16, 4), "mid": (8, 4)}
OVERVIEW_CAP_BYTES = 160 * 1024 * 1024   # hard RAM cap for decoded overviews (LRU); ~1.4 MB each, so ~110 granules
TILE_CACHE_CAP_BYTES = 96 * 1024 * 1024  # hard RAM cap for rendered map tiles (LRU)
MAX_COMPOSE_TILES = 1500             # a request touching more catalog tiles than this is answered "too coarse", not slowly
_GRANULE_RE = re.compile(r"_(\d{2}[A-Z]{3})_")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="basemap")
_band_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="basemap-band")      # a separate pool: band reads run inside _pool workers


class BasemapError(ValueError):
    pass


# --------------------------------------------------------------------------- web-mercator maths


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2 ** z
    size = 2 * ORIGIN / n
    return (-ORIGIN + x * size, ORIGIN - (y + 1) * size, -ORIGIN + (x + 1) * size, ORIGIN - y * size)


def merc_to_lonlat(mx: float, my: float) -> tuple[float, float]:
    lon = mx / ORIGIN * 180.0
    lat = math.degrees(2 * math.atan(math.exp(my / ORIGIN * math.pi)) - math.pi / 2)
    return lon, lat


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.0511287798), -85.0511287798)
    return lon / 180.0 * ORIGIN, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * ORIGIN / math.pi


def tile_bbox_lonlat(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    w, s, e, n = tile_bounds_3857(z, x, y)
    (lo0, la0), (lo1, la1) = merc_to_lonlat(w, s), merc_to_lonlat(e, n)
    return lo0, la0, lo1, la1


def validate_zxy(z: int, x: int, y: int) -> None:
    if not 0 <= z <= 22 or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
        raise BasemapError(f"tile {z}/{x}/{y} is outside the world grid")


# --------------------------------------------------------------------------- PNG / JPEG output

class ByteLRU:
    """Least-recently-used cache with a hard cap on resident bytes (not entry count)."""

    def __init__(self, cap_bytes: int):
        self.cap = cap_bytes
        self._d: "OrderedDict[object, tuple[object, int]]" = OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.peak_bytes = 0
        self.evictions = 0

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                return None
            self._d.move_to_end(key)
            return v[0]

    def put(self, key, value, size: int):
        if size > self.cap:
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self._d[key] = (value, size)
            self.bytes += size
            while self.bytes > self.cap and self._d:
                _, (_, sz) = self._d.popitem(last=False)
                self.bytes -= sz
                self.evictions += 1
            self.peak_bytes = max(self.peak_bytes, self.bytes)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._d), "bytes": self.bytes, "peak_bytes": self.peak_bytes, "cap_bytes": self.cap, "evictions": self.evictions}

    def bytes_by(self, group) -> dict:
        """Resident bytes grouped by ``group(key)`` (e.g. the overview level)."""
        out: dict = {}
        with self._lock:
            for k, (_, sz) in self._d.items():
                out[group(k)] = out.get(group(k), 0) + sz
        return out


_tile_cache = ByteLRU(TILE_CACHE_CAP_BYTES)
_overview_cache = ByteLRU(OVERVIEW_CAP_BYTES)


def cache_stats() -> dict:
    ov = _overview_cache.stats()
    ov["bytes_by_level"] = _overview_cache.bytes_by(lambda k: k[1])
    return {"tiles": _tile_cache.stats(), "overviews": ov}


def _empty_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


EMPTY_TILE = _empty_png()


def _encode(rgb: np.ndarray, alpha: np.ndarray) -> tuple[bytes, str]:
    """Opaque tiles go out as JPEG (small, fast); partly covered ones as RGBA PNG so uncovered pixels stay transparent."""
    from PIL import Image

    buf = io.BytesIO()
    if alpha.min() == 255:
        Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=84)
        return buf.getvalue(), "image/jpeg"
    Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), "image/png"


# --------------------------------------------------------------------------- Sentinel-2 archive summary


class _Archive:
    """One pass over the catalog: per Sentinel-2 observation its date, mean tile cloud fraction, granule and footprint hull."""

    def __init__(self, repo):
        from shapely.geometry import MultiPoint

        acc: dict[str, dict] = {}
        for r in repo.iter_tile_records():
            if r.collection_id != S2_COLLECTION:
                continue
            a = acc.get(r.observation_id)
            if a is None:
                m = _GRANULE_RE.search(r.observation_id)
                a = acc[r.observation_id] = {"observation_id": r.observation_id, "date": r.acq_date, "granule": m.group(1) if m else r.observation_id,
                                              "sensor": r.sensor, "platform": r.platform, "cloud_sum": 0.0, "n": 0, "pts": []}
            a["cloud_sum"] += float(r.cloud_fraction or 0.0)
            a["n"] += 1
            # a few tiles' corners are enough for a convex hull of the granule; parsing every footprint would be wasteful
            if a["n"] % 37 == 1 or a["n"] < 40:
                nums = _NUM_RE.findall(r.geom_wkt_4326)
                a["pts"].extend((float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2))
        self.obs: dict[str, dict] = {}
        for oid, a in acc.items():
            pts = a.pop("pts")
            a["mean_cloud"] = a.pop("cloud_sum") / max(a["n"], 1)
            a["hull"] = MultiPoint(pts).convex_hull if len(pts) >= 3 else None
            self.obs[oid] = a
        self._preferred: dict[str | None, dict[str, str]] = {}

    def preferred(self, year: str | None) -> dict[str, str]:
        """granule -> chosen observation_id: the lowest mean cloud (newest on a tie), restricted to ``year`` if given."""
        if year in self._preferred:
            return self._preferred[year]
        best: dict[str, dict] = {}
        for a in self.obs.values():
            if year and not str(a["date"]).startswith(year):
                continue
            cur = best.get(a["granule"])
            if cur is None or (a["mean_cloud"], -_datekey(a["date"])) < (cur["mean_cloud"], -_datekey(cur["date"])):
                best[a["granule"]] = a
        out = {g: a["observation_id"] for g, a in best.items()}
        self._preferred[year] = out
        return out


def _datekey(d) -> int:
    try:
        return int(str(d).replace("-", "")[:8])
    except ValueError:
        return 0


_archive_cache: dict = {"repo": None, "value": None}
_archive_lock = threading.Lock()


def archive(repo) -> _Archive:
    with _archive_lock:
        if _archive_cache["repo"] is not repo or _archive_cache["value"] is None:
            _archive_cache.update(repo=repo, value=_Archive(repo))
        return _archive_cache["value"]


# --------------------------------------------------------------------------- Sentinel-2 composite


def _corners(wkt: str):
    """(NW, NE, SW) lon/lat of a tile footprint, whatever the polygon's vertex order."""
    nums = [float(v) for v in _NUM_RE.findall(wkt)]
    pts = list(dict.fromkeys((nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)))
    if len(pts) < 4:
        return None
    by_lat = sorted(pts, key=lambda p: -p[1])
    top, bot = sorted(by_lat[:2]), sorted(by_lat[-2:])
    return top[0], top[1], bot[0]


def _paste_thumbnail(rgb: np.ndarray, alpha: np.ndarray, jpeg: bytes, corners, bounds) -> bool:
    from PIL import Image

    mx0, my0, mx1, my1 = bounds
    res = (mx1 - mx0) / TILE_PX
    px = [((m[0] - mx0) / res, (my1 - m[1]) / res) for m in (lonlat_to_merc(*c) for c in corners)]
    (x_nw, y_nw), (x_ne, y_ne), (x_sw, y_sw) = px
    xs = [x_nw, x_ne, x_sw, x_ne + x_sw - x_nw]
    ys = [y_nw, y_ne, y_sw, y_ne + y_sw - y_nw]
    if max(xs) <= 0 or min(xs) >= TILE_PX or max(ys) <= 0 or min(ys) >= TILE_PX:
        return False
    img = Image.open(io.BytesIO(jpeg))
    side = math.hypot(x_ne - x_nw, y_ne - y_nw)                       # on-screen size of the tile's top edge, px
    if side < img.width / 2:                                           # decode already reduced (JPEG scale 1/2, 1/4, 1/8)
        img.draft("RGB", (max(1, int(side * 1.5)), max(1, int(side * 1.5))))
    img = img.convert("RGB")
    w, h = img.size
    # thumbnail pixel (u, v) -> canvas: NW + u/w * (NE-NW) + v/h * (SW-NW); PIL wants the inverse, canvas -> thumbnail
    m = np.array([[(x_ne - x_nw) / w, (x_sw - x_nw) / h], [(y_ne - y_nw) / w, (y_sw - y_nw) / h]])
    det = np.linalg.det(m)
    if abs(det) < 1e-12:
        return False
    inv = np.linalg.inv(m)
    off = -inv @ np.array([x_nw, y_nw])
    coeff = (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1])
    out = img.transform((TILE_PX, TILE_PX), Image.AFFINE, coeff, resample=Image.BILINEAR)
    mask = Image.new("L", (w, h), 255).transform((TILE_PX, TILE_PX), Image.AFFINE, coeff, resample=Image.NEAREST)
    m_np = np.asarray(mask) > 0
    if not m_np.any():
        return False
    rgb[m_np] = np.asarray(out)[m_np]
    alpha[m_np] = 255
    return True


def _select_records(repo, bbox, year):
    arch = archive(repo)
    pref = arch.preferred(year)
    recs = repo.query_tiles(bbox=bbox, collection=S2_COLLECTION)
    keep = [r for r in recs if pref.get(arch.obs.get(r.observation_id, {}).get("granule")) == r.observation_id]
    keep.sort(key=lambda r: (-float(r.cloud_fraction or 0.0), _datekey(r.acq_date)))      # clearest drawn last, on top
    return keep


def render_sentinel2(repo, thumbnail_fn, z: int, x: int, y: int, year: str | None = None) -> tuple[bytes, str, dict]:
    t0 = time.perf_counter()
    bounds = tile_bounds_3857(z, x, y)
    bbox = tile_bbox_lonlat(z, x, y)
    recs = _select_records(repo, bbox, year)
    meta = {"layer": "sentinel-2", "catalog_tiles": len(recs), "composited": 0}
    if not recs:
        meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage"}
    if len(recs) > MAX_COMPOSE_TILES:
        meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return EMPTY_TILE, "image/png", {**meta, "status": "too-coarse"}

    def fetch(r):
        try:
            return r, thumbnail_fn(r.tile_id)
        except KeyError:                       # band files not on this machine -> nothing to draw (never faked)
            return r, None

    rgb = np.zeros((TILE_PX, TILE_PX, 3), np.uint8)
    alpha = np.zeros((TILE_PX, TILE_PX), np.uint8)
    drawn = 0
    for r, jpg in _pool.map(fetch, recs):      # map keeps submission order, so the draw order (clearest on top) holds
        if jpg is None:
            continue
        c = _corners(r.geom_wkt_4326)
        if c and _paste_thumbnail(rgb, alpha, jpg, c, bounds):
            drawn += 1
    meta["composited"] = drawn
    meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if drawn == 0:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage"}
    body, mime = _encode(rgb, alpha)
    return body, mime, {**meta, "status": "ok"}



# --------------------------------------------------------------------------- Sentinel-2 per-granule overviews (low zoom)

_inflight: dict[tuple, threading.Lock] = {}
_inflight_lock = threading.Lock()


def _decode_overview(repo, datasets_dir: Path, observation_id: str, level: str = "fine") -> dict | None:
    """One observation's true-colour image at the level's reduction, from the same band rasters the thumbnails come from and with
    the same per-observation display stretch. Returns None if the bands are not on this machine."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    from geoseek.ingest.embed import make_true_color_uint8, true_color_bounds_for_scene, true_color_offsets_for_scene

    obs = repo.get_observation(observation_id)
    if obs is None:
        return None
    scene = repo.get_scene(obs.scene_id)
    scene_key = scene.scene_id if scene is not None else obs.scene_id
    d = Path(datasets_dir) / (obs.dataset_dir or obs.observation_id)
    paths = {b: d / f"{b}.tif" for b in ("B04", "B03", "B02")}
    if not all(p.is_file() for p in paths.values()):
        return None
    factor, k = OVERVIEW_LEVELS[level]

    def read_band(item):
        b, path = item
        with rasterio.open(path) as ds:
            h, w = max(1, ds.height // factor), max(1, ds.width // factor)
            # The bands are one-row-per-block deflate: an area-average read must inflate every row of a ~170 MB file (measured
            # 1.1-1.9 s per band with a warm OS cache, minutes for a cold whole-archive view). Nearest decimation inflates only
            # the sampled rows. So read every (factor/k)-th row and column - real pixels, no interpolation - and average k x k
            # blocks of those samples down to 1/factor (error vs the full-area average: see OVERVIEW_LEVELS).
            sub = ds.read(1, out_shape=(h * k, w * k), resampling=Resampling.nearest)
            return b, np.rint(sub.reshape(h, k, w, k).mean(axis=(1, 3))).astype(sub.dtype), (ds.nodata, ds.crs, ds.transform, ds.width, ds.height)

    got = list(_band_pool.map(read_band, paths.items()))          # the three bands are independent: read them together
    bands = {b: arr for b, arr, _ in got}
    nodata, crs, transform, full_w, full_h = got[0][2]
    rgb = make_true_color_uint8(bands, nodata=nodata, per_band_offset_dn=true_color_offsets_for_scene(scene_key),
                                per_band_bounds_dn=true_color_bounds_for_scene(scene_key))
    valid = ((bands["B04"] > 0) | (bands["B03"] > 0) | (bands["B02"] > 0)).astype(np.uint8) * 255
    t = transform @ Affine.scale(full_w / rgb.shape[1], full_h / rgb.shape[0])
    return {"rgb": np.ascontiguousarray(np.moveaxis(rgb, -1, 0)), "valid": valid, "transform": t, "crs": crs,
            "bytes": int(rgb.nbytes + valid.nbytes)}


def overview(repo, datasets_dir: Path, observation_id: str, level: str = "fine") -> dict | None:
    """Lazy decode on first use, warm thereafter; concurrent callers for one (observation, level) share a single decode."""
    key = (observation_id, level)
    hit = _overview_cache.get(key)
    if hit is not None:
        return hit if hit != "absent" else None
    with _inflight_lock:
        lock = _inflight.setdefault(key, threading.Lock())
    with lock:
        hit = _overview_cache.get(key)
        if hit is not None:
            return hit if hit != "absent" else None
        ov = _decode_overview(repo, datasets_dir, observation_id, level)
        _overview_cache.put(key, ov if ov is not None else "absent", ov["bytes"] if ov else 1)
        return ov


def render_sentinel2_overview(repo, datasets_dir: Path, z: int, x: int, y: int, year: str | None = None) -> tuple[bytes, str, dict]:
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import box

    t0 = time.perf_counter()
    bounds = tile_bounds_3857(z, x, y)
    bbox = tile_bbox_lonlat(z, x, y)
    arch = archive(repo)
    view = box(*bbox)
    ids = [oid for oid in arch.preferred(year).values() if arch.obs[oid]["hull"] is not None and arch.obs[oid]["hull"].intersects(view)]
    level = "coarse" if z <= COARSE_MAX_ZOOM else "fine" if z <= FINE_MAX_ZOOM else "mid"
    factor = OVERVIEW_LEVELS[level][0]
    meta = {"layer": "sentinel-2-overview", "level": level, "granules": len(ids), "composited": 0}
    if not ids:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": round((time.perf_counter() - t0) * 1000, 1)}
    ids.sort(key=lambda o: (-arch.obs[o]["mean_cloud"], _datekey(arch.obs[o]["date"])))      # clearest drawn last, on top
    ovs = list(_pool.map(lambda o: overview(repo, datasets_dir, o, level), ids))

    dst_t = Affine((bounds[2] - bounds[0]) / TILE_PX, 0, bounds[0], 0, -(bounds[3] - bounds[1]) / TILE_PX, bounds[3])
    m_per_px = (bounds[2] - bounds[0]) / TILE_PX * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    method = Resampling.average if m_per_px > 2 * 10.0 * factor else Resampling.bilinear      # no aliasing when zoomed far out
    rgb = np.zeros((TILE_PX, TILE_PX, 3), np.uint8)
    alpha = np.zeros((TILE_PX, TILE_PX), np.uint8)
    drawn = 0
    for ov in ovs:
        if ov is None:
            continue
        part = np.zeros((3, TILE_PX, TILE_PX), np.uint8)
        cover = np.zeros((TILE_PX, TILE_PX), np.uint8)
        for i in range(3):
            reproject(ov["rgb"][i], part[i], src_transform=ov["transform"], src_crs=ov["crs"], dst_transform=dst_t, dst_crs="EPSG:3857",
                      resampling=method, src_nodata=0, dst_nodata=0)
        reproject(ov["valid"], cover, src_transform=ov["transform"], src_crs=ov["crs"], dst_transform=dst_t, dst_crs="EPSG:3857",
                  resampling=method)
        m = (cover >= 128) & part.any(axis=0)          # a pixel is imagery only if at least half of it is, so edges are not darkened
        if m.any():
            rgb[m] = np.moveaxis(part, 0, -1)[m]
            alpha[m] = 255
            drawn += 1
    meta["composited"] = drawn
    meta["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if drawn == 0:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage"}
    body, mime = _encode(rgb, alpha)
    return body, mime, {**meta, "status": "ok"}


# --------------------------------------------------------------------------- Maxar scene


_scene_lock = threading.Lock()
_scene_cache: "OrderedDict[str, dict]" = OrderedDict()
_OVERVIEW_FACTOR = 8


def _scene_info(repo, scene: str) -> dict:
    """Validate ``scene`` against the catalog (never a caller-supplied path) and load its raster geometry + a reduced copy."""
    with _scene_lock:
        hit = _scene_cache.get(scene)
        if hit is not None:
            _scene_cache.move_to_end(scene)
            return hit
    import rasterio
    from rasterio.warp import transform_bounds

    obs = repo.get_observation(scene)
    if obs is None:
        raise BasemapError(f"no observation {scene!r}")
    sc = repo.get_scene(obs.scene_id)
    coll = repo.get_collection(sc.collection_id) if sc else None
    if coll is None or coll.collection_id != MAXAR_COLLECTION:
        raise BasemapError(f"{scene!r} is not a Maxar scene")
    d = Path(obs.dataset_dir or "")
    if not all((d / f"{b}.tif").is_file() for b in "RGB"):
        raise BasemapError(f"imagery for {scene!r} is not staged on this machine")
    with rasterio.open(d / "R.tif") as ds:
        crs, transform, h, w = ds.crs, ds.transform, ds.height, ds.width
        bounds = ds.bounds
    info = {"dir": str(d), "crs": crs, "transform": transform, "h": h, "w": w, "bounds": bounds,
            "bbox": transform_bounds(crs, "EPSG:4326", *bounds), "date": obs.acquired_at, "overview": None,
            "platform": sc.platform if sc else None, "sensor": coll.sensor, "gsd": coll.native_gsd_m}
    with _scene_lock:
        _scene_cache[scene] = info
        while len(_scene_cache) > 6:
            _scene_cache.popitem(last=False)
    return info


def _scene_overview(info: dict):
    """One reduced RGB copy of the whole scene (1/8 linear), built once and shared: low-zoom tiles read this, not 900 MB of
    strips. The three bands are read in parallel, and concurrent tile requests wait for the one decode instead of repeating it."""
    with _scene_lock:
        if info["overview"] is not None:
            return info["overview"]
        lock = info.setdefault("_decode_lock", threading.Lock())
    with lock:
        if info["overview"] is not None:
            return info["overview"]
        import rasterio
        from rasterio.transform import Affine

        h, w = info["h"] // _OVERVIEW_FACTOR, info["w"] // _OVERVIEW_FACTOR

        def read(b):
            with rasterio.open(Path(info["dir"]) / f"{b}.tif") as ds:
                return ds.read(1, out_shape=(h, w), resampling=rasterio.enums.Resampling.average)

        bands = list(_pool.map(read, "RGB"))
        t = info["transform"]
        ov = {"data": np.stack(bands), "transform": t @ Affine.scale(info["w"] / w, info["h"] / h)}
        with _scene_lock:
            info["overview"] = ov
        return ov


def render_scene(repo, z: int, x: int, y: int, scene: str) -> tuple[bytes, str, dict]:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window, from_bounds
    from rasterio.windows import intersection as win_intersection

    t0 = time.perf_counter()
    info = _scene_info(repo, scene)
    bounds = tile_bounds_3857(z, x, y)
    w_, s_, e_, n_ = tile_bbox_lonlat(z, x, y)
    sw, ss, se, sn = info["bbox"]
    meta = {"layer": "maxar-scene", "scene": scene}
    if e_ < sw or w_ > se or n_ < ss or s_ > sn:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": 0.0}

    from rasterio.warp import transform_bounds

    ub = transform_bounds("EPSG:3857", info["crs"], *bounds, densify_pts=21)
    m_per_px = (bounds[2] - bounds[0]) / TILE_PX * math.cos(math.radians((s_ + n_) / 2))
    use_overview = m_per_px >= abs(info["transform"].a) * _OVERVIEW_FACTOR * 0.9
    if use_overview:
        ov = _scene_overview(info)
        src, t = ov["data"], ov["transform"]
        win = from_bounds(*ub, transform=t).round_offsets().round_lengths()
        if not _overlaps(win, src.shape[2], src.shape[1]):
            return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": 0.0}
        win = win_intersection(win, Window(0, 0, src.shape[2], src.shape[1]))
        r0, c0, hh, ww = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
        arr = src[:, r0:r0 + hh, c0:c0 + ww]
        src_t = t @ Affine.translation(c0, r0)
    else:
        win = from_bounds(*ub, transform=info["transform"]).round_offsets().round_lengths()
        if not _overlaps(win, info["w"], info["h"]):
            return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": 0.0}
        win = win_intersection(win, Window(0, 0, info["w"], info["h"]))
        r0, c0, hh, ww = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
        step = max(1, int(m_per_px / abs(info["transform"].a)) // 2)               # decimate reads to ~2x the output density
        oh, ow = max(1, hh // step), max(1, ww // step)
        planes = []
        for b in "RGB":
            with rasterio.open(Path(info["dir"]) / f"{b}.tif") as ds:
                planes.append(ds.read(1, window=win, out_shape=(oh, ow)))
        arr = np.stack(planes)
        src_t = info["transform"] @ Affine.translation(c0, r0) @ Affine.scale(ww / ow, hh / oh)
    if arr.size == 0:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": 0.0}

    dst_t = Affine((bounds[2] - bounds[0]) / TILE_PX, 0, bounds[0], 0, -(bounds[3] - bounds[1]) / TILE_PX, bounds[3])
    out = np.zeros((3, TILE_PX, TILE_PX), np.uint8)
    cover = np.zeros((TILE_PX, TILE_PX), np.uint8)
    ones = np.full(arr.shape[1:], 255, np.uint8)
    for i in range(3):
        reproject(arr[i], out[i], src_transform=src_t, src_crs=info["crs"], dst_transform=dst_t, dst_crs="EPSG:3857", resampling=Resampling.bilinear)
    reproject(ones, cover, src_transform=src_t, src_crs=info["crs"], dst_transform=dst_t, dst_crs="EPSG:3857", resampling=Resampling.nearest)
    if cover.max() == 0:
        return EMPTY_TILE, "image/png", {**meta, "status": "no-coverage", "ms": round((time.perf_counter() - t0) * 1000, 1)}
    body, mime = _encode(np.moveaxis(out, 0, -1).copy(), cover)
    return body, mime, {**meta, "status": "ok", "ms": round((time.perf_counter() - t0) * 1000, 1), "source": "overview" if use_overview else "full-resolution"}


def _overlaps(win, w: int, h: int) -> bool:
    return win.col_off < w and win.row_off < h and win.col_off + win.width > 0 and win.row_off + win.height > 0 and win.width > 0 and win.height > 0


# --------------------------------------------------------------------------- public entry points


def render_tile(repo, thumbnail_fn, z: int, x: int, y: int, year: str | None = None, scene: str | None = None,
                datasets_dir: Path | None = None) -> tuple[bytes, str, dict]:
    validate_zxy(z, x, y)
    if year is not None and not re.fullmatch(r"\d{4}", year):
        raise BasemapError("year must be four digits")
    key = (z, x, y, year, scene)
    hit = _tile_cache.get(key)
    if hit is not None:
        return hit[0], hit[1], {**hit[2], "cached": True}
    if scene:
        res = render_scene(repo, z, x, y, scene)
    elif z <= OVERVIEW_MAX_ZOOM and datasets_dir is not None:
        res = render_sentinel2_overview(repo, datasets_dir, z, x, y, year)
    else:
        res = render_sentinel2(repo, thumbnail_fn, z, x, y, year)
    if res[2].get("status") != "too-coarse":
        _tile_cache.put(key, res, len(res[0]) + 256)
    return res


def coverage(repo, bbox: tuple[float, float, float, float], year: str | None = None, scene: str | None = None) -> dict:
    """What the basemap holds for a view: which scenes/dates, how much of the view they cover, and the native zoom range."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    w, s, e, n = bbox
    view = box(w, s, e, n)
    if scene:
        info = _scene_info(repo, scene)
        fp = box(*info["bbox"])
        return {"layer": "maxar-scene", "available": fp.intersects(view), "fraction": round(fp.intersection(view).area / view.area, 4),
                "scenes": [{"observation_id": scene, "date": info["date"], "platform": info["platform"], "sensor": info["sensor"], "bbox": list(info["bbox"])}],
                "dates": [info["date"]], "native_max_zoom": MAXAR_NATIVE_MAX_ZOOM, "gsd_m": info["gsd"],
                "source": "staged Maxar Open Data scene bands (R/G/B)"}
    arch = archive(repo)
    pref = set(arch.preferred(year).values())
    hit = [a for oid, a in arch.obs.items() if oid in pref and a["hull"] is not None and a["hull"].intersects(view)]
    union = unary_union([a["hull"] for a in hit]) if hit else None
    frac = float(union.intersection(view).area / view.area) if union is not None else 0.0
    # every acquisition whose granule touches the view, whatever the year filter (so a client can pick the nearest date and then ask for that year)
    acquisitions = sorted(({"date": str(a["date"]), "observation_id": a["observation_id"], "platform": a["platform"], "sensor": a["sensor"],
                            "mean_cloud": round(a["mean_cloud"], 4)} for a in arch.obs.values() if a["hull"] is not None and a["hull"].intersects(view)),
                          key=lambda r: (r["date"], r["observation_id"]))
    return {"layer": "sentinel-2", "acquisitions": acquisitions, "available": bool(hit), "fraction": round(min(frac, 1.0), 4),
            "scenes": [{"observation_id": a["observation_id"], "date": a["date"], "platform": a["platform"], "sensor": a["sensor"],
                        "mean_cloud": round(a["mean_cloud"], 4), "bbox": list(a["hull"].bounds)} for a in sorted(hit, key=lambda a: a["observation_id"])],
            "dates": sorted({str(a["date"]) for a in hit}), "native_max_zoom": S2_NATIVE_MAX_ZOOM, "gsd_m": 10.0,
            "year": year, "overview_max_zoom": OVERVIEW_MAX_ZOOM, "coarse_max_zoom": COARSE_MAX_ZOOM, "fine_max_zoom": FINE_MAX_ZOOM,
            "selection_rule": "one representative acquisition per granule: the lowest mean tile cloud fraction (newest on a tie)",
            "source": "catalog Sentinel-2 L2A true-colour imagery (tile thumbnails at z >= 12; reduced 1/8, 1/16 and 1/64 overviews of the same bands below)"}

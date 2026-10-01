"""SearchEngine: text-to-image and image-to-image semantic search over the catalog.

The embedding model and the vector index are both loaded ONCE (at
``SearchEngine()`` construction) and kept resident for the process lifetime -
every query reuses them. Ranking scans the *entire* flat index (a few thousand
vectors, a sub-millisecond brute-force inner-product scan) and applies filters
as a post-filter on the joined catalog metadata, per spec.

This module touches neither sqlite3 nor faiss directly: metadata goes through
:class:`geoseek.catalog.repository.MetadataRepository`, vectors through
:class:`geoseek.vectorindex.VectorIndex`, embeddings through
:class:`geoseek.models.EmbeddingModel`.
"""

from __future__ import annotations

import functools
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely import wkt as shapely_wkt

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.ingest.embed import (
    make_true_color_uint8,
    true_color_bounds_for_scene,
    true_color_offsets_for_scene,
)
from geoseek.ingest.store import DB_FILENAME, INDEX_FILENAME
from geoseek.ingest.tiler import read_tile_window
from geoseek.models import RemoteCLIPEmbeddingModel
from geoseek.vectorindex import FaissFlatIPIndex

RGB_BANDS = ("B04", "B03", "B02")


THUMBNAIL_MEDIA_TYPE = "image/jpeg"


@functools.lru_cache(maxsize=4096)
def _render_thumbnail_bytes(scene_dir: str, row: int, col: int, scene_key: str) -> bytes:
    """One tile's cross-date-harmonized true-color thumbnail (JPEG q85, ~20 kB).

    Cached (>= the full 3 267-tile catalog): the queue, detail and discovery
    views all revisit the same tiles, and 30 result cards per search would
    otherwise re-read 3 COG windows each (~55 ms/tile) on every view. JPEG (not
    PNG) keeps each thumbnail ~10x smaller on the wire and in the cache."""
    from PIL import Image

    bands, nodata = read_tile_window(Path(scene_dir), row, col, list(RGB_BANDS))
    rgb_uint8 = make_true_color_uint8(
        bands, nodata=nodata,
        per_band_offset_dn=true_color_offsets_for_scene(scene_key),
        per_band_bounds_dn=true_color_bounds_for_scene(scene_key),
    )
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


@dataclass
class SearchResult:
    tile_id: str
    score: float
    lon: float
    lat: float
    acq_date: str
    sensor: str
    cloud_fraction: float
    scene_id: str

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "score": self.score,
            "lon": self.lon,
            "lat": self.lat,
            "acq_date": self.acq_date,
            "sensor": self.sensor,
            "cloud_fraction": self.cloud_fraction,
            "scene_id": self.scene_id,
        }


@dataclass
class SearchFilters:
    bbox: tuple[float, float, float, float] | None = None  # (west, south, east, north)
    date_start: str | None = None  # inclusive, "YYYY-MM-DD"
    date_end: str | None = None  # inclusive, "YYYY-MM-DD"
    sensor: str | None = None
    max_cloud_fraction: float | None = None

    def matches(self, lon: float, lat: float, acq_date: str, sensor: str, cloud_fraction: float) -> bool:
        if self.bbox is not None:
            west, south, east, north = self.bbox
            if not (west <= lon <= east and south <= lat <= north):
                return False
        if self.date_start is not None and acq_date < self.date_start:
            return False
        if self.date_end is not None and acq_date > self.date_end:
            return False
        if self.sensor is not None and sensor != self.sensor:
            return False
        if self.max_cloud_fraction is not None and cloud_fraction > self.max_cloud_fraction:
            return False
        return True


class SearchEngine:
    """Construct once (e.g. at API startup) and reuse for every query."""

    def __init__(self, index_dir: Path | None = None):
        self.settings = get_settings()
        self.index_dir = index_dir or self.settings.index_dir

        # SearchEngine backing a server is queried from a threadpool (FastAPI's
        # sync-endpoint executor). The metadata repository is internally locked;
        # this lock guards the compound path against a concurrent refresh()
        # swapping self._rows.
        self._db_lock = threading.Lock()

        self.embedding_model = RemoteCLIPEmbeddingModel()
        self.embedding_model.load()  # RemoteCLIP resident before the first query is served
        self.vector_index = FaissFlatIPIndex(
            self.index_dir / INDEX_FILENAME if index_dir is not None else self.settings.faiss_index_path)
        self.repo = SQLiteMetadataRepository(
            self.index_dir / DB_FILENAME if index_dir is not None else self.settings.database_path)
        self._rows: dict[int, dict] = {}
        # ALL model encodes run on this ONE thread. FastAPI serves sync endpoints
        # from a threadpool; the first GPU op on each fresh worker thread pays a
        # ~1 s cuBLAS/cuDNN handle init, so a burst of first queries would each
        # be slow. Pinning every encode to a single pre-warmed thread makes every
        # query pay the warm ~8 ms instead.
        self._encode_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geoseek-encode")
        self.refresh()
        self._prewarm()
        print(f"[search] SearchEngine ready: {self.vector_index.count()} vectors, "
              f"{len(self._rows)} tile rows.")

    KEEPWARM_INTERVAL_S = 20.0

    def _prewarm(self) -> None:
        """Pay the one-time cost of the first text/image encode (BPE vocab load
        + first CUDA kernel launch, ~300 ms) and the first thumbnail render (PIL
        + libjpeg init) at startup, then keep the GPU encoder hot with a light
        periodic ping - otherwise, after the GPU goes idle for tens of seconds,
        the NEXT user search pays a ~1 s CUDA wake-up + kernel re-autotune.

        No bulk background thumbnail pre-render: the LRU + JPEG + the browser's
        concurrent fetch already put a cold search view's 30 cards well under a
        second, and a 3 000-tile loop would just contend for the GIL with the
        first minutes of live requests."""
        t0 = time.time()
        try:
            self._encode_text("warm up the text tower")
            self._encode_image(np.zeros((32, 32, 3), dtype=np.uint8))
            first = next(iter(self._rows.values()), None)
            if first is not None:
                self.get_tile_thumbnail_png(first["tile_id"])
            print(f"[search] encoders + thumbnail path pre-warmed in "
                  f"{(time.time() - t0) * 1000:.0f} ms")
        except Exception as e:  # never let a warm-up failure block startup
            print(f"[search] pre-warm skipped: {e}")
            return

        self._keepwarm_stop = threading.Event()

        def _keepwarm():
            while not self._keepwarm_stop.wait(self.KEEPWARM_INTERVAL_S):
                try:
                    self._encode_text("keepwarm")  # keeps the encode thread's GPU handles hot
                except Exception:
                    pass

        self._keepwarm_thread = threading.Thread(target=_keepwarm, name="encoder-keepwarm", daemon=True)
        self._keepwarm_thread.start()

    # -- all model encodes funnel through the single pre-warmed thread --------

    def _encode_text(self, query: str) -> np.ndarray:
        return self._encode_pool.submit(self.embedding_model.encode_text, query).result()

    def _encode_image(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        return self._encode_pool.submit(self.embedding_model.encode_image, image_rgb_uint8).result()

    def refresh(self) -> None:
        """(Re)load the faiss_id -> metadata map from the catalog. Call again after new ingests."""
        rows: dict[int, dict] = {}
        for rec in self.repo.iter_tile_records():
            if rec.faiss_id is None:
                continue
            centroid = shapely_wkt.loads(rec.geom_wkt_4326).centroid
            rows[int(rec.faiss_id)] = {
                "tile_id": rec.tile_id,
                # the flat table's `scene_id` was the AOI-clip id == observation_id
                # in the new model; kept under the same key for result compatibility.
                "scene_id": rec.observation_id,
                "observation_id": rec.observation_id,
                "sensor": rec.platform,
                "acq_date": rec.acq_date,
                "lon": centroid.x,
                "lat": centroid.y,
                "cloud_fraction": rec.cloud_fraction,
            }
        with self._db_lock:
            self._rows = rows

    def _rank_and_filter(self, query_vec: np.ndarray, k: int, filters: SearchFilters | None) -> list[SearchResult]:
        n = self.vector_index.count()
        if n == 0:
            return []
        scores, ids = self.vector_index.search(query_vec, n)

        results: list[SearchResult] = []
        for score, faiss_id in zip(scores, ids):
            if faiss_id < 0:
                continue
            row = self._rows.get(int(faiss_id))
            if row is None:
                continue
            if filters is not None and not filters.matches(
                row["lon"], row["lat"], row["acq_date"], row["sensor"], row["cloud_fraction"]
            ):
                continue
            results.append(
                SearchResult(
                    tile_id=row["tile_id"], score=float(score), lon=row["lon"], lat=row["lat"],
                    acq_date=row["acq_date"], sensor=row["sensor"], cloud_fraction=row["cloud_fraction"],
                    scene_id=row["scene_id"],
                )
            )
            if len(results) >= k:
                break
        return results

    def search_text(
        self, query: str, k: int = 10, filters: SearchFilters | None = None
    ) -> tuple[list[SearchResult], float]:
        """Text tower -> unit-norm vector -> vector search -> filtered, joined results."""
        t0 = time.time()
        vec = self._encode_text(query)
        results = self._rank_and_filter(vec, k, filters)
        latency_ms = (time.time() - t0) * 1000.0
        return results, latency_ms

    def search_image(
        self,
        tile_id: str | None = None,
        image_rgb_uint8: np.ndarray | None = None,
        k: int = 10,
        filters: SearchFilters | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Image-to-image ("find more like this"): from an existing tile_id, or a fresh RGB image."""
        if (tile_id is None) == (image_rgb_uint8 is None):
            raise ValueError("search_image needs exactly one of tile_id or image_rgb_uint8")

        t0 = time.time()
        if tile_id is not None:
            tile = self.repo.get_tile(tile_id)
            if tile is None or tile.faiss_id is None:
                raise KeyError(f"tile_id '{tile_id}' not found in the store")
            vec = self.vector_index.get_vector(tile.faiss_id)
        else:
            vec = self._encode_image(image_rgb_uint8)

        results = self._rank_and_filter(vec, k, filters)
        latency_ms = (time.time() - t0) * 1000.0
        return results, latency_ms

    def get_tile_thumbnail_png(self, tile_id: str) -> bytes:
        """The stretched, cross-date-harmonized true-color PNG for one tile
        (not stored on disk; rendered on demand, then LRU-cached)."""
        tile = self.repo.get_tile(tile_id)
        if tile is None:
            raise KeyError(f"tile_id '{tile_id}' not found in the store")
        obs = self.repo.get_observation(tile.observation_id)
        if obs is None:
            raise KeyError(f"tile_id '{tile_id}' -> observation '{tile.observation_id}' missing")
        scene = self.repo.get_scene(obs.scene_id)
        scene_key = scene.scene_id if scene is not None else obs.scene_id  # keys the radiometry config

        scene_dir = self.settings.datasets_dir / (obs.dataset_dir or obs.observation_id)
        # A catalogued tile whose band rasters are not on this machine (e.g. a Maxar scene whose COGs were never
        # staged) is "not found", not a server fault: KeyError is what the API maps to 404. Only absence is
        # translated - a file that exists but cannot be read still surfaces as the 500 it is.
        missing = [b for b in RGB_BANDS if not (scene_dir / f"{b}.tif").is_file()]
        if missing:
            raise KeyError(f"imagery for tile '{tile_id}' is not staged on this machine "
                           f"(missing {', '.join(missing)} under {scene_dir.name})")
        return _render_thumbnail_bytes(str(scene_dir), tile.row, tile.col, scene_key)

    def count(self) -> int:
        """Number of indexed vectors (== catalog tiles with an embedding)."""
        return self.vector_index.count()

    def close(self) -> None:
        stop = getattr(self, "_keepwarm_stop", None)
        if stop is not None:
            stop.set()
        pool = getattr(self, "_encode_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
        self.repo.close()

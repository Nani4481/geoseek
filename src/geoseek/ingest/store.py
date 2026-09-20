"""Tile store: FAISS vectors + the catalog metadata, persisted to data/index/.

``TileStore`` is a thin coordinator over the two seams:

  * the metadata catalog  -> :class:`geoseek.catalog.repository.MetadataRepository`
    (SQLite impl; the only sqlite3 in geoseek), reachable as ``store.repo``;
  * the vector index      -> :class:`geoseek.vectorindex.VectorIndex`
    (FAISS ``IndexFlatIP`` impl; the only faiss in geoseek), reachable as
    ``store.vector_index``. Each tile is keyed to its vector by ``faiss_id``
    (== insertion position).

Incremental by construction: ``TileStore()`` loads whatever already exists on
disk; ``add_tiles`` only ever appends vectors and inserts catalog rows - it
never rebuilds the index or rewrites existing rows/vectors.

``add_tiles`` still accepts the legacy per-tile record dict
(``tile_id, scene_id, sensor, acq_date, geom_wkt_4326, cloud_fraction,
embedding, processing_history``); it derives and upserts the
collection / scene / observation those tiles belong to, then inserts the
tiles under that observation. The record's ``scene_id`` is the AOI-clip id,
which is the *observation* id in the scene -> observation -> tile model; the
source scene id is that with the AOI suffix stripped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from geoseek.catalog.entities import Collection, Observation, Scene, Tile
from geoseek.catalog.naming import base_scene_id, scene_source_url
from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository
from geoseek.config import get_settings
from geoseek.ingest.tiler import parse_tile_row_col
from geoseek.vectorindex import FaissFlatIPIndex

INDEX_FILENAME = "tiles.faiss"
DB_FILENAME = "tiles.sqlite"

_SENSOR_TO_COLLECTION = {
    "Sentinel-2A": "sentinel-2-l2a",
    "Sentinel-2B": "sentinel-2-l2a",
    "Sentinel-2C": "sentinel-2-l2a",   # launched 2024-09; Phase 8 stages a 2026 scene from it
    "Sentinel-2D": "sentinel-2-l2a",
}
_DEFAULT_BANDS = ("B04", "B03", "B02", "SCL")


def _envelope_wkt(geoms: list[str]) -> str:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for g in geoms:
        x0, y0, x1, y1 = shapely_wkt.loads(g).bounds
        minx, miny, maxx, maxy = min(minx, x0), min(miny, y0), max(maxx, x1), max(maxy, y1)
    return box(minx, miny, maxx, maxy).wkt


class TileStore:
    def __init__(self, index_dir: Path | None = None):
        settings = get_settings()
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / INDEX_FILENAME
        self.db_path = self.index_dir / DB_FILENAME

        self.vector_index = FaissFlatIPIndex(self.index_path)
        self.repo = SQLiteMetadataRepository(self.db_path)

    # -- catalog registration ------------------------------------------------

    def _register_chain(
        self,
        observation_id: str,
        platform: str,
        acq_date: str,
        geoms: list[str],
        cloud_fractions: list[float],
        *,
        dataset_dir: str | None,
        crs: str | None,
        bands: tuple[str, ...] | None,
        source_url: str | None,
        checksums: dict | None,
        aoi_name: str | None,
        collection_id: str | None = None,
        collection_sensor: str | None = None,
        collection_platform: str | None = None,
        collection_native_gsd_m: float | None = None,
        collection_description: str | None = None,
        collection_metadata: dict | None = None,
        scene_license: str | None = None,
        obs_metadata: dict | None = None,
    ) -> None:
        """Register the collection/scene/observation chain for a batch of tiles.

        Defaults to the original Sentinel-2 L2A assumptions (this project's
        first and still primary collection) when the ``collection_*`` overrides
        are not given, so every pre-Phase-8-Step-A caller is unaffected. A
        caller staging a new collection (e.g. ``maxar-opendata``) passes its
        own collection_id/sensor/platform/native_gsd_m/license explicitly
        instead of going through the Sentinel-2 defaults.
        """
        scene_id_base = base_scene_id(observation_id)
        collection_id = collection_id or _SENSOR_TO_COLLECTION.get(platform, "sentinel-2-l2a")
        footprint = _envelope_wkt(geoms) if geoms else "POLYGON EMPTY"

        self.repo.register_collection(Collection(
            collection_id=collection_id,
            sensor=collection_sensor or "MSI",
            platform=collection_platform or "Sentinel-2",
            bands=tuple(bands) if bands else _DEFAULT_BANDS,
            native_gsd_m=collection_native_gsd_m if collection_native_gsd_m is not None else 10.0,
            description=collection_description or "Sentinel-2 L2A surface reflectance COGs.",
            metadata=collection_metadata or {},
        ))
        if self.repo.get_scene(scene_id_base) is None:
            self.repo.register_scene(Scene(
                scene_id=scene_id_base, collection_id=collection_id, platform=platform,
                acquired_at=acq_date, footprint_wkt_4326=footprint, processing_baseline=None,
                source_url=source_url or scene_source_url(scene_id_base),
                license=scene_license or "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)",
                crs=crs, checksums=checksums or {},
                metadata={"footprint_source": "observation extent"},
            ))
        n = len(cloud_fractions)
        s = sorted(cloud_fractions)
        median = (s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])) if n else None
        self.repo.register_observation(Observation(
            observation_id=observation_id, scene_id=scene_id_base, acquired_at=acq_date,
            footprint_wkt_4326=footprint, aoi_name=aoi_name, dataset_dir=dataset_dir or observation_id,
            quality_summary={
                "n_tiles": n,
                "cloud_fraction_mean": round(sum(cloud_fractions) / n, 6) if n else None,
                "cloud_fraction_median": round(median, 6) if median is not None else None,
                "cloud_fraction_max": round(max(cloud_fractions), 6) if n else None,
                "n_clear_tiles_cf_le_0p05": sum(1 for cf in cloud_fractions if cf <= 0.05),
            },
            metadata=obs_metadata or {},
        ))

    def add_tiles(
        self,
        records: list[dict],
        *,
        dataset_dir: str | None = None,
        crs: str | None = None,
        bands: tuple[str, ...] | None = None,
        source_url: str | None = None,
        checksums: dict | None = None,
        aoi_name: str | None = None,
        indices_ref: str | None = None,
        collection_id: str | None = None,
        collection_sensor: str | None = None,
        collection_platform: str | None = None,
        collection_native_gsd_m: float | None = None,
        collection_description: str | None = None,
        collection_metadata: dict | None = None,
        scene_license: str | None = None,
        obs_metadata: dict | None = None,
    ) -> list[int]:
        """Append new tile vectors + catalog rows. Never touches existing vectors/rows.

        Each record needs: tile_id, scene_id, sensor, acq_date, geom_wkt_4326,
        cloud_fraction, embedding (np.float32[EMBEDDING_DIM]), processing_history.

        The ``collection_*``/``scene_license``/``obs_metadata`` overrides let a
        caller register tiles under a NEW collection (different sensor/native
        GSD/license) instead of the Sentinel-2 L2A defaults - e.g. Maxar Open
        Data (sub-metre VHR). ``obs_metadata`` is the place to record the exact
        per-observation GSD (:class:`~geoseek.temporal.matcher.TemporalObservationMatcher`
        prefers it over the collection-wide value, so a mixed-GSD collection
        still gates correctly pair-by-pair).

        Returns the faiss_id assigned to each new record, in order.
        """
        if not records:
            return []

        vectors = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in records])
        faiss_ids = self.vector_index.add(vectors)  # append-only; prior vectors untouched

        # group by observation (== record["scene_id"]) and register the chain
        by_obs: dict[str, list[dict]] = {}
        for r in records:
            by_obs.setdefault(r["scene_id"], []).append(r)
        for observation_id, obs_recs in by_obs.items():
            self._register_chain(
                observation_id=observation_id,
                platform=obs_recs[0].get("sensor", "unknown"),
                acq_date=obs_recs[0].get("acq_date", "unknown"),
                geoms=[r["geom_wkt_4326"] for r in obs_recs],
                cloud_fractions=[float(r["cloud_fraction"]) for r in obs_recs],
                dataset_dir=dataset_dir, crs=crs, bands=bands, source_url=source_url,
                checksums=checksums, aoi_name=aoi_name,
                collection_id=collection_id, collection_sensor=collection_sensor,
                collection_platform=collection_platform,
                collection_native_gsd_m=collection_native_gsd_m,
                collection_description=collection_description,
                collection_metadata=collection_metadata,
                scene_license=scene_license, obs_metadata=obs_metadata,
            )

        tiles: list[Tile] = []
        for r, faiss_id in zip(records, faiss_ids):
            try:
                row_i, col_i = parse_tile_row_col(r["tile_id"])
            except ValueError:
                row_i, col_i = -1, -1
            tiles.append(Tile(
                tile_id=r["tile_id"], observation_id=r["scene_id"], row=row_i, col=col_i,
                geom_wkt_4326=r["geom_wkt_4326"], cloud_fraction=float(r["cloud_fraction"]),
                faiss_id=faiss_id, embedding_ref=INDEX_FILENAME, indices_ref=indices_ref,
                processing_history=r["processing_history"],
            ))
        self.repo.add_tiles(tiles)
        return faiss_ids

    # -- vector access -----------------------------------------------------------

    def reconstruct(self, faiss_id: int) -> np.ndarray:
        """Fetch back the stored vector at a given faiss_id (for byte-identity checks)."""
        return self.vector_index.get_vector(faiss_id)

    def save(self) -> Path:
        return self.vector_index.persist()

    def count(self) -> int:
        return self.vector_index.count()

    def row_count(self) -> int:
        return self.repo.count_tiles()

    def close(self) -> None:
        self.repo.close()

    def __enter__(self) -> "TileStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

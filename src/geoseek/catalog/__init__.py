"""geoseek catalog: the scene -> observation -> tile geospatial/metadata model.

This package owns the metadata catalog (what was ingested, from which source
scene, over which AOI, at what quality) and the ONLY code in geoseek that
opens a ``sqlite3`` connection. Application code talks to it through
:class:`geoseek.catalog.repository.MetadataRepository` (Step 2); the concrete
store is :class:`geoseek.catalog.sqlite_repository.SQLiteMetadataRepository`,
built to be swapped for a PostGIS implementation without touching callers.

Hierarchy (every tile reaches its source scene by foreign key)::

    collections   sensor / platform / bands / native GSD
      scenes      source product id, footprint, acquisition datetime,
                  processing baseline, source URL, license, checksums
        observations   a scene's intersection with our AOI at one
                       acquisition time: quality summary, radiometry /
                       normalization params used, co-registration status
          tiles        256x256, geom_4326, row/col, cloud_fraction,
                       quality flags, embedding ref, indices ref
    derived       per-tile / per-observation products (NDVI/NDWI/NDBI,
                  masks, thumbnails) - referenced by path, never blobbed
"""

from geoseek.catalog.entities import (
    Collection,
    DerivedProduct,
    Observation,
    Scene,
    Tile,
    TileProvenance,
    TileRecord,
)

__all__ = [
    "Collection",
    "Scene",
    "Observation",
    "Tile",
    "DerivedProduct",
    "TileProvenance",
    "TileRecord",
]

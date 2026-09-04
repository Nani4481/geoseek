"""Phase 6 - the analyst interface.

A read/review layer over the Phase 4/5 change pipeline output:

  service.py   AnalystService - candidates queue, full evidence + provenance,
               analyst decisions (through the append-only audit seam), export.
  imagery.py   before / after / change-overlay PNG tiles rendered on demand
               from the cached probability raster + the staged observations.
  geo.py       raster-bbox <-> EPSG:4326 helpers (candidate footprints for the map).
  api.py       FastAPI routes; mounted onto the existing geoseek.search.api app.
  web/         the offline single-page frontend (no CDN, no web map tiles).

Nothing here opens sqlite3 / faiss directly: catalog reads go through the
MetadataRepository seam on the shared SearchEngine, vector search through the
SearchEngine itself.
"""

"""faiss_id <-> tile_id mapping for a RE-EMBEDDED index, kept in its own SQLite file.

The production catalog (``tiles.sqlite``) pins ``tiles.faiss_id`` to the production index and is
never touched by a re-embed. A candidate index built by ``scripts/reembed.py`` therefore ships its
own small mapping database next to it: row ``new_id`` of the candidate FAISS file is tile
``tile_id`` (and was ``source_faiss_id`` in the production index). Promoting a candidate later is
then an explicit, auditable step instead of an in-place mutation.

Part of the catalog package so ``sqlite3`` stays inside the seam; it does not touch the
production schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA = """
CREATE TABLE faiss_map (
    new_id           INTEGER PRIMARY KEY,
    tile_id          TEXT NOT NULL UNIQUE,
    source_faiss_id  INTEGER,
    observation_id   TEXT NOT NULL
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class MappingError(RuntimeError):
    pass


def write_mapping(path: Path | str, rows: Iterable[Sequence], meta: dict[str, str]) -> Path:
    """Create a NEW mapping database. ``rows`` = (new_id, tile_id, source_faiss_id, observation_id).

    Refuses to overwrite an existing file - a candidate index is always written to a new path.
    """
    path = Path(path)
    if path.exists():
        raise MappingError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(str(tmp))
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO faiss_map VALUES (?,?,?,?)", rows)
        conn.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
        conn.commit()
    finally:
        conn.close()
    tmp.replace(path)
    return path


def read_mapping(path: Path | str) -> tuple[list[tuple[int, str, int | None, str]], dict[str, str]]:
    """All rows in ``new_id`` order, plus the meta dict."""
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"mapping database not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT new_id, tile_id, source_faiss_id, observation_id FROM faiss_map ORDER BY new_id").fetchall()
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    finally:
        conn.close()
    return rows, meta

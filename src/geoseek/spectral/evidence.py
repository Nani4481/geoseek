"""Spectral evidence for one tile: per-pixel NDVI / NDWI / NDBI at the native 10 m, plus which indices bear on a search query.

This is what "why did this match?" can honestly be answered with. The retrieval embedding (ViT-B/32 over a 2.58 km tile)
resolves ~366 m patches and cannot localise a structure or a riverbank, so no attention map is offered; the indices below are
*measurements* of the same pixels, from bands (B08 NIR, B11 SWIR) that the embedding model never sees.

Everything is computed by the same functions the per-tile spectral descriptor uses (``geoseek.change.indices`` and
``geoseek.spectral.descriptor.describe_tile``), over the same valid-pixel mask (SCL classes), so the statistics shown here are
numerically identical to the ``tile_spectral`` catalog row for the tile. Read-only; nothing is stored.
"""

from __future__ import annotations

import functools
import io
import re
from pathlib import Path

import numpy as np

from geoseek.change.indices import normalized_difference
from geoseek.ingest.tiler import read_tile_window
from geoseek.spectral.descriptor import SCL_VALID, describe_tile

BANDS = ("B03", "B04", "B08", "B11", "SCL")

# value -> colour stops (the legend and the PNG are both drawn from these, so they cannot disagree) and the fixed physical
# domain each index is shown over - fixed, not per-tile, so two tiles are comparable.
INDEX_SPECS: dict[str, dict] = {
    "ndvi": {
        "label": "NDVI", "meaning": "vegetation vigour", "formula": "(B08 − B04) / (B08 + B04)", "bands": ["B08", "B04"],
        "domain": [-0.2, 0.9], "stops": [[-0.2, "#6b4423"], [0.1, "#d9c27e"], [0.3, "#a8d46f"], [0.55, "#3fa34d"], [0.9, "#0b5d1e"]],
        "high": "dense green vegetation", "low": "bare ground, built surfaces, water",
    },
    "ndwi": {
        "label": "NDWI", "meaning": "open water", "formula": "(B03 − B08) / (B03 + B08)", "bands": ["B03", "B08"],
        "domain": [-0.6, 0.6], "stops": [[-0.6, "#8c510a"], [-0.2, "#dfc27d"], [0.0, "#f5f5f5"], [0.2, "#74add1"], [0.6, "#08306b"]],
        "high": "open water (above 0)", "low": "dry land, vegetation",
    },
    "ndbi": {
        "label": "NDBI", "meaning": "built-up and bare surfaces", "formula": "(B11 − B08) / (B11 + B08)", "bands": ["B11", "B08"],
        "domain": [-0.5, 0.5], "stops": [[-0.5, "#2c7bb6"], [-0.15, "#abd9e9"], [0.0, "#ffffbf"], [0.25, "#fdae61"], [0.5, "#d7191c"]],
        "high": "built-up and bare surfaces", "low": "vegetation, water",
    },
}

# Which index speaks to which words in a query. Plain substring stems; the reason text is shown to the analyst verbatim.
QUERY_LEXICON: dict[str, tuple[tuple[str, ...], str]] = {
    "ndwi": (("water", "river", "lake", "pond", "reservoir", "flood", "wetland", "canal", "stream", "sandbar", "coast", "sea ",
              "lagoon", "backwater", "irrigat", "paddy", "marsh", "dam"),
             "water terms: NDWI is positive over open water and negative over dry land"),
    "ndvi": (("tree", "forest", "vegetation", "crop", "field", "farm", "agricultur", "grass", "green", "orchard", "plantation",
              "canopy", "jungle", "woodland", "shrub", "bare", "barren", "dry ", "desert", "soil", "quarry", "excavation", "sand"),
             "vegetation / bare-ground terms: NDVI is high over green cover and near zero over bare soil"),
    "ndbi": (("building", "urban", "town", "city", "roof", "built", "construction", "industrial", "road", "settlement", "village",
              "house", "airport", "runway", "parking", "infrastructure", "bare", "barren", "dry ", "desert", "soil", "quarry",
              "excavation", "mine", "sand"),
             "built-up / bare-surface terms: NDBI is high over rooftops, roads and bare soil"),
}


def relevant_indices(query: str | None) -> dict:
    """Indices that bear on the words of ``query`` (lower-cased substring match against QUERY_LEXICON)."""
    q = f" {(query or '').lower()} "
    matches = []
    for idx, (stems, why) in QUERY_LEXICON.items():
        hit = sorted({s.strip() for s in stems if s in q})
        if hit:
            matches.append({"index": idx, "terms": hit, "why": why})
    return {"query": query, "matches": matches,
            "note": None if matches else "No spectral index is specifically tied to these words; all three are shown for reference."}


def _hex_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def colorize(values: np.ndarray, valid: np.ndarray, index: str) -> np.ndarray:
    """RGBA uint8 for an index array: colour from the index's stops, fully transparent where the pixel is not valid."""
    spec = INDEX_SPECS[index]
    xs = np.array([s[0] for s in spec["stops"]], dtype=np.float32)
    cols = np.array([_hex_rgb(s[1]) for s in spec["stops"]], dtype=np.float32)
    v = np.clip(values, xs[0], xs[-1])
    out = np.zeros(values.shape + (4,), dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.interp(v, xs, cols[:, c]).round().astype(np.uint8)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return out


# ViT-B/32 on a 224 px input is a 7 x 7 grid of 32 px patches (224 / 32): an architectural constant of the retrieval model.
VIT_B32_GRID = 224 // 32


def _patch_metres(tile) -> float | None:
    """Ground size of one retrieval-model patch, from the tile's own catalogued footprint (None if the footprint is unknown)."""
    import math

    wkt_text = getattr(tile, "geom_wkt_4326", None)
    if not wkt_text:
        return None
    from shapely import wkt as shapely_wkt

    try:
        w, s_, e, n = shapely_wkt.loads(wkt_text).bounds
    except Exception:
        return None
    width_m = (e - w) * 111_320.0 * math.cos(math.radians((s_ + n) / 2))
    return width_m / VIT_B32_GRID


class SpectralUnavailable(KeyError):
    """The tile exists but its NIR/SWIR bands are not staged (e.g. Maxar VHR, Sentinel-1) - the API maps this to 404."""


def _scene_dir(repo, datasets_dir: Path, tile_id: str):
    tile = repo.get_tile(tile_id)
    if tile is None:
        raise KeyError(f"tile_id '{tile_id}' not found in the store")
    obs = repo.get_observation(tile.observation_id)
    if obs is None:
        raise KeyError(f"tile_id '{tile_id}' -> observation '{tile.observation_id}' missing")
    scene_dir = Path(datasets_dir) / (obs.dataset_dir or obs.observation_id)
    missing = [b for b in BANDS if not (scene_dir / f"{b}.tif").is_file()]
    if missing:
        raise SpectralUnavailable(f"spectral bands for tile '{tile_id}' are not staged (missing {', '.join(missing)} under "
                                  f"{scene_dir.name}); only Sentinel-2 tiles with B03/B04/B08/B11/SCL on disk have spectral evidence")
    return tile, obs, scene_dir


@functools.lru_cache(maxsize=256)
def _window(scene_dir: str, row: int, col: int):
    bands, _ = read_tile_window(Path(scene_dir), row, col, list(BANDS))
    b03, b04, b08, b11, scl = (bands[b] for b in BANDS)
    idx = {"ndvi": normalized_difference(b08, b04), "ndwi": normalized_difference(b03, b08), "ndbi": normalized_difference(b11, b08)}
    valid = np.isin(scl, SCL_VALID) & np.isfinite(idx["ndvi"]) & np.isfinite(idx["ndwi"]) & np.isfinite(idx["ndbi"])
    return idx, valid, describe_tile(b03, b04, b08, b11, scl), (int(b03.shape[1]), int(b03.shape[0]))


def tile_spectral(repo, datasets_dir: Path, tile_id: str, query: str | None = None) -> dict:
    tile, obs, scene_dir = _scene_dir(repo, datasets_dir, tile_id)
    idx, valid, desc, (w, h) = _window(str(scene_dir), tile.row, tile.col)
    patch_m = _patch_metres(tile)
    usable = bool(desc.get("usable"))
    layers = {}
    for name, spec in INDEX_SPECS.items():
        stats = None
        if usable:
            stats = {k: desc[f"{name}_{k}"] for k in ("mean", "std", "p10", "p50", "p90")}
        layers[name] = {**spec, "stats": stats, "image_url": f"/ui/tiles/{tile_id}/spectral/{name}.png"}
    classes = {k: desc[k] for k in ("water_frac", "veg_frac", "dense_veg_frac", "bare_frac", "built_frac")} if usable else None
    return {
        "tile_id": tile_id, "width_px": w, "height_px": h, "gsd_m": 10.0, "scene_id": obs.scene_id, "acquired_at": obs.acquired_at,
        "valid_fraction": float(desc["valid_frac"]), "usable": usable,
        "unusable_reason": None if usable else "fewer than 5% of the tile's pixels are valid land/water (cloud, shadow or no data), so no statistics are given",
        "layers": layers, "classes": classes,
        "class_rules": {
            "water_frac": "NDWI > 0", "veg_frac": "NDVI > 0.35", "dense_veg_frac": "NDVI > 0.55",
            "bare_frac": "NDVI < 0.20 and NDWI < 0 and NDBI > −0.15", "built_frac": "NDBI > −0.05 and NDVI < 0.30 and NDWI < 0"},
        "relevance": relevant_indices(query),
        "source": ("computed now from the staged B03/B04/B08/B11 bands and the SCL mask, with the same functions and valid-pixel mask as "
                   "the catalog's per-tile spectral descriptor"),
        "embedding_patch_m": None if patch_m is None else round(patch_m),
        "caveat": ("These are per-pixel measurements at 10 m, not an explanation of the embedding: the retrieval model looks at "
                   + (f"~{round(patch_m)} m patches" if patch_m else "coarse patches")
                   + " of the true-colour image (too coarse to localise a structure) and never sees NIR/SWIR."),
    }


@functools.lru_cache(maxsize=256)
def _png(scene_dir: str, row: int, col: int, index: str) -> bytes:
    from PIL import Image

    idx, valid, _, _ = _window(scene_dir, row, col)
    buf = io.BytesIO()
    Image.fromarray(colorize(idx[index], valid, index), mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def tile_index_png(repo, datasets_dir: Path, tile_id: str, index: str) -> bytes:
    if index not in INDEX_SPECS:
        raise ValueError(f"index must be one of {sorted(INDEX_SPECS)}")
    tile, _, scene_dir = _scene_dir(repo, datasets_dir, tile_id)
    return _png(str(scene_dir), tile.row, tile.col, index)


SAFE_INDEX = re.compile(r"^(ndvi|ndwi|ndbi)$")

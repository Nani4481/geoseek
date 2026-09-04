"""Shared naming / provenance constants for the catalog.

The source of truth for the STAC / COG constants is
``geoseek.staging.download_datasets``; they are duplicated here (not imported)
so the catalog layer carries no dependency on the network-staging module.
"""

from __future__ import annotations

COG_BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/44/R/PQ"
DATA_LICENSE = "Copernicus Sentinel Data (free & open, EU Copernicus Data Policy)"
COLLECTION_ID = "sentinel-2-l2a"
AOI_NAME = "ayodhya_44RPQ_scaled_82km"

# AOI-clip suffixes an observation_id carries on top of its source scene_id.
# '_scaled' = the Sentinel-2 82 km AOI clip; '_grd' = the Sentinel-1 AOI clip.
OBS_SUFFIXES = ("_scaled", "_grd")


def base_scene_id(observation_id: str) -> str:
    """Strip the AOI-clip suffix: 'S2B_..._L2A_scaled' -> 'S2B_..._L2A' (or '..._grd')."""
    for suf in OBS_SUFFIXES:
        if observation_id.endswith(suf):
            return observation_id[: -len(suf)]
    return observation_id


def scene_source_url(base_scene_id_: str) -> str | None:
    """Reconstruct the public COG folder URL for a Sentinel-2 product id."""
    parts = base_scene_id_.split("_")
    if len(parts) < 3 or len(parts[2]) != 8:
        return None
    year, month = parts[2][:4], str(int(parts[2][4:6]))
    return f"{COG_BASE}/{year}/{month}/{base_scene_id_}/"


def platform_from_scene_id(base_scene_id_: str) -> str:
    tok = base_scene_id_.split("_")[0]
    return {"S2A": "Sentinel-2A", "S2B": "Sentinel-2B"}.get(tok, tok or "unknown")

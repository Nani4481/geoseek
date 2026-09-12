"""Phase 8 Step B: build every cached terrain raster (offline, no network).

Requires the DEM already staged (``python -m geoseek.staging.download_dem``)
and the 2024-03-08 reference scene's Phase 3a NDVI/NDBI/NDWI already on disk.

    python -m geoseek.terrain.build [--force]
"""

from __future__ import annotations

import sys

from geoseek.terrain.dem import compute_slope_aspect, terrain_staged
from geoseek.terrain.proximity import compute_distance_rasters


def build(force: bool = False) -> None:
    if not terrain_staged():
        raise SystemExit("DEM not staged - run `python -m geoseek.staging.download_dem` first")
    slope_p, aspect_p = compute_slope_aspect(force=force)
    print(f"[terrain] slope  -> {slope_p}")
    print(f"[terrain] aspect -> {aspect_p}")
    water_p, built_p = compute_distance_rasters(force=force)
    print(f"[terrain] distance-to-water     -> {water_p}")
    print(f"[terrain] distance-to-built-up  -> {built_p}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-terrain-build", description=__doc__)
    p.add_argument("--force", action="store_true", help="recompute even if cached rasters exist")
    args = p.parse_args(argv)
    build(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[terrain.build] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

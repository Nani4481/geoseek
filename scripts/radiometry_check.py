"""Phase 2 follow-up part 1 + part 5: Sentinel-2 baseline / BOA-offset check and
harmonization sanity check.

1. Report the STAC processing baseline + earthsearch:boa_offset_applied for both
   staged dates, and sanity-check by comparing DN distributions over the whole
   overlap and over deep water (the diagnostic for a raw-DN -1000 offset).
5. Pick 3 tiles that are clearly UNCHANGED 2019 -> 2024 (interior, away from the
   seasonally-shifting river) and print mean reflectance per band at both dates.
   After harmonization these must be CLOSE (not offset by ~1000).

Uses the network ONLY for the STAC metadata read in step 1 (a one-off analysis,
not part of the offline ingest/search path). Falls back to cached findings if
offline.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window

from geoseek.config import get_settings
from geoseek.ingest.embed import (
    S2_REFLECTANCE_SCALE,
    TRUE_COLOR_MAX_REFLECTANCE,
    boa_offset_dn_for_scene,
)
from geoseek.ingest.tiler import TILE_SIZE

SCENES = {
    "2019": "S2B_44RPQ_20190330_1_L2A",
    "2024": "S2A_44RPQ_20240308_0_L2A",
}
SCALED = {k: v + "_scaled" for k, v in SCENES.items()}
RGB = ("B04", "B03", "B02")


def stac_report() -> None:
    print("=" * 78)
    print("1a. STAC item metadata (Earth Search v1, collection sentinel-2-l2a)")
    print("=" * 78)
    try:
        import requests

        base = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"
        for label, sid in SCENES.items():
            r = requests.get(f"{base}/{sid}", timeout=30)
            r.raise_for_status()
            j = r.json()
            p = j["properties"]
            red = j["assets"]["red"].get("raster:bands")
            print(f"\n  {label}  {sid}")
            print(f"    datetime                        : {p.get('datetime')}")
            print(f"    s2:processing_baseline          : {p.get('s2:processing_baseline')}")
            print(f"    earthsearch:boa_offset_applied  : {p.get('earthsearch:boa_offset_applied')}")
            print(f"    s2:product_uri                  : {p.get('s2:product_uri')}")
            print(f"    assets.red raster:bands         : {red}")
    except Exception as e:  # offline / blocked - fall back to what was captured at authoring
        print(f"\n  [network unavailable: {e!r}]")
        print("  Cached findings (captured 2026-09-03):")
        print("    2019 S2B_44RPQ_20190330_1_L2A: baseline 05.00, boa_offset_applied=True,")
        print("         product_uri N0500 reprocessed 2022-11-20, raster:bands scale=1e-4 offset=-0.1")
        print("    2024 S2A_44RPQ_20240308_0_L2A: baseline 05.10, boa_offset_applied=True,")
        print("         product_uri N0510, raster:bands scale=1e-4 offset=-0.1")


def _read_full(scaled_dir, band):
    with rasterio.open(scaled_dir / f"{band}.tif") as ds:
        return ds.read(1)


def dn_sanity() -> None:
    ds_dir = get_settings().datasets_dir
    print("\n" + "=" * 78)
    print("1b. DN sanity check over the co-registered overlap (raw staged COG values)")
    print("=" * 78)
    print("    A raw-DN baseline-04.00+ scene with the -1000 offset NOT applied has its")
    print("    dark floor (deep water / shadow) sitting near DN 1000, not near 0.\n")
    for band in RGB:
        a19 = _read_full(ds_dir / SCALED["2019"], band).astype(np.int32)
        a24 = _read_full(ds_dir / SCALED["2024"], band).astype(np.int32)
        m = (a19 != 0) & (a24 != 0)
        v19, v24 = a19[m], a24[m]
        # deep-water proxy = darkest 0.5% of each date
        w19 = v19[v19 <= np.percentile(v19, 0.5)]
        w24 = v24[v24 <= np.percentile(v24, 0.5)]
        print(f"  {band}:")
        print(f"    2019  min={v19.min():5d}  p2={np.percentile(v19,2):6.0f}  median={np.median(v19):6.0f}  "
              f"dark-floor(mean of darkest 0.5%)={w19.mean():6.0f}")
        print(f"    2024  min={v24.min():5d}  p2={np.percentile(v24,2):6.0f}  median={np.median(v24):6.0f}  "
              f"dark-floor(mean of darkest 0.5%)={w24.mean():6.0f}")
        print(f"    scene-wide median (2024-2019) = {np.median(v24 - v19):+.0f} DN")


def _tile_means(scaled_dir, r, c, boa_offset_dn):
    """Mean reflectance per band over valid pixels of one tile."""
    col0, row0 = c * TILE_SIZE, r * TILE_SIZE
    out = {}
    valid_mask = None
    arrs = {}
    with rasterio.open(scaled_dir / "B04.tif") as ds:
        H, W = ds.height, ds.width
    tw = min(TILE_SIZE, W - col0)
    th = min(TILE_SIZE, H - row0)
    win = Window(col0, row0, tw, th)
    for band in RGB:
        with rasterio.open(scaled_dir / f"{band}.tif") as ds:
            a = ds.read(1, window=win).astype(np.float32)
        arrs[band] = a
        vm = a != 0
        valid_mask = vm if valid_mask is None else (valid_mask & vm)
    nvalid = int(valid_mask.sum())
    if nvalid == 0:
        return None, 0
    for band in RGB:
        a = arrs[band]
        refl = (a[valid_mask] - boa_offset_dn) / S2_REFLECTANCE_SCALE
        out[band] = float(refl.mean())
    return out, nvalid


def find_stable_tiles() -> None:
    ds_dir = get_settings().datasets_dir
    d19, d24 = ds_dir / SCALED["2019"], ds_dir / SCALED["2024"]
    off19, off24 = boa_offset_dn_for_scene(SCENES["2019"]), boa_offset_dn_for_scene(SCENES["2024"])

    with rasterio.open(d19 / "B04.tif") as ds:
        H, W = ds.height, ds.width
    n_rows, n_cols = -(-H // TILE_SIZE), -(-W // TILE_SIZE)

    # Scan a coarse grid of full interior tiles, skip the edge strip and the
    # river corridor (low-reflectance, seasonally variable) by requiring a
    # land-like mean reflectance and low cross-date change.
    cands = []
    for r in range(2, n_rows - 2, 2):
        for c in range(2, n_cols - 2, 2):
            m19, nv19 = _tile_means(d19, r, c, off19)
            if m19 is None or nv19 < TILE_SIZE * TILE_SIZE * 0.98:
                continue
            m24, nv24 = _tile_means(d24, r, c, off24)
            if m24 is None or nv24 < TILE_SIZE * TILE_SIZE * 0.98:
                continue
            # land-like: red reflectance clearly above water (~0.03) at both dates
            if min(m19["B04"], m24["B04"]) < 0.07:
                continue
            # rank by the WORST band's cross-date drift, so a "stable" pick is
            # stable in every channel, not just on average
            change = max(abs(m24[b] - m19[b]) for b in RGB)
            bright = np.mean([m19[b] + m24[b] for b in RGB]) / 2
            cands.append((change, r, c, m19, m24, bright))
    cands.sort(key=lambda x: x[0])

    # take 3 spatially spread-out low-change tiles
    picks = []
    for change, r, c, m19, m24, bright in cands:
        if all(abs(r - pr) + abs(c - pc) >= 8 for _, pr, pc, *_ in picks):
            picks.append((change, r, c, m19, m24, bright))
        if len(picks) == 3:
            break

    print("\n" + "=" * 78)
    print("5. Unchanged-tile reflectance comparison (harmonized: fixed bounds, BOA offset 0/0)")
    print("=" * 78)
    print(f"    Scanned {len(cands)} candidate interior land tiles; showing the 3 lowest-change,")
    print("    spatially separated picks. Values are MEAN surface reflectance over the tile.\n")
    for change, r, c, m19, m24, bright in picks:
        tid19 = f"{SCALED['2019']}_r{r:03d}_c{c:03d}"
        tid24 = f"{SCALED['2024']}_r{r:03d}_c{c:03d}"
        print(f"  tile r{r:03d}_c{c:03d}   (worst-band abs reflectance change = {change:.4f})")
        print(f"    {tid19}")
        print(f"    {tid24}")
        for b in RGB:
            d_refl = m24[b] - m19[b]
            d_dn = d_refl * S2_REFLECTANCE_SCALE
            print(f"      {b}:  2019 = {m19[b]:.4f}   2024 = {m24[b]:.4f}   "
                  f"change = {d_refl:+.4f} reflectance ({d_dn:+.0f} DN)")
        print()
    max_abs = max(abs(m24[b] - m19[b]) for _, _, _, m19, m24, _ in picks for b in RGB)
    print(f"    Largest per-band cross-date difference among the 3 picks: "
          f"{max_abs:.4f} reflectance ({max_abs * S2_REFLECTANCE_SCALE:.0f} DN).")
    print(f"    (A missed -1000 BOA offset would show ~0.1000 reflectance / ~1000 DN here.)")


def main() -> None:
    stac_report()
    dn_sanity()
    find_stable_tiles()


if __name__ == "__main__":
    main()

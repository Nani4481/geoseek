"""Stage NASA Blue Marble Earth textures for the interactive globe - a
one-time build-time fetch (same category as scripts/build_basemap.py fetching
Natural Earth data / scripts/stage_threejs.py fetching three.js itself),
never touched at runtime.

The files themselves are NASA Visible Earth's public-domain "Blue Marble"
(day) / "Black Marble" (night lights) / ocean specular mask / cloud composites,
as redistributed inside the three.js repo's own examples (same pinned tag we
already stage three.js from) at 2048x1024 - already a sane "downsample
aggressively" size for a background globe texture, so no re-encoding needed
for the day or night map (both kept at full 2048x1024 so the night-lights
terminator reads as crisply as the day side, not a visibly blurrier patch).
The specular mask is a subtle, low-frequency ocean-glint input only - halved
to 1024x512 to save bundle weight where it costs nothing visually. The cloud
layer ships at its native 1024x512 and keeps its alpha channel (PNG, not
JPEG) since the globe shader uses it as a translucent overlay.

    python scripts/stage_earth_textures.py

Output: src/geoseek/analyst/vendor/earth/day.jpg      (2048x1024)
        src/geoseek/analyst/vendor/earth/night.jpg    (2048x1024)
        src/geoseek/analyst/vendor/earth/specular.jpg (1024x512)
        src/geoseek/analyst/vendor/earth/clouds.png   (1024x512, RGBA)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from PIL import Image

from geoseek.staging.manifest import load_manifest, sha256_of, write_manifest

TAG = "r160"
BASE_URL = f"https://raw.githubusercontent.com/mrdoob/three.js/{TAG}/examples/textures/planets/"
SRC_DIR = Path(__file__).resolve().parent.parent / "data" / "earth_textures_src"
OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "vendor" / "earth"

SOURCES = {
    "day": "earth_atmos_2048.jpg",
    "night": "earth_lights_2048.png",
    "specular": "earth_specular_2048.jpg",
    "clouds": "earth_clouds_1024.png",
}
HALVE = {"specular"}  # day and night both stay full 2048 res; clouds is native 1024; specular is a subtle blend only
KEEP_ALPHA = {"clouds"}  # the cloud layer needs its alpha channel to render as a translucent overlay


def main() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for name, fname in SOURCES.items():
        src_path = SRC_DIR / fname
        if not src_path.is_file():
            print(f"fetching {fname} ...")
            urllib.request.urlretrieve(BASE_URL + fname, src_path)
        img = Image.open(src_path)
        img = img.convert("RGBA") if name in KEEP_ALPHA else img.convert("RGB")
        if name in HALVE:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        ext = "png" if name in KEEP_ALPHA else "jpg"
        out_path = OUT_DIR / f"{name}.{ext}"
        if name in KEEP_ALPHA:
            img.save(out_path, format="PNG", optimize=True)
        else:
            img.save(out_path, format="JPEG", quality=86, optimize=True)
        kb = round(out_path.stat().st_size / 1024, 1)
        print(f"  {name}: {img.width}x{img.height} -> {out_path.name} ({kb} KB)")
        entries.append({
            "name": f"earth_{name}", "source_url": BASE_URL + fname,
            "local_path": str(out_path.resolve()), "sha256": sha256_of(out_path),
            "byte_size": out_path.stat().st_size, "resized_from": fname if name in HALVE else None,
        })

    total = sum(e["byte_size"] for e in entries)
    print(f"\n{len(entries)} Earth textures, {total / 1e6:.2f} MB total")

    manifest = load_manifest()
    artifacts = [a for a in manifest.setdefault("artifacts", []) if a.get("name") != "earth_textures_vendor"]
    artifacts.append({
        "name": "earth_textures_vendor",
        "source_url": f"redistributed by three.js ({BASE_URL}) - originally NASA Visible Earth "
                       "'Blue Marble Next Generation' (day) and 'Black Marble' (night lights) / "
                       "ocean specular mask / MODIS cloud fraction composites",
        "license": "Public domain (NASA imagery, https://visibleearth.nasa.gov/collection/1484/blue-marble "
                    "and https://visibleearth.nasa.gov/collection/1579/city-lights) - "
                    "no restriction on reuse, attribution appreciated but not required",
        "pinned_tag": TAG,
        "files": entries,
        "byte_size": total,
    })
    manifest["artifacts"] = artifacts
    write_manifest(manifest)
    print("recorded provenance for earth_textures_vendor in the manifest")


if __name__ == "__main__":
    main()

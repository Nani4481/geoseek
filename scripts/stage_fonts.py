"""Stage the analyst UI's self-hosted webfonts - a one-time build-time fetch
(same category as scripts/stage_threejs.py / stage_earth_textures.py), never
touched at runtime. css/tokens.css's @font-face rules load these as plain
files from this same FastAPI origin - no CDN, no network call once the app
is running.

Inter (UI prose) and JetBrains Mono (machine-readable values) are both
SIL Open Font License 1.1 - free to redistribute, no restriction beyond
keeping the license notice, which stage_threejs.py's manifest pattern
already does for us.

    python scripts/stage_fonts.py

Output: src/geoseek/analyst/web/fonts/Inter-{Regular,Medium,SemiBold,Bold}.woff2
        src/geoseek/analyst/web/fonts/JetBrainsMono-{Regular,Medium,SemiBold}.woff2
"""
from __future__ import annotations

import urllib.request
from dataclasses import asdict
from pathlib import Path

from geoseek.staging.manifest import build_record, load_manifest, write_manifest

INTER_TAG = "v4.1"
INTER_BASE = f"https://raw.githubusercontent.com/rsms/inter/{INTER_TAG}/docs/font-files"
INTER_LICENSE_URL = f"https://raw.githubusercontent.com/rsms/inter/{INTER_TAG}/LICENSE.txt"
INTER_WEIGHTS = ["Regular", "Medium", "SemiBold", "Bold"]

JBM_TAG = "v2.304"
JBM_BASE = f"https://raw.githubusercontent.com/JetBrains/JetBrainsMono/{JBM_TAG}/fonts/webfonts"
JBM_LICENSE_URL = f"https://raw.githubusercontent.com/JetBrains/JetBrainsMono/{JBM_TAG}/OFL.txt"
JBM_WEIGHTS = ["Regular", "Medium", "SemiBold"]  # matches what tokens.css declares

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "web" / "fonts"


def stage_family(*, family_name: str, base_url: str, weights: list[str], license_url: str,
                  tag: str, manifest_key: str, artifacts: list[dict]) -> None:
    license_text = urllib.request.urlopen(license_url, timeout=30).read().decode("utf-8")
    assert "SIL Open Font License" in license_text, f"expected an OFL license for {family_name}"
    license_line = "SIL OFL 1.1 - " + license_text.strip().splitlines()[0]

    for weight in weights:
        fname = f"{family_name}-{weight}.woff2"
        src_url = f"{base_url}/{fname}"
        out = OUT_DIR / fname
        print(f"fetching {family_name} {tag} :: {weight} ...")
        urllib.request.urlretrieve(src_url, out)
        kb = round(out.stat().st_size / 1024, 1)
        print(f"  wrote {out} - {kb} KB")

        name = f"{manifest_key}_{weight.lower()}"
        record = build_record(name=name, source_url=src_url, local_path=out, license=license_line)
        artifacts[:] = [a for a in artifacts if a.get("name") != name]
        artifacts.append({**asdict(record), "pinned_tag": tag})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    artifacts = manifest.setdefault("artifacts", [])

    stage_family(family_name="Inter", base_url=INTER_BASE, weights=INTER_WEIGHTS,
                 license_url=INTER_LICENSE_URL, tag=INTER_TAG, manifest_key="inter_font_vendor",
                 artifacts=artifacts)
    stage_family(family_name="JetBrainsMono", base_url=JBM_BASE, weights=JBM_WEIGHTS,
                 license_url=JBM_LICENSE_URL, tag=JBM_TAG, manifest_key="jetbrains_mono_font_vendor",
                 artifacts=artifacts)

    manifest["artifacts"] = artifacts
    write_manifest(manifest)
    print("recorded provenance for the staged webfonts in the manifest")


if __name__ == "__main__":
    main()

"""Stage three.js locally for the analyst console's globe and vector-space view -
a one-time build-time fetch, never touched at runtime. The React console
imports both files through the Vite aliases in frontend-react/vite.config.ts
("three" resolves to three.module.min.js, which is also what OrbitControls.js
imports), so the build bundles exactly these bytes - no CDN, no network call
once the app is running.

    python scripts/stage_threejs.py

Output: src/geoseek/analyst/vendor/three.module.min.js
        src/geoseek/analyst/vendor/OrbitControls.js
"""
from __future__ import annotations

import urllib.request
from dataclasses import asdict
from pathlib import Path

from geoseek.staging.manifest import build_record, load_manifest, write_manifest

TAG = "r160"
RAW_BASE = f"https://raw.githubusercontent.com/mrdoob/three.js/{TAG}"
LICENSE_URL = f"{RAW_BASE}/LICENSE"
VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "vendor"

FILES = {
    "threejs_vendor": ("build/three.module.min.js", "three.module.min.js"),
    "threejs_orbitcontrols_vendor": ("examples/jsm/controls/OrbitControls.js", "OrbitControls.js"),
}


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    license_text = urllib.request.urlopen(LICENSE_URL, timeout=30).read().decode("utf-8")
    assert "MIT License" in license_text, "expected three.js's MIT license text"
    license_line = "MIT - " + license_text.strip().splitlines()[0]

    manifest = load_manifest()
    artifacts = manifest.setdefault("artifacts", [])

    for name, (rel_path, out_name) in FILES.items():
        src_url = f"{RAW_BASE}/{rel_path}"
        out = VENDOR_DIR / out_name
        print(f"fetching three.js {TAG} :: {rel_path} ...")
        urllib.request.urlretrieve(src_url, out)
        kb = round(out.stat().st_size / 1024, 1)
        print(f"wrote {out} - {kb} KB")

        record = build_record(name=name, source_url=src_url, local_path=out, license=license_line)
        artifacts[:] = [a for a in artifacts if a.get("name") != name]
        artifacts.append({**asdict(record), "license_full_text": license_text.strip(), "pinned_tag": TAG})
        print(f"recorded provenance for {name} (sha256 {record.sha256[:12]}...) in the manifest")

    manifest["artifacts"] = artifacts
    write_manifest(manifest)


if __name__ == "__main__":
    main()

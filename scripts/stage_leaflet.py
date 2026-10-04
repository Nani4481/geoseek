"""Stage Leaflet locally for the analyst console's maps - a one-time build-time
fetch (same category as scripts/stage_threejs.py), never touched at runtime.
The React console imports leaflet.js and leaflet.css through the Vite aliases in
frontend-react/vite.config.ts, so the build bundles exactly these bytes. No CDN,
no network call once the app is running.

    python scripts/stage_leaflet.py

Output: src/geoseek/analyst/vendor/leaflet/leaflet.js
        src/geoseek/analyst/vendor/leaflet/leaflet.css
        src/geoseek/analyst/vendor/leaflet/images/marker-icon.png
(The console draws every marker as a divIcon, so Leaflet's default 2x / shadow marker images are not staged.)
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

from geoseek.staging.manifest import build_record, load_manifest, write_manifest

VERSION = "1.9.4"
TARBALL_URL = f"https://registry.npmjs.org/leaflet/-/leaflet-{VERSION}.tgz"
EXPECTED_SHA1 = "23fae724e282fa25745aff82ca4d394748db7d8d"  # npm's published shasum for this version
LICENSE = "BSD-2-Clause (Leaflet)"

VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "vendor" / "leaflet"
MEMBERS = {
    "package/dist/leaflet.js": VENDOR_DIR / "leaflet.js",
    "package/dist/leaflet.css": VENDOR_DIR / "leaflet.css",
    "package/dist/images/marker-icon.png": VENDOR_DIR / "images" / "marker-icon.png",
}


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    (VENDOR_DIR / "images").mkdir(parents=True, exist_ok=True)

    print(f"fetching leaflet {VERSION} tarball ...")
    tgz_path = VENDOR_DIR / f"_leaflet-{VERSION}.tgz"
    urllib.request.urlretrieve(TARBALL_URL, tgz_path)
    got_sha1 = hashlib.sha1(tgz_path.read_bytes()).hexdigest()
    assert got_sha1 == EXPECTED_SHA1, f"leaflet tarball sha1 mismatch: got {got_sha1}, expected {EXPECTED_SHA1}"

    with tarfile.open(tgz_path) as tar:
        for member_name, out_path in MEMBERS.items():
            member = tar.getmember(member_name)
            with tar.extractfile(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            kb = round(out_path.stat().st_size / 1024, 1)
            print(f"wrote {out_path} - {kb} KB")
    tgz_path.unlink()

    manifest = load_manifest()
    artifacts = manifest.setdefault("artifacts", [])
    for name, out_path in MEMBERS.items():
        # full file name, not the stem: leaflet.js and leaflet.css share a stem, and the upsert-by-name below would
        # have let the second silently replace the first's provenance record
        artifact_name = "leaflet_" + Path(name).name.replace("-", "_").replace(".", "_") + "_vendor"
        record = build_record(name=artifact_name, source_url=TARBALL_URL, local_path=out_path, license=LICENSE)
        artifacts[:] = [a for a in artifacts if a.get("name") != artifact_name]
        artifacts.append({**asdict(record), "pinned_version": VERSION})
        print(f"recorded provenance for {artifact_name} (sha256 {record.sha256[:12]}...) in the manifest")

    manifest["artifacts"] = artifacts
    write_manifest(manifest)


if __name__ == "__main__":
    main()

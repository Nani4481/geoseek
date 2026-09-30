"""Stage Leaflet locally for the analyst UI's interactive candidate map - a
one-time build-time fetch (same category as scripts/stage_threejs.py), never
touched at runtime. index.html loads vendor/leaflet/leaflet.js as a classic
script (Leaflet's own dist build is a UMD global, not an ES module) and
vendor/leaflet/leaflet.css as a stylesheet; js/components/candidate-map.js
then uses the resulting global `L`. No CDN, no network call once the app is
running.

    python scripts/stage_leaflet.py

Output: src/geoseek/analyst/web/vendor/leaflet/leaflet.js
        src/geoseek/analyst/web/vendor/leaflet/leaflet.css
        src/geoseek/analyst/web/vendor/leaflet/images/marker-icon.png
        src/geoseek/analyst/web/vendor/leaflet/images/marker-icon-2x.png
        src/geoseek/analyst/web/vendor/leaflet/images/marker-shadow.png
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

VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "web" / "vendor" / "leaflet"
MEMBERS = {
    "package/dist/leaflet.js": VENDOR_DIR / "leaflet.js",
    "package/dist/leaflet.css": VENDOR_DIR / "leaflet.css",
    "package/dist/images/marker-icon.png": VENDOR_DIR / "images" / "marker-icon.png",
    "package/dist/images/marker-icon-2x.png": VENDOR_DIR / "images" / "marker-icon-2x.png",
    "package/dist/images/marker-shadow.png": VENDOR_DIR / "images" / "marker-shadow.png",
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
        artifact_name = "leaflet_" + Path(name).stem.replace("-", "_") + "_vendor"
        record = build_record(name=artifact_name, source_url=TARBALL_URL, local_path=out_path, license=LICENSE)
        artifacts[:] = [a for a in artifacts if a.get("name") != artifact_name]
        artifacts.append({**asdict(record), "pinned_version": VERSION})
        print(f"recorded provenance for {artifact_name} (sha256 {record.sha256[:12]}...) in the manifest")

    manifest["artifacts"] = artifacts
    write_manifest(manifest)


if __name__ == "__main__":
    main()

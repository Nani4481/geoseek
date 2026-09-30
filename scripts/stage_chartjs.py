"""Stage Chart.js locally for the analyst UI's confidence/count charts - a
one-time build-time fetch (same category as scripts/stage_threejs.py), never
touched at runtime. index.html loads vendor/chart.umd.js as a classic script
(the UMD build auto-registers every chart type and exposes a global `Chart`,
avoiding the ESM build's separate tree-shaking/registration step for a
no-bundler app); js/components/mini-chart.js then uses that global. No CDN,
no network call once the app is running.

    python scripts/stage_chartjs.py

Output: src/geoseek/analyst/web/vendor/chart.umd.js
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

from geoseek.staging.manifest import build_record, load_manifest, write_manifest

VERSION = "4.4.4"
TARBALL_URL = f"https://registry.npmjs.org/chart.js/-/chart.js-{VERSION}.tgz"
EXPECTED_SHA1 = "b682d2e7249f7a0cbb1b1d31c840266ae9db64b7"  # npm's published shasum for this version
LICENSE = "MIT (Chart.js)"

VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "geoseek" / "analyst" / "web" / "vendor"
OUT_PATH = VENDOR_DIR / "chart.umd.js"
MEMBER = "package/dist/chart.umd.js"


def main() -> None:
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fetching chart.js {VERSION} tarball ...")
    tgz_path = VENDOR_DIR / f"_chartjs-{VERSION}.tgz"
    urllib.request.urlretrieve(TARBALL_URL, tgz_path)
    got_sha1 = hashlib.sha1(tgz_path.read_bytes()).hexdigest()
    assert got_sha1 == EXPECTED_SHA1, f"chart.js tarball sha1 mismatch: got {got_sha1}, expected {EXPECTED_SHA1}"

    with tarfile.open(tgz_path) as tar:
        member = tar.getmember(MEMBER)
        with tar.extractfile(member) as src, open(OUT_PATH, "wb") as dst:
            dst.write(src.read())
    tgz_path.unlink()
    kb = round(OUT_PATH.stat().st_size / 1024, 1)
    print(f"wrote {OUT_PATH} - {kb} KB")

    manifest = load_manifest()
    artifacts = manifest.setdefault("artifacts", [])
    record = build_record(name="chartjs_vendor", source_url=TARBALL_URL, local_path=OUT_PATH, license=LICENSE)
    artifacts[:] = [a for a in artifacts if a.get("name") != "chartjs_vendor"]
    artifacts.append({**asdict(record), "pinned_version": VERSION})
    manifest["artifacts"] = artifacts
    write_manifest(manifest)
    print(f"recorded provenance for chartjs_vendor (sha256 {record.sha256[:12]}...) in the manifest")


if __name__ == "__main__":
    main()

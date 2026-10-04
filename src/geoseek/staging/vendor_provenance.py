"""Provenance for everything vendored under ``analyst/vendor/`` - committed, and regenerable offline.

The offline guarantee (``tests/test_frontend_offline.py``) pins every vendored file by SHA256 in
``data/provenance_manifest.json`` and requires every external-looking URL string in a vendored text file to be
enumerated and classified. The manifest itself is git-ignored (it is generated), so the *content* of that
allowlist lives here, under version control; ``python -m geoseek.staging.vendor_provenance`` rewrites the
manifest entries from it WITHOUT any network access.

The claim this supports is "0 external network requests at runtime", not "0 external URL strings in shipped
assets": vendored libraries legitimately contain attribution banners, XML namespace identifiers, browser-bug
citations and diagnostic text. Each such string is listed below with a classification. Nothing here is a live
runtime fetch (``LIVE_FETCH_KIND`` is forbidden by a test); the app's own code contains no external URL, drives
Leaflet with ``attributionControl: false`` (its one tile layer is the same-origin ``/ui/basemap`` one) and three.js
with same-origin texture URLs only.

Vendored files are byte-pinned: ``.gitattributes`` marks ``vendor/**`` as ``-text`` so no checkout converts line
endings (a CRLF checkout changes the SHA256 of otherwise-identical files).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from geoseek.config import PROJECT_ROOT
from geoseek.staging.manifest import build_record, load_manifest, sha256_of, write_manifest

VENDOR_ROOT = PROJECT_ROOT / "src" / "geoseek" / "analyst" / "vendor"
# Subtrees of VENDOR_ROOT that are not pinned in the manifest. The web fonts are pinned through the React build's own
# pins (frontend-react/build-pins.json records each copied font's SHA256 and its vendor_source); they have never had
# manifest entries, and this module keeps that scope rather than inventing provenance for them.
UNPINNED_SUBTREES = ("fonts",)

LIVE_FETCH_KIND = "live_runtime_fetch"          # a submission blocker: must never appear in this file
ATTRIBUTION = "attribution_comment"
SOURCEMAP = "sourcemap_comment"
OTHER = "other"


@dataclass(frozen=True)
class VendorUrl:
    url: str
    kind: str
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def _attr(url: str, banner: str) -> VendorUrl:
    return VendorUrl(url, ATTRIBUTION, f"{banner} Inert comment text; never dereferenced.")


# path relative to VENDOR_ROOT -> every URL-looking string in the file, classified. (An empty tuple = no URLs.)
VENDOR_URLS: dict[str, tuple[VendorUrl, ...]] = {
    "leaflet/leaflet.css": (
        VendorUrl("https://bugs.chromium.org/p/chromium/issues/detail?id=600120", OTHER,
                  "CSS comment citing a browser bug next to a workaround (documentation only)."),
        VendorUrl("https://bugzilla.mozilla.org/show_bug.cgi?id=888319", OTHER,
                  "CSS comment citing a browser bug next to a workaround (documentation only)."),
    ),
    "leaflet/leaflet.js": (
        _attr("https://leafletjs.com", "`/* @preserve Leaflet 1.9.4 ... */` header."),
        VendorUrl("https://leafletjs.com", OTHER,
                  "`<a href>` inside the default attribution-control HTML string. The app builds its map with "
                  "attributionControl:false so it is never rendered; if it were, it is a user-click navigation, not a fetch."),
        VendorUrl("http://www.w3.org/2000/svg", OTHER,
                  "XML namespace identifier for createElementNS / xmlns attributes; an opaque name, never dereferenced."),
        VendorUrl("sourceMappingURL=leaflet.js.map", SOURCEMAP,
                  "Relative reference to a .map file that is not shipped; devtools-only; not an external URL."),
    ),
    # kept verbatim from the entry that was already in the manifest (kinds are the original, more specific labels)
    "three.module.min.js": (
        VendorUrl("http://www.w3.org/1999/xhtml", "namespace_constant",
                  "XML namespace URI string literal, passed to document.createElementNS() calls inside three.js's own "
                  "DOM-renderer helpers. Never dereferenced over the network - it is an opaque identifier the DOM spec "
                  "requires, not a fetch target."),
        VendorUrl("https?://  (regex fragment, appears twice: /^https?:\\/\\//i and /^(https?:)?\\/\\//i)", "url_detection_regex",
                  "Pattern text inside three.js's own LoaderUtils.resolveURL()-style helper, used to classify whether an "
                  "already-known resource path is absolute vs relative. It is regex source text, not a URL, and is never "
                  "fetched itself."),
        VendorUrl("https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733", "console_warning_text",
                  "Static string inside a console.warn() shown when a deprecated legacy-lighting API path is used. Plain "
                  "diagnostic text - never fetched, never assigned to a src/href, never opened programmatically. Appears "
                  "twice (two call sites emit the same message)."),
    ),
    "OrbitControls.js": (),
}

# Provenance for vendored files whose manifest entry may be missing (staged by scripts/stage_*.py, whose manifest
# writes are not version-controlled). name / source_url / licence match what those scripts record.
_LEAFLET_TGZ = "https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz"


def _leaflet_name(filename: str) -> str:
    return "leaflet_" + filename.replace("-", "_").replace(".", "_") + "_vendor"


SOURCES: dict[str, dict] = {
    **{f"leaflet/{f}": dict(name=_leaflet_name(f), source_url=_LEAFLET_TGZ, license="BSD-2-Clause (Leaflet)",
                                   pinned_version="1.9.4")
       for f in ("leaflet.js", "leaflet.css")},
    **{f"leaflet/images/{f}": dict(name=_leaflet_name(f), source_url=_LEAFLET_TGZ, license="BSD-2-Clause (Leaflet)",
                                          pinned_version="1.9.4")
       for f in ("marker-icon.png",)},
    "three.module.min.js": dict(
        name="threejs_vendor", license="MIT - The MIT License", pinned_version="r160",
        source_url="https://raw.githubusercontent.com/mrdoob/three.js/r160/build/three.module.min.js"),
    "OrbitControls.js": dict(
        name="threejs_orbitcontrols_vendor", license="MIT - The MIT License", pinned_version="r160",
        source_url="https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/jsm/controls/OrbitControls.js"),
    # Earth textures are re-encoded derivatives (PIL, scripts/stage_earth_textures.py), so they are pinned to the bytes
    # that are committed here, not to the upstream files they were made from.
    **{f"earth/{f}": dict(
        name=f"earth_{f.split('.')[0]}", pinned_version="r160",
        license="Public domain (NASA Visible Earth imagery) - re-encoded by scripts/stage_earth_textures.py",
        source_url=f"https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/textures/planets/{up}")
       for f, up in (("day.jpg", "earth_atmos_2048.jpg"), ("night.jpg", "earth_lights_2048.png"))},
}


def vendor_relative(path: str | Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(VENDOR_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def vendor_files() -> list[Path]:
    return sorted(p for p in VENDOR_ROOT.rglob("*")
                  if p.is_file() and p.relative_to(VENDOR_ROOT).parts[0] not in UNPINNED_SUBTREES)


def _find_entries(manifest: dict) -> dict[str, dict]:
    """vendor-relative path -> the manifest dict (artifact or grouped sub-file) that pins it."""
    found: dict[str, dict] = {}
    for art in manifest.get("artifacts", []):
        for entry in (art, *art.get("files", [])):
            rel = vendor_relative(entry.get("local_path", "")) if entry.get("local_path") else None
            if rel:
                found[rel] = entry
    return found


def update_manifest(manifest: dict) -> list[str]:
    """Pin every vendored file by its CURRENT on-disk SHA256 and attach the classified URL allowlist.

    Offline and idempotent. Returns a human-readable change log (empty when nothing changed).
    """
    changes: list[str] = []
    artifacts = manifest.setdefault("artifacts", [])
    entries = _find_entries(manifest)
    for path in vendor_files():
        rel = path.relative_to(VENDOR_ROOT).as_posix()
        sha, size = sha256_of(path), path.stat().st_size
        entry = entries.get(rel)
        if entry is None:
            src = SOURCES.get(rel)
            if src is None:
                raise SystemExit(f"{rel}: vendored file has no manifest entry and no SOURCES record - add one")
            rec = {**asdict(build_record(name=src["name"], source_url=src["source_url"], local_path=path, license=src["license"])),
                   "pinned_version": src["pinned_version"]}
            artifacts[:] = [a for a in artifacts if a.get("name") != src["name"]]
            artifacts.append(rec)
            entries[rel] = entry = rec
            changes.append(f"{rel}: recorded new entry '{src['name']}' sha256 {sha[:12]}")
        elif entry.get("sha256") != sha:
            changes.append(f"{rel}: re-pinned sha256 {str(entry.get('sha256'))[:12]} -> {sha[:12]} "
                           f"({entry.get('byte_size')} -> {size} bytes)")
            entry["sha256"], entry["byte_size"] = sha, size
        spec = VENDOR_URLS.get(rel)
        if spec is not None:
            want = [u.as_dict() for u in spec]
            if entry.get("external_urls") != want:
                entry["external_urls"] = want
                changes.append(f"{rel}: external_urls set ({len(want)} classified)")
    return changes


def check_manifest(manifest: dict) -> list[str]:
    """Problems if the manifest does not pin every vendored file at its current SHA256 (read-only)."""
    entries, problems = _find_entries(manifest), []
    for path in vendor_files():
        rel = path.relative_to(VENDOR_ROOT).as_posix()
        e = entries.get(rel)
        if e is None:
            problems.append(f"{rel}: no manifest entry")
        elif e.get("sha256") != sha256_of(path):
            problems.append(f"{rel}: manifest sha256 {str(e.get('sha256'))[:12]} != file {sha256_of(path)[:12]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pin vendored assets in the provenance manifest (offline).")
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on any problem")
    args = ap.parse_args(argv)
    manifest = load_manifest()
    if args.check:
        problems = check_manifest(manifest)
        print("\n".join(problems) or "all vendored files pinned")
        return 1 if problems else 0
    changes = update_manifest(manifest)
    if changes:
        write_manifest(manifest)
    print("\n".join(changes) or "manifest already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

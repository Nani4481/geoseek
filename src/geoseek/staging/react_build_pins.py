"""Hash pins + URL classification for the React console build (``analyst/web_react/``).

Same two rules the vendored directory is held to by ``tests/test_frontend_offline.py``, applied to the build output:

1. **Hash pin.** Every file the build emits is pinned by SHA-256 in ``frontend-react/build-pins.json`` (committed).
   Any drift - an edited chunk, a stray file, a missing file - fails the test until a human re-runs the build and
   re-reviews. The pin also records which vendored inputs (three.js, Leaflet, OrbitControls, fonts, textures) were
   bundled, so the test can prove those bytes are the already-pinned ones from ``analyst/web/vendor``.
2. **URL scan.** Every external-looking URL string in a shipped text file must be classified in
   :data:`ALLOWED_URLS` below. Protocol-relative references, CSS ``@import`` of a remote URL and ``fetch()`` /
   ``XMLHttpRequest`` / ``Image().src`` to a remote URL are never allowed.

The claim this supports is "0 external network requests at runtime", not "0 URL strings in the bundle": bundled
libraries legitimately contain XML namespace identifiers, diagnostic text and an attribution link. None is ever
dereferenced - and ``index.html`` carries a ``default-src 'self'`` Content-Security-Policy, so the browser would refuse
the request even if one were. ``frontend-react/tools/verify-offline.mjs`` records the real request log from headless
Chrome as the runtime counterpart.

    python -m geoseek.staging.react_build_pins            # re-pin after `npm run build` (offline)
    python -m geoseek.staging.react_build_pins --check    # verify the committed pins match the build on disk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from geoseek.config import PROJECT_ROOT

FRONTEND_DIR = PROJECT_ROOT / "frontend-react"
WEB_ROOT = PROJECT_ROOT / "src" / "geoseek" / "analyst" / "web"
REACT_ROOT = PROJECT_ROOT / "src" / "geoseek" / "analyst" / "web_react"
PINS_PATH = FRONTEND_DIR / "build-pins.json"

TEXT_SUFFIXES = {".js", ".mjs", ".css", ".html", ".svg", ".json"}

# Vendored files bundled INTO the JS/CSS by Vite aliases (see frontend-react/vite.config.ts). Read-only inputs.
VENDOR_INPUTS = (
    "vendor/three.module.min.js",
    "vendor/OrbitControls.js",
    "vendor/leaflet/leaflet.js",
    "vendor/leaflet/leaflet.css",
)
# Directories whose files the build may copy verbatim (hashed filename, identical bytes).
VENDOR_COPY_DIRS = ("vendor", "fonts")

LIVE_FETCH_KIND = "live_runtime_fetch"       # a submission blocker: must never appear in ALLOWED_URLS (a test enforces it)
XML_NAMESPACE = "xml_namespace_identifier"
ERROR_DOC_TEXT = "diagnostic_message_text"
ATTRIBUTION_LINK = "attribution_link"


@dataclass(frozen=True)
class AllowedUrl:
    url: str
    kind: str
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


ALLOWED_URLS: tuple[AllowedUrl, ...] = (
    AllowedUrl("http://www.w3.org/1998/Math/MathML", XML_NAMESPACE,
               "react-dom namespace constant used with createElementNS; an identifier, never fetched."),
    AllowedUrl("http://www.w3.org/1999/xhtml", XML_NAMESPACE, "react-dom namespace constant; an identifier, never fetched."),
    AllowedUrl("http://www.w3.org/1999/xlink", XML_NAMESPACE,
               "react-dom namespace for xlink:* SVG attributes; an identifier, never fetched."),
    AllowedUrl("http://www.w3.org/2000/svg", XML_NAMESPACE,
               "SVG namespace passed to createElementNS by react-dom, three.js and Leaflet; an identifier, never fetched."),
    AllowedUrl("http://www.w3.org/XML/1998/namespace", XML_NAMESPACE,
               "react-dom namespace for xml:* attributes; an identifier, never fetched."),
    AllowedUrl("https://react.dev/errors", ERROR_DOC_TEXT,
               "react-dom concatenates this into the text of a thrown Error in production builds so a developer can look the "
               "code up. String content only; nothing requests it."),
    AllowedUrl("https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733", ERROR_DOC_TEXT,
               "three.js console.warn text about a lighting-units change; string content only."),
    AllowedUrl("https://github.com/geotiffjs/geotiff.js/issues", ERROR_DOC_TEXT,
               "geotiff.js appends this to the text of an Error it throws when a 64-bit TIFF offset exceeds Number.MAX_SAFE_INTEGER "
               "(DataView64). String content only; nothing requests it. The console reads file headers only, via a Blob in this browser."),
    AllowedUrl("https://leafletjs.com", ATTRIBUTION_LINK,
               "href inside Leaflet's default attribution prefix. The console builds every map with attributionControl:false, "
               "so the markup is never created or rendered, and the page CSP would block a navigation fetch anyway."),
)

_LOCAL_HOSTS = ("127.0.0.1", "localhost")
_ABS_URL_RE = re.compile(r"https?://[^\s\"'`()<>,;\\]+")
_PROTO_RELATIVE_RE = re.compile(r"""["'`]//[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}[^\s"'`()<>]*""")
_AT_IMPORT_RE = re.compile(r"@import\s+url\(\s*['\"]?([^'\")\s]+)")
_NETWORK_CALL_RE = re.compile(
    r"(?:\bfetch\(\s*|\.open\(\s*['\"]\w+['\"]\s*,\s*|new\s+Image\(\)\.src\s*=\s*)['\"`]([^'\"`]+)")
_TRAILING = ".,;:)]}>\"' "


def _is_local(url: str) -> bool:
    return any(h in url for h in _LOCAL_HOSTS)


def _looks_remote(target: str) -> bool:
    return target.startswith(("http://", "https://", "//")) and not _is_local(target)


def external_references(text: str) -> dict[str, list[str]]:
    """Every external-looking reference in ``text``, split by how dangerous it is.

    ``absolute`` URL strings may be allow-listed (:data:`ALLOWED_URLS`); the other three buckets - protocol-relative
    strings, remote ``@import``, remote ``fetch()``/XHR/``Image`` targets - must always be empty.
    """
    absolute = sorted({m.group(0).rstrip(_TRAILING) for m in _ABS_URL_RE.finditer(text)
                       if not _is_local(m.group(0))})
    return {
        "absolute": absolute,
        "protocol_relative": sorted({m.group(0)[1:].rstrip(_TRAILING) for m in _PROTO_RELATIVE_RE.finditer(text)
                                     if not _is_local(m.group(0))}),
        "css_import": sorted({m.group(1) for m in _AT_IMPORT_RE.finditer(text) if _looks_remote(m.group(1))}),
        "network_call": sorted({m.group(1) for m in _NETWORK_CALL_RE.finditer(text) if _looks_remote(m.group(1))}),
    }


def is_allowed(url: str) -> bool:
    """True if ``url`` is an allow-listed string or extends one at a path / query boundary."""
    return any(url == a.url or url.startswith((a.url + "/", a.url + "?", a.url + "#")) for a in ALLOWED_URLS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vendor_source_index() -> dict[str, str]:
    """sha256 -> web-relative path, for every file the build could have copied verbatim."""
    out: dict[str, str] = {}
    for d in VENDOR_COPY_DIRS:
        for p in sorted((WEB_ROOT / d).rglob("*")):
            if p.is_file():
                out.setdefault(sha256_file(p), p.relative_to(WEB_ROOT).as_posix())
    return out


def _locked_versions() -> dict:
    lock = FRONTEND_DIR / "package-lock.json"
    if not lock.is_file():
        return {}
    data = json.loads(lock.read_text(encoding="utf-8"))
    pk = data.get("packages", {})
    keep = ("react", "react-dom", "geotiff", "pako", "lerc", "zstddec", "quick-lru", "xml-utils", "web-worker", "parse-headers",
            "@petamoriken/float16", "vite", "typescript", "@vitejs/plugin-react", "@types/leaflet")
    return {"package_lock_sha256": sha256_file(lock),
            "versions": {k: pk.get(f"node_modules/{k}", {}).get("version") for k in keep}}


def compute_pins() -> dict:
    """Derive the pin record from the build on disk. Raises ValueError listing anything unclassified."""
    if not (REACT_ROOT / "index.html").is_file():
        raise ValueError(f"no build at {REACT_ROOT} - run `npm run build` in frontend-react/ first")
    vendor_by_hash = _vendor_source_index()
    files: dict[str, dict] = {}
    problems: list[str] = []
    for p in sorted(REACT_ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REACT_ROOT).as_posix()
        entry: dict = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        if p.suffix.lower() in TEXT_SUFFIXES:
            refs = external_references(p.read_text(encoding="utf-8", errors="replace"))
            for bucket in ("protocol_relative", "css_import", "network_call"):
                if refs[bucket]:
                    problems.append(f"{rel}: {bucket} reference(s) are never allowed: {refs[bucket]}")
            unlisted = [u for u in refs["absolute"] if not is_allowed(u)]
            if unlisted:
                problems.append(f"{rel}: unclassified external URL(s) - add to ALLOWED_URLS only after review: {unlisted}")
            entry["external_urls"] = refs["absolute"]
        else:
            raw = p.read_bytes()
            if b"http://" in raw or b"https://" in raw:
                problems.append(f"{rel}: binary asset contains an http(s):// string")
            src = vendor_by_hash.get(entry["sha256"])
            if src:
                entry["vendor_source"] = src          # byte-identical copy of an already-vendored file
        files[rel] = entry
    if problems:
        raise ValueError("\n".join(problems))
    inputs = {}
    for rel in VENDOR_INPUTS:
        src = WEB_ROOT / rel
        inputs[rel] = sha256_file(src)
    return {
        "generated_by": "python -m geoseek.staging.react_build_pins",
        "note": ("Hash pins for src/geoseek/analyst/web_react/ (the React console build). tests/test_frontend_offline.py "
                 "fails if any file drifts from its pin, if a file is added or missing, or if a URL string is not "
                 "classified in geoseek/staging/react_build_pins.py::ALLOWED_URLS."),
        "build_toolchain": _locked_versions(),
        "vendored_inputs_bundled": inputs,
        "allowed_urls": [a.as_dict() for a in ALLOWED_URLS],
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the committed pins with the build on disk; write nothing")
    args = ap.parse_args(argv)
    try:
        pins = compute_pins()
    except ValueError as e:
        print(f"[react_build_pins] REFUSED:\n{e}", file=sys.stderr)
        return 1
    if args.check:
        committed = json.loads(PINS_PATH.read_text(encoding="utf-8")) if PINS_PATH.is_file() else {}
        same = committed.get("files") == pins["files"] and committed.get("vendored_inputs_bundled") == pins["vendored_inputs_bundled"]
        print("[react_build_pins] pins match the build on disk" if same else "[react_build_pins] pins are STALE - re-run without --check")
        return 0 if same else 1
    PINS_PATH.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    n_urls = sum(len(f.get("external_urls", [])) for f in pins["files"].values())
    print(f"[react_build_pins] pinned {len(pins['files'])} files, {len(pins['vendored_inputs_bundled'])} vendored inputs; "
          f"{n_urls} classified inert URL strings -> {PINS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

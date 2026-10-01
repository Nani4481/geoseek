"""The offline guarantee: nothing shipped under src/geoseek/analyst/web/ may
reference an external URL.

This replaces the check that used to live inside
test_phase6_presentation.py::test_existing_analyst_endpoints_unaffected. That
version had two problems (found during the Phase 1 redesign audit, see
docs/FRONTEND_AUDIT.md Section 9/10):

  1. It only scanned three hand-picked files (app.js, style.css, index.html).
     globe.js and everything under vendor/ (including three.module.min.js,
     which already contains literal "http://"/"https://" substrings - see
     below) were never inspected at all.
  2. It lived inside a test gated on the full production catalog + change
     report being staged, so on a machine where the ML pipeline hasn't been
     run, this "we ship nothing external" guarantee silently never ran.

This version scans every text file under the web root, has no dependency on
the production catalog (it only reads files already in the repo), and treats
a vendored third-party file as exempt ONLY if it is explicitly allow-listed in
the provenance manifest by relative path AND sha256, with every external URL
the file contains enumerated there too - so a vendor file silently changing
(and possibly gaining a new external reference) fails this test until a human
updates the manifest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from geoseek.config import get_settings

WEB_ROOT = get_settings().project_root / "src" / "geoseek" / "analyst" / "web"
MANIFEST_PATH = get_settings().provenance_manifest_path

# Files of these kinds are scanned with the full regex sweep below (URLs,
# protocol-relative references, @import, fetch/XHR/Image targets). "Any new
# file" of one of these kinds is covered automatically - nothing to update
# here when a new .js/.css/.html/.svg/.json file is added to the web root.
TEXT_SUFFIXES = {".js", ".css", ".html", ".svg", ".json"}

# localhost / the app's own backend is explicitly permitted everywhere.
_LOCAL_HOSTS = ("127.0.0.1", "localhost")

_ABS_URL_RE = re.compile(r"https?://[^\s\"'()<>]+")
# protocol-relative //host. A bare "//identifier.method"-shaped JS/CSS line
# comment (e.g. OrbitControls.js's "//scope.dispatchEvent(...)") is
# syntactically indistinguishable from "//host.tld" by character class alone
# - "scope.dispatchEvent" and "cdn.example.com" have the same shape. The
# reliable signal is context: a real protocol-relative reference is always
# inside a quoted string or a CSS url(...), so this only fires when the "//"
# is immediately preceded by a quote character.
_PROTO_RELATIVE_RE = re.compile(r"""["'`]//[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}[^\s"'()<>]*""")
_AT_IMPORT_URL_RE = re.compile(r"@import\s+url\(\s*['\"]?([^'\")\s]+)")
_NETWORK_CALL_RE = re.compile(
    r"(?:\bfetch\(\s*|\.open\(\s*['\"]\w+['\"]\s*,\s*|new\s+Image\(\)\.src\s*=\s*)['\"`]([^'\"`]+)"
)


def _is_local(url: str) -> bool:
    return any(h in url for h in _LOCAL_HOSTS)


def _looks_external(target: str) -> bool:
    return (target.startswith("http://") or target.startswith("https://") or target.startswith("//")) \
        and not _is_local(target)


_TRAILING_PUNCT = ".,;:)]}>\"' "


def _findings(text: str) -> list[str]:
    """Every external-looking reference in `text`, as human-readable strings."""
    out = []
    for m in _ABS_URL_RE.finditer(text):
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        if not _is_local(url):
            out.append(url)
    for m in _PROTO_RELATIVE_RE.finditer(text):
        url = m.group(0)[1:].rstrip(_TRAILING_PUNCT)  # drop the leading quote char that anchored the match
        if not _is_local(url):
            out.append(url)
    for m in _AT_IMPORT_URL_RE.finditer(text):
        if _looks_external(m.group(1)):
            out.append("@import url(" + m.group(1) + ")")
    for m in _NETWORK_CALL_RE.finditer(text):
        if _looks_external(m.group(1)):
            out.append("network call -> " + m.group(1))
    return out


def _iter_scan_files():
    for p in sorted(WEB_ROOT.rglob("*")):
        if p.is_file():
            yield p


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _to_web_relative(local_path: str) -> str | None:
    if not local_path:
        return None
    try:
        return Path(local_path).resolve().relative_to(WEB_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _load_vendor_allowlist() -> dict:
    """{relative/path/from/web_root.ext: {"sha256": ..., "external_urls": {...}}}

    Sourced from data/provenance_manifest.json's own "artifacts" list (the
    same record scripts/stage_threejs.py etc. already write), reading two
    fields this test adds meaning to: "external_urls" (a list of {url, kind,
    note} dicts the file is allowed to contain) and "sha256" (must match the
    file on disk right now, or the allowlist entry is stale and doesn't
    apply). An artifact entry with no "external_urls" key is not an
    allowlisted exemption at all - it's just an ordinary staged-data record
    (the manifest has hundreds of those, e.g. every Sentinel-2 band COG) and
    is ignored here.
    """
    if not MANIFEST_PATH.is_file():
        pytest.fail(
            f"Provenance manifest missing at {MANIFEST_PATH}. The vendor URL "
            f"allowlist lives there - without it every vendored file under "
            f"vendor/ would fail this test as an un-allowlisted external "
            f"reference. Run the staging scripts first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    allow: dict[str, dict] = {}

    def _record(entry: dict) -> None:
        if "external_urls" not in entry:
            return
        rel = _to_web_relative(entry.get("local_path", ""))
        if rel is None:
            return
        allow[rel] = {
            "sha256": entry.get("sha256"),
            "urls": {u["url"] for u in entry["external_urls"]},
        }

    for art in manifest.get("artifacts", []):
        _record(art)
        for sub in art.get("files", []):  # grouped entries, e.g. earth_textures_vendor
            _record(sub)
    return allow


_ALL_FILES = list(_iter_scan_files())
_TEXT_FILES = [p for p in _ALL_FILES if p.suffix.lower() in TEXT_SUFFIXES]
_OTHER_FILES = [p for p in _ALL_FILES if p.suffix.lower() not in TEXT_SUFFIXES]


@pytest.mark.parametrize(
    "relpath", [p.relative_to(WEB_ROOT).as_posix() for p in _TEXT_FILES], ids=lambda s: s
)
def test_no_external_urls_in_text_asset(relpath):
    path = WEB_ROOT / relpath
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = _findings(text)
    if not findings:
        return

    allow = _load_vendor_allowlist()
    entry = allow.get(relpath)
    if entry is None:
        pytest.fail(
            f"{relpath} references what looks like an external URL and is not "
            f"in the vendor allowlist (data/provenance_manifest.json): {findings}"
        )
    current_hash = _sha256(path)
    if current_hash != entry["sha256"]:
        pytest.fail(
            f"{relpath} has changed since it was allow-listed "
            f"(manifest sha256 {entry['sha256']!r} != current {current_hash!r}). "
            f"Re-review its external URLs and update the manifest before this test can pass again."
        )
    undeclared = [f for f in findings if f not in entry["urls"] and not any(f in u for u in entry["urls"])]
    assert not undeclared, (
        f"{relpath} contains external references not enumerated in its "
        f"provenance-manifest allowlist entry: {undeclared}"
    )


def _load_vendor_hashes() -> dict[str, str]:
    """{web-relative path: pinned sha256} for EVERY manifest entry (artifact or grouped sub-file) that points
    into the web root - not only the ones that enumerate external URLs."""
    if not MANIFEST_PATH.is_file():
        pytest.fail(f"Provenance manifest missing at {MANIFEST_PATH}; run `python -m geoseek.staging.vendor_provenance`.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for art in manifest.get("artifacts", []):
        for entry in (art, *art.get("files", [])):
            rel = _to_web_relative(entry.get("local_path", ""))
            if rel is not None and entry.get("sha256"):
                out[rel] = entry["sha256"]
    return out


_VENDOR_FILES = [p for p in _ALL_FILES if p.relative_to(WEB_ROOT).parts[0] == "vendor"]


@pytest.mark.parametrize("relpath", [p.relative_to(WEB_ROOT).as_posix() for p in _VENDOR_FILES], ids=lambda s: s)
def test_every_vendored_file_is_hash_pinned_in_the_manifest(relpath):
    """The hash check used to run only for vendored files that happened to contain a URL, so a file without any
    (OrbitControls.js) could drift from its recorded hash unnoticed. Every file under vendor/ - text or binary,
    URL-bearing or not - must be pinned, and must still hash to the pin."""
    pins = _load_vendor_hashes()
    assert relpath in pins, (f"{relpath} is vendored but not pinned in the provenance manifest "
                             f"(run `python -m geoseek.staging.vendor_provenance`)")
    assert _sha256(WEB_ROOT / relpath) == pins[relpath], (
        f"{relpath} no longer hashes to its pinned sha256. Either it changed (re-review it) or the checkout converted "
        f"line endings (vendor/** must be '-text' in .gitattributes).")


@pytest.mark.parametrize(
    "relpath", [p.relative_to(WEB_ROOT).as_posix() for p in _OTHER_FILES], ids=lambda s: s
)
def test_no_external_urls_in_binary_asset(relpath):
    """Binary assets (images, fonts, ...) can't meaningfully contain a fetch()
    call, but embedded metadata (EXIF, an ICC profile comment, ...) could in
    principle carry a URL. A cheap raw-byte substring check costs nothing and
    closes that gap - "recursively scan EVERY file" means every file."""
    path = WEB_ROOT / relpath
    raw = path.read_bytes()
    hits = [needle for needle in (b"http://", b"https://") if needle in raw]
    if not hits:
        return
    allow = _load_vendor_allowlist()
    entry = allow.get(relpath)
    assert entry is not None, f"{relpath} (binary) contains {hits} and is not in the vendor allowlist"
    assert _sha256(path) == entry["sha256"], f"{relpath} changed since being allow-listed; re-review and update the manifest"


def test_tokens_css_is_clean():
    """Explicit call-out for the design-token layer: it must be clean under
    the exact same rules as everything else - this is redundant with the
    parametrized sweep above (it's under WEB_ROOT and has a scanned suffix)
    but is kept as its own named test so a regression here is unambiguous in
    a test report, not just "one of N parametrized cases failed".

    The analyst UI rebuild replaced the old root-level tokens.css/tokens.js
    (Phase 1 of the prior redesign) with css/tokens.css and no separate JS
    token file - see docs/FRONTEND_AUDIT.md for the superseded layout."""
    path = WEB_ROOT / "css" / "tokens.css"
    assert path.is_file(), f"tokens.css not found under {WEB_ROOT / 'css'}"
    findings = _findings(path.read_text(encoding="utf-8"))
    assert not findings, f"tokens.css contains external references: {findings}"


def test_web_root_has_files_to_scan():
    """Guards against the parametrize lists above silently being empty (e.g.
    WEB_ROOT resolved to the wrong directory) and every test in this module
    trivially "passing" by having nothing to check.

    The analyst UI rebuild vendors fonts (binary) instead of three.js - the
    prior redesign's vendor/three.module.min.js no longer exists by design
    (the globe view it backed was retired), so this checks for a vendored
    font instead of that specific former dependency."""
    assert len(_TEXT_FILES) >= 4  # index.html, tokens.css, api-client.js, shell.js at minimum
    assert any(p.suffix.lower() == ".woff2" for p in _OTHER_FILES), \
        "expected at least one vendored .woff2 font to exist and be scanned"

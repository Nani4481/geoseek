"""The committed vendor-URL allowlist (geoseek.staging.vendor_provenance): complete, classified, never a live
fetch, regenerable offline, and the checkout rule that keeps vendored bytes identical on every platform."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from geoseek.staging import vendor_provenance as VP

ROOT = Path(__file__).resolve().parents[1]


def _offline_test_module():
    spec = importlib.util.spec_from_file_location("offline_guard", ROOT / "tests" / "test_frontend_offline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_vendored_url_is_classified_as_a_live_runtime_fetch():
    kinds = {u.kind for urls in VP.VENDOR_URLS.values() for u in urls}
    assert VP.LIVE_FETCH_KIND not in kinds
    assert all(u.note.strip() for urls in VP.VENDOR_URLS.values() for u in urls), "every URL needs a written classification note"


def test_the_allowlist_covers_every_url_the_offline_guard_finds_in_the_vendored_files():
    guard = _offline_test_module()
    for rel, urls in VP.VENDOR_URLS.items():
        text = (VP.WEB_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        listed = [u.url for u in urls]
        undeclared = [f for f in guard._findings(text) if not any(f in l or l in f for l in listed)]
        assert not undeclared, f"{rel}: URL(s) present in the file but not in VENDOR_URLS: {undeclared}"


def test_every_vendored_file_is_either_url_free_or_has_a_declared_allowlist_entry():
    guard = _offline_test_module()
    for path in VP.vendor_files():
        rel = path.relative_to(VP.WEB_ROOT).as_posix()
        if path.suffix.lower() in guard.TEXT_SUFFIXES and guard._findings(path.read_text(encoding="utf-8", errors="replace")):
            assert VP.VENDOR_URLS.get(rel), f"{rel} contains URL strings but has no classified allowlist entry"


def test_update_manifest_pins_everything_idempotently_and_check_agrees():
    manifest: dict = {"artifacts": []}
    first = VP.update_manifest(manifest)
    assert first and VP.check_manifest(manifest) == []
    assert VP.update_manifest(manifest) == []                                  # idempotent
    # a hash drift is reported by check and repaired (and logged) by update
    entry = next(a for a in manifest["artifacts"] if a["name"] == "chartjs_vendor")
    entry["sha256"] = "0" * 64
    assert any("chart.umd.js" in p for p in VP.check_manifest(manifest))
    assert any("re-pinned" in c for c in VP.update_manifest(manifest))
    assert VP.check_manifest(manifest) == []
    # the classified URLs are attached to the entries
    by_name = {a["name"]: a for a in manifest["artifacts"]}
    assert len(by_name["chartjs_vendor"]["external_urls"]) == 3
    assert by_name["leaflet_leaflet_js_vendor"]["external_urls"][1]["kind"] == "other"


def test_leaflet_artifact_names_do_not_collide():
    names = [s["name"] for k, s in VP.SOURCES.items() if k.startswith("vendor/leaflet/")]
    assert len(names) == len(set(names)) == 5
    assert VP._leaflet_name("leaflet.js") != VP._leaflet_name("leaflet.css")


def test_gitattributes_keeps_vendored_files_byte_identical_across_checkouts():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "src/geoseek/analyst/web/vendor/** -text" in attrs

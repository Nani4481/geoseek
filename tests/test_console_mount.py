"""/app/ serves the React console and cannot serve anything else.

The requirement is unconditional: whatever happens to the build directory, the mount order, the 404 handler or a catch-all
route, a request under /app/ is answered either by the console (200, the committed build) or by a 503 that names the missing
directory and the command that produces it. It is never a bare 404 and never some other page - a silent fallback would be
indistinguishable from a working interface. The retired /react/ path only redirects; it never serves content itself.

Everything runs against the real ``geoseek.search.api.app`` (no lifespan: no model or index is loaded). Where a case needs a
different build directory, the console mount's directory is pointed at a temp directory for the duration of the test.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, WebSocketRequestValidationError
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles

from geoseek.search import api
from geoseek.search.api import _REACT_BUILD_CMD, _REACT_DIR, _ConsoleApp, app

CONSOLE_MARK = "GeoSeek Console"                      # the <title> of the built page
ROOT_MARK = '<div id="root"></div>'
APP_PATHS = ["/app/", "/app/index.html", "/app/assets/index-x.js", "/app/favicon.svg", "/app/some/deep/path", "/app/x.html"]


def _console_mount() -> Mount:
    (m,) = [r for r in app.routes if isinstance(r, Mount) and r.path == "/app"]
    return m


def _point_console_at(monkeypatch, directory: Path) -> None:
    """Make the real mount serve ``directory`` (restored by monkeypatch)."""
    console = _console_mount().app
    monkeypatch.setattr(console, "directory", directory)
    monkeypatch.setattr(console, "_static", StaticFiles(directory=str(directory), html=True, check_dir=False))


def _client() -> TestClient:
    return TestClient(app)                            # no `with`: the lifespan is deliberately not started


def _assert_visible_failure(r, directory: Path) -> None:
    """503, naming the directory and the build command, and carrying no console content."""
    assert r.status_code == 503, (r.request.url.path, r.status_code, r.text[:200])
    assert "text/html" in r.headers["content-type"]
    body = html.unescape(r.text)
    assert str(directory) in body, "the message must name the missing directory"
    assert _REACT_BUILD_CMD in body, "the message must name the build command"
    assert CONSOLE_MARK not in body and ROOT_MARK not in body and "/app/assets/" not in body, "failure page carries console content"


# ---- the console itself ---------------------------------------------------------------------------------------------


def test_app_serves_exactly_the_committed_react_build():
    c = _client()
    r = c.get("/app/")
    assert r.status_code == 200 and CONSOLE_MARK in r.text and ROOT_MARK in r.text
    assert r.content == (_REACT_DIR / "index.html").read_bytes()
    for p in sorted((_REACT_DIR / "assets").glob("*"))[:5] + [_REACT_DIR / "favicon.svg"]:
        rel = p.relative_to(_REACT_DIR).as_posix()
        got = c.get(f"/app/{rel}")
        assert got.status_code == 200 and got.content == p.read_bytes(), rel


def test_a_missing_file_in_a_present_build_is_a_404_not_the_index_page():
    """No single-page fallback: an unknown path must not be answered with index.html (or anything else)."""
    c = _client()
    for p in ("/app/assets/no-such-file-123.js", "/app/no/such/page", "/app/nonexistent.html"):
        r = c.get(p)
        assert r.status_code == 404, (p, r.status_code)
        assert CONSOLE_MARK not in r.text and ROOT_MARK not in r.text, f"{p} was answered with the console page"


# ---- the build is absent --------------------------------------------------------------------------------------------


def _build_dirs(tmp_path: Path) -> dict[str, Path]:
    no_index = tmp_path / "no_index"
    (no_index / "assets").mkdir(parents=True)
    (no_index / "assets" / "index-abc.js").write_text("console.log('a decoy that must not be served')")
    (no_index / "favicon.svg").write_text("<svg/>")
    empty = tmp_path / "empty"
    empty.mkdir()
    index_is_dir = tmp_path / "index_is_dir"
    (index_is_dir / "index.html").mkdir(parents=True)
    return {"missing_directory": tmp_path / "no_such_build_dir", "directory_without_index_html": no_index,
            "empty_directory": empty, "index_html_is_a_directory": index_is_dir}


@pytest.mark.parametrize("case", ["missing_directory", "directory_without_index_html", "empty_directory", "index_html_is_a_directory"])
def test_absent_build_is_a_503_naming_the_directory_and_the_command_for_every_path(case, tmp_path, monkeypatch):
    directory = _build_dirs(tmp_path)[case]
    _point_console_at(monkeypatch, directory)
    c = _client()
    for p in APP_PATHS + ["/app"]:                    # "/app" is redirected to "/app/" first; the final answer must still be the 503
        _assert_visible_failure(c.get(p), directory)
    assert c.head("/app/").status_code == 503                                  # HEAD has no body; the status is what matters


@pytest.mark.parametrize("case", ["missing_directory", "directory_without_index_html"])
def test_the_api_keeps_working_when_the_build_is_absent(case, tmp_path, monkeypatch):
    _point_console_at(monkeypatch, _build_dirs(tmp_path)[case])
    c = _client()
    root = c.get("/")
    assert root.status_code == 200 and root.json()["ui"] == "/app/"
    assert c.get("/openapi.json").status_code == 200 and c.get("/docs").status_code == 200


def test_building_the_app_object_does_not_need_the_build_directory():
    """Mounting is unconditional: a missing directory must not stop the backend from starting."""
    console = _ConsoleApp(Path("/definitely/not/a/build/dir"))
    a = FastAPI()
    a.mount("/app", console, name="console")
    r = TestClient(a).get("/app/")
    assert r.status_code == 503 and "/definitely/not/a/build/dir" in html.unescape(r.text).replace("\\", "/")


def test_the_mount_is_registered_even_when_the_build_is_absent_at_import_time(monkeypatch):
    """A fresh import of the API module with the build directory hidden must still mount /app (and answer 503 there).
    Guards against the mount being made conditional on the directory existing, which would leave /app/ as a bare 404."""
    import importlib.util

    real_is_dir, real_is_file = Path.is_dir, Path.is_file
    monkeypatch.setattr(Path, "is_dir", lambda self: False if "web_react" in str(self) else real_is_dir(self))
    monkeypatch.setattr(Path, "is_file", lambda self: False if "web_react" in str(self) else real_is_file(self))
    spec = importlib.util.spec_from_file_location("geoseek_search_api_without_build", api.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert [r.path for r in fresh.app.routes if isinstance(r, Mount)] == ["/app"]
    r = TestClient(fresh.app).get("/app/")
    assert r.status_code == 503 and "npm run build" in html.unescape(r.text)


def test_a_build_that_disappears_while_the_server_runs_fails_the_same_way_and_recovers(tmp_path, monkeypatch):
    live = tmp_path / "live"
    shutil.copytree(_REACT_DIR, live)
    _point_console_at(monkeypatch, live)
    c = _client()
    assert c.get("/app/").status_code == 200
    (live / "index.html").unlink()                                        # index.html removed under a running server
    _assert_visible_failure(c.get("/app/"), live)
    _assert_visible_failure(c.get("/app/assets/" + next((live / "assets").glob("index-*.js")).name), live)   # even for files that still exist
    shutil.rmtree(live)                                                   # then the whole directory
    _assert_visible_failure(c.get("/app/"), live)
    shutil.copytree(_REACT_DIR, live)                                     # and it recovers when the build comes back
    r = c.get("/app/")
    assert r.status_code == 200 and CONSOLE_MARK in r.text


# ---- mount order, catch-all routes, the 404 handler -----------------------------------------------------------------


def _winners(a: FastAPI, path: str) -> list:
    scope = {"type": "http", "path": path, "method": "GET", "root_path": "", "headers": []}
    return [r for r in a.routes if r.matches(scope)[0] == Match.FULL]


def test_exactly_one_route_can_answer_a_path_under_app_and_it_is_the_console():
    for p in APP_PATHS:
        win = _winners(app, p)
        assert len(win) == 1 and win[0] is _console_mount(), (p, [getattr(w, "path", w) for w in win])
    # a console mount that is not the first thing to match would be shadowed by whatever matches earlier
    assert [r for r in app.routes if isinstance(r, Mount)] == [_console_mount()], "no other static mount may exist"


def test_the_mount_order_guard_detects_a_shadowing_catch_all():
    """The guard above is only worth having if it fails when the order is wrong: register a catch-all BEFORE the mount."""
    bad = FastAPI()
    bad.add_api_route("/{anything:path}", lambda: {"page": "something else"})
    bad.mount("/app", _ConsoleApp(_REACT_DIR), name="console")
    mount = [r for r in bad.routes if isinstance(r, Mount)][0]
    win = _winners(bad, "/app/")
    assert win[0] is not mount, "the catch-all registered first must be what matches first (so the guard can see it)"
    assert TestClient(bad).get("/app/").json() == {"page": "something else"}      # i.e. exactly the failure the guard prevents


def test_a_catch_all_registered_after_the_mount_cannot_take_over_app(tmp_path):
    good = FastAPI()
    good.mount("/app", _ConsoleApp(tmp_path / "no_such_build"), name="console")
    good.add_api_route("/{anything:path}", lambda: {"page": "something else"})
    c = TestClient(good)
    _assert_visible_failure(c.get("/app/"), tmp_path / "no_such_build")
    assert c.get("/elsewhere").json() == {"page": "something else"}              # the catch-all works outside /app


def test_the_real_app_has_no_catch_all_route_and_no_custom_404_handler():
    catch_alls = [r.path for r in app.routes if isinstance(r, APIRoute) and ":path}" in r.path]
    assert catch_alls == ["/react/{path:path}"], f"a catch-all route exists: {catch_alls}"     # the redirect only
    assert set(app.exception_handlers) <= {StarletteHTTPException, RequestValidationError, WebSocketRequestValidationError},         f"custom exception handler(s) registered: {set(app.exception_handlers)}"
    assert 404 not in app.exception_handlers, "a custom 404 handler could turn unknown paths into some other page"
    r = _client().get("/definitely/not/a/route")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}


def test_the_old_interface_is_gone_from_disk_and_from_the_routes():
    assert not (Path(api.__file__).resolve().parents[1] / "analyst" / "web").exists(), "the old interface directory is back"
    assert not hasattr(api, "_WEB_DIR")


# ---- the retired /react/ path ---------------------------------------------------------------------------------------


def test_react_path_only_redirects_and_never_serves_content():
    c = _client()
    asset = next((_REACT_DIR / "assets").glob("index-*.js")).name
    for p in ("/react", "/react/", "/react/index.html", f"/react/assets/{asset}", "/react/favicon.svg", "/react/no/such/thing"):
        r = c.get(p, follow_redirects=False)
        assert r.status_code == 308, (p, r.status_code)
        assert r.headers["location"].startswith("/app/")
        assert CONSOLE_MARK not in r.text and ROOT_MARK not in r.text and len(r.content) < 200, f"{p} served content itself"
    assert [r.path for r in app.routes if isinstance(r, Mount) and r.path.startswith("/react")] == [], "/react must not be a mount"


def test_react_redirect_leads_to_the_503_when_the_build_is_absent(tmp_path, monkeypatch):
    directory = tmp_path / "no_such_build"
    _point_console_at(monkeypatch, directory)
    c = _client()
    assert c.get("/react/", follow_redirects=False).status_code == 308               # the redirect does not depend on the build
    _assert_visible_failure(c.get("/react/", follow_redirects=True), directory)
    _assert_visible_failure(c.get("/react/assets/index-abc.js", follow_redirects=True), directory)

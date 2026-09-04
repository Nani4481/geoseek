"""Phase 6 Step E - prove the analyst UI runs with the network disabled, and
measure per-view interactive latency.

  python scripts/verify_offline_perf.py

The network is hard-disabled IN PROCESS before the app is built: every
``socket.connect`` / ``connect_ex`` / ``getaddrinfo`` for a non-loopback address
raises. RemoteCLIP, FAISS, the SQLite catalog, rasterio and the probability
rasters are all local files, so the whole stack - search, queue, candidate
detail, imagery, decision, audit, export, discovery - has to work offline or
fail loudly here.

Audit writes are redirected to a throwaway COPY of the catalog so this probe
does not touch data/index.
"""

from __future__ import annotations

import shutil
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------- kill the network
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GAI = socket.getaddrinfo
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_leaks: list[str] = []


def _addr_ok(address) -> bool:
    try:
        host = address[0] if isinstance(address, tuple) else str(address)
    except Exception:
        return False
    return host in _LOOPBACK


def _blocked_connect(self, address, *a, **k):
    if not _addr_ok(address):
        _leaks.append(f"connect -> {address}")
        raise OSError("NETWORK DISABLED (Phase 6 offline verification)")
    return _REAL_CONNECT(self, address, *a, **k)


def _blocked_connect_ex(self, address, *a, **k):
    if not _addr_ok(address):
        _leaks.append(f"connect_ex -> {address}")
        return 111
    return _REAL_CONNECT_EX(self, address, *a, **k)


def _blocked_gai(host, *a, **k):
    if host not in _LOOPBACK and host is not None:
        _leaks.append(f"getaddrinfo -> {host}")
        raise socket.gaierror("NETWORK DISABLED (Phase 6 offline verification)")
    return _REAL_GAI(host, *a, **k)


socket.socket.connect = _blocked_connect
socket.socket.connect_ex = _blocked_connect_ex
socket.getaddrinfo = _blocked_gai

# prove the block is live
try:
    socket.create_connection(("8.8.8.8", 53), timeout=1)
    print("!! network block FAILED - a real outbound connection succeeded")
    raise SystemExit(1)
except OSError:
    print("network block active: outbound connect to 8.8.8.8:53 rejected in process\n")
_leaks.clear()   # forget this deliberate probe; from here on any entry is a real leak

# ---------------------------------------------------------------- build the app offline
from fastapi.testclient import TestClient  # noqa: E402

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.change.analyze import OUT_DIR  # noqa: E402
from geoseek.config import get_settings  # noqa: E402

if not (get_settings().index_dir / "tiles.faiss").is_file():
    print("no production index - run the ingest pipeline first"); raise SystemExit(1)
if not (OUT_DIR / "ayodhya_change_report.json").is_file():
    print("no change report - run `python -m geoseek.change.analyze` first"); raise SystemExit(1)

t_boot = time.perf_counter()
from geoseek.search import api as api_mod  # noqa: E402

client = TestClient(api_mod.app)
client.__enter__()      # runs lifespan: SearchEngine (RemoteCLIP + FAISS) + AnalystService
boot_s = time.perf_counter() - t_boot

# redirect audit writes to a temp copy of the catalog
_tmp = Path(tempfile.mkdtemp(prefix="geoseek_offline_")) / "tiles.sqlite"
shutil.copy2(get_settings().index_dir / "tiles.sqlite", _tmp)
api_mod._analyst.repo = SQLiteMetadataRepository(_tmp)

print(f"app booted offline in {boot_s:.1f}s "
      f"(RemoteCLIP + FAISS + catalog + change report all from local files)\n")


# ---------------------------------------------------------------- timing helper
def bench(label, fn, n=12, budget_ms=1000.0):
    fn()  # warm
    xs = []
    for _ in range(n):
        t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000)
    p50, p95, mx = statistics.median(xs), sorted(xs)[max(0, int(n * 0.95) - 1)], max(xs)
    ok = p95 < budget_ms
    RESULTS.append((label, p50, p95, mx, ok))
    print(f"  {label:<34} p50 {p50:7.1f} ms   p95 {p95:7.1f} ms   max {mx:7.1f} ms   "
          f"{'OK' if ok else 'OVER BUDGET'}")
    return ok


RESULTS: list = []
CID = None


def run():
    global CID
    print("per-view interactive latency (network disabled, N=12, budget < 1000 ms):\n")

    print(" SEARCH view")
    bench("GET /health", lambda: client.get("/health"))
    bench("GET /stats", lambda: client.get("/stats"))
    bench("GET /search/text", lambda: client.get("/search/text", params={"q": "an open water reservoir", "k": 30}))
    png1 = (b"\x89PNG\r\n\x1a\n")  # not used; image search uses a synthetic array
    import base64
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 96, 3), "uint8")).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    bench("POST /search/image", lambda: client.post("/search/image", json={"image_base64": b64, "k": 20}))

    print("\n REVIEW QUEUE view")
    bench("GET /candidates (queue, 400)", lambda: client.get("/candidates", params={"limit": 400}))
    bench("GET /candidates (filtered)",
          lambda: client.get("/candidates", params={"change_type": "construction", "min_confidence": 0.9}))

    CID = client.get("/candidates", params={"limit": 1}).json()["candidates"][0]["candidate_id"]

    print("\n CANDIDATE DETAIL view")
    bench("GET /candidates/{id} (detail)", lambda: client.get(f"/candidates/{CID}"))
    # imagery: 'cold' = first render of a (candidate,date,view); 'warm' = LRU hit
    from geoseek.analyst.imagery import _render_cached

    def imagery_cold():
        _render_cached.cache_clear()
        client.get(f"/candidates/{CID}/imagery", params={"date": "2019", "view": "rgb"})

    bench("GET .../imagery rgb (cold)", imagery_cold, n=8)
    bench("GET .../imagery rgb (warm/LRU)",
          lambda: client.get(f"/candidates/{CID}/imagery", params={"date": "2019", "view": "rgb"}))
    bench("GET .../imagery overlay (cold)", lambda: (_render_cached.cache_clear(),
          client.get(f"/candidates/{CID}/imagery", params={"date": "2024", "view": "overlay"}))[1], n=8)

    print("\n DECISION + AUDIT + EXPORT")
    bench("POST /candidates/{id}/decision",
          lambda: client.post(f"/candidates/{CID}/decision",
                              json={"decision": "confirm", "note": "offline perf probe", "analyst": "probe"}))
    bench("GET /audit", lambda: client.get("/audit"))
    bench("POST /export (filtered ~30)", lambda: client.post(
        "/export", json={"filters": {"change_type": "construction", "min_confidence": 0.9}, "format": "both"}),
        n=8)
    bench("POST /export (all 1104)", lambda: client.post("/export", json={"format": "geojson"}),
          n=5, budget_ms=3000.0)

    print("\n DISCOVERY view")
    bench("GET /discovery/clusters", lambda: client.get("/discovery/clusters"))
    bench("GET /candidates/{id}/similar", lambda: client.get(f"/candidates/{CID}/similar", params={"k": 8}))
    bench("GET /discovery/similar (lon,lat)",
          lambda: client.get("/discovery/similar", params={"lon": 82.19, "lat": 26.79, "k": 8}))


def functional_checks():
    print("\nfunctional checks (every view actually works offline):")
    checks = []
    h = client.get("/health").json()
    checks.append(("health / stats", h["status"] == "ok" and client.get("/stats").json()["index"]["tiles"] > 0))
    s = client.get("/search/text", params={"q": "river", "k": 5}).json()
    checks.append(("search returns hits", s["count"] >= 1))
    q = client.get("/candidates", params={"limit": 20}).json()
    checks.append(("queue lists candidates + footprints",
                   q["total"] > 1000 and q["candidates"][0]["geometry"]["type"] == "Polygon"))
    d = client.get(f"/candidates/{CID}").json()
    checks.append(("detail has evidence + suppression trace + provenance",
                   bool(d["confidence_breakdown"]) and len(d["suppression"]["trace"]) == 5
                   and d["provenance"]["observations"][0]["scene"]["source_url"].startswith("https://")))
    im = client.get(f"/candidates/{CID}/imagery", params={"date": "2021", "view": "overlay"})
    checks.append(("imagery renders PNG", im.headers["content-type"] == "image/png"
                   and im.content[:8] == b"\x89PNG\r\n\x1a\n"))
    aud = client.get("/audit", params={"candidate_id": CID}).json()
    checks.append(("decision written to append-only audit",
                   aud["append_only"] and any(x["analyst"] == "probe" for x in aud["decisions"])))
    ex = client.post("/export", json={"candidate_ids": [CID], "format": "both"}).json()
    checks.append(("export carries provenance",
                   ex["geojson"]["features"][0]["properties"]["weights_sha256"]
                   and "csv" in ex))
    sim = client.get(f"/candidates/{CID}/similar", params={"k": 5}).json()
    cl = client.get("/discovery/clusters").json()
    checks.append(("discovery KNN + clusters", len(sim["results"]) >= 1 and cl["available"]))
    ui = client.get("/app/")
    checks.append(("frontend bundle served", ui.status_code == 200 and "<canvas" in ui.text))

    allok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        allok = allok and ok
    return allok


if __name__ == "__main__":
    run()
    fn_ok = functional_checks()

    print("\n" + "=" * 78)
    over = [r for r in RESULTS if not r[4]]
    print("LATENCY SUMMARY (p95):")
    for label, p50, p95, mx, ok in RESULTS:
        print(f"  {'ok ' if ok else 'OVER'}  {label:<34} {p95:8.1f} ms")
    print("-" * 78)
    if _leaks:
        print(f"!! {len(_leaks)} network access attempt(s) were blocked: {_leaks[:5]}")
    else:
        print("no process made (or attempted) a non-loopback network call")
    print(f"functional checks: {'ALL PASS' if fn_ok else 'FAILURES ABOVE'}")
    if over:
        print(f"NOTE: {len(over)} path(s) over the 1 s interactive budget: "
              + ", ".join(f"{o[0]} ({o[2]:.0f} ms max)" for o in over))
    else:
        print("every interactive path is well under the 1 s budget")
    print("=" * 78)
    client.__exit__(None, None, None)
    raise SystemExit(0 if (fn_ok and not _leaks) else 2)

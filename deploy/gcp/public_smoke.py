"""Validate the anonymously accessible deployed HTTPS service."""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request


BASE = "https://geoseek.8.234.117.216.sslip.io"
TLS = ssl.create_default_context()


def request(path: str, *, timeout: int = 120):
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(BASE + path),
            timeout=timeout,
            context=TLS,
        )
    except urllib.error.HTTPError as error:
        response = error
    body = response.read()
    return response.status, response.headers, body, round((time.perf_counter() - started) * 1000, 1)


def main() -> None:
    result: dict[str, object] = {"base_url": BASE}
    status, _, body, elapsed = request("/app/")
    result["public_frontend"] = {"status": status, "ms": elapsed, "bytes": len(body)}
    assert status == 200 and b"GeoSeek" in body

    status, _, body, elapsed = request("/health")
    health = json.loads(body)
    result["health"] = {"status": status, "ms": elapsed, **health}
    assert status == 200 and health["status"] == "ok"

    query = urllib.parse.urlencode({"q": "a river with sandbars", "k": 3})
    status, _, body, elapsed = request(f"/search/text?{query}")
    search = json.loads(body)
    result["semantic_search"] = {"status": status, "ms": elapsed, "count": search["count"]}
    assert status == 200 and search["count"] > 0

    tile_id = search["results"][0]["tile_id"]
    status, headers, body, elapsed = request(f"/tile/{tile_id}/thumbnail")
    result["thumbnail"] = {
        "status": status,
        "ms": elapsed,
        "content_type": headers.get_content_type(),
        "bytes": len(body),
        "tile_id": tile_id,
    }
    assert status == 200 and headers.get_content_type().startswith("image/")

    status, _, body, elapsed = request("/candidates?limit=1")
    candidates = json.loads(body)
    candidate = candidates["candidates"][0]["candidate_id"]
    result["candidates"] = {"status": status, "ms": elapsed, "candidate_id": candidate}

    status, headers, body, elapsed = request(
        f"/candidates/{urllib.parse.quote(candidate)}/imagery?view=overlay"
    )
    result["change_overlay"] = {
        "status": status,
        "ms": elapsed,
        "content_type": headers.get_content_type(),
        "bytes": len(body),
    }
    assert status == 200 and headers.get_content_type().startswith("image/")

    status, headers, body, elapsed = request("/discovery/cluster-map.png")
    result["cluster_map"] = {
        "status": status,
        "ms": elapsed,
        "content_type": headers.get_content_type(),
        "bytes": len(body),
    }
    assert status == 200 and headers.get_content_type().startswith("image/")

    status, _, body, elapsed = request("/detect/observations")
    detections = json.loads(body)
    result["detections"] = {
        "status": status,
        "ms": elapsed,
        "observations": len(detections["observations"]),
    }
    assert status == 200
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

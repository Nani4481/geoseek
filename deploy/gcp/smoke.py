"""Exercise real CPU API paths against a disposable catalog snapshot.

Run from repo root. Original analyst decisions/watch areas remain untouched.
Uses local staged reports/imagery and never downloads or rebuilds anything.
"""
import json
from contextlib import closing
import os
import sqlite3
import tempfile
import time
from pathlib import Path

os.environ["GEOSEEK_DEVICE"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"


def main():
    from fastapi.testclient import TestClient
    result = {}
    with tempfile.TemporaryDirectory(prefix="geoseek-smoke-") as temp:
        db = Path(temp) / "tiles.sqlite"
        source = Path("data/index/tiles.sqlite").resolve()
        with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(db)) as dest:
                src.backup(dest)
        os.environ["DATABASE_URL"] = "sqlite:///" + str(db)
        from geoseek.search.api import app
        start = time.perf_counter()
        with TestClient(app) as client:
            result["startup_seconds"] = time.perf_counter() - start
            def get(path, **kwargs):
                t = time.perf_counter()
                r = client.get(path, **kwargs)
                result[path] = {"status": r.status_code, "ms": (time.perf_counter()-t)*1000}
                r.raise_for_status()
                return r
            get("/health")
            get("/app/")
            search = get("/search/text", params={"q": "a river with sandbars", "k": 3}).json()
            assert search["count"] > 0
            get(f"/tile/{search['results'][0]['tile_id']}/thumbnail")
            get("/stats")
            from geoseek.search.api import _analyst
            assert _analyst is not None and len(_analyst.details) > 0
            candidate = _analyst.details[0]["candidate_id"]
            get(f"/candidates/{candidate}/imagery", params={"view": "overlay"})
            r = client.post(f"/candidates/{candidate}/decision", json={"decision": "confirm", "note": "isolated deployment smoke"})
            r.raise_for_status()
            result["decision_write_to_snapshot"] = r.status_code
            get("/audit")
            r = client.post("/watch-areas", json={"name": "deployment smoke", "bbox": [82,26,83,28]})
            r.raise_for_status()
            result["watch_write_to_snapshot"] = r.status_code
            get("/watch-areas")
            get("/discovery/cluster-map.png")
            observations = get("/detect/observations").json()["observations"]
            result["detection_observations"] = len(observations)
    Path("deploy/gcp/smoke-results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

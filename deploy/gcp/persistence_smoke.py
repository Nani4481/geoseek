"""Create and verify labeled SQLite records across a container restart."""
from __future__ import annotations

import argparse
import json
import pathlib
import ssl
import urllib.parse
import urllib.request


BASE = "https://geoseek.8.234.117.216.sslip.io"
STATE = pathlib.Path("deploy/gcp/.persistence-smoke.json")


def call(path: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120, context=ssl.create_default_context()) as response:
        return json.load(response)


def create() -> None:
    candidate = call("/candidates?limit=1")["candidates"][0]["candidate_id"]
    decision = call(
        f"/candidates/{urllib.parse.quote(candidate)}/decision",
        {
            "decision": "reopen",
            "note": "GCE persistent-disk deployment validation",
            "analyst": "deployment-smoke",
        },
    )
    watch = call(
        "/watch-areas",
        {
            "name": "GeoSeek deployment persistence check",
            "bbox": [82.0, 26.0, 83.0, 28.0],
            "text_query": "persistent-disk validation",
            "created_by": "deployment-smoke",
        },
    )
    state = {"decision_id": decision["decision_id"], "watch_id": watch["watch_id"]}
    STATE.write_text(json.dumps(state) + "\n", encoding="utf-8")
    print(json.dumps({"created": state}))


def verify() -> None:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    decision = call(f"/audit/{urllib.parse.quote(state['decision_id'])}")
    watch = call(f"/watch-areas/{urllib.parse.quote(state['watch_id'])}")
    assert decision["analyst"] == "deployment-smoke"
    assert decision["analyst_note"] == "GCE persistent-disk deployment validation"
    assert watch["created_by"] == "deployment-smoke"
    print(json.dumps({"persisted": state}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"))
    args = parser.parse_args()
    create() if args.mode == "create" else verify()

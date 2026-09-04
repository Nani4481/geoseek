"""Capture the current production search results as a regression fixture.

Run this ONCE, before the Phase 3.5 refactor, against the live production
index (data/index/). It freezes the exact ranked output of a battery of
text + image queries into tests/fixtures/search_baseline.json so a
post-refactor test can prove the seams did not change a single result.
"""

from __future__ import annotations

import json
from pathlib import Path

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine, SearchFilters

TEXT_QUERIES = [
    "a river with sandbars",
    "dense urban buildings",
    "agricultural fields",
    "open bare ground",
    "a bridge over a river",
    "green cropland next to a town",
]

# (label, query, filter kwargs) - exercises the post-filter path too
FILTERED_QUERIES = [
    ("river|S2B", "a river", dict(sensor="Sentinel-2B")),
    ("river|2024", "a river", dict(date_start="2024-01-01", date_end="2024-12-31")),
    ("fields|bbox", "agricultural fields", dict(bbox=(82.1, 26.7, 82.4, 26.95))),
    ("urban|clear", "dense urban buildings", dict(max_cloud_fraction=0.01)),
]

K = 15

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "search_baseline.json"


def main() -> None:
    settings = get_settings()
    engine = SearchEngine()
    n = engine.count()
    print(f"[baseline] production index has {n} vectors")

    out: dict = {"index_vectors": n, "k": K, "text": {}, "filtered": {}, "image": {}}

    for q in TEXT_QUERIES:
        results, _ = engine.search_text(q, k=K)
        out["text"][q] = [
            {"tile_id": r.tile_id, "score": round(float(r.score), 6),
             "scene_id": r.scene_id, "acq_date": r.acq_date,
             "lon": round(r.lon, 6), "lat": round(r.lat, 6)}
            for r in results
        ]
        print(f"[baseline] text {q!r}: {len(results)} hits, top={results[0].tile_id} {results[0].score:.4f}")

    for label, q, fkw in FILTERED_QUERIES:
        results, _ = engine.search_text(q, k=K, filters=SearchFilters(**fkw))
        out["filtered"][label] = {
            "query": q, "filters": fkw,
            "results": [{"tile_id": r.tile_id, "score": round(float(r.score), 6)} for r in results],
        }
        top = results[0].tile_id if results else None
        print(f"[baseline] filtered {label}: {len(results)} hits, top={top}")

    # image-to-image from a few deterministic anchor tiles (first, middle, last faiss_id)
    anchor_faiss_ids = sorted(engine._rows)[:: max(1, len(engine._rows) // 4)][:4]
    for fid in anchor_faiss_ids:
        tid = engine._rows[fid]["tile_id"]
        results, _ = engine.search_image(tile_id=tid, k=K)
        out["image"][tid] = [
            {"tile_id": r.tile_id, "score": round(float(r.score), 6)} for r in results
        ]
        print(f"[baseline] image {tid}: top={results[0].tile_id} {results[0].score:.4f}")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[baseline] wrote {FIXTURE}")

    engine.close()


if __name__ == "__main__":
    main()

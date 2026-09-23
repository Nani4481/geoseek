"""Object Detection tab (Phase 8F): the /detect/* FastAPI endpoints and the
lon/lat -> tile-pixel reprojection they depend on. Real stored detections
only - nothing here re-runs inference; every box comes from
data/detections/<observation_id>/detections.geojson, written by
scripts/detect_maxar.py.

Integration tests gated on the staged Maxar detections existing (skip
otherwise, same convention as tests/test_phase6.py's endpoint tests).
"""

from __future__ import annotations

import pytest

from geoseek.config import get_settings

VAN_NUYS = "103001010C12B000_031311102120"


def _detections_staged() -> bool:
    return (get_settings().data_dir / "detections" / VAN_NUYS / "summary.json").is_file()


@pytest.fixture(scope="module")
def client():
    if not _detections_staged():
        pytest.skip("no staged Maxar detections - run scripts/detect_maxar.py")
    if not (get_settings().models_dir / "detector" / "geoseek_obb_v15_yolo26s.card.json").is_file():
        pytest.skip("no trained detector model card")

    from fastapi.testclient import TestClient

    from geoseek.search.api import app

    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# reprojection unit test (pure function, no server needed)
# --------------------------------------------------------------------------


def test_tile_detections_reprojects_onto_the_tile_and_matches_summary_count():
    if not _detections_staged():
        pytest.skip("no staged Maxar detections")
    from geoseek.analyst.detections import list_tiles_with_detections, tile_detections
    from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository

    repo = SQLiteMetadataRepository(get_settings().index_dir / "tiles.sqlite")
    try:
        tiles = list_tiles_with_detections(VAN_NUYS)
        assert tiles and tiles[0]["n_detections"] > 0
        row, col = tiles[0]["row"], tiles[0]["col"]
        out = tile_detections(repo, VAN_NUYS, row, col)
        assert out["tile_id"] == f"{VAN_NUYS}_r{row:03d}_c{col:03d}"
        assert len(out["detections"]) == tiles[0]["n_detections"]
        for d in out["detections"]:
            assert len(d["polygon_px"]) == 4
            for x, y in d["polygon_px"]:
                # a genuine on-tile reprojection lands inside (or very near) the 1024x1024 frame;
                # a real bug (CRS/affine mixup) produces wildly out-of-range coordinates
                assert -50 <= x <= out["width"] + 50 and -50 <= y <= out["height"] + 50
            assert 0.0 <= d["score"] <= 1.0
            assert d["class"] in {"small-vehicle", "large-vehicle", "ship", "plane",
                                  "helicopter", "storage-tank", "harbor", "bridge"}
    finally:
        repo.close()


# --------------------------------------------------------------------------
# FastAPI endpoints
# --------------------------------------------------------------------------


def test_model_info_has_operating_points_and_the_large_vehicle_caveat(client):
    r = client.get("/detect/model-info")
    assert r.status_code == 200
    body = r.json()
    assert set(body["classes"]) == {"small-vehicle", "large-vehicle", "ship", "plane",
                                    "helicopter", "storage-tank", "harbor", "bridge"}
    sv = body["operating_points"]["small-vehicle"]
    assert 0.0 < sv["conf"] < 1.0 and 0.0 <= sv["AP50"] <= 1.0
    assert any("large-vehicle" in c for c in body["caveats"])


def test_observations_lists_maxar_aois_with_counts(client):
    r = client.get("/detect/observations")
    assert r.status_code == 200
    obs = r.json()["observations"]
    assert any(o["observation_id"] == VAN_NUYS for o in obs)
    van_nuys = next(o for o in obs if o["observation_id"] == VAN_NUYS)
    assert van_nuys["n_detections"] > 0
    assert van_nuys["by_class"]["small-vehicle"] > 0


def test_tiles_are_sorted_by_detection_count_descending(client):
    r = client.get(f"/detect/observations/{VAN_NUYS}/tiles")
    assert r.status_code == 200
    tiles = r.json()["tiles"]
    assert len(tiles) > 1
    counts = [t["n_detections"] for t in tiles]
    assert counts == sorted(counts, reverse=True)


def test_tile_detections_endpoint_matches_the_summary_count(client):
    tiles = client.get(f"/detect/observations/{VAN_NUYS}/tiles").json()["tiles"]
    row, col = tiles[0]["row"], tiles[0]["col"]
    r = client.get(f"/detect/observations/{VAN_NUYS}/tiles/{row}/{col}")
    assert r.status_code == 200
    body = r.json()
    assert len(body["detections"]) == tiles[0]["n_detections"]


def test_tile_image_is_a_real_1024_png(client):
    tiles = client.get(f"/detect/observations/{VAN_NUYS}/tiles").json()["tiles"]
    row, col = tiles[0]["row"], tiles[0]["col"]
    r = client.get(f"/detect/observations/{VAN_NUYS}/tiles/{row}/{col}/image.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(r.content) > 100_000       # a real photo, not a stub


def test_unknown_observation_is_404(client):
    assert client.get("/detect/observations/nope/tiles").status_code == 404
    assert client.get("/detect/observations/nope/tiles/0/0").status_code == 404
    assert client.get("/detect/observations/nope/tiles/0/0/image.png").status_code == 404

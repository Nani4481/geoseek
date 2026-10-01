"""NIR/SWIR staging: resumable downloads, exact-grid crops, skip-what-exists. A local HTTP server stands in for the
COG bucket (no network)."""

from __future__ import annotations

import http.server
import threading
from functools import partial

import numpy as np
import pytest
import rasterio
from affine import Affine

from geoseek.staging import download_nir_swir as dl


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static file server WITH Range support (SimpleHTTPRequestHandler has none) and a switch to ignore Range."""

    ignore_range = False
    truncate_after: int | None = None          # simulate a connection dropped after N body bytes
    truncate_always = False                    # False: only the next request is cut; True: every request is
    served = []

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        path = self.translate_path(self.path)
        size = __import__("os").path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        data = open(path, "rb").read()
        rng = self.headers.get("Range")
        start = 0
        if rng and not type(self).ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        type(self).served.append(start)
        cut = type(self).truncate_after
        if cut is not None:
            if not type(self).truncate_always:
                type(self).truncate_after = None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.close()                         # drop mid-body
            return
        self.wfile.write(body)


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    _Handler.ignore_range, _Handler.truncate_after, _Handler.truncate_always, _Handler.served = False, None, False, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_Handler, directory=str(root)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


PAYLOAD = bytes(range(256)) * 4096                           # 1 MiB


def test_download_writes_the_file_and_returns_size_and_sha(server, tmp_path):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    info = dl.download_resumable(f"{base}/a.tif", tmp_path / "out" / "a.tif", chunk=64 * 1024)
    assert (tmp_path / "out" / "a.tif").read_bytes() == PAYLOAD and not (tmp_path / "out" / "a.tif.part").exists()
    assert info["bytes"] == len(PAYLOAD) and info["sha256"] == dl.sha256_file(tmp_path / "out" / "a.tif")


def test_an_interrupted_download_resumes_from_the_partial_file_with_a_range_request(server, tmp_path):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    dest = tmp_path / "a.tif"
    (tmp_path / "a.tif.part").write_bytes(PAYLOAD[:300_000])          # what a killed run left behind
    info = dl.download_resumable(f"{base}/a.tif", dest, chunk=64 * 1024)
    assert dest.read_bytes() == PAYLOAD and info["resumed_from"] == 300_000
    assert 300_000 in _Handler.served and 0 not in _Handler.served, "must ask only for the missing bytes"


def test_a_connection_dropped_mid_body_is_retried_and_completes(server, tmp_path):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    _Handler.truncate_after = 200_000
    info = dl.download_resumable(f"{base}/a.tif", tmp_path / "a.tif", chunk=32 * 1024, retries=4)
    assert (tmp_path / "a.tif").read_bytes() == PAYLOAD and info["bytes"] == len(PAYLOAD)
    assert any(s > 0 for s in _Handler.served), "the retry must have resumed, not restarted"


def test_a_server_that_ignores_range_restarts_cleanly_instead_of_corrupting(server, tmp_path):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    _Handler.ignore_range = True
    (tmp_path / "a.tif.part").write_bytes(PAYLOAD[:100_000])
    dl.download_resumable(f"{base}/a.tif", tmp_path / "a.tif", chunk=64 * 1024)
    assert (tmp_path / "a.tif").read_bytes() == PAYLOAD                  # not partial-prefix + full body


def test_a_partial_larger_than_the_source_is_discarded(server, tmp_path):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    (tmp_path / "a.tif.part").write_bytes(PAYLOAD + b"junk")
    dl.download_resumable(f"{base}/a.tif", tmp_path / "a.tif", chunk=64 * 1024)
    assert (tmp_path / "a.tif").read_bytes() == PAYLOAD


def test_a_short_download_is_an_error_and_keeps_the_partial_for_resume(server, tmp_path, monkeypatch):
    root, base = server
    (root / "a.tif").write_bytes(PAYLOAD)
    _Handler.truncate_after, _Handler.truncate_always = 10_000, True          # every attempt dies after 10 kB
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with pytest.raises(dl.StagingError, match="of .* bytes"):
        dl.download_resumable(f"{base}/a.tif", tmp_path / "a.tif", chunk=4096, retries=3)
    assert not (tmp_path / "a.tif").exists()
    assert (tmp_path / "a.tif.part").stat().st_size > 10_000, "progress is kept so a later run can resume"


# ------------------------------------------------------------------------------ cropping onto the existing grid


def _tif(path, data, transform, dtype="uint16", crs="EPSG:32644", nodata=0):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=dtype,
                       crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(data.astype(dtype), 1)


def _reference(tmp_path, h=64, w=96, x0=500_000.0, y0=3_000_000.0):
    ref = tmp_path / "scene" / "B04.tif"
    ref.parent.mkdir()
    _tif(ref, np.full((h, w), 100), Affine(10, 0, x0, 0, -10, y0))
    return ref


def test_b08_is_cut_on_the_reference_grid_without_resampling(tmp_path):
    ref = _reference(tmp_path)                                           # a 96x64 crop starting 20 px right, 12 px down
    full = np.arange(200 * 300, dtype=np.uint32).reshape(200, 300) % 60000
    _tif(tmp_path / "full.tif", full, Affine(10, 0, 500_000.0 - 200, 0, -10, 3_000_000.0 + 120))
    dl.crop_to_reference(tmp_path / "full.tif", ref, "B08", tmp_path / "scene" / "B08.tif")
    with rasterio.open(tmp_path / "scene" / "B08.tif") as out, rasterio.open(ref) as r:
        assert out.shape == r.shape and out.transform == r.transform and out.crs == r.crs and out.dtypes[0] == "uint16"
        assert np.array_equal(out.read(1), full[12:12 + 64, 20:20 + 96]), "pixels must be the source pixels, unresampled"
    assert dl.band_on_reference_grid(tmp_path / "scene" / "B08.tif", ref)


def test_b11_is_resampled_from_20m_to_the_10m_reference_grid(tmp_path):
    ref = _reference(tmp_path)
    coarse = np.full((60, 80), 1800)
    _tif(tmp_path / "full20.tif", coarse, Affine(20, 0, 500_000.0 - 200, 0, -20, 3_000_000.0 + 120))
    dl.crop_to_reference(tmp_path / "full20.tif", ref, "B11", tmp_path / "scene" / "B11.tif")
    with rasterio.open(tmp_path / "scene" / "B11.tif") as out:
        assert out.shape == (64, 96) and out.transform == rasterio.open(ref).transform
        assert np.all(out.read(1) == 1800)                               # constant surface stays constant under bilinear


def test_a_source_that_is_not_pixel_aligned_to_the_reference_is_refused(tmp_path):
    ref = _reference(tmp_path)
    _tif(tmp_path / "full.tif", np.zeros((200, 300)), Affine(10, 0, 500_000.0 - 203, 0, -10, 3_000_000.0 + 120))  # 3 m off
    with pytest.raises(dl.StagingError, match="pixel-aligned"):
        dl.crop_to_reference(tmp_path / "full.tif", ref, "B08", tmp_path / "scene" / "B08.tif")
    assert not (tmp_path / "scene" / "B08.tif").exists() and not list((tmp_path / "scene").glob("*.tmp.tif"))


def test_a_reference_crop_that_does_not_fit_inside_the_source_is_refused(tmp_path):
    ref = _reference(tmp_path)
    _tif(tmp_path / "full.tif", np.zeros((30, 30)), Affine(10, 0, 500_000.0 - 200, 0, -10, 3_000_000.0 + 120))
    with pytest.raises(dl.StagingError, match="does not fit"):
        dl.crop_to_reference(tmp_path / "full.tif", ref, "B08", tmp_path / "scene" / "B08.tif")


def test_band_on_reference_grid_rejects_missing_empty_and_misaligned_files(tmp_path):
    ref = _reference(tmp_path)
    p = tmp_path / "scene" / "B08.tif"
    assert not dl.band_on_reference_grid(p, ref)                          # missing
    p.write_bytes(b"")
    assert not dl.band_on_reference_grid(p, ref)                          # empty
    _tif(p, np.zeros((64, 96)), Affine(10, 0, 500_010.0, 0, -10, 3_000_000.0))
    assert not dl.band_on_reference_grid(p, ref)                          # shifted one pixel
    _tif(p, np.zeros((64, 96)), Affine(10, 0, 500_000.0, 0, -10, 3_000_000.0))
    assert dl.band_on_reference_grid(p, ref)


# ------------------------------------------------------------------------------ incremental


def test_ensure_bands_downloads_nothing_when_both_bands_already_exist(tmp_path, monkeypatch):
    ref = _reference(tmp_path)
    for b in dl.BANDS:
        _tif(ref.parent / f"{b}.tif", np.zeros((64, 96)), rasterio.open(ref).transform)
    monkeypatch.setattr(dl, "fetch_item", lambda *_a, **_k: pytest.fail("must not even query STAC"))
    monkeypatch.setattr(dl, "download_resumable", lambda *_a, **_k: pytest.fail("must not download"))
    out = dl.ensure_bands("S2A_X_L2A", ref.parent, tmp_path / "tmp", log=lambda _m: None)
    assert set(out) == {"B08", "B11"} and not any(v["staged_now"] for v in out.values())


def test_ensure_bands_fetches_only_the_missing_band(tmp_path, monkeypatch):
    ref = _reference(tmp_path)
    tf = rasterio.open(ref).transform
    _tif(ref.parent / "B08.tif", np.zeros((64, 96)), tf)                   # B08 present, B11 missing
    full20 = tmp_path / "src20.tif"
    _tif(full20, np.full((60, 80), 1500), Affine(20, 0, 500_000.0 - 200, 0, -20, 3_000_000.0 + 120))
    requested = []
    monkeypatch.setattr(dl, "fetch_item", lambda sid, **k: {"assets": {"nir": {"href": "http://x/B08"}, "swir16": {"href": "http://x/B11"}}})

    def fake_dl(url, dest, **k):
        requested.append(url)
        dest.write_bytes(full20.read_bytes())
        return {"bytes": dest.stat().st_size, "sha256": dl.sha256_file(dest), "seconds": 0.1, "resumed_from": 0}
    monkeypatch.setattr(dl, "download_resumable", fake_dl)
    out = dl.ensure_bands("S2A_X_L2A", ref.parent, tmp_path / "tmp", log=lambda _m: None)
    assert requested == ["http://x/B11"]
    assert out["B11"]["staged_now"] and out["B11"]["resampling"].startswith("20 m") and not out["B08"]["staged_now"]
    assert out["B11"]["source_url"] == "http://x/B11" and len(out["B11"]["sha256"]) == 64
    assert not (tmp_path / "tmp" / "S2A_X_L2A" / "B11_full.tif").exists(), "the full-tile download is deleted after cropping"

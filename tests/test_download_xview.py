"""xView staging: vehicle class mapping, GeoJSON -> ImageGT loading, and the streaming zip recovery
extractor used to work around train_images.zip's corrupted zip64 central directory (numpy/zlib only)."""

from __future__ import annotations

import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

from geoseek.detect.classes import CLASS_TO_INDEX
from geoseek.staging.download_xview import (
    VEHICLE_TYPE_ID_MAP,
    class_histogram,
    extract_streaming,
    load_xview_gt,
    verify_extracted,
)


def _geojson(tmp_path, features):
    p = tmp_path / "xView_train.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    return p


def _feature(type_id, bounds, image_id="1.tif"):
    return {"type": "Feature", "properties": {"type_id": type_id, "image_id": image_id, "bounds_imcoords": bounds},
            "geometry": {"type": "Polygon", "coordinates": []}}


# --------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------


def test_vehicle_type_id_map_only_uses_known_classes():
    assert set(VEHICLE_TYPE_ID_MAP.values()) == {"small-vehicle", "large-vehicle"}
    assert all(name in CLASS_TO_INDEX for name in VEHICLE_TYPE_ID_MAP.values())


def test_vehicle_type_id_map_excludes_non_vehicle_categories():
    # rail (33-38), maritime (40-52), aircraft (11-15), tower crane (54, a fixed structure),
    # infrastructure (71+) are all out of scope for this small/large-vehicle-only mapping.
    for excluded in (11, 15, 34, 38, 40, 51, 54, 73, 86):
        assert excluded not in VEHICLE_TYPE_ID_MAP


# --------------------------------------------------------------------------
# GeoJSON -> ImageGT
# --------------------------------------------------------------------------


def test_load_xview_gt_maps_classes_and_builds_axis_aligned_quads(tmp_path):
    p = _geojson(tmp_path, [
        _feature(18, "10,20,30,50", image_id="2355.tif"),   # Small Car -> small-vehicle
        _feature(23, "0,0,40,20", image_id="2355.tif"),     # Truck -> large-vehicle
        _feature(73, "5,5,15,15", image_id="2355.tif"),     # Building -> unmapped, dropped
    ])
    gt, stats = load_xview_gt(p)
    assert set(gt) == {"2355"}
    g = gt["2355"]
    assert list(g.classes) == [CLASS_TO_INDEX["small-vehicle"], CLASS_TO_INDEX["large-vehicle"]]
    assert list(g.difficult) == [False, False]
    np.testing.assert_allclose(g.quads[0], [[10, 20], [30, 20], [30, 50], [10, 50]])
    assert stats["n_skipped_unmapped_class"] == 1
    assert stats["n_skipped_degenerate_box"] == 0


def test_load_xview_gt_skips_degenerate_boxes(tmp_path):
    p = _geojson(tmp_path, [
        _feature(18, "10,10,10,20"),   # zero width
        _feature(18, "10,10,20,10"),   # zero height
        _feature(18, "10,10,20,20"),   # valid
    ])
    gt, stats = load_xview_gt(p)
    assert len(gt["1"].classes) == 1
    assert stats["n_skipped_degenerate_box"] == 2


def test_load_xview_gt_stems_filter_includes_empty_images(tmp_path):
    p = _geojson(tmp_path, [_feature(18, "0,0,10,10", image_id="1.tif")])
    gt, _ = load_xview_gt(p, stems={"1", "2"})
    assert set(gt) == {"1", "2"}
    assert len(gt["2"].classes) == 0


def test_load_xview_gt_reads_from_zip(tmp_path):
    geojson_bytes = json.dumps({"type": "FeatureCollection",
                                 "features": [_feature(18, "0,0,10,10")]}).encode()
    zpath = tmp_path / "train_labels.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("xView_train.geojson", geojson_bytes)
    gt, _ = load_xview_gt(zpath)
    assert set(gt) == {"1"}


def test_class_histogram_counts_only_mapped_classes(tmp_path):
    p = _geojson(tmp_path, [
        _feature(18, "0,0,10,10"), _feature(18, "0,0,10,10"), _feature(23, "0,0,10,10"),
        _feature(73, "0,0,10,10"),  # unmapped
    ])
    hist = class_histogram(p)
    assert hist["mapped_vehicle_instance_counts"] == {"small-vehicle": 2, "large-vehicle": 1}
    assert hist["total_instances_all_classes"] == 4
    assert hist["total_mapped_vehicle_instances"] == 3


# --------------------------------------------------------------------------
# streaming recovery extractor
# --------------------------------------------------------------------------


def _make_corrupt_offset_zip(path, members: dict[str, bytes], bogus_shift: int = 123_456_789):
    """Build a real zip (correct local headers + data), then hand-corrupt every central-directory
    header_offset by +bogus_shift - reproducing train_images.zip's actual failure mode, where
    zipfile can still list names/CRC/sizes but cannot open() any entry. (The real corruption is an
    exact +3x4GiB in a zip64 field; any wrong-but-different 4-byte offset reproduces the same
    *symptom* - zipfile.open() lands on garbage - which is all extract_streaming needs to handle.
    NOTE: a shift that's a multiple of 2**32 would be a no-op after the 4-byte field's mod-2**32
    wraparound - keep this value NOT a multiple of 2**32.)"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    raw = bytearray(path.read_bytes())
    # naive rewrite: relocate every central-directory "local header offset" field (a 4-byte LE uint
    # at a fixed position within each 46-byte-plus-name central directory record).
    i = 0
    while True:
        i = raw.find(b"PK\x01\x02", i)
        if i == -1:
            break
        off_pos = i + 42
        current = struct.unpack_from("<I", raw, off_pos)[0]
        struct.pack_into("<I", raw, off_pos, (current + bogus_shift) & 0xFFFFFFFF)
        i += 4
    path.write_bytes(bytes(raw))


def test_extract_streaming_recovers_files_zipfile_cannot_open(tmp_path):
    members = {"a.txt": b"hello world " * 100, "dir/b.txt": b"second file, some other bytes " * 50}
    zpath = tmp_path / "broken.zip"
    _make_corrupt_offset_zip(zpath, members)

    # sanity: prove the corruption actually breaks stdlib zipfile the same way train_images.zip is broken
    zf = zipfile.ZipFile(zpath)
    with pytest.raises(zipfile.BadZipFile):
        zf.read("a.txt")

    dest = tmp_path / "out"
    entries = extract_streaming(zpath, dest)
    names = {e.filename for e in entries}
    assert names == set(members)
    assert all(e.crc_ok for e in entries)
    for name, data in members.items():
        assert (dest / name).read_bytes() == data


def test_extract_streaming_handles_stored_and_deflate(tmp_path):
    zpath = tmp_path / "mixed.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(zipfile.ZipInfo("stored.bin"), b"raw bytes, no compression" * 10,
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr(zipfile.ZipInfo("deflated.bin"), b"compressible " * 500,
                    compress_type=zipfile.ZIP_DEFLATED)
    dest = tmp_path / "out"
    entries = extract_streaming(zpath, dest)
    assert {e.filename for e in entries} == {"stored.bin", "deflated.bin"}
    assert all(e.crc_ok for e in entries)


def test_verify_extracted_reports_missing_and_bad(tmp_path):
    zpath = tmp_path / "z.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("good.txt", b"content")
        zf.writestr("missing.txt", b"content2")
        zf.writestr("corrupted.txt", b"content3")

    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "good.txt").write_bytes(b"content")
    (dest / "corrupted.txt").write_bytes(b"WRONG BYTES")

    report = verify_extracted(zpath, dest)
    assert report["n_expected"] == 3
    assert report["n_ok"] == 1
    assert "missing.txt" in report["missing"]
    assert "corrupted.txt" in report["bad"]

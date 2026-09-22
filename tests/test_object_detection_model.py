"""Phase 8F-2 Step 7: the ObjectDetectionModel ABC and its YOLO-OBB implementation, exercised with a FAKE predictor
(no ultralytics, no weights): channel order, thresholds, class filtering, geo-referencing, windowed inference on large
tiles, and that the AGPL-3.0 framework is never imported by importing geoseek."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from geoseek.models.base import Detection, ObjectDetectionModel, TileGeoRef
from geoseek.models.yolo_obb import YoloObbDetectionModel

NAMES = ("small-vehicle", "large-vehicle", "ship", "plane", "helicopter", "storage-tank", "harbor", "bridge")


class _T:                                    # tiny stand-in for a torch tensor: .cpu().numpy()
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Result:
    def __init__(self, xywhr, conf, cls):
        self.obb = None if xywhr is None else type("O", (), {"xywhr": _T(xywhr), "conf": _T(conf), "cls": _T(cls)})()
        if self.obb is not None:
            self.obb.__class__.__len__ = lambda s: len(s.conf.a)


class FakeYolo:
    """Reports one object per image at a fixed GLOBAL location, in the coordinates of whatever crop it is handed."""

    def __init__(self, world_objects=None):
        self.names = {i: n for i, n in enumerate(NAMES)}
        self.calls: list[dict] = []
        self.world_objects = world_objects or []        # [(gx, gy, w, h, angle, conf, cls)] in the (single) big tile's frame

    def predict(self, imgs, **kw):
        self.calls.append({"n": len(imgs), "kw": kw, "first": imgs[0].copy()})
        res = []
        for k, im in enumerate(imgs):
            h, w = im.shape[:2]
            x0 = y0 = 0
            if "origins" in kw:
                x0, y0 = kw["origins"][k]
            rows = []
            for gx, gy, bw, bh, ang, conf, cls in self.world_objects:
                lx, ly = gx - self._origin(im), gy
                if 0 <= lx < w and 0 <= ly < h and conf >= kw["conf"]:
                    rows.append((lx, ly, bw, bh, ang, conf, cls))
            if not rows and not self.world_objects:
                rows = [(50.0, 60.0, 30.0, 10.0, 0.5, 0.9, 0), (200.0, 80.0, 40.0, 12.0, 1.0, 0.3, 2)]
                rows = [r for r in rows if r[5] >= kw["conf"]]
            if not rows:
                res.append(_Result(None, None, None))
            else:
                a = np.array(rows)
                res.append(_Result(a[:, :5], a[:, 5], a[:, 6]))
        return res

    def _origin(self, im):                              # window origin is encoded in pixel (0,0) of the crop by the test
        return int(im[0, 0, 0]) * 8


def make_model(tmp_path, fake=None, **kw):
    m = YoloObbDetectionModel(tmp_path / "w.pt", **kw)
    m._model = fake or FakeYolo()
    m._names = NAMES
    m._device, m._half = "cpu", False
    return m


def rgb_tile(h=128, w=256, rgb=(200, 100, 50)):
    t = np.zeros((h, w, 3), dtype=np.uint8)
    t[...] = rgb
    return t


# --------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------


def test_abc_cannot_be_instantiated_and_the_concrete_class_implements_it():
    with pytest.raises(TypeError):
        ObjectDetectionModel()                          # abstract
    assert issubclass(YoloObbDetectionModel, ObjectDetectionModel)


def test_detection_value_object():
    d = Detection("small-vehicle", 0, 0.87654, (10.0, 20.0, 30.0, 8.0, 0.0), ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert d.long_side_px == 30.0 and d.heading_deg == pytest.approx(0.0)
    d2 = Detection("ship", 2, 0.5, (0.0, 0.0, 8.0, 30.0, 0.0), ())               # long axis along the h side -> 90 deg
    assert d2.heading_deg == pytest.approx(90.0)
    assert d.as_dict()["score"] == 0.8765 and d.as_dict()["geom_wkt_4326"] is None


# --------------------------------------------------------------------------
# detect()
# --------------------------------------------------------------------------


def test_rgb_tiles_are_flipped_to_bgr_before_reaching_ultralytics(tmp_path):
    fake = FakeYolo()
    m = make_model(tmp_path, fake)
    m.detect(rgb_tile(rgb=(200, 100, 50)))
    seen = fake.calls[0]["first"]
    assert tuple(seen[0, 0]) == (50, 100, 200)          # BGR: ultralytics flips it back to RGB itself


def test_min_score_default_override_and_class_filter(tmp_path):
    fake = FakeYolo()
    m = make_model(tmp_path, fake, min_score=0.5)
    dets = m.detect(rgb_tile())
    assert [d.class_name for d in dets] == ["small-vehicle"] and fake.calls[-1]["kw"]["conf"] == 0.5
    dets = m.detect(rgb_tile(), min_score=0.1)
    assert {d.class_name for d in dets} == {"small-vehicle", "ship"} and fake.calls[-1]["kw"]["conf"] == 0.1
    assert [d.class_name for d in m.detect(rgb_tile(), min_score=0.1, classes=["ship"])] == ["ship"]
    assert dets == sorted(dets, key=lambda d: -d.score)                          # highest score first
    with pytest.raises(ValueError):
        m.detect(rgb_tile(), classes=["tennis-court"])


def test_per_class_min_score_uses_the_lowest_class_as_the_ultralytics_floor_and_filters_afterwards(tmp_path):
    # small-vehicle (conf 0.9) and ship (conf 0.3) both come back from FakeYolo's default world objects.
    fake = FakeYolo()
    m = make_model(tmp_path, fake, min_score={"small-vehicle": 0.95, "ship": 0.2})
    dets = m.detect(rgb_tile())
    # ultralytics is asked for the LOWEST threshold in effect (0.2), so nothing is dropped before NMS ...
    assert fake.calls[-1]["kw"]["conf"] == 0.2
    # ... but small-vehicle's own 0.95 threshold then filters out its 0.9-confidence candidate
    assert [d.class_name for d in dets] == ["ship"]


def test_per_class_min_score_from_the_card(tmp_path):
    w = tmp_path / "geoseek.pt"
    (w.with_suffix(".card.json")).write_text(json.dumps({
        "classes": list(NAMES), "weights_sha256": "abc", "architecture": "YOLO26s-OBB",
        "operating_point": {"per_class": {"small-vehicle": {"conf": 0.95}, "ship": {"conf": 0.2}}},
    }), encoding="utf-8")
    m = YoloObbDetectionModel(w)  # constructor reads the card -> no override needed
    assert m.min_score == {"small-vehicle": 0.95, "ship": 0.2}
    fake = FakeYolo()
    m._model, m._names, m._device, m._half = fake, NAMES, "cpu", False
    dets = m.detect(rgb_tile())
    assert fake.calls[-1]["kw"]["conf"] == 0.2                                   # floor = min of the two entries
    assert [d.class_name for d in dets] == ["ship"]                              # small-vehicle's 0.9 < its own 0.95
    # a class absent from the card's per_class map falls back to DEFAULT_MIN_SCORE at lookup time, not stored
    assert "plane" not in m.min_score


def test_pixel_geometry_and_orientation_are_carried_through(tmp_path):
    m = make_model(tmp_path)
    (d,) = m.detect(rgb_tile(), min_score=0.5)
    assert d.obb_px == (50.0, 60.0, 30.0, 10.0, 0.5) and len(d.polygon_px) == 4
    cx = np.mean([p[0] for p in d.polygon_px])
    cy = np.mean([p[1] for p in d.polygon_px])
    assert (cx, cy) == pytest.approx((50.0, 60.0))
    assert d.geom_wkt_4326 is None                                               # no georeference given


def test_geo_reference_produces_a_lonlat_footprint(tmp_path):
    m = make_model(tmp_path)
    geo = TileGeoRef(transform=(1e-5, 0.0, 88.5, 0.0, -1e-5, 27.6), crs="EPSG:4326")   # 1e-5 deg per pixel
    (d,) = m.detect(rgb_tile(), min_score=0.5, geo=geo)
    assert d.geom_wkt_4326.startswith("POLYGON ((")
    coords = [tuple(map(float, p.split())) for p in d.geom_wkt_4326[len("POLYGON (("):-2].split(", ")]
    assert coords[0] == coords[-1] and len(coords) == 5                          # closed ring
    lon = np.mean([c[0] for c in coords[:4]])
    lat = np.mean([c[1] for c in coords[:4]])
    assert lon == pytest.approx(88.5 + 50 * 1e-5, abs=1e-6) and lat == pytest.approx(27.6 - 60 * 1e-5, abs=1e-6)


def test_batch_detect_batches_small_tiles_and_matches_single_calls(tmp_path):
    fake = FakeYolo()
    m = make_model(tmp_path, fake, batch_size=2)
    tiles = [rgb_tile() for _ in range(5)]
    out = m.detect_batch(tiles, min_score=0.5)
    assert len(out) == 5 and [c["n"] for c in fake.calls] == [2, 2, 1]
    assert all(len(o) == 1 for o in out)


def test_upscale_runs_on_the_resampled_tile_and_reports_in_original_pixels(tmp_path):
    fake = FakeYolo()
    m = make_model(tmp_path, fake, min_score=0.5, upscale=2.0)
    geo = TileGeoRef(transform=(1e-5, 0.0, 88.5, 0.0, -1e-5, 27.6), crs="EPSG:4326")
    (d,) = m.detect(rgb_tile(h=128, w=256), geo=geo)
    assert fake.calls[0]["first"].shape[:2] == (256, 512)                        # the predictor saw the 2x tile ...
    assert d.obb_px == pytest.approx((25.0, 30.0, 15.0, 5.0, 0.5))               # ... and its box comes back in ORIGINAL tile pixels
    assert np.mean([p[0] for p in d.polygon_px]) == pytest.approx(25.0)
    coords = [tuple(map(float, p.split())) for p in d.geom_wkt_4326[len("POLYGON (("):-2].split(", ")]
    lon, lat = np.mean([c[0] for c in coords[:4]]), np.mean([c[1] for c in coords[:4]])
    assert lon == pytest.approx(88.5 + 25 * 1e-5, abs=1e-6) and lat == pytest.approx(27.6 - 30 * 1e-5, abs=1e-6)
    assert m.info["upscale"] == 2.0 and YoloObbDetectionModel(tmp_path / "w.pt").upscale == 1.0
    with pytest.raises(ValueError):
        YoloObbDetectionModel(tmp_path / "w.pt", upscale=0.0)


def test_upscale_makes_a_tile_that_now_exceeds_imgsz_run_as_windows(tmp_path):
    fake = FakeYolo()
    m = make_model(tmp_path, fake, min_score=0.5, upscale=2.0, imgsz=300, batch_size=8)
    m.detect(rgb_tile(h=128, w=256))                                            # 256x512 after upscaling > imgsz 300 -> windowed
    assert sum(c["n"] for c in fake.calls) > 1 and all(max(c["first"].shape[:2]) <= 300 for c in fake.calls)


# --------------------------------------------------------------------------
# large tiles: windows + cross-window de-duplication
# --------------------------------------------------------------------------


def test_large_tile_is_windowed_and_an_object_in_the_overlap_is_reported_once(tmp_path):
    # imgsz 256, overlap 200 -> stride 56. Windows start at x0 = 0, 56, 112, ... An object at x=150 lies in several windows.
    world = [(150.0, 100.0, 30.0, 10.0, 0.0, 0.9, 0)]
    m = YoloObbDetectionModel(tmp_path / "w.pt", imgsz=256, batch_size=8, min_score=0.5)

    class WindowAware(FakeYolo):
        def predict(self, imgs, **kw):
            self.calls.append({"n": len(imgs), "kw": kw, "first": imgs[0]})
            res = []
            for im in imgs:
                x0 = int(round(im[0, 0, 2] * 1.0))                               # window origin is stamped into pixel (0,0), BGR
                rows = []
                for gx, gy, bw, bh, ang, conf, cls in world:
                    lx = gx - x0 * 8
                    if 0 <= lx < im.shape[1] and 0 <= gy < im.shape[0]:
                        rows.append((lx, gy, bw, bh, ang, conf, cls))
                a = np.array(rows) if rows else None
                res.append(_Result(None if a is None else a[:, :5], None if a is None else a[:, 5], None if a is None else a[:, 6]))
            return res

    m._model, m._names, m._device, m._half = WindowAware(), NAMES, "cpu", False
    tile = np.zeros((256, 600, 3), dtype=np.uint8)
    from geoseek.detect.chipping import chip_origins
    for x0, y0 in chip_origins(600, 256, 256, 56):                               # stamp each window's origin into its top-left pixel
        tile[y0, x0, :] = (x0 // 8, x0 // 8, x0 // 8)
    n_windows = len(chip_origins(600, 256, 256, 56))
    dets = m.detect(tile)
    assert n_windows > 3 and len(dets) == 1                                      # seen in many windows, kept once
    assert dets[0].obb_px[0] == pytest.approx(150.0, abs=1e-6)                   # window-local + origin == global


# --------------------------------------------------------------------------
# model card + licence isolation
# --------------------------------------------------------------------------


def test_operating_point_and_classes_come_from_the_model_card_without_loading_ultralytics(tmp_path):
    w = tmp_path / "geoseek.pt"
    (tmp_path / "geoseek.card.json").write_text(json.dumps(
        {"classes": list(NAMES), "weights_sha256": "abc", "architecture": "YOLO26s-OBB",
         "operating_point": {"per_class": {n: {"conf": 0.1 * (i + 1)} for i, n in enumerate(NAMES)}}}), encoding="utf-8")
    m = YoloObbDetectionModel(w)
    expected = {n: pytest.approx(0.1 * (i + 1)) for i, n in enumerate(NAMES)}
    assert m.min_score == expected and m.class_names == NAMES and m._model is None
    assert m.info["weights_sha256"] == "abc" and m.info["min_score_default"] == expected
    # explicit float override beats the card, applied uniformly to every class the card knows about
    assert YoloObbDetectionModel(w, min_score=0.9).min_score == {n: 0.9 for n in NAMES}
    # explicit dict override is used as-is
    assert YoloObbDetectionModel(w, min_score={"ship": 0.7}).min_score == {"ship": 0.7}
    # no card -> empty map; every lookup falls back to the documented default at call time
    assert YoloObbDetectionModel(tmp_path / "no_card.pt").min_score == {}


def test_info_surfaces_a_low_precision_caveat_for_any_class_below_50_percent(tmp_path):
    w = tmp_path / "geoseek.pt"
    (tmp_path / "geoseek.card.json").write_text(json.dumps({
        "classes": list(NAMES), "weights_sha256": "abc", "architecture": "YOLO26s-OBB",
        "operating_point": {"per_class": {
            "small-vehicle": {"conf": 0.343, "precision": 0.612},   # >= 50% -> no caveat
            "large-vehicle": {"conf": 0.105, "precision": 0.445},   # < 50% -> caveat
            "ship": {"conf": 0.686, "precision": 0.9},
        }},
    }), encoding="utf-8")
    m = YoloObbDetectionModel(w)
    caveats = m.info["caveats"]
    assert len(caveats) == 1 and caveats[0].startswith("large-vehicle:")
    assert "44%" in caveats[0] or "45%" in caveats[0]         # ~44-45% depending on rounding
    assert "small-vehicle" not in "".join(caveats) and "ship" not in "".join(caveats)


def test_legacy_single_conf_card_still_applies_uniformly(tmp_path):
    """A card written before this per-class change (operating_point.conf, no per_class) still works,
    applied uniformly to every class - so an old card on disk doesn't break inference."""
    w = tmp_path / "geoseek.pt"
    (tmp_path / "geoseek.card.json").write_text(json.dumps(
        {"classes": list(NAMES), "weights_sha256": "abc", "architecture": "YOLO26s-OBB",
         "operating_point": {"conf": 0.42}}), encoding="utf-8")
    m = YoloObbDetectionModel(w)
    assert m.min_score == {n: pytest.approx(0.42) for n in NAMES}


def test_importing_geoseek_models_does_not_import_the_agpl_framework():
    code = "import sys, geoseek.models, geoseek.detect; print('ultralytics' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().endswith("False"), out.stderr[-400:]

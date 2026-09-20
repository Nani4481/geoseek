"""Phase 8F-2 Step 7 companion (EXPLORATORY): does test-time upscaling change what the detector finds on the Maxar tiles?

Why this exists. On the held-out DOTA val the detector finds 75-90 % of the labelled small vehicles that are 10-32 px long at
the operating confidence (``scripts/eval_size_operating_point.py``), yet on the Maxar tiles - where a car is also ~12-16 px -
the manual audit found only a few percent of the visible cars (docs/PHASE8F2.md sec. 7). A controlled blur (the tiles are
~1.8x upsampled from a 0.5-0.6 m sensor) costs the DOTA monitor split only ~4 points of vehicle recall
(``scripts/eval_domain_shift_proxy.py``), so softness alone does not explain it. This probe tests the next-simplest
hypothesis - object scale - by running the SAME weights on 1.0x / 1.5x / 2.0x cubic-resampled tiles
(``YoloObbDetectionModel(upscale=...)``; boxes are reported back in original pixels).

What it can and cannot say. There is still NO ground truth on these tiles: the output is detection counts per scale and side-by-side
crops (raw | native | upscaled) for the SAME seeded crops that ``scripts/audit_maxar_crops.py`` drew before any scale was tried,
so a human can tally correct / wrong / missed per scale on identical pixels. Nothing here is a metric and no setting is
adopted from it: the default stays ``upscale=1.0`` (the configuration whose operating point was chosen on DOTA).

    python scripts/maxar_scale_probe.py --observations vannuys --scales 1.0,1.5,2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from geoseek.catalog.sqlite_repository import SQLiteMetadataRepository  # noqa: E402
from geoseek.config import get_settings  # noqa: E402
from geoseek.models import YoloObbDetectionModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_maxar import CLASS_BGR, read_rgb_tile  # noqa: E402

CROP = 256
ZOOM = 3


def crop_panel(rgb: np.ndarray, dets, x0: int, y0: int, label: str) -> np.ndarray:
    crop = np.ascontiguousarray(rgb[y0:y0 + CROP, x0:x0 + CROP, ::-1])
    big = cv2.resize(crop, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_CUBIC)
    n = 0
    for d in dets:
        pts = np.array(d.polygon_px, dtype=np.float64)
        cx, cy = pts.mean(axis=0)
        if x0 <= cx < x0 + CROP and y0 <= cy < y0 + CROP:
            n += 1
            p = ((pts - [x0, y0]) * ZOOM).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(big, [p], True, CLASS_BGR[d.class_name], 1, cv2.LINE_AA)
    cv2.rectangle(big, (0, 0), (CROP * ZOOM, 22), (0, 0, 0), -1)
    cv2.putText(big, f"{label}: {n} detections in crop", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return big


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", default="vannuys", help="comma-separated substrings of the observation id / aoi")
    ap.add_argument("--scales", default="1.0,1.5,2.0")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--min-score", type=float, default=None, help="default: the operating point in the model card (same for every scale)")
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    settings = get_settings()
    weights = args.weights or settings.models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt"
    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")
    keys = args.observations.split(",")
    obs_list = [o for o in repo.list_observations(collection="maxar-opendata")
                if any(k in o.observation_id or k in (o.aoi_name or "") for k in keys)]
    out_dir = settings.data_dir / "detections" / "scale_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {s: YoloObbDetectionModel(weights, min_score=args.min_score, batch_size=2, upscale=s) for s in scales}
    for m in models.values():
        m.load()

    for obs in obs_list:
        scene_dir = Path(obs.dataset_dir or (settings.datasets_dir / obs.observation_id))
        tiles = sorted(repo.list_tiles(observation_id=obs.observation_id), key=lambda t: (t.row, t.col))
        audit_file = settings.data_dir / "detections" / "audit" / f"audit_{obs.observation_id}.json"
        crops = json.loads(audit_file.read_text(encoding="utf-8"))["crops"] if audit_file.is_file() else []
        crop_tiles = {(int(c["stem"].split("__")[1].split("_")[0]), int(c["stem"].split("__")[1].split("_")[1])) for c in crops}
        print(f"\n[probe] === {obs.observation_id}: {len(tiles)} tiles, scales {scales}, {len(crops)} audit crops ===", flush=True)
        per_scale = {s: {"by_class": Counter(), "n": 0, "scores": [], "seconds": 0.0, "tiles_with": 0} for s in scales}
        kept: dict[tuple[int, int], dict] = {}
        for t in tiles:
            rgb, geo = read_rgb_tile(scene_dir, t.row, t.col)
            for s, m in models.items():
                t0 = time.time()
                dets = m.detect(rgb, geo=geo)
                acc = per_scale[s]
                acc["seconds"] += time.time() - t0
                acc["by_class"].update(d.class_name for d in dets)
                acc["n"] += len(dets)
                acc["scores"].extend(d.score for d in dets)
                acc["tiles_with"] += bool(dets)
                if (t.row, t.col) in crop_tiles:
                    kept.setdefault((t.row, t.col), {"rgb": rgb})[s] = dets
        summary = {"observation_id": obs.observation_id, "min_score": next(iter(models.values())).min_score, "scales": {}}
        for s, acc in per_scale.items():
            sc = acc["scores"]
            summary["scales"][str(s)] = {
                "n_detections": acc["n"], "by_class": dict(acc["by_class"].most_common()), "tiles_with_detections": acc["tiles_with"],
                "score_quantiles": {q: round(float(np.quantile(sc, q)), 3) for q in (0.1, 0.5, 0.9)} if sc else None,
                "seconds_per_tile": round(acc["seconds"] / max(len(tiles), 1), 3)}
            print(f"[probe]   scale {s}: {acc['n']} detections in {acc['tiles_with']}/{len(tiles)} tiles {dict(acc['by_class'].most_common())} "
                  f"({acc['seconds'] / max(len(tiles), 1):.2f}s/tile)", flush=True)
        # the same seeded crops the audit drew, raw | native | largest scale
        first, last = scales[0], scales[-1]
        summary["crops"] = []
        for c in crops:
            row, col = int(c["stem"].split("__")[1].split("_")[0]), int(c["stem"].split("__")[1].split("_")[1])
            k = kept[(row, col)]
            counts = {}
            for s_ in scales:
                inside = [d for d in k[s_] if c["x0"] <= np.mean([q[0] for q in d.polygon_px]) < c["x0"] + CROP
                          and c["y0"] <= np.mean([q[1] for q in d.polygon_px]) < c["y0"] + CROP]
                counts[str(s_)] = dict(Counter(d.class_name for d in inside))
            summary["crops"].append({"stem": c["stem"], "by_scale": counts})
            raw = cv2.resize(np.ascontiguousarray(k["rgb"][c["y0"]:c["y0"] + CROP, c["x0"]:c["x0"] + CROP, ::-1]), (CROP * ZOOM, CROP * ZOOM),
                             interpolation=cv2.INTER_CUBIC)
            panels = [raw, crop_panel(k["rgb"], k[first], c["x0"], c["y0"], f"x{first:g}"), crop_panel(k["rgb"], k[last], c["x0"], c["y0"], f"x{last:g}")]
            cv2.imwrite(str(out_dir / f"{c['stem']}__scales.png"), np.hstack(panels))
        (out_dir / f"scale_probe_{obs.observation_id}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[probe]   wrote {len(crops)} raw | x{first:g} | x{last:g} panels and the per-crop counts to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

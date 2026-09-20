"""Phase 8F-2 Step 7: crops for a MANUAL audit of the Maxar detections (there is no ground truth to score against).

Reads ``data/detections/<observation>/summary.json`` (written by scripts/detect_maxar.py), picks tiles and 256x256 windows
with a FIXED SEED (never by how good or bad the result looks), re-runs the detector on just those tiles, and writes each crop
twice at 3x zoom - the raw imagery and the same imagery with the detections drawn - so a human can tally, per crop:

    correct detections / wrong detections (false positives) / visible objects the detector missed

That tally is a small, subjective sample, NOT a metric, and is reported as such. Windows are drawn from tiles that have at
least ``--min-dets`` detections (so there is something to audit) - except for the negative-control observation, where the
tiles with the MOST detections are audited (worst case for false positives).

    python scripts/audit_maxar_crops.py --observation vannuys --n-tiles 3 --crops-per-tile 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

SEED = 20260920
CROP = 256
ZOOM = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observation", required=True, help="substring of the observation id / aoi (e.g. vannuys, chungthang, 120220030330)")
    ap.add_argument("--n-tiles", type=int, default=3)
    ap.add_argument("--crops-per-tile", type=int, default=2)
    ap.add_argument("--min-dets", type=int, default=5)
    ap.add_argument("--worst-case", action="store_true", help="audit the tiles with the MOST detections (negative-control observations)")
    ap.add_argument("--weights", type=Path, default=None)
    args = ap.parse_args()

    settings = get_settings()
    repo = SQLiteMetadataRepository(settings.index_dir / "tiles.sqlite")
    obs = [o for o in repo.list_observations(collection="maxar-opendata")
           if args.observation in o.observation_id or args.observation in (o.aoi_name or "")]
    if not obs:
        raise SystemExit(f"no maxar observation matches {args.observation!r}")
    o = obs[0]
    summ = json.loads((settings.data_dir / "detections" / o.observation_id / "summary.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    pool = [t for t in summ["per_tile"] if t["n_detections"] >= args.min_dets]
    if args.worst_case:
        chosen = sorted(summ["per_tile"], key=lambda t: -t["n_detections"])[: args.n_tiles]
    else:
        chosen = rng.sample(pool, min(args.n_tiles, len(pool)))

    model = YoloObbDetectionModel(args.weights or settings.models_dir / "detector" / "geoseek_obb_v15_yolo26s.pt")
    model.load()
    scene_dir = Path(o.dataset_dir)
    out_dir = settings.data_dir / "detections" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for t in chosen:
        rgb, geo = read_rgb_tile(scene_dir, t["row"], t["col"])
        dets = model.detect(rgb, geo=geo)
        h, w = rgb.shape[:2]
        for c in range(args.crops_per_tile):
            x0 = rng.randrange(0, max(1, w - CROP))
            y0 = rng.randrange(0, max(1, h - CROP))
            crop = np.ascontiguousarray(rgb[y0:y0 + CROP, x0:x0 + CROP, ::-1])
            big = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM), interpolation=cv2.INTER_CUBIC)
            marked = big.copy()
            inside = []
            for d in dets:
                pts = np.array(d.polygon_px, dtype=np.float64)
                cx, cy = pts.mean(axis=0)
                if x0 <= cx < x0 + CROP and y0 <= cy < y0 + CROP:
                    inside.append(d)
                    p = ((pts - [x0, y0]) * ZOOM).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(marked, [p], True, CLASS_BGR[d.class_name], 1, cv2.LINE_AA)
            counts = Counter(d.class_name for d in inside)
            stem = f"{o.observation_id}__{t['row']:03d}_{t['col']:03d}__x{x0}_y{y0}"
            cv2.imwrite(str(out_dir / f"{stem}__raw.png"), big)
            cv2.imwrite(str(out_dir / f"{stem}__det.png"), marked)
            manifest.append({"stem": stem, "tile_id": t["tile_id"], "x0": x0, "y0": y0, "n_detections_in_crop": len(inside),
                             "by_class": dict(counts), "mean_score": round(float(np.mean([d.score for d in inside])), 3) if inside else None})
            print(f"[audit] {stem}: {len(inside)} detections {dict(counts)}", flush=True)
    (out_dir / f"audit_{o.observation_id}.json").write_text(json.dumps({"observation": o.observation_id, "seed": SEED, "crops": manifest}, indent=1),
                                                            encoding="utf-8")
    print(f"[audit] wrote {len(manifest)} crop pairs to {out_dir}")


if __name__ == "__main__":
    main()

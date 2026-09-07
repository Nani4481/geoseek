"""Phase 7b follow-up, Part 1 (diagnosis): are the high-ranked tiles the frozen
judge cannot score actually irrelevant, or just out of the judge's domain?

For every Phase 7a query, take RemoteCLIP's current global top-N at 101,911
tiles and split it into:
  * judged_relevant     - in judgments.json with grade > 0
  * judged_zero         - in judgments.json with grade 0 (judge says: not a match)
  * unjudged_ayodhya    - an Ayodhya tile that was never pooled/judged
  * unjudged_other      - a tile from one of the 8 new regions (un-judgeable with
                          the frozen NDVI/NDWI/NDBI + Ayodhya-river-mask signal)

It renders one contact sheet per query (thumbnail + rank + region + score +
bucket) for the tiles that carry NO positive judgement (judged_zero +
unjudged_*), so they can be eyeballed: of the tiles dragging measured
precision down, how many are genuinely wrong vs. plausibly-right-but-unjudged.

Artifacts under data/eval_retrieval/judge_diagnosis/:
  <query_slug>.png          contact sheet
  manifest.json             every shown tile: query, rank, region, score, bucket
  ANNOTATIONS_TEMPLATE.json  {query: {tile_id: null}}  <- fill with 1/0/'?' by eye
  (if ANNOTATIONS.json exists) diagnosis_summary.json  scored against your calls

Usage:  python scripts/diagnose_judge_transfer.py [--top-n 12]
"""

from __future__ import annotations

import argparse
import io
import json

from PIL import Image, ImageDraw

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine
from geoseek.search.rerank import region_key

EVAL_DIR = get_settings().data_dir / "eval_retrieval"
OUT = EVAL_DIR / "judge_diagnosis"
AYODHYA_JUDGED_OBS = (
    "S2B_44RPQ_20190330_1_L2A_scaled",
    "S2A_44RPQ_20210304_1_L2A_scaled",
    "S2A_44RPQ_20240308_0_L2A_scaled",
)
THUMB = 128
COLS = 6


def _slug(q: str) -> str:
    return q.replace(" ", "_").replace("/", "-")


def _is_ay(tid: str) -> bool:
    return any(tid.startswith(o) for o in AYODHYA_JUDGED_OBS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    queries = json.loads((EVAL_DIR / "queries.json").read_text())
    judgments = json.loads((EVAL_DIR / "judgments.json").read_text())

    eng = SearchEngine()
    n = eng.count()
    aoi_by_obs = {o.observation_id: region_key(o.aoi_name) for o in eng.repo.list_observations()}
    id_to_tile = {fid: r["tile_id"] for fid, r in eng._rows.items()}
    obs_of = {r["tile_id"]: r["observation_id"] for r in eng._rows.values()}

    manifest: dict = {}
    ann_template: dict = {}
    for q in queries:
        vec = eng._encode_text(q)
        scores, ids = eng.vector_index.search(vec, args.top_n)
        judg = judgments[q]
        rows = []
        for rank, (i, s) in enumerate(zip(ids, scores), 1):
            i = int(i)
            if i < 0 or i not in id_to_tile:
                continue
            tid = id_to_tile[i]
            region = aoi_by_obs.get(obs_of.get(tid, ""), "unknown")
            if tid in judg:
                bucket = "judged_relevant" if judg[tid] > 0 else "judged_zero"
            elif _is_ay(tid):
                bucket = "unjudged_ayodhya"
            else:
                bucket = "unjudged_other"
            rows.append({"rank": rank, "tile_id": tid, "region": region,
                         "score": round(float(s), 4), "bucket": bucket})
        manifest[q] = rows

        # contact sheet: everything that is NOT judged_relevant
        show = [r for r in rows if r["bucket"] != "judged_relevant"]
        if not show:
            continue
        ncols = min(COLS, len(show))
        nrows = -(-len(show) // ncols)
        sheet = Image.new("RGB", (THUMB * ncols, 40 + nrows * (THUMB + 30)), "white")
        d = ImageDraw.Draw(sheet)
        d.text((6, 6), f"query: '{q}'   (tiles with NO positive judgement, from global top-{args.top_n})",
               fill="black")
        d.text((6, 20), "label = rank | region | score | bucket", fill=(90, 90, 90))
        for j, r in enumerate(show):
            rr, cc = divmod(j, ncols)
            x, y = cc * THUMB, 40 + rr * (THUMB + 30)
            try:
                png = eng.get_tile_thumbnail_png(r["tile_id"])
                im = Image.open(io.BytesIO(png)).convert("RGB").resize((THUMB, THUMB))
            except Exception as e:
                im = Image.new("RGB", (THUMB, THUMB), "gray")
                print(f"  thumb fail {r['tile_id']}: {e}")
            sheet.paste(im, (x, y))
            d.text((x + 2, y + THUMB + 1), f"#{r['rank']} {r['region']}", fill="black")
            d.text((x + 2, y + THUMB + 14), f"{r['score']:.3f} {r['bucket'].replace('_',' ')}", fill=(90, 90, 90))
            ann_template.setdefault(q, {})[r["tile_id"]] = None
        sheet.save(OUT / f"{_slug(q)}.png")
        print(f"[diag] {q!r}: {len(show)} un-positively-judged tiles -> {_slug(q)}.png")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "ANNOTATIONS_TEMPLATE.json").write_text(json.dumps(ann_template, indent=2))

    ann_path = OUT / "ANNOTATIONS.json"
    if ann_path.is_file():
        ann = json.loads(ann_path.read_text())
        by_bucket: dict[str, list[int]] = {}
        per_query = {}
        for q, rows in manifest.items():
            calls = ann.get(q, {})
            pq = {"plausibly_relevant": 0, "irrelevant": 0, "ambiguous": 0, "n": 0}
            for r in rows:
                if r["bucket"] == "judged_relevant":
                    continue
                c = calls.get(r["tile_id"])
                if c in (1, "1", True):
                    key = "plausibly_relevant"
                elif c in (0, "0", False):
                    key = "irrelevant"
                else:
                    key = "ambiguous"
                pq[key] += 1
                pq["n"] += 1
                by_bucket.setdefault(r["bucket"], []).append(1 if key == "plausibly_relevant" else 0)
            per_query[q] = pq
        summary = {
            "per_query": per_query,
            "by_bucket_fraction_plausibly_relevant": {
                b: round(sum(v) / len(v), 4) for b, v in by_bucket.items() if v},
            "by_bucket_n": {b: len(v) for b, v in by_bucket.items()},
            "overall_fraction_plausibly_relevant": round(
                sum(sum(v) for v in by_bucket.values()) / sum(len(v) for v in by_bucket.values()), 4),
        }
        (OUT / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2))
        print("\n[diag] diagnosis_summary.json:")
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n[diag] fill {ann_path.name} (1=plausibly relevant, 0=irrelevant, '?'=ambiguous) "
              f"then re-run to score.")
    eng.close()


if __name__ == "__main__":
    main()

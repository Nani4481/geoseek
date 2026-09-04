"""Regenerate RemoteCLIP contact sheets after the fixed-bounds true-color rebuild.

Two queries only, RemoteCLIP (the live search engine) only. Saved under
data/prove_semantic/corrected_true_color/.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

from geoseek.config import get_settings
from geoseek.search.engine import SearchEngine

QUERIES = ["newly built structures near a river", "settlement along a riverbank"]
TOP_K = 5
THUMB_PX = 200
OUT_SUBDIR = "prove_semantic/corrected_true_color"


def contact_sheet(images, labels, title, out_path: Path) -> None:
    n = max(len(images), 1)
    header_h, footer_h = 28, 18
    sheet = Image.new("RGB", (THUMB_PX * n, THUMB_PX + header_h + footer_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill="black")
    for i, (img, label) in enumerate(zip(images, labels)):
        sheet.paste(img.resize((THUMB_PX, THUMB_PX)), (i * THUMB_PX, header_h))
        draw.text((i * THUMB_PX + 4, header_h + THUMB_PX + 2), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    out_dir = get_settings().data_dir / OUT_SUBDIR
    engine = SearchEngine()
    print("=" * 78)
    print("RemoteCLIP contact sheets - corrected (fixed-bounds) true-color")
    print("=" * 78)
    saved = []
    for query in QUERIES:
        results, latency_ms = engine.search_text(query, k=TOP_K)
        print(f"\nquery: '{query}'  (latency={latency_ms:.1f} ms)")
        images, labels = [], []
        for r in results:
            print(f"  tile_id={r.tile_id}  score={r.score:.4f}  centroid=({r.lon:.5f}, {r.lat:.5f})  "
                  f"acq_date={r.acq_date}  scene_id={r.scene_id}  cloud_fraction={r.cloud_fraction:.3f}")
            png = engine.get_tile_thumbnail_png(r.tile_id)
            images.append(Image.open(io.BytesIO(png)).convert("RGB"))
            labels.append(f"{r.score:.3f}")
        out_path = out_dir / f"remoteclip_{query.replace(' ', '_')}.png"
        contact_sheet(images, labels, f"RemoteCLIP (fixed true-color): {query}", out_path)
        saved.append(out_path)
    engine.close()
    print("\nSaved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()

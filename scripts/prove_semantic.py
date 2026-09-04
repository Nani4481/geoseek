"""Step C: prove the staged weights are really RemoteCLIP (remote-sensing CLIP).

Runs 4 real text queries against the live search index (RemoteCLIP - the
actual search engine) and, as a control, against vanilla OpenCLIP ViT-B-32
(pretrained='openai', a natural-image model) embedding the SAME tiles. Both
sets of top-5 results are printed and saved as contact-sheet PNGs so the
difference can be eyeballed directly - RemoteCLIP should visibly understand
satellite-specific vocabulary ("sandbars", "dense urban buildings") that
vanilla CLIP, never trained on overhead imagery, does not.

Usage:
    python scripts/prove_semantic.py
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geoseek.config import get_settings
from geoseek.ingest.embed import embed_text_vanilla, embed_tiles_batch_vanilla, make_true_color_uint8
from geoseek.ingest.reader import read_scene
from geoseek.ingest.tiler import tile_scene
from geoseek.search.engine import SearchEngine

QUERIES = [
    "a river with sandbars",
    "dense urban buildings",
    "agricultural fields",
    "open bare ground",
]

RGB_BANDS = ("B04", "B03", "B02")
SCL_BAND = "SCL"
ALL_BANDS = (*RGB_BANDS, SCL_BAND)

TOP_K = 5
THUMB_PX = 200
LARGE_AOI_DIR_SUFFIX = "_scaled"
SCENE_IDS = ("S2B_44RPQ_20190330_1_L2A", "S2A_44RPQ_20240308_0_L2A")


def make_contact_sheet(images: list[Image.Image], labels: list[str], title: str, out_path: Path) -> None:
    n = max(len(images), 1)
    header_h, footer_h = 28, 18
    sheet = Image.new("RGB", (THUMB_PX * n, THUMB_PX + header_h + footer_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 6), title, fill="black")
    for i, (img, label) in enumerate(zip(images, labels)):
        resized = img.resize((THUMB_PX, THUMB_PX))
        sheet.paste(resized, (i * THUMB_PX, header_h))
        draw.text((i * THUMB_PX + 4, header_h + THUMB_PX + 2), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"  saved contact sheet -> {out_path}")


def run_remoteclip(engine: SearchEngine, out_dir: Path) -> None:
    print("=" * 78)
    print("RemoteCLIP (staged weights) - the real search engine")
    print("=" * 78)
    for query in QUERIES:
        results, latency_ms = engine.search_text(query, k=TOP_K)
        print(f"\nquery: '{query}'  (latency={latency_ms:.1f} ms)")
        images, labels = [], []
        for r in results:
            print(
                f"  tile_id={r.tile_id}  score={r.score:.4f}  centroid=({r.lon:.5f}, {r.lat:.5f})  "
                f"acq_date={r.acq_date}  scene_id={r.scene_id}  cloud_fraction={r.cloud_fraction:.3f}"
            )
            png_bytes = engine.get_tile_thumbnail_png(r.tile_id)
            images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            labels.append(f"{r.score:.3f}")
        safe_name = query.replace(" ", "_")
        make_contact_sheet(images, labels, f"RemoteCLIP: {query}", out_dir / f"remoteclip_{safe_name}.png")


def build_vanilla_corpus(scene_dirs: list[Path]):
    """Read + tile + stretch + embed (VANILLA CLIP) every tile across the given scenes."""
    vector_chunks, rows, rgb_cache = [], [], []
    for scene_dir in scene_dirs:
        print(f"[prove]   (vanilla) reading + tiling {scene_dir.name} ...")
        scene = read_scene(scene_dir, list(ALL_BANDS))
        tiles = tile_scene(scene)
        print(f"[prove]   (vanilla) {len(tiles)} tiles - stretching + batch-embedding ...")
        rgb_list = [make_true_color_uint8({b: t.bands[b] for b in RGB_BANDS}, nodata=t.nodata) for t in tiles]
        vectors = embed_tiles_batch_vanilla(rgb_list)
        vector_chunks.append(vectors)
        for t, rgb in zip(tiles, rgb_list):
            rows.append({"tile_id": t.tile_id, "scene_id": scene.scene_id})
            rgb_cache.append(rgb)
    return np.concatenate(vector_chunks, axis=0), rows, rgb_cache


def run_vanilla(scene_dirs: list[Path], out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("VANILLA OpenCLIP (control, pretrained='openai') - same tiles, same queries")
    print("=" * 78)
    vectors, rows, rgb_cache = build_vanilla_corpus(scene_dirs)
    print(f"[prove] vanilla corpus: {len(rows)} tiles embedded")

    for query in QUERIES:
        t0 = time.time()
        qvec = embed_text_vanilla(query)
        scores = vectors @ qvec
        order = np.argsort(-scores)[:TOP_K]
        latency_ms = (time.time() - t0) * 1000.0

        print(f"\nquery: '{query}'  (latency={latency_ms:.1f} ms)")
        images, labels = [], []
        for idx in order:
            row = rows[idx]
            print(f"  tile_id={row['tile_id']}  score={scores[idx]:.4f}  scene_id={row['scene_id']}")
            images.append(Image.fromarray(rgb_cache[idx]))
            labels.append(f"{scores[idx]:.3f}")
        safe_name = query.replace(" ", "_")
        make_contact_sheet(images, labels, f"Vanilla CLIP: {query}", out_dir / f"vanilla_{safe_name}.png")


def main() -> None:
    settings = get_settings()
    scaled_dirs = [settings.datasets_dir / (sid + LARGE_AOI_DIR_SUFFIX) for sid in SCENE_IDS]
    missing = [d for d in scaled_dirs if not d.is_dir()]
    if missing:
        raise SystemExit(
            f"Scaled AOI scene(s) not staged: {missing}. "
            "Run the Phase 2 STEP A scale-up staging first."
        )

    out_dir = settings.data_dir / "prove_semantic"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = SearchEngine()
    run_remoteclip(engine, out_dir)
    engine.close()

    run_vanilla(scaled_dirs, out_dir)

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE NOTE")
    print("=" * 78)
    print(
        "Open data/prove_semantic/remoteclip_<query>.png next to vanilla_<query>.png\n"
        "for each of the 4 queries. RemoteCLIP (remote-sensing-tuned) is expected to\n"
        "retrieve tiles whose true-color appearance visibly matches the query text\n"
        "(river/sandbars, dense urban grid, farmland parcels, bare ground), while\n"
        "vanilla CLIP - trained only on everyday photos, never overhead imagery - is\n"
        "expected to be visibly less consistent on these satellite-specific terms.\n"
        "That gap is the proof the staged checkpoint is really the RS-tuned model."
    )
    print(f"\nAll contact sheets saved under: {out_dir}")


if __name__ == "__main__":
    main()

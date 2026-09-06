"""Phase 7b Tier 3: latency-vs-scale and storage-vs-scale plots across the
four measured points (3267 baseline / tier1 / tier2 / tier3).

Baseline (3267 vectors) predates this measurement harness (Phase 7b started
from the already-built Phase 7a production catalog), so its row is assembled
from two sources, both labeled in the figure notes:
  - latency: Phase 7a's OWN documented measurements (scripts/bench_spatial_index.py
    + scripts/eval_retrieval_prepare.py at the time, see PHASE7.md / memory) -
    a real measurement, different harness, same methodology family (median/p95
    over repeated warm queries).
  - storage: faiss_index_mb is exact (3267*512*4 bytes, deterministic from the
    flat-index format); sqlite_db_mb is ESTIMATED by linear extrapolation from
    the current DB's bytes/row (not a direct measurement) since the pre-scale-up
    sqlite file was not preserved - marked as estimated on the chart.

Palette: dataviz skill categorical slots 1-5 (blue/orange/aqua/yellow/magenta),
fixed order, light-mode chart surface. Log-log axes (index size spans ~30x;
latency spans ~3 orders of magnitude across operations).

Usage: python scripts/plot_scale.py
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from geoseek.config import get_settings

EVAL_DIR = get_settings().data_dir / "eval_retrieval"

# dataviz skill categorical slots (light mode), fixed order
SLOT_BLUE, SLOT_ORANGE, SLOT_AQUA, SLOT_YELLOW, SLOT_MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

# Phase 7a documented baseline (see memory geoseek-phase7a / PHASE7.md) -
# a real measurement under a slightly different harness, not this script.
BASELINE_N = 3267
BASELINE_LATENCY_MS = {  # {op: (median, p95)}
    "text_search_k20": (17.4, 21.8),
    "image_search_k20": (1.0, 1.5),       # == tile-seeded KNN's core cost (same search_image path)
    "point_seeded_knn": (1.00, 1.5),
    "tile_seeded_knn": (1.0, 1.5),
    "bbox_filter": (0.28, 0.5),
}
BASELINE_FAISS_MB = round(BASELINE_N * 512 * 4 / 1e6, 3)  # exact: flat index, deterministic


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_LINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, which="major", axis="both", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _load_tier_reports() -> list[dict]:
    rows = []
    baseline_storage_estimated = None
    for tier in ("tier1", "tier2", "tier3"):
        p = EVAL_DIR / f"scale_report_{tier}.json"
        if p.is_file():
            rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("No scale_report_tier*.json found - run measure_tier.py for at least tier1 first.")

    # estimate baseline sqlite size by linear extrapolation from the earliest
    # available real measurement's bytes/row (clearly marked as an estimate)
    first = rows[0]
    bytes_per_row = (first["storage"]["sqlite_db_mb"] * 1e6) / first["query"]["n_vectors"]
    baseline_storage_estimated = round(bytes_per_row * BASELINE_N / 1e6, 3)

    baseline = {
        "tier": "baseline", "is_baseline": True,
        "query": {"n_vectors": BASELINE_N, "latency": {
            op: {"median_ms": m, "p95_ms": p} for op, (m, p) in BASELINE_LATENCY_MS.items()
        }},
        "storage": {"faiss_index_mb": BASELINE_FAISS_MB, "sqlite_db_mb": baseline_storage_estimated,
                    "sqlite_db_mb_is_estimated": True},
    }
    return [baseline] + rows


def plot_latency(rows: list[dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    _style_axes(ax)
    series = [
        ("text_search_k20", "text search (k=20)", SLOT_BLUE),
        ("image_search_k20", "image search (k=20)", SLOT_ORANGE),
        ("point_seeded_knn", "point-seeded KNN", SLOT_AQUA),
        ("tile_seeded_knn", "tile-seeded KNN", SLOT_YELLOW),
        ("bbox_filter", "bbox filter", SLOT_MAGENTA),
    ]
    xs_all = [r["query"]["n_vectors"] for r in rows]
    for key, label, color in series:
        xs, ys_med, ys_p95 = [], [], []
        for r in rows:
            lat = r["query"]["latency"].get(key)
            if lat is None or lat.get("median_ms") is None:
                continue
            xs.append(r["query"]["n_vectors"])
            ys_med.append(lat["median_ms"])
            ys_p95.append(lat["p95_ms"])
        if not xs:
            continue
        ax.plot(xs, ys_med, color=color, linewidth=2, marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, zorder=3)
        ax.plot(xs, ys_p95, color=color, linewidth=1.2, linestyle="--", alpha=0.55,
                marker="o", markersize=4, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=min(xs_all) * 0.55)  # headroom so the legend doesn't sit on the first points
    ax.set_xlabel("index size (vectors, log scale)")
    ax.set_ylabel("latency, ms (log scale) — solid = median, dashed = p95")
    ax.set_title("Query latency vs. index size (FaissFlatIP, warm)", color=INK_PRIMARY, fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    fig.text(0.01, 0.01,
              "baseline (3267) latency: Phase 7a's own measurement, different harness — see PHASE7B.md",
              fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_storage(rows: list[dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    _style_axes(ax)
    series = [
        ("faiss_index_mb", "FAISS index", SLOT_BLUE),
        ("sqlite_db_mb", "SQLite catalog", SLOT_ORANGE),
        ("raw_datasets_mb", "raw staged imagery", SLOT_AQUA),
        ("total_data_dir_mb", "total data/ footprint", SLOT_YELLOW),
    ]
    for key, label, color in series:
        xs, ys = [], []
        for r in rows:
            v = r["storage"].get(key)
            if v is None:
                continue
            xs.append(r["query"]["n_vectors"])
            ys.append(v)
        if len(xs) < 2 and key != "faiss_index_mb":
            continue
        est_flag = " (baseline pt. estimated)" if key == "sqlite_db_mb" else ""
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label + est_flag, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("index size (vectors, log scale)")
    ax.set_ylabel("storage, MB (log scale)")
    ax.set_title("Storage footprint vs. index size", color=INK_PRIMARY, fontsize=13, pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    rows = _load_tier_reports()
    print(f"[plot] {len(rows)} scale points: {[r['query']['n_vectors'] for r in rows]}")
    plot_latency(rows, EVAL_DIR / "plot_latency_vs_scale.png")
    plot_storage(rows, EVAL_DIR / "plot_storage_vs_scale.png")


if __name__ == "__main__":
    main()

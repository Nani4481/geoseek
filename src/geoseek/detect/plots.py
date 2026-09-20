"""Matplotlib figures for the detector: training curves (from Ultralytics' results.csv), per-class AP bars, PR curves.

Kept in the library (not in the evaluation script) so the training curves can be produced straight after training, before
the much longer official-val evaluation. matplotlib is imported lazily and forced onto the non-interactive Agg backend.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from geoseek.detect.classes import CLASS_GROUPS
from geoseek.detect.evaluate import pr_curve


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_results_csv(run_dir: Path) -> dict[str, np.ndarray]:
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    head = [h.strip() for h in rows[0]]
    cols = {h: np.array([float(r[i]) if r[i].strip() not in ("", "nan") else np.nan for r in rows[1:]]) for i, h in enumerate(head)}
    return cols


def plot_training_curves(run_dir: Path, out_dir: Path, *, baseline: dict | None, official_points: list[dict]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    c = read_results_csv(run_dir)
    ep = c["epoch"]
    written = []

    losses = [k[len("train/"):] for k in c if k.startswith("train/") and k.endswith("_loss") and np.nanmax(c[k]) > 0]
    n = len(losses)
    fig, axes = _plt().subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, name in zip(axes[0], losses):
        ax.plot(ep, c[f"train/{name}"], "o-", ms=3, label="train (on-the-fly augmented)")
        if f"val/{name}" in c:
            ax.plot(ep, c[f"val/{name}"], "s-", ms=3, label="val (monitor holdout)")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Loss per epoch: train vs monitor-holdout validation (official val is NOT used per epoch)")
    fig.tight_layout()
    p = out_dir / "detector_loss_curves.png"
    fig.savefig(p, dpi=130)
    _plt().close(fig)
    written.append(p)

    fig, axes = _plt().subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in ((axes[0], "metrics/mAP50(B)", "mAP@0.5"), (axes[1], "metrics/mAP50-95(B)", "mAP@0.5:0.95")):
        ax.plot(ep, c[key], "o-", ms=3, label="monitor holdout (chip level), per epoch")
        if baseline:
            ax.axhline(baseline["monitor"]["mAP50" if "50-95" not in key else "mAP50_95"], ls="--", c="gray",
                       label="epoch 0: pretrained, no fine-tuning")
        pts = [(o["epoch"], o["mAP50" if "50-95" not in key else "mAP50_95"]) for o in official_points]
        if pts:
            ax.plot(*zip(*pts), "D", c="crimson", label="OFFICIAL val, post-hoc (reporting only)")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = out_dir / "detector_map_curves.png"
    fig.savefig(p, dpi=130)
    _plt().close(fig)
    written.append(p)
    return written


def plot_per_class(res_ft: dict, res_base: dict, out_dir: Path, label_set: str) -> Path:
    fig, axes = _plt().subplots(1, 2, figsize=(12, 4.2), sharey=True)
    order = [n for g in CLASS_GROUPS.values() for n in g]
    colors = {"ground_vehicles": "tab:red", "ships_and_aircraft": "tab:blue", "infrastructure": "tab:green"}
    for ax, key, title in ((axes[0], "AP50", "AP@0.5"), (axes[1], "AP50_95", "AP@0.5:0.95")):
        x = np.arange(len(order))
        base = [res_base["per_class"][n][key] for n in order]
        ft = [res_ft["per_class"][n][key] for n in order]
        ax.bar(x - 0.2, base, 0.4, color="lightgray", label="pretrained, no fine-tuning")
        ax.bar(x + 0.2, ft, 0.4, color=[colors[[g for g, m in CLASS_GROUPS.items() if n in m][0]] for n in order],
               label="fine-tuned (colour = group)")
        for xi, v in zip(x + 0.2, ft):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title(f"{title} - official val, DOTA {label_set} GT, full-image protocol")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p = out_dir / f"detector_per_class_ap_{label_set.replace('.', '')}.png"
    fig.savefig(p, dpi=130)
    _plt().close(fig)
    return p


def plot_pr(matches_ft: dict, matches_base: dict, out_dir: Path) -> Path:
    fig, axes = _plt().subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("small-vehicle", "large-vehicle")):
        for tag, mm, sty in (("fine-tuned", matches_ft, "-"), ("pretrained", matches_base, "--")):
            r, p, _ = pr_curve(mm[name], 0.5)
            ax.plot(r, p, sty, label=tag)
        ax.set_title(f"{name}: precision-recall @ IoU 0.5 (official val, v1.5 GT)")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    p = out_dir / "detector_pr_vehicles.png"
    fig.savefig(p, dpi=130)
    _plt().close(fig)
    return p

"""Diff two baseline snapshots: metric, baseline, candidate, delta, and whether the delta exceeds noise.

Any phase that claims an improvement must produce this diff (``scripts/compare_to_baseline.py``).

Snapshot contract (``data/eval/baseline_v1.json`` and every candidate snapshot):

* numeric leaves anywhere under the metric sections (``retrieval``, ``change_detection``, ``detector``,
  ``latency``, ``footprint``, ``ingestion``) are compared by dotted path;
* ``noise_bands`` maps a dotted metric path to ``{"band": <float>, ...}`` - the smallest |delta| that is
  distinguishable from run-to-run / sampling noise for that metric;
* the ``meta``, ``environment``, ``reproduction`` and ``noise_bands`` sections are never compared, and any
  ``{"status": "NOT_REPRODUCED", ...}`` object is a placeholder, not a number, so it is skipped (and listed).

A delta is "significant" only when ``|delta| > band``. Direction is judged with the metric's polarity
(lower is better for FPR, latency, sizes, ...); a metric with no polarity rule reports ``changed``.

The band is an independent-sample uncertainty and is wide for 16-query / 10-region metrics. Where both snapshots
scored the same queries or regions, a PAIRED bootstrap of the delta is also computed; if the band calls a delta noise
but the paired 95% CI excludes zero, the verdict is ``improved_paired`` / ``regressed_paired`` (never silently
upgraded to plain ``improved``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

SIGNIFICANT_VERDICTS = ("improved", "regressed", "changed", "improved_paired", "regressed_paired", "changed_paired")
SKIP_SECTIONS = frozenset({"meta", "environment", "reproduction", "noise_bands", "noise_detail", "schema"})

# last path segment (or suffix) => lower is better
_HIGHER_EXACT = frozenset({"tp", "map", "iou", "accuracy"})
_HIGHER_SUBSTR = ("recall", "precision", "ndcg", "f1", "ap50", "per_s", "throughput", "agreement", "f0.5")
_LOWER_EXACT = frozenset({"fpr", "fp", "fn", "false_positive_rate", "false_positives", "false_negatives"})
_LOWER_SUFFIXES = ("_ms", "_seconds", "_mb", "_mib", "_gb", "_bytes", "_per_tile", "_wall_s", "_vram", "_rss")
_LOWER_PREFIXES = ("latency", "peak_", "bytes_per", "seconds_per", "ms_per", "cost")


def lower_is_better(path: str) -> bool | None:
    """True (lower is better) / False (higher is better) / None (no known polarity -> reported 'changed').

    Throughput (``*_per_s``) is checked before the ``*_s`` family so ``tiles_per_s`` is higher-is-better.
    """
    last = path.rsplit(".", 1)[-1].lower()
    if last in _HIGHER_EXACT or any(s in last for s in _HIGHER_SUBSTR):
        return False
    if last in _LOWER_EXACT or last.endswith(_LOWER_SUFFIXES) or last.startswith(_LOWER_PREFIXES):
        return True
    return None


def is_placeholder(x) -> bool:
    return isinstance(x, Mapping) and x.get("status") == "NOT_REPRODUCED"


def flatten_metrics(snapshot: Mapping) -> tuple[dict[str, float], list[str]]:
    """Dotted-path -> float for every numeric leaf under the metric sections; plus the paths that are
    NOT_REPRODUCED placeholders (so the diff can say they were skipped, not silently drop them)."""
    numeric: dict[str, float] = {}
    placeholders: list[str] = []

    def walk(node, path: str) -> None:
        if is_placeholder(node):
            placeholders.append(path)
        elif isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, bool) or node is None or isinstance(node, str):
            return
        elif isinstance(node, (int, float, np.integer, np.floating)):
            v = float(node)
            if math.isfinite(v):
                numeric[path] = v
        # lists are not compared element-wise (curves, raw samples live in them)

    for section, body in snapshot.items():
        if section in SKIP_SECTIONS:
            continue
        walk(body, section)
    return numeric, placeholders


def noise_band(path: str, baseline: Mapping, candidate: Mapping) -> float | None:
    """The band for ``path``: the larger of the two snapshots' recorded bands (None if neither has one)."""
    bands = []
    for snap in (baseline, candidate):
        entry = (snap.get("noise_bands") or {}).get(path)
        if entry is not None:
            bands.append(float(entry["band"] if isinstance(entry, Mapping) else entry))
    return max(bands) if bands else None


@dataclass(frozen=True)
class DiffRow:
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative: float | None          # delta / |baseline|; None when baseline is 0 or a side is missing
    band: float | None
    exceeds_noise: bool | None      # None: no band recorded, or a side is missing
    verdict: str                    # improved | regressed | changed | within_noise | no_band | only_in_baseline | only_in_candidate
                                    # | improved_paired | regressed_paired (band says noise, the paired test says otherwise)
    paired_ci: tuple[float, float] | None = None   # 95% CI of the paired delta (same queries / regions in both snapshots)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["paired_ci"] = list(self.paired_ci) if self.paired_ci else None
        return d


def judge(delta: float, band: float | None, lower_better: bool | None) -> tuple[bool | None, str]:
    """(exceeds_noise, verdict) for a delta against a band. Strict ``>``: a delta equal to the band is noise."""
    if band is None:
        return None, "no_band"
    exceeds = abs(delta) > band
    if not exceeds:
        return False, "within_noise"
    if lower_better is None:
        return True, "changed"
    better = (delta < 0) if lower_better else (delta > 0)
    return True, "improved" if better else "regressed"



# --------------------------------------------------------------------------
# paired tests (same queries / same regions scored in both snapshots)
# --------------------------------------------------------------------------
# The recorded noise band is an INDEPENDENT-sample uncertainty (query / region bootstrap of one snapshot) and is
# deliberately conservative: with 16 queries or 10 regions it is wide. When both snapshots scored the same units,
# the difference can be tested far more tightly by resampling the units jointly. A metric the band calls noise but
# whose paired 95% CI excludes zero gets verdict ``improved_paired`` / ``regressed_paired``.

_RETR_ANY = re.compile(r"^(retrieval\.(?:ayodhya_3267\.[^.]+|full_101911\.[^.]+\.[^.]+))\.(?:(all|excluding_low_confidence)\.)?k(\d+)\.(recall|precision|ndcg)$")
_OSCD = re.compile(r"^(change_detection\.oscd_heldout\.thr_[^.]+)\.pooled\.(precision|recall|f1|iou|fpr)$")


def _get(node, dotted: str):
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def paired_bootstrap_ci(base: Sequence[float], cand: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of mean(cand - base), resampling the paired units jointly."""
    d = np.asarray(cand, dtype=float) - np.asarray(base, dtype=float)
    if d.size < 2:
        return (float(d.mean()), float(d.mean())) if d.size else (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def _counts_metric(c: np.ndarray, name: str) -> float:
    tp, fp, fn, tn = (float(x) for x in c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0}[name]


def paired_delta_ci(path: str, baseline: Mapping, candidate: Mapping, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """95% CI of (candidate - baseline) for ``path`` when both snapshots carry the per-unit data; else None."""
    m = _RETR_ANY.match(path)
    if m:
        prefix, subset, k, metric = m.group(1), m.group(2), m.group(3), m.group(4)
        # per-query scores sit next to the metric blocks: <prefix>.per_query
        # (ayodhya: <prefix> = ...ayodhya_3267.<system>; full corpus: ...full_101911.<system>.<condition>)
        pq_b, pq_c = _get(baseline, f"{prefix}.per_query"), _get(candidate, f"{prefix}.per_query")
        if not (isinstance(pq_b, list) and isinstance(pq_c, list)):
            return None
        key = f"{metric}@{k}"
        rows_b = {r["query"]: r for r in pq_b}
        rows_c = {r["query"]: r for r in pq_c}
        qs = [q for q in rows_b if q in rows_c and not (subset == "excluding_low_confidence" and rows_b[q].get("low_confidence"))]
        if len(qs) < 2 or set(rows_b) != set(rows_c):
            return None
        return paired_bootstrap_ci([rows_b[q][key] for q in qs], [rows_c[q][key] for q in qs], n_boot=n_boot, seed=seed)
    m = _OSCD.match(path)
    if m:
        prefix, name = m.group(1), m.group(2)
        reg_b, reg_c = _get(baseline, f"{prefix}.per_region"), _get(candidate, f"{prefix}.per_region")
        if not (isinstance(reg_b, Mapping) and isinstance(reg_c, Mapping)) or set(reg_b) != set(reg_c) or len(reg_b) < 2:
            return None
        regions = sorted(reg_b)
        cb = np.array([[reg_b[r][x] for x in ("tp", "fp", "fn", "tn")] for r in regions], dtype=float)
        cc = np.array([[reg_c[r][x] for x in ("tp", "fp", "fn", "tn")] for r in regions], dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(regions), size=(n_boot, len(regions)))
        diffs = [_counts_metric(cc[i].sum(axis=0), name) - _counts_metric(cb[i].sum(axis=0), name) for i in idx]
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        return float(lo), float(hi)
    return None


def diff_snapshots(baseline: Mapping, candidate: Mapping, *, only_prefix: str | None = None,
                   paired: bool = True) -> list[DiffRow]:
    b, b_ph = flatten_metrics(baseline)
    c, c_ph = flatten_metrics(candidate)
    rows: list[DiffRow] = []
    for path in sorted(set(b) | set(c)):
        if only_prefix and not path.startswith(only_prefix):
            continue
        if path not in c:
            rows.append(DiffRow(path, b[path], None, None, None, noise_band(path, baseline, candidate), None, "only_in_baseline"))
            continue
        if path not in b:
            rows.append(DiffRow(path, None, c[path], None, None, noise_band(path, baseline, candidate), None, "only_in_candidate"))
            continue
        delta = c[path] - b[path]
        band = noise_band(path, baseline, candidate)
        lower = lower_is_better(path)
        exceeds, verdict = judge(delta, band, lower)
        ci = paired_delta_ci(path, baseline, candidate) if paired else None
        if ci is not None and verdict in ("within_noise", "no_band") and (ci[0] > 0 or ci[1] < 0):
            better = (ci[1] < 0) if lower else (ci[0] > 0)
            verdict = "changed_paired" if lower is None else ("improved_paired" if better else "regressed_paired")
        rel = delta / abs(b[path]) if b[path] != 0 else None
        rows.append(DiffRow(path, b[path], c[path], delta, rel, band, exceeds, verdict, ci))
    return rows


def skipped_placeholders(baseline: Mapping, candidate: Mapping) -> dict[str, list[str]]:
    return {"baseline": flatten_metrics(baseline)[1], "candidate": flatten_metrics(candidate)[1]}


def summarize(rows: Sequence[DiffRow]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.verdict] = out.get(r.verdict, 0) + 1
    out["total"] = len(rows)
    return out


def _fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "-"
    if x != 0 and abs(x) < 10 ** -nd:
        return f"{x:.2e}"
    return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:,.1f}"


def render_markdown(rows: Sequence[DiffRow], *, only_significant: bool = False,
                    skipped: Mapping[str, Sequence[str]] | None = None) -> str:
    shown = [r for r in rows if not only_significant or r.verdict in SIGNIFICANT_VERDICTS]
    lines = ["| metric | baseline | candidate | delta | rel | noise band | exceeds noise | paired 95% CI of delta | verdict |",
             "|---|--:|--:|--:|--:|--:|:--:|--:|---|"]
    for r in shown:
        rel = "-" if r.relative is None else f"{r.relative * 100:+.1f}%"
        ex = "-" if r.exceeds_noise is None else ("yes" if r.exceeds_noise else "no")
        d = "-" if r.delta is None else f"{r.delta:+.4f}" if abs(r.delta) < 1000 else f"{r.delta:+,.1f}"
        pc = "-" if r.paired_ci is None else f"[{r.paired_ci[0]:+.4f}, {r.paired_ci[1]:+.4f}]"
        lines.append(f"| `{r.metric}` | {_fmt(r.baseline)} | {_fmt(r.candidate)} | {d} | {rel} | {_fmt(r.band)} | {ex} | {pc} | {r.verdict} |")
    s = summarize(rows)
    lines += ["", "**Summary:** " + ", ".join(f"{k}={v}" for k, v in sorted(s.items()))]
    if skipped and (skipped.get("baseline") or skipped.get("candidate")):
        lines += ["", "**Skipped (NOT_REPRODUCED placeholders, not numbers):** "
                  + "; ".join(f"{side}: {', '.join(paths) or 'none'}" for side, paths in skipped.items())]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# noise-band estimators (shared by scripts/snapshot_baseline.py and its tests)
# --------------------------------------------------------------------------


def spread(values: Iterable[float]) -> float:
    """max - min across repeated runs of the same measurement."""
    v = [float(x) for x in values]
    return (max(v) - min(v)) if v else 0.0


def bootstrap_halfwidth(per_unit_scores: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                        level: float = 0.95) -> float:
    """Half-width of the percentile bootstrap CI of the MEAN over independent units (queries, images, regions)."""
    x = np.asarray(per_unit_scores, dtype=float)
    if x.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float((hi - lo) / 2)

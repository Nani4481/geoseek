"""Numbers the existing reports CLAIM, and the machinery that checks a fresh snapshot against them.

These values are transcribed from ``docs/EVALUATION_REPORT.md`` (section noted per entry) and are used
for ONE purpose: deciding whether a freshly measured number reproduces what was reported. They are
never copied into a snapshot as measurements - a snapshot holds only numbers produced by a run.

Tolerances are per metric family and deliberately plain:
  * fractions the report prints to 3 decimals  -> 0.0015 (rounding of the report + of ``report.json``'s 4 dp)
  * percentages printed to 1 decimal            -> 0.0006 (as fractions)
  * exact counts (TP/FP/FN, integer corpus sizes) -> 0
  * latency -> relative 40% (the live index is ~4% larger and the host has run-to-run variance), bbox filter
    additionally +-0.5 ms absolute because it is sub-millisecond
  * throughput -> relative 20%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from geoseek.eval.compare import flatten_metrics

R3 = 0.0015
PCT = 0.0006

OSCD_PER_REGION_080 = {   # section 8.3: P, R, F1 in percent at the deployed 0.80 operating point
    "lasvegas": (74.3, 74.3, 74.3), "montpellier": (84.2, 58.3, 68.9), "brasilia": (53.9, 62.6, 57.9),
    "chongqing": (74.5, 45.6, 56.6), "dubai": (59.1, 35.4, 44.3), "rio": (50.2, 34.8, 41.1),
    "milano": (77.0, 22.4, 34.8), "norcia": (20.8, 46.5, 28.7), "saclay_w": (17.2, 41.1, 24.3),
    "valencia": (2.2, 13.8, 3.8),
}


@dataclass(frozen=True)
class Reported:
    path: str                  # dotted path of the corresponding measurement in a snapshot
    value: float
    tol_abs: float = 0.0
    tol_rel: float = 0.0       # allowed |measured - reported| <= tol_abs + tol_rel * |reported|
    source: str = ""
    note: str = ""

    def within(self, measured: float) -> bool:
        return abs(measured - self.value) <= self.tol_abs + self.tol_rel * abs(self.value) + 1e-12


def _k(k: int) -> str:
    return f"k{k}"


def _retrieval() -> list[Reported]:
    out: list[Reported] = []
    # section 7.2, all 16 queries: K -> (RC R, P, NDCG), (VA R, P, NDCG)
    all_q = {1: ((.040, .563, .438), (.019, .313, .281)), 5: ((.228, .425, .397), (.062, .263, .231)),
             10: ((.365, .381, .419), (.115, .238, .223)), 20: ((.705, .356, .526), (.227, .231, .254))}
    hi_q = {5: ((.244, .492, .454), (.067, .308, .257)), 10: ((.403, .446, .480), (.122, .277, .244)),
            20: ((.667, .381, .547), (.261, .277, .282))}
    for subset, table in (("all", all_q), ("excluding_low_confidence", hi_q)):
        for k, (rc, va) in table.items():
            for system, vals in (("remoteclip", rc), ("vanilla", va)):
                for metric, v in zip(("recall", "precision", "ndcg"), vals):
                    out.append(Reported(f"retrieval.ayodhya_3267.{system}.{subset}.{_k(k)}.{metric}", v, R3,
                                        source="EVALUATION_REPORT 7.2"))
    # section 7.3 (RemoteCLIP, frozen 101,911-tile corpus, frozen judgements)
    raw = {10: (.027, .031, .026), 20: (.106, .028, .048)}
    reg = {10: (.365, .381, .418), 20: (.705, .356, .526)}
    for cond, table in (("global", raw), ("region_filtered_ayodhya", reg)):
        for k, vals in table.items():
            for metric, v in zip(("recall", "precision", "ndcg"), vals):
                out.append(Reported(f"retrieval.full_101911.remoteclip.{cond}.{_k(k)}.{metric}", v, R3,
                                    source="EVALUATION_REPORT 7.3"))
    for bucket, v in (("judged_relevant", .028), ("judged_zero", .066), ("unjudged_ayodhya", 0.0), ("unjudged_other_region", .906)):
        out.append(Reported(f"retrieval.full_101911.remoteclip.global.topk_composition_k20.{bucket}", v, R3,
                            source="EVALUATION_REPORT 7.3 composition table"))
    return out


def _oscd() -> list[Reported]:
    out: list[Reported] = []
    base = "change_detection.oscd_heldout"
    pooled = {"0p50": dict(precision=51.7, recall=61.0, f1=56.0, iou=38.8, fpr=3.10, tp=96997, fp=90601, fn=62080),
              "0p80": dict(precision=60.3, recall=51.0, f1=55.3, iou=38.2, fpr=1.83, tp=81146, fp=53342, fn=77931)}
    for thr, vals in pooled.items():
        for name, v in vals.items():
            exact = name in ("tp", "fp", "fn")
            out.append(Reported(f"{base}.thr_{thr}.pooled.{name}", v if exact else v / 100.0, 0.0 if exact else PCT,
                                source="EVALUATION_REPORT 8.1"))
    out.append(Reported(f"{base}.test_change_pixel_fraction", 0.0517, PCT, source="EVALUATION_REPORT 8.1"))
    out.append(Reported(f"{base}.test_pixels_total", 3077936, source="EVALUATION_REPORT 8.1"))
    out.append(Reported(f"{base}.chosen_threshold_from_validation", 0.80, 0.0, source="EVALUATION_REPORT 5.3 / 8"))
    for region, (p, r, f) in OSCD_PER_REGION_080.items():
        for name, v in (("precision", p), ("recall", r), ("f1", f)):
            out.append(Reported(f"{base}.thr_0p80.per_region.{region}.{name}", v / 100.0, PCT, source="EVALUATION_REPORT 8.3"))
    return out


def _detector() -> list[Reported]:
    base = "detector.dota_val_v15_full_image"
    rows = {  # group: (AP50, AP50 CI lo, hi, AP50:95)       section 15
        "ground_vehicles": (.854, .786, .891, .471), "ships_and_aircraft": (.859, .735, .911, .557),
        "infrastructure": (.751, .704, .784, .414), "all_kept_classes": (.817, .764, .845, .482)}
    out = []
    for g, (ap, lo, hi, ap95) in rows.items():
        for name, v in (("ap50", ap), ("ap50_ci_lo", lo), ("ap50_ci_hi", hi), ("ap50_95", ap95)):
            out.append(Reported(f"{base}.{g}.{name}", v, R3, source="EVALUATION_REPORT 15"))
    out.append(Reported(f"{base}.small_vehicle.ap50_95", .413, R3, source="EVALUATION_REPORT 15"))
    return out


def _latency() -> list[Reported]:
    # section 3.2, 100,887-vector tier; the live index has 105,245 vectors
    rows = {"text_search_k20": (27.8, 38.3), "image_search_k20": (21.7, 29.8), "point_seeded_knn": (20.0, 28.9),
            "tile_seeded_knn": (18.0, 24.5), "bbox_filter": (0.29, 0.54)}
    out = []
    for op, (med, p95) in rows.items():
        tol_abs = 0.5 if op == "bbox_filter" else 0.0
        for name, v in (("median_ms", med), ("p95_ms", p95)):
            out.append(Reported(f"latency.{op}.{name}", v, tol_abs, 0.40, source="EVALUATION_REPORT 3.2",
                                note="reported at 100,887 vectors; live index is larger"))
    return out


def _footprint_and_ingest() -> list[Reported]:
    return [
        Reported("footprint.faiss_bytes_per_vector", 2048.0, 1.0, source="EVALUATION_REPORT 2.3 (n x 512 x 4 B)",
                 note="file bytes / vectors; the FAISS header adds a few dozen bytes in total"),
        Reported("footprint.datasets_mb", 32396.4, 0.0, 0.05, source="EVALUATION_REPORT 2.3 (raw staged imagery)"),
        Reported("footprint.models_mb", 1210.4, 0.0, 0.05, source="EVALUATION_REPORT 2.3 (RemoteCLIP + vanilla CLIP)"),
        Reported("footprint.data_dir_mb", 34449.9, 0.0, 0.05, source="EVALUATION_REPORT 2.3 (total data/)"),
        Reported("footprint.sqlite_bytes_per_catalog_tile", 900.0, 0.0, 0.25, source="EVALUATION_REPORT 2.3 (~0.9 KB/tile)"),
        Reported("footprint.index_bytes_per_embedded_tile", 2900.0, 0.0, 0.20, source="EVALUATION_REPORT 2.3 (~2.9 KB/tile)"),
        Reported("ingestion.end_to_end_tiles_per_s", 207.5, 0.0, 0.20, source="EVALUATION_REPORT 2.1 (AC, ~205-210)"),
        Reported("ingestion.pure_embed_tiles_per_s", 384.0, 0.0, 0.20, source="EVALUATION_REPORT 2.1 (AC, ~375-393)"),
        Reported("ingestion.incremental_append_byte_identical_fraction", 1.0, 0.0, source="EVALUATION_REPORT 2.2"),
    ]


def _corpus_and_suite() -> list[Reported]:
    return [
        Reported("corpus.embedded_tiles_frozen_prefix", 101911, source="EVALUATION_REPORT 1.1 (faiss_id < 101911)"),
        Reported("corpus.faiss_vectors", 101911, source="EVALUATION_REPORT 1.1 (FAISS vectors)",
                 note="the index has grown since the report (Phase 8 appended Ayodhya 2025/2026 + Maxar)"),
        Reported("corpus.sentinel2_embedded_tiles", 101911, source="EVALUATION_REPORT 1.1 (Sentinel-2 tiles, embedded)"),
        Reported("corpus.catalog_tiles_total", 104990, source="EVALUATION_REPORT 1.1"),
        Reported("corpus.scenes_total", 75, source="EVALUATION_REPORT 1.1"),
        Reported("corpus.sar_tiles_not_embedded", 3079, source="EVALUATION_REPORT 1.1"),
        Reported("suite.passed", 228, source="EVALUATION_REPORT 13 / project rules file, rule 4"),
        Reported("suite.failed", 0, source="project rules file, rule 4 ('never commit red')"),
        Reported("suite.skipped", 3, source="EVALUATION_REPORT 13"),
    ]


REPORTED: list[Reported] = (_retrieval() + _oscd() + _detector() + _latency() + _footprint_and_ingest()
                            + _corpus_and_suite())


def reproduction_report(snapshot: Mapping, reasons: Mapping[str, str] | None = None) -> dict:
    """Compare a snapshot's measurements to the reported numbers.

    Returns ``{"summary", "entries", "not_reproduced"}``. Every entry carries reported / measured /
    tolerance / status. ``not_reproduced`` is the list in the required
    ``{"status": "NOT_REPRODUCED", "reported", "reason", ...}`` shape. A reported number whose section
    was not run in this snapshot is ``NOT_RUN`` (neither claim is made).
    """
    flat, placeholders = flatten_metrics(snapshot)
    reasons = reasons or {}
    entries, bad = [], []
    for r in REPORTED:
        measured = flat.get(r.path)
        base = {"metric": r.path, "reported": r.value, "source": r.source}
        if measured is None:
            if r.path in placeholders:
                status, why = "NOT_REPRODUCED", "measurement could not be produced in this session"
            else:
                entries.append({**base, "measured": None, "status": "NOT_RUN"})
                continue
        elif r.within(measured):
            entries.append({**base, "measured": measured, "abs_diff": abs(measured - r.value),
                            "tolerance": r.tol_abs + r.tol_rel * abs(r.value), "status": "REPRODUCED",
                            **({"note": r.note} if r.note else {})})
            continue
        else:
            status = "NOT_REPRODUCED"
            why = reasons.get(r.path) or (f"measured {measured:.6g} differs from reported {r.value:.6g} by "
                                          f"{abs(measured - r.value):.3g} (tolerance {r.tol_abs + r.tol_rel * abs(r.value):.3g})")
        entry = {**base, "measured": measured, "status": status, "reason": why,
                 **({"abs_diff": abs(measured - r.value), "tolerance": r.tol_abs + r.tol_rel * abs(r.value)} if measured is not None else {})}
        entries.append(entry)
        bad.append({"status": "NOT_REPRODUCED", "metric": r.path, "reported": r.value, "measured": measured,
                    "reason": why, "source": r.source})
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"summary": counts, "entries": entries, "not_reproduced": bad}

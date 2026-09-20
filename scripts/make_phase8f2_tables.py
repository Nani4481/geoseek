"""Generate the Markdown results tables for docs/PHASE8F2.md straight from the run's JSON artifacts, so no number in the report
is hand-transcribed. Missing artifacts / keys simply omit their section.

Inputs (all under data/):
  runs/detector/<run>/progress.jsonl, results.csv      training
  detect_eval/eval_results.json                         official-val + monitor evaluation (scripts/eval_detector.py)
  detect_eval/evaluator_selftest.json                   oracle / jitter / degraded self-test
  detect_eval/domain_shift_proxy.json                   blur + test-time-upscale proxy (monitor split)
  detections/maxar_inference_report.json                Maxar runs
  detect_preflight/epoch0_monitor_baseline.json         pretrained (restricted) monitor baseline

    python scripts/make_phase8f2_tables.py > data/detect_eval/phase8f2_tables.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoseek.config import get_settings  # noqa: E402
from geoseek.detect.classes import CLASS_GROUPS, KEPT_CLASSES  # noqa: E402

# Oriented R-CNN R-50-FPN, DOTA-v1.0 TEST, single scale, VOC07 AP50 (arXiv 2108.05699 Table; header order PL BD BR GTF SV LV SH TC BC ST SBF RA HA SP HC)
PUBLISHED_ORIENTED_RCNN = {"plane": 89.46, "ship": 88.20, "storage-tank": 84.68, "harbor": 74.94, "bridge": 54.78,
                           "large-vehicle": 83.00, "small-vehicle": 78.93, "helicopter": 52.28}
GROUP_TITLES = {"ground_vehicles": "ground vehicles (small + large)", "ships_and_aircraft": "ships + aircraft",
                "infrastructure": "infrastructure (tanks, harbors, bridges)", "all_kept_classes": "all 8 classes"}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def f(x, d=3, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100 * x:.{max(d - 2, 0)}f}" if pct else f"{x:.{d}f}"


def ci(c, d=3):
    return "" if not c or c[0] is None else f" [{c[0]:.{d}f}–{c[1]:.{d}f}]"


def table(head: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    return "\n".join(["| " + " | ".join(head) + " |", align] + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def training_tables(run_dir: Path) -> str:
    out = []
    prog = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (run_dir / "progress.jsonl").is_file() else []
    res = {}
    if (run_dir / "results.csv").is_file():
        with open(run_dir / "results.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        head = [h.strip() for h in rows[0]]
        for r in rows[1:]:
            res[int(float(r[0]))] = dict(zip(head, r))
    base = load(get_settings().data_dir / "detect_preflight" / "epoch0_monitor_baseline.json")
    body = []
    if base:
        b = base["nms_one2many"]
        body.append(["0 (pretrained, no fine-tuning)", "—", "—", "—", "—", f(b["P"]), f(b["R"]), f(b["mAP50"], 4), f(b["mAP50-95"], 4), "—"])
    best = max(prog, key=lambda r: 0.1 * r["metrics"]["metrics/mAP50(B)"] + 0.9 * r["metrics"]["metrics/mAP50-95(B)"], default=None)
    for r in prog:
        if r["epoch"] > r["epochs"]:
            continue
        m, e = r["metrics"], res.get(r["epoch"], {})
        tag = " **←best (monitor fitness)**" if best is r else ""
        body.append([f"{r['epoch']}{tag}", f"{r['epoch_seconds'] / 60:.1f}", f(float(e["train/box_loss"])) if e else "—", f(float(e["train/cls_loss"])) if e else "—",
                     f(float(e["train/angle_loss"])) if e and "train/angle_loss" in e else "—", f(m["metrics/precision(B)"]), f(m["metrics/recall(B)"]),
                     f(m["metrics/mAP50(B)"], 4), f(m["metrics/mAP50-95(B)"], 4), f(m["val/cls_loss"])])
    out.append(table(["epoch", "min", "train box", "train cls", "train angle", "monitor P", "monitor R", "monitor mAP50", "monitor mAP50-95", "monitor val cls loss"], body))
    if prog:
        secs = [r["epoch_seconds"] for r in prog]
        out.append(f"\nTotal wall clock (train + per-epoch monitor validation, includes any suspension): **{sum(secs) / 3600:.2f} h** over {len(prog)} epoch records; "
                   f"peak torch VRAM {max(r['peak_vram_reserved_gb'] for r in prog):.2f} GB; epochs run on battery: "
                   f"{[r['epoch'] for r in prog if r.get('ac_power') is False] or 'none'}.\n")
    return "\n".join(out)


def val_tables(ev: dict) -> str:
    out = []
    ft, base = ev.get("val_full_image_v15"), ev.get("val_full_image_v15_pretrained_restricted")
    bs = ev.get("val_bootstrap_v15", {})
    if ft:
        rows = []
        for g, names in CLASS_GROUPS.items():
            for n in names:
                a = ft["per_class"][n]
                b = base["per_class"][n] if base else None
                c = bs.get("per_class", {}).get(n, {})
                rows.append([f"**{n}**" if g == "ground_vehicles" else n, str(a["n_gt"]), f(a["AP50"]) + ci(c.get("AP50_ci")), f(a["AP50_voc07"]),
                             f(a["AP50_95"]) + ci(c.get("AP50_95_ci")), f(b["AP50"]) if b else "—", f(b["AP50_95"]) if b else "—",
                             f(a["AP50"] - b["AP50"], 3) if b else "—"])
        out.append("**Per class — official val, DOTA v1.5 GT, full-image protocol** (difficult ignored; 95 % bootstrap CI over val images in brackets):\n")
        out.append(table(["class", "n GT", "AP50 (fine-tuned)", "AP50 VOC07", "AP50:95 (fine-tuned)", "AP50 pretrained", "AP50:95 pretrained", "ΔAP50"], rows))
        rows = []
        for g in list(CLASS_GROUPS) + ["all_kept_classes"]:
            a = ft["groups"].get(g, {})
            b = base["groups"].get(g, {}) if base else {}
            c = bs.get("groups", {}).get(g, {})
            rows.append([GROUP_TITLES[g], str(a.get("n_gt", "—")), f(a.get("macro_AP50")) + ci(c.get("macro_AP50_ci")), f(a.get("weighted_AP50")),
                         f(a.get("macro_AP50_95")) + ci(c.get("macro_AP50_95_ci")), f(b.get("macro_AP50")), f(b.get("macro_AP50_95"))])
        out.append("\n**By reporting group** (macro = mean over the group's classes; weighted = by instance count):\n")
        out.append(table(["group", "n GT", "macro AP50", "weighted AP50", "macro AP50:95", "macro AP50 pretrained", "macro AP50:95 pretrained"], rows))
        rows = []
        for n in CLASS_GROUPS["ground_vehicles"]:
            for k, v in ft["per_class"][n].get("by_size_px", {}).items():
                bb = base["per_class"][n].get("by_size_px", {}).get(k, {}) if base else {}
                rows.append([n, k, str(v["n_gt"]), f(v["AP50"]), f(v["AP50_95"]), f(bb.get("AP50")), f(bb.get("AP50_95"))])
        if rows:
            out.append("\n**Vehicles by object size** (long side in px; a real car at the Maxar 0.305 m grid is ~15 px; GT outside the bucket is ignored):\n")
            out.append(table(["class", "long side px", "n GT", "AP50", "AP50:95", "AP50 pretrained", "AP50:95 pretrained"], rows))
    last = ev.get("val_full_image_v15_last_epoch")
    if ft and last:
        rows = []
        for n in KEPT_CLASSES:
            rows.append([n, f(ft["per_class"][n]["AP50"]), f(last["per_class"][n]["AP50"]), f(ft["per_class"][n]["AP50_95"]), f(last["per_class"][n]["AP50_95"])])
        for g in list(CLASS_GROUPS) + ["all_kept_classes"]:
            rows.append([f"**{GROUP_TITLES[g]}** (macro)", f(ft["groups"][g]["macro_AP50"]), f(last["groups"][g]["macro_AP50"]),
                         f(ft["groups"][g]["macro_AP50_95"]), f(last["groups"][g]["macro_AP50_95"])])
        out.append("**Selected checkpoint (epoch %s, chosen on the monitor split) vs the final epoch (%s) - official val, v1.5 GT, "
                   "full-image (reporting only):**\n" % (ev.get("best_checkpoint_epoch_by_monitor_fitness", "?"), ev.get("last_epoch", "?")))
        out.append(table(["class / group", "AP50 selected", "AP50 final epoch", "AP50:95 selected", "AP50:95 final epoch"], rows))
    v10, b10 = ev.get("val_full_image_v10"), ev.get("val_full_image_v10_pretrained_restricted")
    if v10:
        rows = []
        for n in KEPT_CLASSES:
            a = v10["per_class"][n]
            rows.append([n, str(a["n_gt"]), f(a["AP50_voc07"]), f(a["AP50"]), f(a["AP50_95"]), f(b10["per_class"][n]["AP50_voc07"]) if b10 else "—",
                         f"{PUBLISHED_ORIENTED_RCNN[n] / 100:.3f}"])
        out.append("\n**Against DOTA v1.0 GT** (the label set every published number uses; v1.0 leaves most real vehicles unlabeled, so vehicle precision — and hence AP — is depressed by the labels, not the model):\n")
        out.append(table(["class", "n GT (v1.0)", "AP50 VOC07 (fine-tuned)", "AP50 all-point", "AP50:95", "AP50 VOC07 pretrained", "published Oriented R-CNN R50 (v1.0 *test*)"], rows))
    for tag, key in (("v1.5", "val_operating_point_v15"), ("v1.0", "val_operating_point_v10")):
        op = ev.get(key)
        if op:
            rows = []
            for n in KEPT_CLASSES:
                r = op["per_class"][n]
                rows.append([n, f(r["precision"]), f(r["recall"]), f(r["f1"]), str(r["TP"]), str(r["FP"]), str(r["FN"])])
            for g in list(CLASS_GROUPS) + ["all_kept_classes"]:
                r = op["groups"][g]
                rows.append([f"**{GROUP_TITLES[g]}**", f(r["precision"]), f(r["recall"]), f(r["f1"]), str(r["TP"]), str(r["FP"]), str(r["FN"])])
            out.append(f"\n**Operating point on official val, {tag} GT** — confidence {op['conf']} (chosen on the MONITOR split, IoU 0.5, difficult ignored):\n")
            out.append(table(["class / group", "precision", "recall", "F1", "TP", "FP", "FN"], rows))
    chip = ev.get("val_chip_level_ultralytics")
    if chip:
        rows = []
        for n in KEPT_CLASSES:
            rows.append([n, f(chip["finetuned"]["per_class_AP50"][n]), f(chip["finetuned"]["per_class_AP50_95"][n]),
                         f(chip["pretrained_restricted"]["per_class_AP50"][n]), f(chip["pretrained_restricted"]["per_class_AP50_95"][n])])
        rows.append(["**all (mean)**", f(chip["finetuned"]["mAP50"]), f(chip["finetuned"]["mAP50_95"]), f(chip["pretrained_restricted"]["mAP50"]), f(chip["pretrained_restricted"]["mAP50_95"])])
        out.append("\n**Chip-level protocol (Ultralytics' validator; every instance incl. `difficult` counts; per-chip; ProbIoU; 101-point AP)** — comparable with the checkpoints' self-reported numbers:\n")
        out.append(table(["class", "AP50 fine-tuned", "AP50:95 fine-tuned", "AP50 pretrained", "AP50:95 pretrained"], rows))
    cc = ev.get("evaluator_crosscheck_vs_ultralytics")
    if cc:
        att = cc.get("attribution")
        if att:
            rows = [[n, f(r["ultralytics_AP50"]), f(r["rescored_AP50"]), f(r["ours_probiou_AP50"]), f(r["ours_AP50"]),
                     f(r["ultralytics_AP50_95"]), f(r["rescored_AP50_95"]), f(r["ours_probiou_AP50_95"]), f(r["ours_AP50_95"])] for n, r in cc["per_class"].items()]
            out.append("\n**Evaluator cross-check — where do the two evaluators differ?** Same fine-tuned detector, same val chips, chip-level protocol "
                       "(chip rectangles as GT, `difficult` counted, no cross-chip merge). Columns: the Ultralytics validator; Ultralytics' own overlap/matching/AP code run on "
                       "*our* saved predictions (plumbing check); our evaluator with only the overlap measure swapped to ProbIoU; our evaluator as used for the headline numbers.\n")
            out.append(table(["class", "validator AP50", "UL code on our preds", "ours + ProbIoU", "ours (polygon IoU)",
                              "validator AP50:95", "UL code on our preds", "ours + ProbIoU", "ours (polygon IoU)"], rows))
            arows = [[label, f"{a['mean_abs_diff_AP50']:.3f} / {a['max_abs_diff_AP50']:.3f}", f"{a['mean_abs_diff_AP50_95']:.3f} / {a['max_abs_diff_AP50_95']:.3f}"]
                     for label, a in (("ours (polygon IoU)", att["ours_polygon_iou_vs_validator"]), ("ours + ProbIoU only", att["ours_probiou_vs_validator"]),
                                      ("Ultralytics code on our predictions", att["ultralytics_code_on_our_predictions_vs_validator"]))]
            out.append("\nGap to the validator (mean / max over the 8 classes):\n")
            out.append(table(["evaluator", "|ΔAP50|", "|ΔAP50:95|"], arows))
            out.append(f"\n`probiou_polys` vs `ultralytics.utils.metrics.batch_probiou` on 300 random overlapping rectangles: max |diff| "
                       f"{ev.get('probiou_implementation_max_abs_error_vs_ultralytics', float('nan')):.1e}.")
        else:
            rows = [[n, f(r["ours_AP50"]), f(r["ultralytics_AP50"]), f(r["ours_AP50_95"]), f(r["ultralytics_AP50_95"])] for n, r in cc["per_class"].items()]
            out.append(f"\n**Evaluator cross-check** (our evaluator under the chip-level protocol vs Ultralytics on identical predictions): max |ΔAP50| {cc['max_abs_diff_AP50']:.3f}, "
                       f"mean {cc['mean_abs_diff_AP50']:.3f}; max |ΔAP50:95| {cc['max_abs_diff_AP50_95']:.3f}.\n")
            out.append(table(["class", "ours AP50", "Ultralytics AP50", "ours AP50:95", "Ultralytics AP50:95"], rows))
    pts = ev.get("official_val_by_checkpoint")
    if pts:
        out.append("\n**Official val by checkpoint (post-hoc, chip level, reporting only — never used to choose anything):**\n")
        out.append(table(["after epoch", "checkpoint", "mAP50", "mAP50:95"], [[str(p["epoch"]), p.get("label", ""), f(p["mAP50"]), f(p["mAP50_95"])] for p in pts],
                         "|---:|---|---:|---:|"))
    q = ev.get("qualitative")
    if q:
        out.append("\n**Qualitative examples** (chips chosen by a fixed seed from the chip index, before any prediction was looked at; left = ground truth, right = predictions):\n")
        out.append(table(["#", "stratum", "chip", "GT", "pred", "TP", "FP", "FN", "file"],
                         [[str(i + 1), r["stratum"], r["chip_id"], str(r["n_gt"]), str(r["n_pred"]), str(r["TP"]), str(r["FP"]), str(r["FN"]), f"`{r['path']}`"] for i, r in enumerate(q)]))
    return "\n".join(out)


def selftest_table(st: dict) -> str:
    rows = []
    for ls, block in st["label_sets"].items():
        for name in ("oracle", "jitter", "degraded"):
            b = block[name]
            rows.append([ls, name, f(b["macro_AP50"], 4), f(b["macro_AP50_95"], 4), f(b["operating_point_conf0.5"]["precision"]), f(b["operating_point_conf0.5"]["recall"]),
                         f(block[name]["per_class"]["small-vehicle"]["AP50_95"])])
    return table(["GT label set", "synthetic detector", "macro AP50", "macro AP50:95", "precision", "recall", "small-vehicle AP50:95"], rows)


def proxy_table(px: dict) -> str:
    rows = []
    for name, r in px["runs"].items():
        g = r["operating_point"]["ground_vehicles"]
        rows.append([name, f(r["macro_AP50"]), f(r["macro_AP50_95"]), f(r["ground_vehicles_macro_AP50"]), f(r["per_class_AP50"]["small-vehicle"]), f(g["precision"]), f(g["recall"])])
    return table(["run (monitor chips)", "macro AP50", "macro AP50:95", "vehicles AP50", "small-vehicle AP50", "vehicle precision", "vehicle recall"], rows)


def maxar_tables(rep: dict) -> str:
    rows = []
    for oid, o in rep["observations"].items():
        by = ", ".join(f"{k} {v}" for k, v in list(o["by_class"].items())[:5]) or "—"
        rows.append([o.get("aoi", oid).replace("maxar_", ""), (o.get("role") or "")[:40], str(o["n_tiles"]), str(o["n_detections"]),
                     f(o["detections_per_tile"]["mean"], 2), str(o["detections_per_tile"]["max"]), by, f"{o['tile_seconds_mean']:.2f}"])
    return table(["observation", "role", "tiles", "detections", "per tile (mean)", "per tile (max)", "by class (top)", "s/tile"], rows,
                 "|---|---|---:|---:|---:|---:|---|---:|")


def main() -> None:
    s = get_settings()
    d = s.data_dir
    run = d / "runs" / "detector" / "geoseek_obb_v15_yolo26s"
    print("<!-- generated by scripts/make_phase8f2_tables.py; do not edit by hand -->\n")
    if run.is_dir():
        print("### T1 — Training per epoch\n")
        print(training_tables(run))
    ev = load(d / "detect_eval" / "eval_results.json")
    if ev:
        print("\n### T2 — Official-val evaluation\n")
        print(val_tables(ev))
        if ev.get("operating_point"):
            op = ev["operating_point"]
            print(f"\nOperating confidence chosen on the monitor split: **{op['conf']}** (macro-F1 {op['macro_f1_on_monitor']:.3f}); per-class optima: "
                  + ", ".join(f"{n} {v['conf']:.2f}" for n, v in op["per_class_best"].items()) + "\n")
        print("\n### T3 — Integrity checks\n")
        print("```json\n" + json.dumps(ev.get("integrity", {}), indent=1) + "\n```\n")
    st = load(d / "detect_eval" / "evaluator_selftest.json")
    if st:
        print("\n### T4 — Evaluator self-test on real val geometry\n")
        print(selftest_table(st))
    px = load(d / "detect_eval" / "domain_shift_proxy.json")
    if px:
        print(f"\n### T5 — Domain-shift proxy (k = {px['k']}, conf {px['conf']}, {px['split']})\n")
        print(proxy_table(px))
    mx = load(d / "detections" / "maxar_inference_report.json")
    if mx:
        print("\n### T6 — Maxar inference\n")
        print(maxar_tables(mx))


if __name__ == "__main__":
    main()

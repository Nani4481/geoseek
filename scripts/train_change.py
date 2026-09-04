"""Phase 3b: TRAIN FC-Siam-diff on the OSCD train split, then honestly evaluate it.

    python scripts/train_change.py                 # train (11 regions) + eval (10 held-out test regions)
    python scripts/train_change.py --eval-only     # re-evaluate the saved checkpoint

Offline: reads only ``data/datasets/oscd/`` (staged by
``python -m geoseek.staging.download_oscd``). Nothing here touches the network.

Leakage rules enforced in code
------------------------------
* Gradient updates use ONLY 11 of the 14 OSCD *train* regions. 3 train regions
  are held out for validation (loss curve + threshold selection).
* The 10 OSCD *test* regions are NEVER read until :func:`evaluate`, and the
  precision-favouring operating point is chosen on the VALIDATION regions, then
  frozen, then applied once to the test regions. ``assert`` guards make a
  train/val/test region overlap a hard error.

Outputs (under ``data/change_model/``): ``fc_siam_diff.pt`` (weights + full model
card), ``norm_stats.json``, ``loss_curve.png``, ``pr_curve_val.png``,
``pr_curve_test.png``, ``qual_<region>.png`` x3, and manifest sections
``oscd_change_model`` (model card) + ``oscd_change_model_eval`` (metrics).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

# OSCD imgs_*_rect tiles are plain rasters (no CRS/transform) - expected, not a problem.
warnings.filterwarnings("ignore", message=".*no geotransform.*")
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from geoseek.change.models.fc_siam_diff import DEFAULT_BANDS, FCSiamDiff, count_parameters
from geoseek.config import get_settings
from geoseek.staging.download_oscd import (
    PRODUCTION_BANDS,
    REFLECTANCE_SCALE,
    change_mask_path,
    enumerate_split,
    oscd_root,
)
from geoseek.staging.manifest import record_analysis_section, record_artifact

_IMAGES_ROOT = "Onera Satellite Change Detection dataset - Images"

# 3 of the 14 OSCD train regions held out for validation (loss curve + threshold
# tuning). Spread across continents and change densities; fixed + recorded.
DEFAULT_VAL_REGIONS = ("bordeaux", "cupertino", "beirut")

# 3 held-out TEST regions for the qualitative [before|after|GT|pred] panels.
QUAL_REGIONS = ("lasvegas", "montpellier", "chongqing")

OUT_DIR = get_settings().data_dir / "change_model"


# ==========================================================================
# data
# ==========================================================================


def _region_dir(region: str) -> Path:
    return oscd_root() / _IMAGES_ROOT / region


def load_region(region: str, bands=PRODUCTION_BANDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (img1, img2, change) - img* are (C, H, W) reflectance float32, change is (H, W) {0,1} float32."""
    rd = _region_dir(region)

    def _stack(date_idx: int) -> np.ndarray:
        chans = []
        for b in bands:
            with rasterio.open(rd / f"imgs_{date_idx}_rect" / f"{b}.tif") as ds:
                chans.append(ds.read(1).astype(np.float32))
        return np.stack(chans, axis=0) / REFLECTANCE_SCALE

    img1, img2 = _stack(1), _stack(2)
    with rasterio.open(change_mask_path(region)) as ds:
        cm = ds.read(1)
    change = (cm == 2).astype(np.float32)  # OSCD: 1 = no change, 2 = change
    if change.shape != img1.shape[1:]:
        h = min(change.shape[0], img1.shape[1])
        w = min(change.shape[1], img1.shape[2])
        img1, img2, change = img1[:, :h, :w], img2[:, :h, :w], change[:h, :w]
    return img1, img2, change


def compute_norm_stats(regions: list[str], bands=PRODUCTION_BANDS) -> dict:
    """Per-band mean/std over the given regions (both dates, pixels valid in every band)."""
    n = np.zeros(len(bands))
    s = np.zeros(len(bands))
    ss = np.zeros(len(bands))
    for region in regions:
        img1, img2, _ = load_region(region, bands)
        for img in (img1, img2):
            valid = np.all(img > 0, axis=0)
            v = img[:, valid]  # (C, npix)
            n += v.shape[1]
            s += v.sum(axis=1)
            ss += (v ** 2).sum(axis=1)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return {"bands": list(bands), "mean": mean.tolist(), "std": std.tolist(),
            "space": f"reflectance (DN / {REFLECTANCE_SCALE:.0f})", "n_regions": len(regions)}


def _standardize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (img - mean[:, None, None]) / std[:, None, None]


class OSCDPatchSampler:
    """On-the-fly random 96x96 patch sampler over the training regions.

    Class imbalance is ~3% change pixels, some regions < 1%. ``change_oversample``
    of the patches are forced to contain at least one change pixel (rejection
    sampling); the rest are uniform. Augmentation is the dihedral group D4
    (random h/v flip + k*90 rotation), applied identically to both dates and the
    mask - geometric only, so no spurious radiometric change is taught.
    """

    def __init__(self, regions: list[str], mean, std, *, patch: int = 96,
                 change_oversample: float = 0.5, seed: int = 0):
        self.patch = patch
        self.change_oversample = change_oversample
        self.rng = np.random.default_rng(seed)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.data = []
        for region in regions:
            img1, img2, change = load_region(region)
            img1 = _standardize(img1, self.mean, self.std).astype(np.float32)
            img2 = _standardize(img2, self.mean, self.std).astype(np.float32)
            ys, xs = np.where(change > 0)
            self.data.append({"region": region, "img1": img1, "img2": img2, "change": change,
                              "hw": change.shape, "change_yx": (ys, xs)})
        areas = np.array([d["hw"][0] * d["hw"][1] for d in self.data], np.float64)
        self.region_p = areas / areas.sum()

    def _rand_topleft(self, d, force_change: bool) -> tuple[int, int]:
        h, w = d["hw"]
        p = self.patch
        if force_change and len(d["change_yx"][0]) > 0:
            i = self.rng.integers(len(d["change_yx"][0]))
            cy, cx = int(d["change_yx"][0][i]), int(d["change_yx"][1][i])
            y0 = int(np.clip(cy - self.rng.integers(0, p), 0, max(h - p, 0)))
            x0 = int(np.clip(cx - self.rng.integers(0, p), 0, max(w - p, 0)))
            return y0, x0
        return int(self.rng.integers(0, max(h - p, 0) + 1)), int(self.rng.integers(0, max(w - p, 0) + 1))

    def sample_batch(self, batch_size: int):
        p = self.patch
        x1b, x2b, yb = [], [], []
        for _ in range(batch_size):
            di = int(self.rng.choice(len(self.data), p=self.region_p))
            d = self.data[di]
            force = self.rng.random() < self.change_oversample
            for _try in range(8):
                y0, x0 = self._rand_topleft(d, force)
                sl = (slice(y0, y0 + p), slice(x0, x0 + p))
                a1 = d["img1"][:, sl[0], sl[1]]
                if a1.shape[1:] == (p, p):
                    break
            a2 = d["img2"][:, sl[0], sl[1]]
            m = d["change"][sl]
            # dihedral D4 augmentation (identical transform for both dates + mask)
            k = int(self.rng.integers(4))
            a1, a2, m = np.rot90(a1, k, (1, 2)), np.rot90(a2, k, (1, 2)), np.rot90(m, k)
            if self.rng.random() < 0.5:
                a1, a2, m = a1[:, ::-1], a2[:, ::-1], m[::-1]
            if self.rng.random() < 0.5:
                a1, a2, m = a1[:, :, ::-1], a2[:, :, ::-1], m[:, ::-1]
            x1b.append(np.ascontiguousarray(a1))
            x2b.append(np.ascontiguousarray(a2))
            yb.append(np.ascontiguousarray(m))
        return (torch.from_numpy(np.stack(x1b)), torch.from_numpy(np.stack(x2b)),
                torch.from_numpy(np.stack(yb))[:, None, :, :])


# ==========================================================================
# loss
# ==========================================================================


def combo_loss(logits, target, pos_weight, bce_w=0.5, dice_w=1.0, eps=1.0):
    """Weighted BCE + soft Dice.

    Dice directly optimises region overlap (an F1 surrogate) and is inherently
    robust to a 30:1 class imbalance; the weighted-BCE term keeps per-pixel
    gradients well-conditioned and sharpens boundaries. ``pos_weight`` is capped
    (see --pos-weight-cap) so BCE cannot collapse to predicting all-change.
    Focal loss is a reasonable alternative; this pairing was chosen for training
    stability and direct alignment with the F1 / IoU we report.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum() + eps
    den = p.sum() + target.sum() + eps
    dice = 1.0 - num / den
    return bce_w * bce + dice_w * dice, bce.detach(), dice.detach()


# ==========================================================================
# inference / metrics
# ==========================================================================


@torch.no_grad()
def infer_region(model, region, mean, std, device, *, want_logits: bool = False):
    """Full-image change probability + ground-truth for one region (optionally the raw logits too)."""
    img1, img2, change = load_region(region)
    x1 = torch.from_numpy(_standardize(img1, mean, std).astype(np.float32))[None].to(device)
    x2 = torch.from_numpy(_standardize(img2, mean, std).astype(np.float32))[None].to(device)
    with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        logits = model(x1, x2).float()
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if want_logits:
        return prob, change, logits[0, 0].cpu().numpy()
    return prob, change


def _counts_at(prob: np.ndarray, gt: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = prob >= thr
    g = gt > 0.5
    tp = int(np.sum(pred & g)); fp = int(np.sum(pred & ~g))
    fn = int(np.sum(~pred & g)); tn = int(np.sum(~pred & ~g))
    return tp, fp, fn, tn


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pr_curve(probs: list[np.ndarray], gts: list[np.ndarray], thresholds: np.ndarray) -> list[dict]:
    flat_p = np.concatenate([p.ravel() for p in probs])
    flat_g = np.concatenate([(g > 0.5).ravel() for g in gts])
    pos = int(flat_g.sum())
    out = []
    for t in thresholds:
        pred = flat_p >= t
        tp = int(np.sum(pred & flat_g)); fp = int(np.sum(pred & ~flat_g))
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / pos if pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fbeta = _fbeta(prec, rec, 0.5)
        out.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, "f0.5": fbeta})
    return out


def _fbeta(prec: float, rec: float, beta: float) -> float:
    b2 = beta * beta
    d = b2 * prec + rec
    return (1 + b2) * prec * rec / d if d else 0.0


# ==========================================================================
# plotting (hand-rolled PIL - matplotlib is deliberately not a geoseek dep)
# ==========================================================================


def _font(size: int = 13):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _xy_plot(series: list[dict], out: Path, *, xlabel: str, ylabel: str, title: str,
             xlim=None, ylim=None, w=760, h=520):
    m = {"l": 70, "r": 170, "t": 50, "b": 55}
    pw, ph = w - m["l"] - m["r"], h - m["t"] - m["b"]
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f = _font(13); fb = _font(15)
    allx = np.concatenate([np.asarray(s["x"], float) for s in series]) if series else np.array([0, 1])
    ally = np.concatenate([np.asarray(s["y"], float) for s in series]) if series else np.array([0, 1])
    x0, x1 = xlim or (float(allx.min()), float(allx.max()))
    y0, y1 = ylim or (float(ally.min()), float(ally.max()))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1

    def px(x): return m["l"] + (x - x0) / (x1 - x0) * pw
    def py(y): return m["t"] + (1 - (y - y0) / (y1 - y0)) * ph

    for i in range(6):
        gy = m["t"] + ph * i / 5
        d.line([(m["l"], gy), (m["l"] + pw, gy)], fill=(232, 232, 232))
        d.text((m["l"] - 46, gy - 7), f"{y1 - (y1 - y0) * i / 5:.3f}", font=f, fill=(90, 90, 90))
        gx = m["l"] + pw * i / 5
        d.line([(gx, m["t"]), (gx, m["t"] + ph)], fill=(232, 232, 232))
        d.text((gx - 14, m["t"] + ph + 8), f"{x0 + (x1 - x0) * i / 5:.2f}", font=f, fill=(90, 90, 90))
    d.rectangle([m["l"], m["t"], m["l"] + pw, m["t"] + ph], outline=(60, 60, 60))
    d.text((w / 2 - len(title) * 4, 16), title, font=fb, fill=(20, 20, 20))
    d.text((m["l"] + pw / 2 - len(xlabel) * 3, h - 22), xlabel, font=f, fill=(20, 20, 20))
    d.text((10, m["t"] + ph / 2 - 30), ylabel, font=f, fill=(20, 20, 20))

    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    for i, s in enumerate(series):
        col = colors[i % len(colors)]
        pts = [(px(x), py(y)) for x, y in zip(s["x"], s["y"])]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
        else:
            d.ellipse([pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3], fill=col)
        d.line([(m["l"] + pw + 14, m["t"] + 18 + i * 20), (m["l"] + pw + 34, m["t"] + 18 + i * 20)],
               fill=col, width=3)
        d.text((m["l"] + pw + 38, m["t"] + 10 + i * 20), s["label"], font=f, fill=(20, 20, 20))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def _stretch_rgb(img_chw: np.ndarray, bands=PRODUCTION_BANDS) -> np.ndarray:
    """B04/B03/B02 -> uint8 RGB for display (per-image 2-98 pct; visualization only, not model input)."""
    idx = {b: i for i, b in enumerate(bands)}
    rgb = np.stack([img_chw[idx["B04"]], img_chw[idx["B03"]], img_chw[idx["B02"]]], axis=-1)
    lo, hi = np.percentile(rgb[np.all(img_chw > 0, axis=0)], (2, 98)) if np.any(img_chw > 0) else (0, 1)
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def qualitative_panel(region: str, prob: np.ndarray, gt: np.ndarray, thr: float, out: Path):
    img1, img2, _ = load_region(region)
    before, after = _stretch_rgb(img1), _stretch_rgb(img2)
    h, w = gt.shape
    gt_img = np.stack([(gt > 0.5).astype(np.uint8) * 255] * 3, axis=-1)
    pred = prob >= thr
    g = gt > 0.5
    err = np.zeros((h, w, 3), np.uint8)
    err[pred & g] = (60, 200, 60)      # TP green
    err[pred & ~g] = (220, 50, 50)     # FP red
    err[~pred & g] = (60, 120, 230)    # FN blue
    pad = 8
    top, bot = 26, 22
    labels = ["before (date 1)", "after (date 2)", "ground truth", f"prediction @ {thr:.2f}  (G=TP R=FP B=FN)"]
    panels = [before, after, gt_img, err]
    W = w * 4 + pad * 5
    H = h + top + bot
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f = _font(13)
    for i, (pn, lb) in enumerate(zip(panels, labels)):
        x = pad + i * (w + pad)
        canvas.paste(Image.fromarray(pn), (x, top))
        d.text((x, 7), lb, font=f, fill=(15, 15, 15))
    d.text((pad, H - bot + 4), f"OSCD test region: {region}   ({w}x{h} px)   "
           f"change pixels: {100 * (gt > 0.5).mean():.2f}%", font=f, fill=(90, 90, 90))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ==========================================================================
# train
# ==========================================================================


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _source_hashes() -> dict:
    root = get_settings().project_root
    files = ["src/geoseek/change/models/fc_siam_diff.py",
             "src/geoseek/change/models/fc_siam_diff_model.py",
             "src/geoseek/staging/download_oscd.py",
             "scripts/train_change.py"]
    return {f: _sha256(root / f) for f in files if (root / f).is_file()}


def train(args) -> dict:
    settings = get_settings()
    device = settings.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = enumerate_split()
    if split["consistency_problems"]:
        raise RuntimeError(f"OSCD split problems: {split['consistency_problems']}")
    all_train = split["train_regions"]
    test_regions = split["test_regions"]
    val_regions = list(args.val_regions)
    fit_regions = [r for r in all_train if r not in val_regions]

    # ---- leakage guards -------------------------------------------------
    assert set(val_regions).issubset(all_train), f"val regions not in OSCD train split: {val_regions}"
    assert not (set(fit_regions) & set(val_regions)), "fit/val overlap"
    assert not (set(fit_regions + val_regions) & set(test_regions)), "TRAIN/VAL leaked into TEST split"
    assert len(fit_regions) == 11 and len(val_regions) == 3, (len(fit_regions), len(val_regions))

    print("=" * 78)
    print("geoseek Phase 3b - training FC-Siam-diff on OSCD")
    print("=" * 78)
    print(f"  device            : {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"  OSCD train split  : {len(all_train)} regions")
    print(f"  -> fit (gradient) : {fit_regions}")
    print(f"  -> validation     : {val_regions}   (loss curve + threshold selection)")
    print(f"  HELD-OUT test     : {test_regions}   (never seen until evaluation)")

    per_region_change = {r: round(float((rasterio.open(change_mask_path(r)).read(1) == 2).mean()), 4)
                         for r in all_train}
    print(f"  change fraction per train region: {per_region_change}")

    norm = compute_norm_stats(fit_regions)
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "norm_stats.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    print(f"  per-band standardization (reflectance): mean={np.round(mean,4).tolist()} "
          f"std={np.round(std,4).tolist()}")

    sampler = OSCDPatchSampler(fit_regions, mean, std, patch=args.patch,
                               change_oversample=args.change_oversample, seed=args.seed)

    fit_change_frac = float(np.mean([per_region_change[r] for r in fit_regions]))
    pos_weight_val = min(max((1 - fit_change_frac) / max(fit_change_frac, 1e-6), 1.0), args.pos_weight_cap)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"  fit-set change fraction ~{fit_change_frac:.4f}  -> raw neg/pos "
          f"{(1-fit_change_frac)/fit_change_frac:.1f}, BCE pos_weight capped at {pos_weight_val:.1f}")

    model = FCSiamDiff(in_channels=len(PRODUCTION_BANDS), base_channels=args.base_channels,
                       depth=args.depth, dropout=args.dropout).to(device)
    n_params = model.num_parameters()
    print(f"\n  MODEL: FCSiamDiff  base={args.base_channels} depth={args.depth} "
          f"channels={model.channels}")
    print(f"  PARAMETER COUNT: {n_params:,} ({n_params/1e6:.2f} M)  [target ~1-2 M]")
    print(f"  skip/bottleneck combine: |f1 - f2| (order-invariant)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    steps_per_epoch = max(1, args.patches_per_epoch // args.batch_size)

    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs_since_best = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_bce = tr_dice = 0.0
        for _ in range(steps_per_epoch):
            x1, x2, y = sampler.sample_batch(args.batch_size)
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(x1, x2)
                loss, bce, dice = combo_loss(logits, y, pos_weight, args.bce_weight, args.dice_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += float(loss); tr_bce += float(bce); tr_dice += float(dice)
        sched.step()
        tr_loss /= steps_per_epoch; tr_bce /= steps_per_epoch; tr_dice /= steps_per_epoch

        # ---- validation (full-image, on the 3 held-out train regions) ----
        model.eval()
        v_loss = 0.0
        v_tp = v_fp = v_fn = v_tn = 0
        for region in val_regions:
            prob, gt, logits = infer_region(model, region, mean, std, device, want_logits=True)
            with torch.no_grad():
                vl, _, _ = combo_loss(torch.from_numpy(logits)[None, None].float(),
                                      torch.from_numpy(gt)[None, None].float(),
                                      pos_weight.cpu(), args.bce_weight, args.dice_weight)
            v_loss += float(vl)
            tp, fp, fn, tn = _counts_at(prob, gt, 0.5)
            v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn
        v_loss /= len(val_regions)
        vm = _metrics(v_tp, v_fp, v_fn, v_tn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_bce": tr_bce,
                        "train_dice": tr_dice, "val_loss": v_loss, "val_f1@0.5": vm["f1"],
                        "val_iou@0.5": vm["iou"], "lr": sched.get_last_lr()[0]})
        flag = ""
        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
            flag = "  <- best"
        else:
            epochs_since_best += 1
        print(f"  epoch {epoch:3d}/{args.epochs}  train {tr_loss:.4f} (bce {tr_bce:.3f} dice {tr_dice:.3f})"
              f"  val {v_loss:.4f}  val_F1@0.5 {vm['f1']:.3f}  val_IoU {vm['iou']:.3f}"
              f"  lr {sched.get_last_lr()[0]:.2e}{flag}")
        if args.patience and epochs_since_best >= args.patience:
            print(f"  early stop: no val-loss improvement for {args.patience} epochs "
                  f"(best was epoch {best_epoch}, val {best_val:.4f})")
            break

    wall = time.time() - t0
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    model.load_state_dict(best_state)
    epochs_run = len(history)
    print(f"\n  training done in {wall:.1f}s ({wall/60:.1f} min)  |  peak VRAM {peak_vram_gb:.2f} GB"
          f"  |  {epochs_run} epochs run  |  best val loss {best_val:.4f} @ epoch {best_epoch}")

    _xy_plot(
        [{"x": [h["epoch"] for h in history], "y": [h["train_loss"] for h in history], "label": "train loss"},
         {"x": [h["epoch"] for h in history], "y": [h["val_loss"] for h in history], "label": "val loss"}],
        OUT_DIR / "loss_curve.png", xlabel="epoch", ylabel="combo loss (BCE+Dice)",
        title="FC-Siam-diff / OSCD - training vs validation loss")
    print(f"  loss curve -> {OUT_DIR / 'loss_curve.png'}")

    # ---- checkpoint + model card -------------------------------------
    ckpt_path = OUT_DIR / "fc_siam_diff.pt"
    card = {
        "architecture": {
            "name": "FCSiamDiff", "family": "FC-Siam-diff (Daudt et al., ICIP/IGARSS 2018)",
            "in_channels": len(PRODUCTION_BANDS), "base_channels": args.base_channels,
            "depth": args.depth, "channels": model.channels,
            "encoder": "Siamese, weights SHARED across both dates",
            "skip_combine": "abs(f1 - f2) at every level + bottleneck (order-invariant)",
            "decoder": "U-Net: transposed-conv upsample, concat differenced skip, 2x conv3x3",
            "head": "1x1 conv -> 1 change logit / pixel",
            "parameters": n_params, "parameters_M": round(n_params / 1e6, 3),
            "param_breakdown": {"encoder": count_parameters(model.enc_blocks),
                                "decoder": count_parameters(model.dec_blocks) + count_parameters(model.upconvs),
                                "head": count_parameters(model.classifier)},
        },
        "bands": list(PRODUCTION_BANDS),
        "bands_rationale": "RGB+NIR+SWIR1 = exactly geoseek's production bands; encoder transfers unmapped.",
        "input_space": f"reflectance (DN / {REFLECTANCE_SCALE:.0f}), per-band standardized",
        "norm_stats": norm,
        "training_data": {
            "dataset": "OSCD (Onera Satellite Change Detection)",
            "manifest_ref": "data/provenance_manifest.json # 'oscd' section",
            "oscd_train_split_regions": all_train,
            "fit_regions_gradient_updates": fit_regions,
            "validation_regions_heldout_from_train": val_regions,
            "test_regions_never_used": test_regions,
            "per_region_change_fraction": per_region_change,
        },
        "hyperparameters": {
            "patch": args.patch, "batch_size": args.batch_size, "epochs_max": args.epochs,
            "early_stop_patience": args.patience,
            "patches_per_epoch": args.patches_per_epoch, "steps_per_epoch": steps_per_epoch,
            "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
            "lr_schedule": "CosineAnnealingLR", "dropout": args.dropout,
            "loss": f"{args.bce_weight}*weighted_BCE(pos_weight={pos_weight_val:.2f}) + {args.dice_weight}*soft_Dice",
            "pos_weight_cap": args.pos_weight_cap, "change_oversample": args.change_oversample,
            "augmentation": "dihedral D4 (h/v flip + k*90 rot), identical for both dates + mask",
            "mixed_precision": device == "cuda", "seed": args.seed,
        },
        "training_run": {
            "wall_clock_s": round(wall, 1), "wall_clock_min": round(wall / 60, 2),
            "epochs_run": epochs_run, "best_epoch": best_epoch,
            "peak_vram_gb": round(peak_vram_gb, 3), "best_val_loss": round(best_val, 5),
            "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "epoch_history": history,
        },
        "provenance": {
            "git_commit": None,
            "git_note": "project directory is not under version control at training time; "
                        "source integrity is pinned by source_sha256 below.",
            "source_sha256": _source_hashes(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        "artifacts": {
            "checkpoint": str(ckpt_path), "loss_curve_png": str(OUT_DIR / "loss_curve.png"),
            "norm_stats_json": str(OUT_DIR / "norm_stats.json"),
        },
    }
    card["provenance"]["weights_sha256"] = None  # set + recorded by evaluate() on the FINAL checkpoint
    card["provenance"]["weights_sha256_note"] = ("SHA256 of the delivered fc_siam_diff.pt is recorded in "
                                                 "the manifest sections 'oscd_change_model' and "
                                                 "'artifacts' after evaluate() writes the final file "
                                                 "(a file cannot embed its own hash).")
    torch.save({"state_dict": best_state, "model_card": card}, ckpt_path)
    record_analysis_section("oscd_change_model", card)
    print(f"  checkpoint -> {ckpt_path}")
    print(f"  model card -> manifest section 'oscd_change_model' (weights SHA256 added after evaluation)")
    return {"ckpt_path": str(ckpt_path), "val_regions": val_regions, "test_regions": test_regions,
            "mean": mean, "std": std, "history": history, "peak_vram_gb": peak_vram_gb, "wall": wall}


# ==========================================================================
# evaluate  (STEP D)
# ==========================================================================

# published OSCD baselines (Daudt et al. 2018, change class) for the reality check
PUBLISHED_BASELINES = {
    "FC-EF (Daudt 2018)": {"precision": 0.4489, "recall": 0.5165, "f1": 0.4803},
    "FC-Siam-conc (Daudt 2018)": {"precision": 0.4212, "recall": 0.6217, "f1": 0.5021},
    "FC-Siam-diff (Daudt 2018)": {"precision": 0.4880, "recall": 0.5748, "f1": 0.5281},
}


def evaluate(ckpt_path: Path, mean=None, std=None) -> dict:
    settings = get_settings()
    device = settings.device
    blob = torch.load(ckpt_path, map_location=device)
    card = blob["model_card"]
    arch = card["architecture"]
    model = FCSiamDiff(in_channels=arch["in_channels"], base_channels=arch["base_channels"],
                       depth=arch["depth"], dropout=0.0).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    if mean is None:
        mean = np.asarray(card["norm_stats"]["mean"], np.float32)
        std = np.asarray(card["norm_stats"]["std"], np.float32)

    val_regions = card["training_data"]["validation_regions_heldout_from_train"]
    test_regions = card["training_data"]["test_regions_never_used"]
    fit_regions = card["training_data"]["fit_regions_gradient_updates"]
    assert not (set(test_regions) & set(fit_regions + val_regions)), "LEAK: test overlaps train/val"

    print("\n" + "=" * 78)
    print("STEP D - HONEST EVALUATION")
    print("=" * 78)
    print(f"  threshold is selected on VALIDATION regions {val_regions}, then frozen and applied")
    print(f"  ONCE to the {len(test_regions)} held-out TEST regions {test_regions}.")

    # ---- validation PR curve -> precision-favouring operating point ----
    val_probs, val_gts = zip(*[infer_region(model, r, mean, std, device) for r in val_regions])
    thr_grid = np.round(np.arange(0.02, 0.985, 0.01), 3)
    val_curve = pr_curve(list(val_probs), list(val_gts), thr_grid)
    # precision-favouring per PS 2.2.3: maximise F0.5 (precision weighted 2x recall),
    # with a floor on recall so we don't pick a degenerate high-precision/no-recall point.
    RECALL_FLOOR = 0.15
    viable = [c for c in val_curve if c["recall"] >= RECALL_FLOOR]
    best_val_pt = max(viable or val_curve, key=lambda c: c["f0.5"])
    thr_pf = best_val_pt["threshold"]
    print(f"\n  validation PR curve -> precision-favouring threshold = {thr_pf:.2f}")
    print(f"    (val @ {thr_pf:.2f}: P={best_val_pt['precision']:.3f} R={best_val_pt['recall']:.3f} "
          f"F1={best_val_pt['f1']:.3f} F0.5={best_val_pt['f0.5']:.3f})")
    _xy_plot([{"x": [c["recall"] for c in val_curve], "y": [c["precision"] for c in val_curve],
               "label": "val PR"}],
             OUT_DIR / "pr_curve_val.png", xlabel="recall", ylabel="precision",
             title=f"OSCD validation PR curve (threshold picked here: {thr_pf:.2f})",
             xlim=(0, 1), ylim=(0, 1))

    # ---- test set: inference once ----------------------------------
    test_probs, test_gts = zip(*[infer_region(model, r, mean, std, device) for r in test_regions])
    per_region = {}
    agg = {t: [0, 0, 0, 0] for t in ("0.5", "pf")}
    for r, prob, gt in zip(test_regions, test_probs, test_gts):
        row = {}
        for tag, thr in (("0.5", 0.5), ("pf", thr_pf)):
            tp, fp, fn, tn = _counts_at(prob, gt, thr)
            for i, v in enumerate((tp, fp, fn, tn)):
                agg[tag][i] += v
            row[tag] = _metrics(tp, fp, fn, tn)
        per_region[r] = row
    m05 = _metrics(*agg["0.5"])
    mpf = _metrics(*agg["pf"])

    # ---- test PR curve (report only - NOT used to pick the threshold) ----
    test_curve = pr_curve(list(test_probs), list(test_gts), thr_grid)
    test_oracle = max(test_curve, key=lambda c: c["f1"])
    _xy_plot([{"x": [c["recall"] for c in test_curve], "y": [c["precision"] for c in test_curve],
               "label": "test PR"}],
             OUT_DIR / "pr_curve_test.png", xlabel="recall", ylabel="precision",
             title="OSCD held-out test PR curve (diagnostic only)", xlim=(0, 1), ylim=(0, 1))

    def _row(name, m):
        print(f"    {name:34s} P={m['precision']*100:5.1f}  R={m['recall']*100:5.1f}  "
              f"F1={m['f1']*100:5.1f}  IoU={m['iou']*100:5.1f}  FPR={m['fpr']*100:5.2f}")

    print("\n  HELD-OUT TEST METRICS (aggregated over all 10 test regions):")
    print(f"    {'operating point':34s} {'P%':>6} {'R%':>6} {'F1%':>6} {'IoU%':>6} {'FPR%':>7}")
    _row("default threshold 0.50", m05)
    _row(f"precision-favouring {thr_pf:.2f} (val-set)", mpf)
    print(f"\n  [diagnostic, NOT used] test-set oracle max-F1 threshold {test_oracle['threshold']:.2f}: "
          f"P={test_oracle['precision']*100:.1f} R={test_oracle['recall']*100:.1f} F1={test_oracle['f1']*100:.1f}")

    print("\n  vs published OSCD baselines (change class):")
    for name, b in PUBLISHED_BASELINES.items():
        print(f"    {name:34s} P={b['precision']*100:5.1f}  R={b['recall']*100:5.1f}  F1={b['f1']*100:5.1f}")
    ours_f1 = max(m05["f1"], mpf["f1"])
    best_pub = max(b["f1"] for b in PUBLISHED_BASELINES.values())
    verdict = ("in range of published FC-Siam-diff/conc (~48-53% F1) - expected"
               if ours_f1 <= best_pub + 0.08 else
               f"ABOVE published by >{(ours_f1-best_pub)*100:.0f} pts - TREAT AS A LEAKAGE RED FLAG, investigate")
    print(f"  -> our best F1 {ours_f1*100:.1f}%  ({verdict})")

    print("\n  per-region test F1 @ precision-favouring threshold:")
    for r in test_regions:
        mr = per_region[r]["pf"]
        print(f"    {r:14s} P={mr['precision']*100:5.1f} R={mr['recall']*100:5.1f} F1={mr['f1']*100:5.1f} "
              f"(change {100*(test_gts[test_regions.index(r)]>0.5).mean():.2f}%)")

    # ---- qualitative panels for 3 test regions ---------------------
    panels = []
    for r in QUAL_REGIONS:
        i = test_regions.index(r)
        p = qualitative_panel(r, test_probs[i], test_gts[i], thr_pf, OUT_DIR / f"qual_{r}.png")
        panels.append(str(p))
        print(f"  qualitative panel -> {p}")

    eval_card = {
        "protocol": {
            "inference": "full-image (reflection-padded to /8, cropped back), sigmoid, per-pixel",
            "threshold_selection": (f"precision-favouring per PS 2.2.3: argmax F0.5 (beta=0.5, precision "
                                    f"weighted 2x recall) over the VALIDATION regions, recall floor "
                                    f"{RECALL_FLOOR}; then FROZEN and applied once to the held-out test regions"),
            "validation_regions": val_regions,
            "test_regions": test_regions,
            "leakage_statement": ("The 10 OSCD test regions were not read during training, normalization-"
                                  "statistic fitting, or threshold selection. Threshold came from the 3 "
                                  "validation regions (themselves held out of gradient updates). Confirmed "
                                  "in code by assert guards in train() and evaluate()."),
        },
        "chosen_threshold_precision_favouring": thr_pf,
        "validation_at_chosen_threshold": best_val_pt,
        "test_metrics": {
            "at_0.50": m05,
            "at_precision_favouring": {"threshold": thr_pf, **mpf},
            "per_region": per_region,
        },
        "test_pr_curve": test_curve,
        "validation_pr_curve": val_curve,
        "test_oracle_max_f1_NOT_USED": test_oracle,
        "published_baselines": PUBLISHED_BASELINES,
        "baseline_comparison_verdict": verdict,
        "qualitative_panels": panels,
        "plots": {"pr_curve_val": str(OUT_DIR / "pr_curve_val.png"),
                  "pr_curve_test": str(OUT_DIR / "pr_curve_test.png"),
                  "loss_curve": str(OUT_DIR / "loss_curve.png")},
    }
    # make the checkpoint self-contained: embed the frozen operating point so
    # FCSiamDiffChangeModel does not have to read the manifest to know its threshold.
    card["eval"] = {
        "chosen_threshold_precision_favouring": thr_pf,
        "recall_floor": RECALL_FLOOR,
        "validation_at_chosen_threshold": best_val_pt,
        "test_at_0.50": m05, "test_at_precision_favouring": {"threshold": thr_pf, **mpf},
        "baseline_comparison_verdict": verdict,
    }
    # write the FINAL checkpoint, then hash the on-disk file and record it everywhere.
    torch.save({"state_dict": blob["state_dict"], "model_card": card}, ckpt_path)
    weights_sha = _sha256(ckpt_path)
    card["provenance"]["weights_sha256"] = weights_sha
    eval_card["checkpoint_sha256"] = weights_sha
    eval_card["checkpoint_path"] = str(ckpt_path)

    record_analysis_section("oscd_change_model_eval", eval_card)
    record_analysis_section("oscd_change_model", card)
    record_artifact(
        name="fc-siam-diff-oscd-change-model",
        source_url="trained in-repo by scripts/train_change.py on the staged OSCD 11-region fit subset",
        local_path=ckpt_path,
        license="Apache-2.0 (geoseek code); weights are a derivative of OSCD (CC-BY-NC-SA-4.0) - "
                "non-commercial + ShareAlike",
    )
    print(f"\n  eval card  -> manifest section 'oscd_change_model_eval'")
    print(f"  model card -> manifest section 'oscd_change_model' (weights_sha256 {weights_sha[:16]}...)")
    print(f"  artifact   -> manifest 'artifacts': fc-siam-diff-oscd-change-model")
    print(f"  frozen operating point {thr_pf:.2f} embedded in {ckpt_path.name}")
    return eval_card


# ==========================================================================
# CLI
# ==========================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_change", description=__doc__)
    p.add_argument("--eval-only", action="store_true", help="skip training; evaluate the saved checkpoint")
    p.add_argument("--val-regions", nargs=3, default=list(DEFAULT_VAL_REGIONS),
                   help="3 OSCD-train regions held out for validation + threshold tuning")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15, help="early stop after N epochs w/o val-loss gain (0=off)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--patches-per-epoch", type=int, default=4000)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    p.add_argument("--change-oversample", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=1234)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    ckpt = OUT_DIR / "fc_siam_diff.pt"
    if args.eval_only:
        if not ckpt.is_file():
            raise SystemExit(f"no checkpoint at {ckpt} - train first")
        evaluate(ckpt)
        return
    res = train(args)
    evaluate(Path(res["ckpt_path"]), mean=res["mean"], std=res["std"])
    print("\n" + "=" * 78)
    print("Phase 3b training + evaluation complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()

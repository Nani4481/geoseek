"""``FCSiamDiffChangeModel`` - the trained FC-Siam-diff behind the ChangeDetectionModel seam.

This is the Phase 3b implementation of
:class:`geoseek.models.base.ChangeDetectionModel`. It consumes a
:class:`geoseek.temporal.contract.ObservationPair` exactly as
:class:`geoseek.temporal.matcher.TemporalObservationMatcher` emits it - no
reshaping at the boundary - reads the two observations' band rasters from
``data/datasets/<observation_id>/``, runs the trained
:class:`geoseek.change.models.fc_siam_diff.FCSiamDiff` tile by tile, and returns
a :class:`geoseek.models.base.ChangeResult` with a per-tile change score, a
per-tile change label, and a georeferenced full-AOI change mask written to disk.

Offline. Depends only on torch / numpy / rasterio + the catalog entities - it
does not import sqlite3 or faiss (the repository / vector-index seams are
untouched).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geoseek.change.models.fc_siam_diff import FCSiamDiff
from geoseek.config import get_settings
from geoseek.models.base import ChangeDetectionModel, ChangeResult

REFLECTANCE_SCALE = 10000.0
DEFAULT_CHECKPOINT = get_settings().data_dir / "change_model" / "fc_siam_diff.pt"
_TILE_PX = 256


@dataclass
class ChangeProbabilityRaster:
    """Full-AOI per-pixel change probability from :class:`FCSiamDiffChangeModel`.

    ``prob`` is float32 in [0, 1] (0 where either date is nodata); ``valid`` is
    the boolean per-pixel validity mask; ``transform`` / ``crs`` are the later
    observation's georeferencing. ``threshold`` is the model's frozen
    precision-favouring operating point (for a default binarisation).
    """

    prob: "object"                     # np.ndarray (H, W) float32
    valid: "object"                    # np.ndarray (H, W) bool
    transform: "object"
    crs: "object"
    earlier_observation_id: str
    later_observation_id: str
    threshold: float
    path: str | None = None


@dataclass
class _Loaded:
    model: "object"
    bands: list[str]
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    weights_sha256: str | None
    card: dict


class FCSiamDiffChangeModel(ChangeDetectionModel):
    """Trained FC-Siam-diff exposed as a :class:`ChangeDetectionModel`.

    Args:
        checkpoint_path: ``.pt`` written by ``scripts/train_change.py``
            (defaults to ``data/change_model/fc_siam_diff.pt``).
        device: torch device; defaults to the geoseek-selected device.
        threshold: binarisation threshold; defaults to the precision-favouring
            operating point recorded in the checkpoint's eval card, else 0.5.
        tile_px: tile size for windowed inference (matches the catalog tile grid).
    """

    def __init__(self, checkpoint_path: str | Path | None = None, *, device: str | None = None,
                 threshold: float | None = None, tile_px: int = _TILE_PX):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.device = device or get_settings().device
        self._override_threshold = threshold
        self.tile_px = tile_px
        self._loaded: _Loaded | None = None

    # -- lifecycle -------------------------------------------------------

    def load(self) -> None:
        if self._loaded is not None:
            return
        import torch

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FC-Siam-diff checkpoint not found at {self.checkpoint_path}. "
                "Train it first:  python scripts/train_change.py")
        blob = torch.load(self.checkpoint_path, map_location=self.device)
        card = blob.get("model_card", {})
        arch = card.get("architecture", {})
        model = FCSiamDiff(
            in_channels=arch.get("in_channels", 5),
            base_channels=arch.get("base_channels", 24),
            depth=arch.get("depth", 4),
            dropout=0.0,
        ).to(self.device)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        ns = card.get("norm_stats", {})
        bands = list(card.get("bands", arch.get("bands", ["B02", "B03", "B04", "B08", "B11"])))
        mean = np.asarray(ns.get("mean", [0.0] * len(bands)), np.float32)
        std = np.asarray(ns.get("std", [1.0] * len(bands)), np.float32)

        thr = self._override_threshold
        if thr is None:
            thr = (card.get("eval", {}) or {}).get("chosen_threshold_precision_favouring")
        if thr is None:
            # not stored on the card itself (eval writes a separate manifest section) - fall back
            thr = _threshold_from_manifest() or 0.5

        # the delivered .pt cannot embed its own hash; compute it from the file if absent
        sha = (card.get("provenance", {}) or {}).get("weights_sha256")
        if not sha:
            import hashlib

            h = hashlib.sha256()
            with open(self.checkpoint_path, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            sha = h.hexdigest()
        self._loaded = _Loaded(model=model, bands=bands, mean=mean, std=std, threshold=float(thr),
                               weights_sha256=sha, card=card)

    @property
    def threshold(self) -> float:
        self.load()
        return self._loaded.threshold

    # -- inference ------------------------------------------------------

    def predict_change(self, pair, *, tiles=None, aoi_wkt=None, bands=None) -> ChangeResult:
        import rasterio
        import torch

        method = "fc_siam_diff"
        base_notes = [f"model={self.checkpoint_path.name}", f"threshold={self.threshold:.2f}"]

        if not pair.comparable:
            return ChangeResult(
                earlier_observation_id=pair.earlier.observation_id,
                later_observation_id=pair.later.observation_id,
                method=method, comparable=False, confidence=0.0,
                notes=base_notes + ["pair not comparable - refused; "
                                    + "; ".join(pair.comparability.blocking_reasons)])

        self.load()
        L = self._loaded
        use_bands = list(bands) if bands else L.bands
        if any(b not in L.bands for b in use_bands):
            raise ValueError(f"model was trained on {L.bands}; cannot run on {use_bands}")
        b_idx = [L.bands.index(b) for b in use_bands]
        mean, std = L.mean[b_idx], L.std[b_idx]

        settings = get_settings()
        e_dir = settings.datasets_dir / (pair.earlier.dataset_dir or pair.earlier.observation_id)
        l_dir = settings.datasets_dir / (pair.later.dataset_dir or pair.later.observation_id)
        for d in (e_dir, l_dir):
            missing = [b for b in use_bands if not (d / f"{b}.tif").is_file()]
            if missing:
                raise FileNotFoundError(f"{d} missing band(s) {missing}")

        with rasterio.open(l_dir / f"{use_bands[0]}.tif") as ref:
            H, W = ref.height, ref.width
            transform, crs, nodata = ref.transform, ref.crs, ref.nodata

        # which tiles to score
        n_rt, n_ct = -(-H // self.tile_px), -(-W // self.tile_px)
        if tiles:
            want = {(int(t.row), int(t.col)) for t in tiles if hasattr(t, "row")}
            grid = [(r, c) for r in range(n_rt) for c in range(n_ct) if (r, c) in want]
        else:
            grid = [(r, c) for r in range(n_rt) for c in range(n_ct)]

        e_src = {b: rasterio.open(e_dir / f"{b}.tif") for b in use_bands}
        l_src = {b: rasterio.open(l_dir / f"{b}.tif") for b in use_bands}
        full_mask = np.zeros((H, W), np.uint8)
        scores: dict[str, float] = {}
        labels: dict[str, str] = {}
        confs: list[float] = []
        try:
            for (r, c) in grid:
                y0, x0 = r * self.tile_px, c * self.tile_px
                y1, x1 = min(y0 + self.tile_px, H), min(x0 + self.tile_px, W)
                win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                a = np.stack([e_src[b].read(1, window=win).astype(np.float32) for b in use_bands])
                bb = np.stack([l_src[b].read(1, window=win).astype(np.float32) for b in use_bands])
                valid = np.all(bb > 0, axis=0) & np.all(a > 0, axis=0)
                if valid.sum() < 16:
                    continue
                a = (a / REFLECTANCE_SCALE - mean[:, None, None]) / std[:, None, None]
                bb = (bb / REFLECTANCE_SCALE - mean[:, None, None]) / std[:, None, None]
                t1 = torch.from_numpy(a)[None].to(self.device)
                t2 = torch.from_numpy(bb)[None].to(self.device)
                with torch.no_grad(), torch.autocast(device_type="cuda",
                                                     enabled=(self.device == "cuda")):
                    prob = torch.sigmoid(self._loaded.model(t1, t2)).float()[0, 0].cpu().numpy()
                prob[~valid] = 0.0
                tile_pred = (prob >= self.threshold) & valid
                full_mask[y0:y1, x0:x1] = tile_pred.astype(np.uint8)
                tid = f"{pair.later.observation_id}_r{r:03d}_c{c:03d}"
                frac = float(tile_pred.sum() / max(valid.sum(), 1))
                scores[tid] = round(float(prob[valid].mean()), 5)
                labels[tid] = "change" if frac >= 0.02 else "no_change"
                confs_val = prob[valid]
                confs_val = np.where(confs_val >= 0.5, confs_val, 1.0 - confs_val)
                confs.append(float(confs_val.mean()))
        finally:
            for s in list(e_src.values()) + list(l_src.values()):
                s.close()

        # write the georeferenced full-AOI change mask
        out_dir = l_dir
        mask_path = out_dir / f"change_fc_siam_diff__{pair.earlier.observation_id}__to__{pair.later.observation_id}.tif"
        profile = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "uint8",
                   "crs": crs, "transform": transform, "nodata": 0, "compress": "deflate"}
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(full_mask, 1)

        n_change_tiles = sum(1 for v in labels.values() if v == "change")
        return ChangeResult(
            earlier_observation_id=pair.earlier.observation_id,
            later_observation_id=pair.later.observation_id,
            method=method,
            change_score_by_tile=scores,
            change_label_by_tile=labels,
            change_mask_path=str(mask_path),
            confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
            comparable=True,
            notes=base_notes + [
                f"bands={use_bands}",
                f"tiles_scored={len(scores)}", f"tiles_flagged_change={n_change_tiles}",
                f"changed_px_fraction={float(full_mask.mean()):.4f}",
                f"weights_sha256={L.weights_sha256}",
                "trained on OSCD 11-region fit subset; OSCD test split never used in training or "
                "threshold tuning",
            ],
        )


    # -- full-AOI probability raster (Phase 4 pipeline entry point) --------

    def infer_probability_raster(self, pair, *, bands=None, out_path=None,
                                 progress_every: int = 0) -> ChangeProbabilityRaster:
        """Run the trained net over the whole later-observation grid and return the
        per-pixel change probability (not thresholded).

        Same tile-by-tile inference as :meth:`predict_change`; this keeps the
        float probabilities so a downstream stage can build candidates / apply
        its own operating point. Optionally writes a float32 GeoTIFF to
        ``out_path``.
        """
        import rasterio
        import torch

        if not pair.comparable:
            raise ValueError(f"pair not comparable: {'; '.join(pair.comparability.blocking_reasons)}")
        self.load()
        L = self._loaded
        use_bands = list(bands) if bands else L.bands
        if any(b not in L.bands for b in use_bands):
            raise ValueError(f"model was trained on {L.bands}; cannot run on {use_bands}")
        b_idx = [L.bands.index(b) for b in use_bands]
        mean, std = L.mean[b_idx], L.std[b_idx]

        settings = get_settings()
        e_dir = settings.datasets_dir / (pair.earlier.dataset_dir or pair.earlier.observation_id)
        l_dir = settings.datasets_dir / (pair.later.dataset_dir or pair.later.observation_id)
        for d in (e_dir, l_dir):
            missing = [b for b in use_bands if not (d / f"{b}.tif").is_file()]
            if missing:
                raise FileNotFoundError(f"{d} missing band(s) {missing}")

        with rasterio.open(l_dir / f"{use_bands[0]}.tif") as ref:
            H, W = ref.height, ref.width
            transform, crs = ref.transform, ref.crs

        prob_full = np.zeros((H, W), np.float32)
        valid_full = np.zeros((H, W), bool)
        n_rt, n_ct = -(-H // self.tile_px), -(-W // self.tile_px)
        e_src = {b: rasterio.open(e_dir / f"{b}.tif") for b in use_bands}
        l_src = {b: rasterio.open(l_dir / f"{b}.tif") for b in use_bands}
        done = 0
        try:
            for r in range(n_rt):
                for c in range(n_ct):
                    y0, x0 = r * self.tile_px, c * self.tile_px
                    y1, x1 = min(y0 + self.tile_px, H), min(x0 + self.tile_px, W)
                    win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                    a = np.stack([e_src[b].read(1, window=win).astype(np.float32) for b in use_bands])
                    bb = np.stack([l_src[b].read(1, window=win).astype(np.float32) for b in use_bands])
                    valid = np.all(bb > 0, axis=0) & np.all(a > 0, axis=0)
                    valid_full[y0:y1, x0:x1] = valid
                    if valid.sum() < 16:
                        continue
                    a = (a / REFLECTANCE_SCALE - mean[:, None, None]) / std[:, None, None]
                    bb = (bb / REFLECTANCE_SCALE - mean[:, None, None]) / std[:, None, None]
                    t1 = torch.from_numpy(a)[None].to(self.device)
                    t2 = torch.from_numpy(bb)[None].to(self.device)
                    with torch.no_grad(), torch.autocast(device_type="cuda",
                                                         enabled=(self.device == "cuda")):
                        p = torch.sigmoid(self._loaded.model(t1, t2)).float()[0, 0].cpu().numpy()
                    p[~valid] = 0.0
                    prob_full[y0:y1, x0:x1] = p
                    done += 1
                    if progress_every and done % progress_every == 0:
                        print(f"    [infer] {done}/{n_rt * n_ct} tiles", flush=True)
        finally:
            for s in list(e_src.values()) + list(l_src.values()):
                s.close()

        path = None
        if out_path is not None:
            path = str(out_path)
            prof = {"driver": "GTiff", "height": H, "width": W, "count": 1, "dtype": "float32",
                    "crs": crs, "transform": transform, "nodata": 0.0, "compress": "deflate",
                    "predictor": 3, "tiled": True, "blockxsize": 256, "blockysize": 256}
            with rasterio.open(path, "w", **prof) as dst:
                dst.write(prob_full, 1)

        return ChangeProbabilityRaster(
            prob=prob_full, valid=valid_full, transform=transform, crs=crs,
            earlier_observation_id=pair.earlier.observation_id,
            later_observation_id=pair.later.observation_id,
            threshold=float(self.threshold), path=path,
        )


def _threshold_from_manifest() -> float | None:
    try:
        from geoseek.staging.manifest import load_manifest

        return (load_manifest().get("oscd_change_model_eval", {}) or {}).get(
            "chosen_threshold_precision_favouring")
    except Exception:
        return None

"""Phase 5 Step C: turn co-located Sentinel-1 dB change into a corroboration factor.

Physical expectation per optical change type (C-band, VV/VH):

    water_gain   land -> open water     backscatter DROPS hard (specular)   driver VV, sign -
    water_loss   open water -> land     backscatter RISES                   driver VV, sign +
    construction bare/veg -> built      backscatter RISES (double-bounce)   driver VV, sign +
    clearance    veg -> bare            VH DROPS (loss of volume scatter)   driver VH, sign -
    road / other no confident C-band expectation                           -> NEUTRAL

The observed dB change is compared **against the scene-wide dB trend** (2019
was a drought March -> lower soil backscatter; 2024 wetter -> higher), exactly
as the optical side is anomaly-framed. Agreement with the expected direction
raises the confidence factor (up to +10 %), a clear opposite lowers it
(down to -20 %), within-noise or SAR-unavailable is exactly neutral (x1.0).
This is a WEIGHT, never an override - we have no ground truth to validate an
override.

SAR rasters are staged already on the S2 10 m grid
(``geoseek.staging.download_sentinel1``), so a candidate's S2 pixel bbox
indexes the dB-change raster directly - no reprojection here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoseek.sar.backscatter import db_change, summarize_db

# change_type -> (driver polarization, expected sign: +1 rise / -1 drop)
EXPECTED_SAR: dict[str, tuple[str, int]] = {
    "water_gain": ("vv", -1),
    "water_loss": ("vv", +1),
    "construction": ("vv", +1),
    "clearance": ("vh", -1),
}
_STRONG_DB, _WEAK_DB = 3.0, 1.0


def sar_factor(change_type: str, dvv_db: float | None, dvh_db: float | None, *,
               scene_dvv_db: float = 0.0, scene_dvh_db: float = 0.0,
               available: bool = True) -> tuple[float, str]:
    """(confidence factor in [0.8, 1.10], human-readable detail)."""
    if not available or change_type not in EXPECTED_SAR or dvv_db is None or dvh_db is None:
        return 1.0, "no usable co-located SAR" if not available else f"no C-band expectation for '{change_type}'"
    pol, sign = EXPECTED_SAR[change_type]
    obs = (dvv_db - scene_dvv_db) if pol == "vv" else (dvh_db - scene_dvh_db)
    agree = sign * obs                       # dB in the expected direction (scene-detrended)
    exp = "drop" if sign < 0 else "rise"
    if agree >= _STRONG_DB:
        f, v = 1.10, "strong agreement"
    elif agree >= _WEAK_DB:
        f, v = 1.04, "weak agreement"
    elif agree <= -_STRONG_DB:
        f, v = 0.80, "clear disagreement"
    elif agree <= -_WEAK_DB:
        f, v = 0.93, "weak disagreement"
    else:
        f, v = 1.0, "within speckle noise (neutral)"
    return f, (f"{pol.upper()} anomaly {obs:+.1f} dB vs expected {exp} -> {v}")


@dataclass
class SarPairChange:
    """VV / VH dB-change rasters for one S2 date-pair (via the nearest S1 pair)."""

    s1_earlier_obs: str
    s1_later_obs: str
    db_vv: np.ndarray
    db_vh: np.ndarray
    valid: np.ndarray
    scene_dvv_db: float
    scene_dvh_db: float
    coverage_fraction: float
    notes: list[str] = field(default_factory=list)

    def lookup(self, bbox_rc: tuple[int, int, int, int], comp_mask: np.ndarray) -> dict:
        r0, c0, r1, c1 = bbox_rc
        m = comp_mask & self.valid[r0:r1, c0:c1]
        vv = summarize_db(self.db_vv[r0:r1, c0:c1], m)
        vh = summarize_db(self.db_vh[r0:r1, c0:c1], m)
        n = vv["n_px"]
        return {"available": n >= 10, "n_px": n,
                "vv_median_db": vv["median_db"], "vh_median_db": vh["median_db"],
                "vv_p16_db": vv["p16_db"], "vv_p84_db": vv["p84_db"]}


class SarCorroborator:
    """Resolves + caches the S1 dB-change raster for a given S2 date-pair."""

    def __init__(self, manifest: dict | None = None):
        from geoseek.staging.manifest import load_manifest

        m = manifest or load_manifest()
        self.section = m.get("sentinel1")
        self._cache: dict[tuple[str, str], SarPairChange] = {}
        # S2 observation id -> S1 observation id
        self.s2_to_s1: dict[str, str] = {}
        if self.section:
            for d in self.section["dates"].values():
                s2_obs = _near_s2_observation(d)
                if s2_obs:
                    self.s2_to_s1[s2_obs] = d["observation_id"]

    @property
    def available(self) -> bool:
        return bool(self.section) and len(self.s2_to_s1) >= 2

    def for_pair(self, s2_earlier_obs: str, s2_later_obs: str) -> SarPairChange | None:
        if not self.available:
            return None
        e1, l1 = self.s2_to_s1.get(s2_earlier_obs), self.s2_to_s1.get(s2_later_obs)
        if not e1 or not l1:
            return None
        key = (e1, l1)
        if key in self._cache:
            return self._cache[key]
        import rasterio

        from geoseek.config import get_settings

        d = get_settings().datasets_dir
        with rasterio.open(d / e1 / "VV.tif") as ds:
            vv_e = ds.read(1)
        with rasterio.open(d / l1 / "VV.tif") as ds:
            vv_l = ds.read(1)
        with rasterio.open(d / e1 / "VH.tif") as ds:
            vh_e = ds.read(1)
        with rasterio.open(d / l1 / "VH.tif") as ds:
            vh_l = ds.read(1)
        db_vv, val_vv = db_change(vv_e, vv_l)
        db_vh, val_vh = db_change(vh_e, vh_l)
        valid = val_vv & val_vh
        # scene-wide dB trend (median over a subsample of valid pixels)
        rng = np.random.default_rng(0)
        idx = rng.choice(int(valid.sum()), size=min(400_000, int(valid.sum())), replace=False) \
            if valid.any() else np.array([], int)
        vv_flat = db_vv[valid]
        vh_flat = db_vh[valid]
        scene_dvv = float(np.median(vv_flat[idx])) if idx.size else 0.0
        scene_dvh = float(np.median(vh_flat[idx])) if idx.size else 0.0
        pc = SarPairChange(
            s1_earlier_obs=e1, s1_later_obs=l1, db_vv=db_vv, db_vh=db_vh, valid=valid,
            scene_dvv_db=scene_dvv, scene_dvh_db=scene_dvh,
            coverage_fraction=float(valid.mean()),
            notes=[f"S1 {e1} -> {l1}", f"scene dB trend VV {scene_dvv:+.2f} / VH {scene_dvh:+.2f}",
                   "speckle: adaptive Lee 7x7 (ENL 4.4), intensity domain, before the ratio"])
        self._cache[key] = pc
        return pc


def _near_s2_observation(date_entry: dict) -> str | None:
    return (date_entry.get("metadata") or {}).get("near_s2_observation") \
        or date_entry.get("near_s2_observation")

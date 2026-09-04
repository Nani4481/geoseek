"""Stage the Onera Satellite Change Detection (OSCD) dataset for Phase 3b training.

This is one of the ONLY modules in geoseek allowed to touch the network (see the
project-wide rule in ``README.md`` / ``config.py``). Everything downstream
(``scripts/train_change.py``, ``geoseek.change.models.*``) reads exclusively from
the local files staged here under ``data/datasets/oscd/``.

What OSCD is
------------
24 co-registered Sentinel-2 image PAIRS (two acquisition dates each, 2015-2018)
with human-drawn binary change masks (0 = no change, 255 = change). Standard
split: **14 train pairs / 10 test pairs** (Daudt et al., IGARSS 2018). The split
is defined by the ``train.txt`` / ``test.txt`` manifests shipped inside the
Images archive and mirrored by which regions carry a label folder in the Train
vs Test Labels archives - this module reads both and cross-checks them.

Source, confirmed LIVE at authoring time (not recalled from memory)
------------------------------------------------------------------
The canonical home is IEEE DataPort (DOI ``10.21227/asqe-7s69``,
https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection),
which now sits behind an IEEE DataPort subscription. The dataset author's own
page (https://rcdaudt.github.io/oscd/) links IMT file-share mirrors. The
practical canonical download used here is the **pinned Hugging Face mirror**
that the maintained ``torchgeo`` library resolves against:

    HF dataset repo : hkristen/oscd
    Pinned revision : 4958d786c1389ede1511d91a6ecf1a75c4074933
    Files           : "Onera Satellite Change Detection dataset - Images.zip"
                      "Onera Satellite Change Detection dataset - Train Labels.zip"
                      "Onera Satellite Change Detection dataset - Test Labels.zip"

Queried live at authoring time via the HF API
(``/api/datasets/hkristen/oscd/tree/<rev>``): the three archives are LFS blobs
whose git-LFS SHA256 oids match :data:`EXPECTED_SHA256` below, and the repo card
declares ``license: cc-by-nc-sa-4.0``. Each download is re-verified against
:data:`EXPECTED_SHA256` after transfer and the *computed* digest is what goes in
the provenance manifest.

License
-------
CC-BY-NC-SA-4.0 (per the HF mirror repo card). The upstream dataset is
distributed for research use on the condition that the IGARSS 2018 paper is
cited (see :data:`CITATION`). This project (a Smart India Hackathon research /
educational prototype) is non-commercial, so the NC term is satisfied; the
trained weights and any redistributed derivative inherit ShareAlike.

Radiometry / harmonization vs the geoseek pipeline
-------------------------------------------------
OSCD imagery is **Sentinel-2 Level-1C top-of-atmosphere (TOA) reflectance**,
uint16, ``reflectance = DN / 10000`` (additive offset 0). geoseek's Ayodhya
pipeline uses **Level-2A bottom-of-atmosphere (BOA / surface) reflectance**,
also uint16 ``DN / 10000`` offset 0. Same *encoding*, different *physical
quantity*: L1C carries the atmospheric (path-radiance + transmittance) term that
L2A has removed, so L1C absolute values run higher, most in the blue.

How this is harmonized so the trained model transfers:
  1. Both are ``reflectance * 10000`` uint16 -> a single ``/1e4`` puts them on
     one nominal reflectance scale (recorded, asserted here).
  2. Training standardizes every band by per-band mean/std computed over the
     OSCD *train* pairs only; inference on Ayodhya applies the *same* transform.
     A constant per-band bias (the bulk of the TOA/BOA gap) is absorbed by the
     mean subtraction.
  3. FC-Siam-diff consumes the *difference* of the two dates' encoder features;
     any atmospheric term shared by both dates of a pair largely cancels in that
     difference regardless of its absolute level.
  4. On the Ayodhya side the two dates are additionally put into a common
     radiometric frame first by ``geoseek.change.prep`` (Phase 3a relative
     normalization), shrinking the residual inter-date atmospheric difference
     the model has to be robust to.

Re-running after the archives are staged skips the network entirely.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import requests

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_analysis_section, record_artifact

HF_REPO_ID = "hkristen/oscd"
HF_REVISION = "4958d786c1389ede1511d91a6ecf1a75c4074933"
_HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{HF_REVISION}"

IMAGES_ZIP = "Onera Satellite Change Detection dataset - Images.zip"
TRAIN_LABELS_ZIP = "Onera Satellite Change Detection dataset - Train Labels.zip"
TEST_LABELS_ZIP = "Onera Satellite Change Detection dataset - Test Labels.zip"
ARCHIVES = (IMAGES_ZIP, TRAIN_LABELS_ZIP, TEST_LABELS_ZIP)

# git-LFS SHA256 oids from the live HF API tree listing at HF_REVISION (also the
# values the torchgeo OSCD loader pins). Re-verified against the transferred bytes.
EXPECTED_SHA256 = {
    IMAGES_ZIP: "940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656",
    TRAIN_LABELS_ZIP: "89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5",
    TEST_LABELS_ZIP: "2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d",
}
EXPECTED_BYTES = {IMAGES_ZIP: 512_716_711, TRAIN_LABELS_ZIP: 137_873, TEST_LABELS_ZIP: 83_614}

DATA_LICENSE = "CC-BY-NC-SA-4.0 (OSCD; Daudt et al., IGARSS 2018 - non-commercial, ShareAlike)"
CITATION = (
    "Daudt, R.C., Le Saux, B., Boulch, A., Gousseau, Y. "
    "'Urban Change Detection for Multispectral Earth Observation Using Convolutional "
    "Neural Networks.' IGARSS 2018, pp. 2115-2118. doi:10.1109/IGARSS.2018.8518015"
)

# The full 13-band Sentinel-2 set OSCD ships, in the canonical band order.
OSCD_ALL_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07",
                  "B08", "B8A", "B09", "B10", "B11", "B12")
# Bands geoseek uses in production (RGB + NIR + SWIR-1) - the ones the trained
# model must transfer on. B10 (cirrus) is L1C-only and is deliberately excluded.
PRODUCTION_BANDS = ("B02", "B03", "B04", "B08", "B11")

REFLECTANCE_SCALE = 10000.0  # OSCD L1C: reflectance = DN / 10000 (offset 0)

_IMAGES_ROOT = "Onera Satellite Change Detection dataset - Images"
_TRAIN_LABELS_ROOT = "Onera Satellite Change Detection dataset - Train Labels"
_TEST_LABELS_ROOT = "Onera Satellite Change Detection dataset - Test Labels"


def oscd_root() -> Path:
    return get_settings().datasets_dir / "oscd"


def _archives_dir() -> Path:
    return oscd_root() / "_archives"


# --------------------------------------------------------------------------
# download + verify
# --------------------------------------------------------------------------


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(name: str, dest: Path) -> None:
    url = f"{_HF_RESOLVE}/{requests.utils.quote(name)}"
    print(f"[staging]   GET {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                f.write(block)
                got += len(block)
                if total:
                    pct = 100.0 * got / total
                    print(f"\r[staging]   {name}: {got / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        if total:
            print()
    tmp.replace(dest)


def download_and_verify(force: bool = False) -> dict[str, dict]:
    """Fetch the three OSCD archives to ``data/datasets/oscd/_archives/`` and check SHA256."""
    adir = _archives_dir()
    adir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for name in ARCHIVES:
        path = adir / name
        if force and path.exists():
            path.unlink()
        if not path.is_file():
            print(f"\n[staging] Downloading {name} ...")
            _download(name, path)
        else:
            print(f"\n[staging] {name} already present ({path.stat().st_size / 1e6:.1f} MB) - skipping network.")

        digest = _sha256_of(path)
        size = path.stat().st_size
        exp_sha = EXPECTED_SHA256[name]
        if digest != exp_sha:
            raise RuntimeError(
                f"SHA256 mismatch for {name}\n  expected {exp_sha}\n  got      {digest}\n"
                f"Refusing to proceed with a corrupt / wrong archive. Delete {path} and re-run."
            )
        if size != EXPECTED_BYTES[name]:
            raise RuntimeError(f"Byte-size mismatch for {name}: expected {EXPECTED_BYTES[name]}, got {size}")
        print(f"[staging]   verified sha256={digest[:16]}... size={size}B  OK")
        records[name] = {"sha256": digest, "byte_size": size,
                         "source_url": f"{_HF_RESOLVE}/{requests.utils.quote(name)}"}
    return records


def extract(force: bool = False) -> None:
    root = oscd_root()
    marker = root / _IMAGES_ROOT / "test.txt"
    if marker.is_file() and not force:
        print(f"[staging] OSCD already extracted under {root} - skipping.")
        return
    for name in ARCHIVES:
        zpath = _archives_dir() / name
        print(f"[staging] Extracting {name} ...")
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(root)
    print(f"[staging] Extracted to {root}")


# --------------------------------------------------------------------------
# split enumeration + band / radiometry report
# --------------------------------------------------------------------------


def _read_split_file(kind: str) -> list[str]:
    p = oscd_root() / _IMAGES_ROOT / f"{kind}.txt"
    if not p.is_file():
        return []
    raw = p.read_text(encoding="utf-8").strip()
    # the file is a single comma-separated line in the OSCD distribution
    parts = [tok.strip() for chunk in raw.splitlines() for tok in chunk.split(",")]
    return [x for x in parts if x]


def _label_regions(root_name: str) -> list[str]:
    d = oscd_root() / root_name
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "cm").is_dir())


def change_mask_path(region: str) -> Path | None:
    """Path to a region's canonical single-band change map (``<region>-cm.tif``, values 1=no-change 2=change)."""
    for root_name in (_TRAIN_LABELS_ROOT, _TEST_LABELS_ROOT):
        p = oscd_root() / root_name / region / "cm" / f"{region}-cm.tif"
        if p.is_file():
            return p
    return None


def enumerate_split() -> dict:
    """Resolve the 14/10 split from both the ``*.txt`` manifests and the label folders, and cross-check."""
    train_txt, test_txt = _read_split_file("train"), _read_split_file("test")
    train_lbl, test_lbl = _label_regions(_TRAIN_LABELS_ROOT), _label_regions(_TEST_LABELS_ROOT)

    train = sorted(train_lbl or train_txt)
    test = sorted(test_lbl or test_txt)

    problems = []
    if train_txt and sorted(train_txt) != train:
        problems.append(f"train.txt {sorted(train_txt)} != train label folders {train}")
    if test_txt and sorted(test_txt) != test:
        problems.append(f"test.txt {sorted(test_txt)} != test label folders {test}")
    if set(train) & set(test):
        problems.append(f"train/test overlap: {sorted(set(train) & set(test))}")
    if len(train) != 14 or len(test) != 10:
        problems.append(f"expected 14 train / 10 test, got {len(train)} / {len(test)}")

    return {
        "train_regions": train,
        "test_regions": test,
        "n_train": len(train),
        "n_test": len(test),
        "source": "OSCD train.txt / test.txt cross-checked against Train/Test Labels folders",
        "consistency_problems": problems,
    }


def _band_path(region: str, date_idx: int, band: str) -> Path | None:
    d = oscd_root() / _IMAGES_ROOT / region / f"imgs_{date_idx}_rect"
    hits = sorted(d.glob(f"*{band}.tif"))
    return hits[0] if hits else None


def report_bands_and_radiometry(sample_region: str) -> dict:
    """Report available bands + resolutions and confirm the reflectance scale on one train region."""
    reg_dir = oscd_root() / _IMAGES_ROOT / sample_region
    rect_files = sorted((reg_dir / "imgs_1_rect").glob("*.tif"))
    native_dir = reg_dir / "imgs_1"
    native_files = sorted(native_dir.glob("*.tif")) if native_dir.is_dir() else []

    def _band_of(p: Path) -> str:
        stem = p.stem
        return stem.split("_")[-1] if "_" in stem else stem

    rect_report = {}
    for p in rect_files:
        with rasterio.open(p) as ds:
            rect_report[_band_of(p)] = {"shape": [ds.height, ds.width], "dtype": str(ds.dtypes[0]),
                                        "crs": str(ds.crs) if ds.crs else None}
    native_report = {}
    for p in native_files:
        with rasterio.open(p) as ds:
            native_report[_band_of(p)] = {"shape": [ds.height, ds.width], "dtype": str(ds.dtypes[0])}

    # DN stats + reflectance-scale confirmation over the production bands, both dates
    dn_stats = {}
    scale_ok = True
    for band in PRODUCTION_BANDS:
        vals = []
        for di in (1, 2):
            bp = _band_path(sample_region, di, band)
            if bp is None:
                continue
            with rasterio.open(bp) as ds:
                a = ds.read(1).astype(np.float64)
            a = a[a > 0]
            if a.size:
                vals.append(a)
        if not vals:
            continue
        v = np.concatenate(vals)
        st = {"min": float(v.min()), "p1": float(np.percentile(v, 1)),
              "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
              "max": float(v.max()), "reflectance_median": round(float(np.median(v)) / REFLECTANCE_SCALE, 4)}
        dn_stats[band] = st
        # L1C TOA reflectance*1e4: median DN should land in a plausible 0.01-0.6 reflectance band
        if not (100.0 <= st["median"] <= 8000.0):
            scale_ok = False

    # per-region change-pixel fraction (severe imbalance the training must handle)
    change_frac = {}
    for reg in _label_regions(_TRAIN_LABELS_ROOT) + _label_regions(_TEST_LABELS_ROOT):
        cmp_ = change_mask_path(reg)
        if cmp_ is None:
            continue
        with rasterio.open(cmp_) as ds:
            cm = ds.read(1)
        change_frac[reg] = round(float((cm == 2).mean()), 5)

    return {
        "sample_region": sample_region,
        "rect_bands_grid": rect_report,
        "rect_grid_note": ("imgs_*_rect tiles carry NO CRS / geotransform (plain rasters). Every band is "
                           "already resampled onto the Sentinel-2 10 m grid and the two dates are "
                           "co-registered by the dataset authors - so all 13 bands + the change mask of "
                           "a region share one HxW pixel grid."),
        "native_resolution_bands": native_report or "imgs_1 (native-resolution) folder not present",
        "native_resolution_note": ("Sentinel-2 native GSD: B02/B03/B04/B08 = 10 m; "
                                   "B05/B06/B07/B8A/B11/B12 = 20 m; B01/B09/B10 = 60 m."),
        "change_mask_format": ("<region>-cm.tif, single band uint8, 1 = no change, 2 = change "
                               "(canonical; used for training). cm.png is an inconsistent RGB/RGBA/L "
                               "rendering of the same map and is only a fallback."),
        "change_pixel_fraction_by_region": change_frac,
        "change_pixel_fraction_overall": round(
            float(np.mean(list(change_frac.values()))) if change_frac else 0.0, 5),
        "production_bands_used": list(PRODUCTION_BANDS),
        "production_bands_rationale": ("RGB (B02/B03/B04) + NIR (B08) + SWIR-1 (B11) - exactly the bands "
                                       "geoseek stages for Ayodhya, so the trained encoder transfers with "
                                       "no band remapping. B10 (cirrus) is L1C-only and excluded."),
        "dn_stats_over_valid_pixels": dn_stats,
        "reflectance_scale_dn_per_unit": REFLECTANCE_SCALE,
        "boa_add_offset_dn": 0.0,
        "product_level": "Sentinel-2 L1C (top-of-atmosphere reflectance)",
        "geoseek_pipeline_level": "Sentinel-2 L2A (bottom-of-atmosphere / surface reflectance)",
        "radiometry_match": ("SAME encoding (uint16, reflectance*10000, offset 0); DIFFERENT physical "
                             "quantity (TOA vs BOA). Harmonized by per-band standardization fit on the "
                             "OSCD train split and re-applied at inference, plus FC-Siam-diff feature "
                             "differencing which cancels a per-pair shared atmospheric term. See module "
                             "docstring."),
        "reflectance_scale_confirmed": bool(scale_ok),
    }


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


def stage_oscd(force: bool = False) -> dict:
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== OSCD (Onera Satellite Change Detection) -> {oscd_root()} ===")

    dl = download_and_verify(force=force)
    extract(force=force)

    for name in ARCHIVES:
        rec = record_artifact(
            name=f"oscd-{name.replace('Onera Satellite Change Detection dataset - ', '').replace('.zip', '')}"
                 .strip().lower().replace(" ", "-"),
            source_url=dl[name]["source_url"],
            local_path=_archives_dir() / name,
            license=DATA_LICENSE,
        )
        print(f"[staging]   provenance: {rec.name} sha256={rec.sha256[:16]}... size={rec.byte_size}B")

    split = enumerate_split()
    print(f"\n[staging] Split: {split['n_train']} train / {split['n_test']} test")
    print(f"[staging]   train: {', '.join(split['train_regions'])}")
    print(f"[staging]   test : {', '.join(split['test_regions'])}")
    if split["consistency_problems"]:
        raise RuntimeError(f"OSCD split inconsistency: {split['consistency_problems']}")

    bands = report_bands_and_radiometry(split["train_regions"][0])
    print(f"\n[staging] Bands on the co-registered 10 m grid ({bands['sample_region']}, "
          f"{len(bands['rect_bands_grid'])} bands available):")
    for b, meta in bands["rect_bands_grid"].items():
        print(f"[staging]   {b}: {meta['shape']}  {meta['dtype']}  crs={meta['crs']}")
    print(f"[staging] Native GSD: B02/B03/B04/B08=10 m, B05/B06/B07/B8A/B11/B12=20 m, B01/B09/B10=60 m "
          f"(all resampled to the 10 m grid in imgs_*_rect).")
    print(f"[staging] Production bands used for training: {bands['production_bands_used']}")
    print(f"[staging] Change-pixel fraction per region (severe imbalance): "
          f"overall {100 * bands['change_pixel_fraction_overall']:.2f}%")
    for b, st in bands["dn_stats_over_valid_pixels"].items():
        print(f"[staging]   {b}: median DN {st['median']:.0f} (reflectance ~{st['reflectance_median']:.3f}), "
              f"p1..p99 {st['p1']:.0f}..{st['p99']:.0f}")
    if not bands["reflectance_scale_confirmed"]:
        raise RuntimeError(f"OSCD reflectance-scale check failed: {bands['dn_stats_over_valid_pixels']}")
    print(f"[staging] Reflectance scale confirmed: reflectance = DN / {REFLECTANCE_SCALE:.0f} (offset 0). "
          f"L1C TOA vs geoseek L2A BOA - harmonization recorded in the manifest.")

    section = {
        "purpose": ("Phase 3b: labelled Sentinel-2 change-detection pairs to TRAIN a real "
                    "FC-Siam-diff change model that is exposed through "
                    "geoseek.models.base.ChangeDetectionModel."),
        "dataset": "Onera Satellite Change Detection (OSCD)",
        "n_pairs_total": split["n_train"] + split["n_test"],
        "source": {
            "canonical_home": "https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection",
            "canonical_doi": "10.21227/asqe-7s69",
            "author_page": "https://rcdaudt.github.io/oscd/",
            "download_mirror": f"Hugging Face dataset {HF_REPO_ID} @ {HF_REVISION} "
                               f"(pinned; the mirror the torchgeo OSCD loader resolves against)",
            "archives": {name: dl[name] for name in ARCHIVES},
        },
        "license": DATA_LICENSE,
        "citation": CITATION,
        "split": split,
        "bands_and_radiometry": bands,
        "extracted_root": str(oscd_root()),
    }
    record_analysis_section("oscd", section)
    print(f"\n[staging] OSCD staged + recorded. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-oscd")
    p.add_argument("--force", action="store_true", help="re-download + re-extract even if already staged")
    args = p.parse_args(argv)
    stage_oscd(force=args.force)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

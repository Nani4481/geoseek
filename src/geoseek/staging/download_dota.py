"""Stage the DOTA (Dataset for Object deTection in Aerial images) benchmark for
the Phase 8F object-detection track - STAGING ONLY, no detector trained here.

This is one of the ONLY modules in geoseek allowed to touch the network (see
the project-wide rule in ``README.md`` / ``config.py``). Everything downstream
reads exclusively from the local files staged here under
``data/datasets/dota/``.

Why DOTA and not xView (confirmed live at authoring time, not recalled)
-------------------------------------------------------------------------
xView (0.3 m WorldView-3, 60 classes) would have matched our staged Maxar
imagery's GSD (0.305 m) exactly - it was the PRIORITY candidate. Checked live:

  https://xviewdataset.org/           -> 200 OK, real marketing page, every
                                          "Download" path routes through
                                          "REGISTER FOR THE DIUx xVIEW 2018
                                          DETECTION CHALLENGE" ->
  https://challenge.xviewdataset.org/signup -> 200 OK, a real account-
                                          registration app (not a dead link),
                                          gated by agreeing to Terms and
                                          Conditions before any download link
                                          is issued.

xView has NO no-login mirror of the full dataset: the Hugging Face mirror
found (``CDAO/xview-subset-classification``) is a derived CLASSIFICATION
subset, not the full 847-image detection dataset with the 60-class ontology;
the Kaggle mirror requires a Kaggle account/API key to fetch. Registering a
new account on an external U.S. government-adjacent (DIUx/NGA) challenge
platform on the user's behalf is outside what this staging step does
autonomously - reported honestly rather than silently worked around.
CONCLUSION: xView is NOT directly downloadable without creating an account.
Falling back to DOTA v2.0 per the task's own contingency.

DOTA v2.0 accessibility (also confirmed live, with a real complication)
-------------------------------------------------------------------------
DOTA v2.0 = DOTA v1.0's base images (RELABELED) + NEW extra images, plus
v2.0-only annotations (adds ``airport``/``helipad`` to the 16-class v1.5
ontology). The official page (https://captain-whu.github.io/DOTA/dataset.html)
hosts these in two different places:

  * DOTA-v1.0 / v1.5 base images + annotations -> Google Drive folders
    (``usp=sharing``). Verified LIVE: ``curl`` to all 3 folder URLs (train/
    val/test) returns HTTP 200 with real Drive folder-listing titles ("train
    - Google Drive" etc, not a "Sign in" page) - i.e. genuinely
    NO-LOGIN-REQUIRED public folders.
  * DOTA-v2.0-EXCLUSIVE extra images/annotations -> a WHU SharePoint/OneDrive
    link (primary) or Baidu Netdisk with extraction code ``ck24``
    (secondary). Verified LIVE: the OneDrive link 302-redirects to
    ``login.microsoftonline.com`` (a real Microsoft-account OAuth prompt for
    the WHU tenant) - i.e. genuinely LOGIN-GATED, not freely downloadable.
    Baidu Netdisk share links are well-established to require a Baidu
    account (and often their desktop client / rate limits for non-VIP users)
    to actually pull files, even when the share page itself loads.

CONCLUSION, reported honestly: only DOTA-v1.0 images + DOTA-v1.5 annotations
(a strict superset of v1.0's 15 classes, adding ``container_crane`` -> 16
classes total) are directly staged here, no account needed anywhere in the
chain. The 2 v2.0-EXCLUSIVE classes (``airport``, ``helipad``) and v2.0's
additional new images are NOT staged - they sit behind the same kind of
account wall as xView, just on a different provider (Microsoft/Baidu instead
of DIUx). This is disclosed in the recorded provenance section, not silently
dropped. The vehicle classes this track cares about (``large-vehicle``,
``small-vehicle``) are present in v1.0/v1.5 already - unaffected by this gap.

License
-------
"All images and their associated annotations in DOTA can be used for
academic purposes only, but any commercial use is prohibited" (official page,
quoted verbatim). Google Earth-sourced images additionally carry Google's own
terms of use. Non-commercial - fine for this SIH research/educational
prototype, same posture as OSCD and Maxar Open Data.

Annotation format
------------------
Oriented bounding box (OBB): each instance is one line
``x1,y1,x2,y2,x3,y3,x4,y4,category,difficult`` in a per-image ``.txt`` file,
vertices clockwise from a category-specific "starting point". The 4 points are a
general quadrilateral (hand-clicked corners), NOT an exact rectangle. ``difficult``
is 1/0. Each split ships label archives, some of them axis-aligned derivatives:

  labelTxt-v1.0/labelTxt.zip          OBB, 15 classes   (flat P*.txt)             -> labelTxt-v1.0/
  labelTxt-v1.0/{Train,Val}_Task2_gt  HBB derivative    (inside its own folder)   -> labelTxt-v1.0/<folder>/
  labelTxt-v1.5/DOTA-v1.5_<split>.zip     OBB, 16 classes (flat P*.txt)           -> labelTxt-v1.5/
  labelTxt-v1.5/DOTA-v1.5_<split>_hbb.zip HBB derivative  (flat P*.txt, SAME names) -> labelTxt-v1.5-hbb/

EXTRACTION-COLLISION BUG (found and fixed 2026-09-20). The two v1.5 zips carry flat,
identically named members, and the original extractor unzipped both into
``labelTxt-v1.5/``; the HBB zip came last and silently overwrote every OBB file, so the
staged ``labelTxt-v1.5/`` was 100.00% axis-aligned (210,631/210,631 train, 69,565/69,565
val instances) while the archive ``DOTA-v1.5_train.zip`` is really 95.3% non-axis-aligned.
(v1.0 was never affected: its HBB zip lives in a subfolder; the on-disk v1.0 files are
byte-identical to ``labelTxt.zip``.) ``EXTRACT_TARGET_OVERRIDES`` now keeps colliding
archives apart and :func:`repair_v15_label_dirs` re-extracts from the retained archives.

Split convention
------------------
Official DOTA train/val/test - test images ship WITHOUT public ground truth
(held by the CodaLab/eval-server for submission-based scoring), so only
train (1830 images, DOTA v2.0 paper) and val (593 images, DOTA v2.0 paper;
DOTA-v1.0/v1.5 alone report 458 val images since v2.0 added extra val images
this module does NOT have access to - see the accessibility note above) are
useful to stage for training. This module stages val fully; see
:func:`stage_dota` docstring for train's status.

Re-running after the archives are staged skips the network entirely.
"""

from __future__ import annotations

import sys
import zipfile
from collections import Counter
from pathlib import Path

from geoseek.config import get_settings, print_startup_banner
from geoseek.staging.manifest import record_analysis_section, record_artifact

# --------------------------------------------------------------------------
# Source: DOTA-v1.0 Google Drive folders (no login) - official page links,
# resolved to individual file IDs via `gdown.download_folder(..., skip_download=True)`
# against the live folders at authoring time.
# --------------------------------------------------------------------------
DOTA_PAGE_URL = "https://captain-whu.github.io/DOTA/dataset.html"

GDRIVE_FOLDERS = {
    "train": "1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK",
    "val": "1n5w45suVOyaqY84hltJhIZdtVFD9B224",
    "test": "1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl",
}

# file_id -> relative path under data/datasets/dota/_archives/<split>/...
VAL_FILES = {
    "1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP": "images/part1.zip",
    "1uFwxA4B7H8zcI1oD11bj0U8z88qroMlG": "labelTxt-v1.0/labelTxt.zip",
    "1roMkDBK9753uS5tCmtYlRTyzrObjjJ83": "labelTxt-v1.0/Val_Task2_gt.zip",
    "1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC": "labelTxt-v1.5/DOTA-v1.5_val.zip",
    "1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse": "labelTxt-v1.5/DOTA-v1.5_val_hbb.zip",
}
# train mirrors the same layout across 3 image parts - recorded for a future
# staging pass (see module + stage_dota docstrings); NOT downloaded by this
# module's default run (large: 3 image-zip parts vs val's 1).
TRAIN_FILES = {
    "1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2": "images/part1.zip",
    "1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v": "images/part2.zip",
    "1pEmwJtugIWhiwgBqOtplNUtTG2T454zn": "images/part3.zip",
    "1I-faCP-DOxf6mxcjUTc8mYVPqUgSQxx6": "labelTxt-v1.0/labelTxt.zip",
    "1sS9hveKtYAiTsGVxC4msF5qJjhn3wYpY": "labelTxt-v1.0/Train_Task2_gt.zip",
    "12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX": "labelTxt-v1.5/DOTA-v1.5_train.zip",
    "1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb": "labelTxt-v1.5/DOTA-v1.5_train_hbb.zip",
}

DATA_LICENSE = "Academic use only, commercial use prohibited (DOTA; Google Earth imagery additionally under Google's terms of use)"
CITATION = (
    "Xia, G.S. et al. 'DOTA: A Large-scale Dataset for Object Detection in Aerial Images.' "
    "CVPR 2018. Ding, J. et al. 'Object Detection in Aerial Images: A Large-Scale Benchmark and "
    "Challenges.' TPAMI 2021 (DOTA-v1.5 / v2.0)."
)

CLASSES_V1_0 = ("plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
                "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool")
CLASSES_V1_5 = CLASSES_V1_0 + ("container-crane",)
CLASSES_V2_0_ONLY = ("airport", "helipad")  # NOT staged - see module docstring
VEHICLE_CLASSES = ("large-vehicle", "small-vehicle")

# Explicit split policy, recorded into the manifest so it survives independent
# of any training code that may or may not exist yet: val is HELD OUT for
# evaluation only - it must never be used for training or for threshold
# selection (that would leak the only independent measure of generalization
# this benchmark gives us). Any future ObjectDetectionModel / training script
# must read this split's images from `train` alone; `val` is read-only for
# metrics.
SPLIT_POLICY = {
    "train": "training data - free to use for fitting model weights.",
    "val": ("HELD OUT for evaluation only. Must NOT be used for training or for "
            "threshold/hyperparameter selection (that leaks the one independent "
            "generalization check this benchmark provides). If a dev/threshold-"
            "tuning split is ever needed, carve it out of `train`, not `val`."),
}


# Archives whose members would collide with a sibling archive's (flat, identically named P*.txt) if unzipped
# into the same directory. They get their own target dir. See the "EXTRACTION-COLLISION BUG" note above.
EXTRACT_TARGET_OVERRIDES = {
    "labelTxt-v1.5/DOTA-v1.5_train_hbb.zip": "labelTxt-v1.5-hbb",
    "labelTxt-v1.5/DOTA-v1.5_val_hbb.zip": "labelTxt-v1.5-hbb",
}


def dota_root() -> Path:
    return get_settings().datasets_dir / "dota"


def _archives_dir(split: str) -> Path:
    return dota_root() / "_archives" / split


def _extracted_dir(split: str) -> Path:
    return dota_root() / split


# --------------------------------------------------------------------------
# download (Google Drive, via gdown - handles the large-file
# virus-scan-interstitial confirm-token dance plain `requests` chokes on)
# --------------------------------------------------------------------------


def download_split(split: str, files: dict[str, str], force: bool = False) -> dict[str, dict]:
    import gdown

    adir = _archives_dir(split)
    records: dict[str, dict] = {}
    for file_id, rel_path in files.items():
        dest = adir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if force and dest.exists():
            dest.unlink()
        if not dest.is_file():
            print(f"[staging] Downloading {split}/{rel_path} (gdrive id={file_id}) ...")
            gdown.download(id=file_id, output=str(dest), quiet=False)
        else:
            print(f"[staging] {dest} already present ({dest.stat().st_size / 1e6:.1f} MB) - skipping network.")
        records[rel_path] = {"gdrive_file_id": file_id, "source_url": f"https://drive.google.com/uc?id={file_id}"}
    return records


def extract_split(split: str, files: dict[str, str], force: bool = False) -> None:
    out_dir = _extracted_dir(split)
    adir = _archives_dir(split)
    for rel_path in files.values():
        zpath = adir / rel_path
        # unzip each archive into a subfolder named after its own stem so
        # images/part1.zip + images/part2.zip + images/part3.zip merge into
        # one out_dir/images/ directory (DOTA ships train images pre-split
        # into parts purely to stay under Drive's per-file size limits).
        target = out_dir / EXTRACT_TARGET_OVERRIDES.get(rel_path, str(Path(rel_path).parent))
        marker = target / ".extracted" / zpath.name
        if marker.is_file() and not force:
            continue
        print(f"[staging] Extracting {split}/{rel_path} -> {target} ...")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


# --------------------------------------------------------------------------
# repair: restore the v1.5 OBB labels the extraction collision overwrote
# --------------------------------------------------------------------------


def _axis_aligned_share(texts) -> tuple[int, int]:
    """(axis-aligned instances, total instances) over an iterable of DOTA label-file texts."""
    aligned = total = 0
    for text in texts:
        for ln in text.splitlines():
            p = ln.split()
            if len(p) < 10:
                continue
            c = [float(v) for v in p[:8]]
            total += 1
            aligned += int(len({round(x, 3) for x in c[0::2]}) == 2 and len({round(y, 3) for y in c[1::2]}) == 2)
    return aligned, total


def repair_v15_label_dirs() -> dict:
    """Offline. Re-extract the v1.5 OBB zip into ``labelTxt-v1.5/`` and the HBB zip into ``labelTxt-v1.5-hbb/``
    from the retained archives, then VERIFY the result (OBB dir mostly non-axis-aligned and byte-identical to its
    zip; HBB dir 100% axis-aligned). Idempotent."""
    report: dict = {}
    for split in ("train", "val"):
        adir = _archives_dir(split) / "labelTxt-v1.5"
        pairs = {
            "obb": (adir / f"DOTA-v1.5_{split}.zip", _extracted_dir(split) / "labelTxt-v1.5"),
            "hbb": (adir / f"DOTA-v1.5_{split}_hbb.zip", _extracted_dir(split) / "labelTxt-v1.5-hbb"),
        }
        split_rep: dict = {}
        for kind, (zpath, target) in pairs.items():
            if not zpath.is_file():
                raise FileNotFoundError(f"{zpath} missing - cannot repair {split}/{kind} without the archive")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                zf.extractall(target)
                n_identical = sum(1 for n in members if (target / n).read_bytes() == zf.read(n))
                aligned, total = _axis_aligned_share(zf.read(n).decode("utf-8", "replace") for n in members)
            on_disk = _axis_aligned_share(p.read_text(encoding="utf-8", errors="replace")
                                          for p in target.glob("P*.txt"))
            pct = 100.0 * on_disk[0] / max(on_disk[1], 1)
            ok = (pct < 50.0) if kind == "obb" else (pct == 100.0)
            split_rep[kind] = {"archive": zpath.name, "extracted_to": str(target), "n_members": len(members),
                               "n_byte_identical_to_archive": n_identical, "instances": on_disk[1],
                               "pct_axis_aligned_on_disk": round(pct, 2),
                               "pct_axis_aligned_in_archive": round(100.0 * aligned / max(total, 1), 2), "ok": bool(ok)}
            if not ok or n_identical != len(members):
                raise RuntimeError(f"repair verification failed for {split}/{kind}: {split_rep[kind]}")
        report[split] = split_rep
    record_analysis_section("dota_label_repair", {
        "issue": ("DOTA-v1.5 <split>.zip (OBB) and <split>_hbb.zip (HBB) have flat, identically named members; the original "
                  "extractor unzipped both into labelTxt-v1.5/ so the HBB files overwrote the OBB files (100% axis-aligned)."),
        "fix": "colliding archives now extract to separate dirs (EXTRACT_TARGET_OVERRIDES); on-disk dirs re-extracted from the retained archives.",
        "detected": "2026-09-20, while pre-flighting Phase 8F-2 training chips (boxes on 45-degree trucks were axis-aligned).",
        "v1_0_unaffected": "labelTxt-v1.0/P*.txt are byte-identical to labelTxt.zip (OBB) for all 1411 train + 458 val files.",
        "result": report,
    })
    return report


# --------------------------------------------------------------------------
# report: image count, class histogram (from v1.5 OBB annotations)
# --------------------------------------------------------------------------


def _find_images_dir(split: str) -> Path:
    base = _extracted_dir(split) / "images"
    # Each part*.zip's own internal top-level folder is itself "images/", so
    # extracting into <split>/images/ (named after our own archive layout)
    # produces a double-nested <split>/images/images/*.png - check that first.
    nested = base / "images"
    if nested.is_dir() and any(nested.glob("*.png")):
        return nested
    return base


def enumerate_images(split: str) -> list[str]:
    d = _find_images_dir(split)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.png"))


def _labelTxt_dir(split: str, version: str) -> Path:
    if version == "v1.5":
        # DOTA-v1.5_val.zip extracts to a folder named after the zip stem
        for cand in (_extracted_dir(split) / "labelTxt-v1.5" / f"DOTA-v1.5_{split}",
                     _extracted_dir(split) / "labelTxt-v1.5"):
            if cand.is_dir() and any(cand.glob("*.txt")):
                return cand
        return _extracted_dir(split) / "labelTxt-v1.5"
    return _extracted_dir(split) / "labelTxt-v1.0" / "labelTxt"


def class_histogram(split: str, version: str = "v1.5") -> dict:
    d = _labelTxt_dir(split, version)
    counts: Counter[str] = Counter()
    n_files = 0
    n_images_with_vehicle = 0
    if not d.is_dir():
        return {"labelTxt_dir": str(d), "found": False}
    for txt in sorted(d.glob("*.txt")):
        n_files += 1
        has_vehicle = False
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 10:
                continue  # skips the 2-line imagesource/gsd header some files carry
            category = parts[8]
            counts[category] += 1
            if category in VEHICLE_CLASSES:
                has_vehicle = True
        if has_vehicle:
            n_images_with_vehicle += 1
    return {
        "labelTxt_dir": str(d),
        "found": True,
        "n_label_files": n_files,
        "instance_counts_by_class": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "total_instances": sum(counts.values()),
        "vehicle_instances": sum(counts[c] for c in VEHICLE_CLASSES if c in counts),
        "n_images_with_a_vehicle_instance": n_images_with_vehicle,
    }


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------


def stage_dota(force: bool = False, include_train: bool = False) -> dict:
    """Stage DOTA-v1.0 images + DOTA-v1.5 annotations for val (and optionally train).

    ``include_train=False`` by default: val (1 image-zip part, ~593 imgs in the
    v2.0 paper's count / 458 in the v1.0/v1.5-only count this module actually
    has access to) is staged as the immediately-usable, fully-verified slice.
    Train (1830 images across 3 image-zip parts, several times val's size) is
    equally accessible via the same no-login Google Drive mechanism
    (``TRAIN_FILES`` above) but is a materially larger download - pass
    ``include_train=True`` (or run again later) to also stage it.
    """
    settings = get_settings()
    print_startup_banner(settings)
    print(f"\n=== DOTA (Dataset for Object deTection in Aerial images) -> {dota_root()} ===")

    splits_to_stage = {"val": VAL_FILES}
    if include_train:
        splits_to_stage["train"] = TRAIN_FILES

    staged_splits = {}
    for split, files in splits_to_stage.items():
        print(f"\n[staging] --- {split} ---")
        dl = download_split(split, files, force=force)
        extract_split(split, files, force=force)

        for rel_path, meta in dl.items():
            local_path = _archives_dir(split) / rel_path
            rec = record_artifact(
                name=f"dota-{split}-{rel_path.replace('/', '-').replace('.zip', '')}",
                source_url=meta["source_url"],
                local_path=local_path,
                license=DATA_LICENSE,
            )
            print(f"[staging]   provenance: {rec.name} sha256={rec.sha256[:16]}... size={rec.byte_size}B")

        images = enumerate_images(split)
        hist = class_histogram(split, version="v1.5")
        print(f"[staging]   {len(images)} images extracted")
        print(f"[staging]   class histogram (v1.5, {hist.get('n_label_files', 0)} label files, "
              f"{hist.get('total_instances', 0)} instances):")
        for cls, cnt in hist.get("instance_counts_by_class", {}).items():
            print(f"[staging]     {cls}: {cnt}")
        staged_splits[split] = {"n_images": len(images), "class_histogram": hist,
                                 "split_policy": SPLIT_POLICY[split]}
        if split == "val":
            print(f"[staging]   POLICY: {SPLIT_POLICY['val']}")

    section = {
        "purpose": ("Phase 8F-1: stage a VHR object-detection training dataset (oriented bounding "
                    "boxes incl. vehicle classes) for a future ObjectDetectionModel. "
                    "STAGING ONLY - no detector built/trained here."),
        "dataset": "DOTA (v1.0 images + v1.5 annotations - see accessibility note below for what "
                   "v2.0-exclusive content was NOT staged and why)",
        "xview_accessibility": (
            "xView was the PRIORITY candidate (0.3 m GSD exactly matches our staged Maxar imagery, "
            "0.305 m). Confirmed LIVE: xviewdataset.org and challenge.xviewdataset.org/signup are "
            "both live, working pages, but every download path requires creating an account and "
            "agreeing to Terms and Conditions on the DIUx/NGA challenge platform first - NOT "
            "directly downloadable. No complete no-login mirror was found (a Hugging Face hit is a "
            "derived classification subset, not the full 60-class detection dataset; a Kaggle "
            "mirror needs a Kaggle account). Falling back to DOTA v2.0 per this task's own "
            "contingency plan."
        ),
        "dota_v2_accessibility": (
            "DOTA-v1.0 base images + DOTA-v1.5 annotations (16 classes, superset of v1.0's 15, "
            "adding container-crane) are on public Google Drive folders - confirmed LIVE with no "
            "login wall (HTTP 200, real folder-listing titles, not a sign-in redirect). DOTA-v2.0's "
            "OWN exclusive content (new extra images + the 2 extra classes airport/helipad) lives "
            "on a WHU OneDrive link that 302-redirects to a real login.microsoftonline.com OAuth "
            "prompt (Microsoft-account-gated) plus a Baidu Netdisk mirror (also account-gated by "
            "well-established Baidu Netdisk policy) - NOT directly downloadable, same category of "
            "access barrier as xView, just a different provider. Staged v1.0/v1.5 instead: it is "
            "the largest slice actually accessible with no account anywhere in the chain, and "
            "already carries the large-vehicle/small-vehicle classes this track needs."
        ),
        "split_policy": SPLIT_POLICY,
        "download_source": DOTA_PAGE_URL,
        "license": DATA_LICENSE,
        "citation": CITATION,
        "classes_staged_v1_5": list(CLASSES_V1_5),
        "classes_v2_0_exclusive_not_staged": list(CLASSES_V2_0_ONLY),
        "vehicle_classes": list(VEHICLE_CLASSES),
        "annotation_format": ("Oriented bounding box (OBB): 'x1,y1,x2,y2,x3,y3,x4,y4,category,difficult' "
                              "per instance per line, vertices clockwise from a category-specific "
                              "starting point (a general quadrilateral, not an exact rectangle). difficult in {0,1}. "
                              "OBB: labelTxt-v1.0/ and labelTxt-v1.5/. HBB derivatives (axis-aligned): "
                              "labelTxt-v1.5-hbb/ and labelTxt-v1.0/<Task2 folder>/. NOTE labelTxt-v1.5/ was "
                              "overwritten by HBB by an extraction collision until repaired 2026-09-20 - see "
                              "the dota_label_repair manifest section."),
        "split_convention": ("Official DOTA train/val/test; test ships WITHOUT public ground truth "
                             "(CodaLab eval-server submission scoring only) so not staged here. "
                             "include_train=True stages train too (3x val's image-zip-part count)."),
        "splits_staged": staged_splits,
        "gdrive_folders": GDRIVE_FOLDERS,
    }
    record_analysis_section("dota", section)
    print(f"\n[staging] DOTA staged + recorded. Manifest: {settings.provenance_manifest_path}")
    return section


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="geoseek-stage-dota")
    p.add_argument("--force", action="store_true", help="re-download + re-extract even if already staged")
    p.add_argument("--include-train", action="store_true", help="also stage the (larger) train split")
    p.add_argument("--repair-v15-labels", action="store_true",
                   help="offline: re-extract the v1.5 OBB / HBB label zips into their own dirs and verify")
    args = p.parse_args(argv)
    if args.repair_v15_labels:
        import json
        print(json.dumps(repair_v15_label_dirs(), indent=1))
        return
    stage_dota(force=args.force, include_train=args.include_train)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[staging] FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e

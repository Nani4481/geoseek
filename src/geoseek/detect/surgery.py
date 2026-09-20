"""Warm-start the 8-class detector from the DOTA-pretrained 15-class OBB weights by TRANSPLANTING class rows.

Fine-tuning ``yolo26s-obb.pt`` onto our 8 classes normally re-initialises every classification output (Ultralytics drops
the shape-mismatched last conv). Eight of the pretrained 15 classes are exactly our classes, and the rare ones (helicopter:
574 train instances) are the ones a from-scratch head learns worst. So instead:

  * everything whose shape still matches (backbone, neck, box / angle branches, the classification branches up to their
    last 1x1 conv) is copied as-is, and
  * the last classification conv (``[nc, c3, 1, 1]`` weight + ``[nc]`` bias, in BOTH the one-to-many ``cv3`` and, for
    YOLO26, the one-to-one ``one2one_cv3`` head, at every FPN level) gets its 8 rows copied from the matching pretrained
    class rows (small vehicle -> our small-vehicle, ...), by class NAME.

The result is verifiable, not trusted: for the same feature maps the transplanted model's class logits must equal the
pretrained model's logits at those rows, and every non-class head output must be identical
(:func:`verify_transplant`). It also gives a free "epoch 0" baseline - the pretrained knowledge restricted to our classes.

ultralytics (AGPL-3.0) is imported lazily, inside the functions.
"""

from __future__ import annotations

import copy
from pathlib import Path

from geoseek.detect.classes import CLASS_NAMES, dota15_index_to_ours

CLS_HEAD_ATTRS = ("cv3", "one2one_cv3")     # classification branches; each is a ModuleList over FPN levels


def _head(model):
    return model.model[-1]


def transplant_head(pretrained_ckpt: Path, out_path: Path) -> dict:
    """Build the 8-class model from the 15-class checkpoint and save an Ultralytics-loadable checkpoint."""
    import torch
    import ultralytics
    from ultralytics.nn.tasks import OBBModel

    ck = torch.load(str(pretrained_ckpt), map_location="cpu", weights_only=False)
    m15 = ck["model"].float().eval()
    cfg = copy.deepcopy(m15.yaml)
    cfg["nc"] = len(CLASS_NAMES)

    m8 = OBBModel(cfg, nc=len(CLASS_NAMES), ch=3, verbose=False)
    m8.load(m15, verbose=False)                # copies every tensor whose name AND shape match
    m8.names = {i: n for i, n in enumerate(CLASS_NAMES)}
    m8.yaml["nc"] = len(CLASS_NAMES)

    mapping = dota15_index_to_ours()          # pretrained index -> our index
    h15, h8 = _head(m15), _head(m8)
    copied = 0
    with torch.no_grad():
        for attr in CLS_HEAD_ATTRS:
            b15, b8 = getattr(h15, attr, None), getattr(h8, attr, None)
            if b15 is None or b8 is None:
                continue
            for lvl in range(len(b15)):
                c15, c8 = b15[lvl][-1], b8[lvl][-1]          # the final nn.Conv2d(c3, nc, 1)
                for i15, j in mapping.items():
                    c8.weight[j].copy_(c15.weight[i15])
                    c8.bias[j].copy_(c15.bias[i15])
                    copied += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": -1, "best_fitness": None, "model": copy.deepcopy(m8).half(), "ema": None, "updates": None,
        "optimizer": None, "train_args": {"task": "obb", "model": str(pretrained_ckpt.name)},
        "date": None, "version": ultralytics.__version__,
        "license": ck.get("license", "AGPL-3.0 License (https://ultralytics.com/license)"),
        "docs": ck.get("docs", "https://docs.ultralytics.com"),
    }, str(out_path))
    return {"out": str(out_path), "rows_copied": copied, "mapping_pretrained_to_ours": mapping,
            "n_params": sum(p.numel() for p in m8.parameters())}


def verify_transplant(pretrained_ckpt: Path, transplanted_ckpt: Path, *, n_images: int = 4, seed: int = 0,
                      device: str = "cpu") -> dict:
    """Same features in -> the 8 transplanted class logits equal the pretrained ones, everything else identical."""
    import torch

    m15 = torch.load(str(pretrained_ckpt), map_location="cpu", weights_only=False)["model"].float().eval().to(device)
    m8 = torch.load(str(transplanted_ckpt), map_location="cpu", weights_only=False)["model"].float().eval().to(device)
    mapping = dota15_index_to_ours()

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_images, 3, 256, 256, generator=g).to(device)

    feats: dict[str, list] = {}
    for tag, m in (("m15", m15), ("m8", m8)):
        h = _head(m)
        handle = h.register_forward_pre_hook(lambda mod, inp, tag=tag: feats.__setitem__(tag, [t.detach() for t in inp[0]]))
        with torch.no_grad():
            m(x)
        handle.remove()

    report = {"max_abs_diff_class_rows": 0.0, "max_abs_diff_non_class_outputs": 0.0, "heads_compared": []}
    with torch.no_grad():
        h15, h8 = _head(m15), _head(m8)
        for a, b in zip(feats["m15"], feats["m8"]):      # identical backbone+neck -> identical features
            report["max_abs_diff_non_class_outputs"] = max(report["max_abs_diff_non_class_outputs"], float((a - b).abs().max()))
        for attr in CLS_HEAD_ATTRS:
            b15, b8 = getattr(h15, attr, None), getattr(h8, attr, None)
            if b15 is None or b8 is None:
                continue
            report["heads_compared"].append(attr)
            for lvl in range(len(b15)):
                l15, l8 = b15[lvl](feats["m15"][lvl]), b8[lvl](feats["m8"][lvl])
                for i15, j in mapping.items():
                    report["max_abs_diff_class_rows"] = max(report["max_abs_diff_class_rows"],
                                                            float((l15[:, i15] - l8[:, j]).abs().max()))
        for attr in ("cv2", "one2one_cv2", "cv4", "one2one_cv4"):        # box / angle branches must be untouched
            b15, b8 = getattr(h15, attr, None), getattr(h8, attr, None)
            if b15 is None or b8 is None:
                continue
            for lvl in range(len(b15)):
                d = (b15[lvl](feats["m15"][lvl]) - b8[lvl](feats["m8"][lvl])).abs().max()
                report["max_abs_diff_non_class_outputs"] = max(report["max_abs_diff_non_class_outputs"], float(d))
    report["ok"] = report["max_abs_diff_class_rows"] < 1e-4 and report["max_abs_diff_non_class_outputs"] < 1e-4
    return report

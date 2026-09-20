"""Phase 8F-2: training configuration, run-control and class-head-transplant tests (hermetic: no staged data, no
downloads; the transplant test builds a tiny random OBB model from the packaged ultralytics yaml)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from geoseek.detect.classes import CLASS_NAMES, dota15_index_to_ours
from geoseek.detect.train import RunLock, build_train_args, find_resume_checkpoint, training_finished


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def _cfg(tmp_path, **kw):
    return build_train_args(tmp_path / "dataset.yaml", project=tmp_path / "runs", **kw)


def test_config_is_explicit_and_includes_rotation_amp_and_a_bounded_schedule(tmp_path):
    c = _cfg(tmp_path)
    assert c["degrees"] == 180.0                    # rotation augmentation is mandatory for oriented aerial imagery
    assert c["flipud"] > 0 and c["fliplr"] > 0 and c["mosaic"] == 1.0
    assert c["amp"] is True                          # mixed precision
    assert c["optimizer"] == "AdamW"                 # never 'auto' (which could silently switch to MuSGD)
    assert c["cos_lr"] is True and 0 < c["lrf"] < 1 and c["epochs"] > c["close_mosaic"] > 0
    assert c["warmup_bias_lr"] == 0.0                # required for Adam-family optimisers
    assert c["imgsz"] == 1024 and c["nbs"] % c["batch"] == 0
    assert c["patience"] >= c["epochs"]              # no early stopping decided by the data
    assert c["max_det"] >= 1000                      # dense parking lots


def test_training_yaml_never_mentions_the_official_val(tmp_path):
    from geoseek.detect.convert import write_dataset_yamls

    yamls = write_dataset_yamls(tmp_path)
    c = build_train_args(yamls["train"], project=tmp_path / "runs")
    txt = Path(c["data"]).read_text(encoding="utf-8")
    assert "monitor/monitor.txt" in txt and "val/val.txt" not in txt.replace("monitor/monitor.txt", "")
    assert Path(c["data"]).name == "dataset.yaml"    # the training yaml, not dataset_official_val.yaml


# --------------------------------------------------------------------------
# run control
# --------------------------------------------------------------------------


def _fake_ckpt(path: Path, *, epoch: int, optimizer=True):
    torch = pytest.importorskip("torch")
    torch.save({"epoch": epoch, "optimizer": {"state": 1} if optimizer else None, "model": None}, str(path))


def test_resume_prefers_last_and_skips_finished_or_stripped_checkpoints(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    assert find_resume_checkpoint(run, epochs=10) is None                     # nothing yet
    _fake_ckpt(run / "weights" / "last.pt", epoch=3)
    assert find_resume_checkpoint(run, epochs=10) == run / "weights" / "last.pt"
    _fake_ckpt(run / "weights" / "last.pt", epoch=9)                          # the FINAL epoch: nothing left to resume
    assert find_resume_checkpoint(run, epochs=10) is None
    _fake_ckpt(run / "weights" / "last.pt", epoch=-1, optimizer=False)        # stripped by a finished run
    assert find_resume_checkpoint(run, epochs=10) is None


def test_resume_falls_back_to_a_periodic_checkpoint_when_last_is_corrupt(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"not a torch file (process killed mid-save)")
    _fake_ckpt(run / "weights" / "epoch2.pt", epoch=2)
    assert find_resume_checkpoint(run, epochs=10) == run / "weights" / "epoch2.pt"


def test_resume_ignores_an_epochN_checkpoint_saved_on_the_last_epoch(tmp_path):
    # the first supervisor was fooled by exactly this: epoch2.pt (last epoch, optimizer still inside) -> 'nothing to resume'
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    _fake_ckpt(run / "weights" / "epoch2.pt", epoch=2)
    assert find_resume_checkpoint(run, epochs=3) is None


def test_training_finished_needs_the_last_epoch_logged_and_a_last_checkpoint(tmp_path):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    log = run / "progress.jsonl"
    log.write_text(json.dumps({"epoch": 2}) + "\n", encoding="utf-8")
    assert not training_finished(run, 3)
    log.write_text(json.dumps({"epoch": 2}) + "\n" + json.dumps({"epoch": 3}) + "\n", encoding="utf-8")
    assert not training_finished(run, 3)                                      # logged, but the checkpoint is not there yet
    (run / "weights" / "last.pt").write_bytes(b"x")
    assert training_finished(run, 3)


def test_run_lock_blocks_a_second_supervisor_and_takes_over_a_stale_lock(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    lock = run / "supervisor.lock"
    lock.write_text(str(os.getppid()))                                        # a live process that is not us
    with pytest.raises(SystemExit):
        with RunLock(run):
            pass
    assert lock.exists()                                                      # the refused launch must not steal the lock
    lock.write_text("999999999")                                              # dead pid -> stale -> taken over
    with RunLock(run):
        assert lock.read_text().strip() == str(os.getpid())
    assert not lock.exists()                                                  # released on exit


# --------------------------------------------------------------------------
# class-head transplant (tiny random OBB model built from the packaged yaml)
# --------------------------------------------------------------------------


def test_transplant_head_is_exact_for_the_kept_classes_and_verifies(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import OBBModel

    from geoseek.detect.classes import DOTA15_NAMES
    from geoseek.detect.surgery import transplant_head, verify_transplant

    torch.manual_seed(0)
    m15 = OBBModel("yolo26n-obb.yaml", nc=15, ch=3, verbose=False)
    m15.names = {i: n for i, n in enumerate(DOTA15_NAMES)}
    m15 = m15.half().float()                        # fp16-exact weights, like a real released checkpoint (stored half)
    src = tmp_path / "fake15.pt"
    torch.save({"model": m15, "train_args": {}, "license": "test"}, str(src))

    out = tmp_path / "init8.pt"
    rep = transplant_head(src, out)
    assert rep["rows_copied"] == len(CLASS_NAMES) * 3 * 2                     # 8 classes x 3 FPN levels x (o2m + o2o heads)
    assert rep["mapping_pretrained_to_ours"] == dota15_index_to_ours()

    v = verify_transplant(src, out, n_images=2)
    assert v["ok"] and v["max_abs_diff_class_rows"] == 0.0 and v["max_abs_diff_non_class_outputs"] == 0.0
    assert set(v["heads_compared"]) == {"cv3", "one2one_cv3"}

    m8 = torch.load(str(out), map_location="cpu", weights_only=False)["model"]
    assert m8.names == {i: n for i, n in enumerate(CLASS_NAMES)} and m8.model[-1].nc == len(CLASS_NAMES)

    # and it is NOT a no-op: perturbing the SOURCE class row must break verification. (A random-init network's feature
    # maps are ~0, so a WEIGHT perturbation would be invisible here; the bias always reaches the logit.)
    h = m15.model[-1]
    with torch.no_grad():
        h.cv3[0][-1].bias[10].add_(1.0)                                        # pretrained 'small vehicle'
    torch.save({"model": m15, "train_args": {}, "license": "test"}, str(src))
    assert not verify_transplant(src, out, n_images=2)["ok"]

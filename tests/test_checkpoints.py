import os
import stat
import subprocess
import sys

import pandas as pd
import torch

from plasticity import checkpoints
from plasticity.checkpoints import (append_manifest, git_version, load_checkpoint,
                                    load_training_state, save_checkpoint)
from plasticity.config import RunConfig
from plasticity.models import resnet18_cifar


def test_roundtrip_same_outputs(tmp_path):
    m = resnet18_cifar().eval(); x = torch.rand(2, 3, 32, 32)
    save_checkpoint(tmp_path / "s.pt", m, RunConfig("cifar_fresh"), step=8, phase="train",
                    phase_step=8, epoch=0.02, data_recipe={"dataset": "cifar10"})
    m2, info = load_checkpoint(tmp_path / "s.pt")
    assert torch.allclose(m(x), m2(x)) and info["step"] == 8
    assert info["config"]["lr"] == 0.05 and "commit" in info["code_version"]
    assert info["full_state"] is False
    assert type(info["config"]) is dict
    assert info["run_id"] and info["data_recipe"] == {"dataset": "cifar10"}
    assert "model_state" not in info


def test_load_checkpoint_on_cpu(tmp_path):
    m = resnet18_cifar().to(torch.float16).to(torch.float32)   # mimics AMP-trained weights
    save_checkpoint(tmp_path / "s.pt", m, RunConfig("cifar_fresh"), step=0, phase="train",
                    phase_step=0, epoch=0.0, data_recipe={})
    model, _ = load_checkpoint(tmp_path / "s.pt", device="cpu")
    assert next(model.parameters()).device.type == "cpu"
    assert not model.training


def test_full_state_restores_optimizer_and_rng(tmp_path):
    cfg = RunConfig("cifar_fresh")
    m = resnet18_cifar()
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    m(torch.rand(2, 3, 32, 32)).sum().backward()
    opt.step()
    save_checkpoint(tmp_path / "f.pt", m, cfg, step=1, phase="train", phase_step=1, epoch=0.0,
                    data_recipe={}, optimizer=opt, extra_state={"k": 3})
    a = torch.rand(1)
    _, info = load_checkpoint(tmp_path / "f.pt")
    assert info["full_state"] is True and "optimizer_state" not in info

    m2 = resnet18_cifar()
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.05, momentum=0.9)
    extra = load_training_state(tmp_path / "f.pt", m2, opt2, "cpu")
    b = torch.rand(1)
    assert torch.equal(a, b) and extra == {"k": 3}
    for p, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p, p2)
        assert torch.equal(opt.state[p]["momentum_buffer"], opt2.state[p2]["momentum_buffer"])


def test_load_training_state_rejects_weights_only(tmp_path):
    m = resnet18_cifar()
    save_checkpoint(tmp_path / "s.pt", m, RunConfig("cifar_fresh"), step=0, phase="train",
                    phase_step=0, epoch=0.0, data_recipe={})
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    try:
        load_training_state(tmp_path / "s.pt", m, opt, "cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_saved_file_is_group_readable_and_atomic(tmp_path):
    old = os.umask(0o022)
    try:
        save_checkpoint(tmp_path / "s.pt", resnet18_cifar(), RunConfig("cifar_fresh"), step=0,
                        phase="train", phase_step=0, epoch=0.0, data_recipe={})
        append_manifest(tmp_path, {"path": "x"})
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "s.pt").stat().st_mode) == 0o644
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoints", "s.pt"]


def test_git_version_outside_repo(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(checkpoints.subprocess, "run", boom)
    assert git_version() == {"commit": "unknown", "dirty": True}


def test_git_version_in_repo():
    v = git_version()
    assert isinstance(v["commit"], str) and isinstance(v["dirty"], bool)


def test_manifest_appends_rows(tmp_path):
    row = {"path": "checkpoints/cifar_fresh/r/step_000000008.pt", "run_id": "r",
           "experiment": "cifar_fresh", "phase": "train", "step": 8, "phase_step": 8,
           "epoch": 0.02, "full_state": False, "commit": "abc123",
           "saved_at": "2026-01-01T00:00:00", "ignored_extra": 1}
    append_manifest(tmp_path, row)
    append_manifest(tmp_path, {**row, "step": 16, "full_state": True})
    df = pd.read_csv(tmp_path / "checkpoints" / "manifest.csv")
    assert len(df) == 2 and list(df.columns)[:3] == ["path", "run_id", "experiment"]
    assert list(df.columns) == ["path", "run_id", "experiment", "phase", "step", "phase_step",
                                "epoch", "full_state", "commit", "saved_at"]
    assert list(df["step"]) == [8, 16]

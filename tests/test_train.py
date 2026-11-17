"""Training loop tests, all on CPU with fake data.

ResNet18 on a laptop CPU costs ~0.4 s per training step (batch 16) and ~0.01 s per evaluated
image, so the tests share one uninterrupted CIFAR run (and its cached test sets), evaluate
10 images per test set, and use a 32-image train set (2 steps per epoch) for colored MNIST.
"""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

from plasticity import train as train_mod
from plasticity.checkpoints import load_checkpoint
from plasticity.config import RunConfig, run_dirs
from plasticity.data import build_test_sets
from plasticity.metrics import COLUMNS, evaluate
from plasticity.train import main, train

REPO = Path(__file__).resolve().parents[1]
steps_per_epoch_tiny = 200 // 16  # 12


def tiny(experiment, tmp_path, **overrides):
    values = dict(root=str(tmp_path), batch_size=16, train_subset=200, eval_size=10,
                  total_steps=20, phase1_epochs=2, phase2_epochs=1, durations=[0, 1, 2],
                  amp=False)
    return RunConfig(experiment, **{**values, **overrides})


@pytest.fixture(scope="module")
def cifar_run(tmp_path_factory, fake_raw):
    """One uninterrupted 20-step CIFAR run shared by several tests."""
    root = tmp_path_factory.mktemp("cifar")
    cfg = tiny("cifar_fresh", root)
    logs = train(cfg, raw=fake_raw("cifar"), device="cpu")
    return cfg, logs


def reuse_test_sets(cifar_run, tmp_path):
    """Copy the shared run's cached CIFAR test sets (same cache key) to skip rebuilding."""
    shutil.copytree(Path(cifar_run[0].root) / "data" / "testsets",
                    tmp_path / "data" / "testsets")


def test_cifar_smoke_writes_logs_checkpoints_manifest(cifar_run):
    cfg, logs = cifar_run
    root = Path(cfg.root)
    df = pd.read_csv(logs / "metrics.csv")
    assert list(df.columns) == COLUMNS
    assert set(df.test_set) == {"real", "gaussian", "independent"}
    assert set(df.phase) == {"train"} and set(df.run_id) == {"resnet18_a1.0_lr0.05_wd0.0005_bs16_seed0"}
    assert df.global_step.min() == 0 and df.global_step.max() == 20
    assert not df.duplicated(["global_step", "test_set"]).any()
    assert df.loc[df.global_step == 0, "train_loss_avg"].isna().all()
    assert df.loc[df.global_step > 0, "train_loss_avg"].notna().all()
    assert ((df.epoch - df.global_step / steps_per_epoch_tiny).abs() < 1e-9).all()
    ckpts = sorted((root / "checkpoints" / "cifar_fresh").glob("*/step_*.pt"))
    assert ckpts and (root / "checkpoints" / "manifest.csv").exists()
    assert ckpts[0].name == "step_000000000.pt" and ckpts[-1].name == "step_000000020.pt"
    assert ckpts[-1].stat().st_mode & 0o044 == 0o044          # group/other readable
    manifest = pd.read_csv(root / "checkpoints" / "manifest.csv")
    assert len(manifest) == len(ckpts)
    assert not manifest.path.str.startswith("/").any()
    assert all((root / p).exists() for p in manifest.path)
    assert manifest.full_state.tolist()[-1] and not manifest.full_state.tolist()[0]
    for d in (run_dirs(cfg)["ckpt"], logs):
        assert json.loads((d / "run_config.json").read_text())["experiment"] == "cifar_fresh"
    log_text = (logs / "train.log").read_text()
    assert "device cpu" in log_text and "resnet18_" in log_text and "real " in log_text
    _, info = load_checkpoint(ckpts[-1])
    assert info["full_state"] is True                         # end of run is full state
    recipe = info["data_recipe"]
    assert recipe["experiment"] == "cifar_fresh" and recipe["testset_seed"] == 1234
    assert len(recipe["colors"]) == 10 and recipe["eval_size"] == 10
    assert recipe["train_subset"] == 200 and recipe["color_corr"] is None


def test_reloaded_checkpoint_reproduces_logged_accuracy(cifar_run, fake_raw):
    cfg, logs = cifar_run
    df = pd.read_csv(logs / "metrics.csv")
    for name in ("step_000000016.pt", "step_000000020.pt"):
        model, info = load_checkpoint(run_dirs(cfg)["ckpt"] / name)
        res = evaluate(model, build_test_sets(cfg, fake_raw("cifar")), "cpu")
        rows = df[df.global_step == info["step"]].set_index("test_set")
        assert len(rows) == 3
        for test_set, r in res.items():
            assert abs(r["acc"] - rows.loc[test_set, "acc"]) < 1e-6
            assert abs(r["loss"] - rows.loc[test_set, "loss"]) < 1e-4


def test_cmnist_phase1_then_phase2_branches(tmp_path, fake_raw):
    spe = 32 // 16
    p1 = tiny("cmnist_phase1", tmp_path, train_subset=32, save_schedule="sparse")
    logs1 = train(p1, raw=fake_raw("mnist"), device="cpu")
    df1 = pd.read_csv(logs1 / "metrics.csv")
    assert set(df1.phase) == {"phase1"} and df1.global_step.max() == 2 * spe
    _, info1 = load_checkpoint(run_dirs(p1)["ckpt"] / "step_000000000.pt")
    assert info1["data_recipe"]["color_corr"] == 1.0
    for n in [0, 1, 2]:
        cfg = tiny("cmnist_phase2", tmp_path, train_subset=32, branch_n=n,
                   save_schedule="sparse")
        logs = train(cfg, raw=fake_raw("mnist"), device="cpu")
        df = pd.read_csv(logs / "metrics.csv")
        assert set(df.phase) == {"phase2"} and df.phase_step.min() == 0
        assert df.global_step.min() == n * spe
        assert df.global_step.max() == (n + 1) * spe
        assert (df.global_step - df.phase_step == n * spe).all()
        assert set(df.test_set) == {"color_matched", "color_random", "color_conflict",
                                    "gaussian"}
        assert df.loc[df.test_set == "color_conflict", "acc_color_label"].notna().all()
        assert df.loc[df.test_set != "color_conflict", "acc_color_label"].isna().all()
        # phase-2 step 0 evaluates exactly the branch weights phase 1 logged at that step
        start1 = df1[df1.global_step == n * spe].set_index("test_set").acc
        start2 = df[df.phase_step == 0].set_index("test_set").acc
        assert (start1 - start2).abs().max() < 1e-6
        # distance from init is measured from phase 1's step-0 weights
        d0 = df[df.phase_step == 0].dist_from_init.iloc[0]
        assert (d0 == 0) == (n == 0)
        ckpts = sorted(run_dirs(cfg)["ckpt"].glob("step_*.pt"))
        assert [c.name for c in ckpts] == [f"step_{(n + 1) * spe:09d}.pt"]
        _, info = load_checkpoint(ckpts[-1])
        assert info["phase"] == "phase2" and info["phase_step"] == spe
        assert info["step"] == (n + 1) * spe and info["data_recipe"]["color_corr"] == 0.0


def test_phase2_missing_branch_checkpoint_errors(tmp_path, fake_raw):
    with pytest.raises(FileNotFoundError, match="step_000000012.pt"):
        train(tiny("cmnist_phase2", tmp_path, branch_n=1), raw=fake_raw("mnist"), device="cpu")


def test_phase2_requires_branch_n(tmp_path, fake_raw):
    with pytest.raises(ValueError, match="branch_n"):
        train(tiny("cmnist_phase2", tmp_path), raw=fake_raw("mnist"), device="cpu")


def test_too_few_examples_for_one_batch_errors(tmp_path, fake_raw):
    with pytest.raises(ValueError, match="batch"):
        train(tiny("cifar_fresh", tmp_path, train_subset=10), raw=fake_raw("cifar"),
              device="cpu")


def test_resume_continues_without_duplicate_rows(tmp_path, fake_raw, cifar_run):
    reuse_test_sets(cifar_run, tmp_path)
    cfg = tiny("cifar_fresh", tmp_path, full_state_every_s=0)   # full save at every check
    train(replace(cfg, total_steps=10), raw=fake_raw("cifar"), device="cpu")
    logs = train(cfg, raw=fake_raw("cifar"), device="cpu")       # same run id, longer
    df = pd.read_csv(logs / "metrics.csv")
    assert not df.duplicated(["global_step", "test_set"]).any() and df.global_step.max() == 20
    assert "resumed from step 10" in (logs / "train.log").read_text()
    assert df.wall_time_s.is_monotonic_increasing                # wall time carried over
    # same batches and RNG after resume: the step-20 weights equal the uninterrupted run's
    resumed, _ = load_checkpoint(run_dirs(cfg)["ckpt"] / "step_000000020.pt")
    straight, _ = load_checkpoint(run_dirs(cifar_run[0])["ckpt"] / "step_000000020.pt")
    for (k, va), vb in zip(resumed.state_dict().items(), straight.state_dict().values()):
        assert torch.equal(va, vb), k
    # a finished run is not retrained
    n_rows = len(df)
    train(cfg, raw=fake_raw("cifar"), device="cpu")
    assert len(pd.read_csv(logs / "metrics.csv")) == n_rows


def test_cli_main(tmp_path, fake_raw, monkeypatch, cifar_run):
    reuse_test_sets(cifar_run, tmp_path)
    monkeypatch.setattr(train_mod, "load_raw", lambda cfg: fake_raw("cifar"))
    logs = main(["--config", str(REPO / "configs" / "cifar_fresh.json"), "--root", str(tmp_path),
                 "--batch-size", "16", "--train-subset", "200", "--eval-size", "10",
                 "--total-steps", "2", "--no-amp", "--save-schedule", "sparse",
                 "--durations", "0", "1", "--lr", "0.01"])
    df = pd.read_csv(Path(logs) / "metrics.csv")
    assert df.global_step.max() == 2 and set(df.lr) == {0.01}
    cfg = json.loads((Path(logs) / "run_config.json").read_text())
    assert cfg["durations"] == [0, 1] and cfg["amp"] is False and cfg["batch_size"] == 16
    assert cfg["root"] == str(tmp_path)


def test_cli_requires_config():
    with pytest.raises(SystemExit):
        main([])


def test_resume_does_not_resave_replayed_checkpoints(tmp_path, fake_raw):
    """A job killed after its last full save replays some steps on resume; files written
    before the kill (which match the logged rows) must not be overwritten or re-listed."""
    cfg = tiny("cmnist_phase1", tmp_path, train_subset=32)  # spe 2, total 4, dense saves
    train(cfg, raw=fake_raw("mnist"), device="cpu")          # full state at steps 2 and 4
    ckpt, root = run_dirs(cfg)["ckpt"], Path(cfg.root)
    # simulate a kill just before step 4: drop its file, manifest row and metrics rows
    (ckpt / "step_000000004.pt").unlink()
    man_path = root / "checkpoints" / "manifest.csv"
    man = pd.read_csv(man_path)
    man[man.step != 4].to_csv(man_path, index=False)
    met_path = run_dirs(cfg)["logs"] / "metrics.csv"
    met = pd.read_csv(met_path)
    met[met.global_step < 4].to_csv(met_path, index=False)
    _, info3 = load_checkpoint(ckpt / "step_000000003.pt")
    assert info3["full_state"] is False                      # weights-only, in replay window
    before = (ckpt / "step_000000003.pt").read_bytes()
    mtime = (ckpt / "step_000000003.pt").stat().st_mtime_ns

    logs = train(cfg, raw=fake_raw("mnist"), device="cpu")   # resumes from step 2
    assert "resumed from step 2" in (logs / "train.log").read_text()
    man = pd.read_csv(man_path)
    assert not man.path.duplicated().any()
    assert sorted(man.step) == [0, 1, 2, 3, 4]
    assert (ckpt / "step_000000003.pt").read_bytes() == before
    assert (ckpt / "step_000000003.pt").stat().st_mtime_ns == mtime
    assert (ckpt / "step_000000004.pt").exists()
    df = pd.read_csv(met_path)
    assert not df.duplicated(["global_step", "test_set"]).any() and df.global_step.max() == 4

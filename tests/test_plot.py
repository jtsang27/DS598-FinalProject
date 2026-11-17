import math

import numpy as np
import pandas as pd
import pytest

from plasticity.plot import (
    load_runs,
    main,
    plasticity_table,
    plot_cifar,
    plot_cmnist_branch,
    plot_cmnist_overlay,
)

COLUMNS = ["run_id", "experiment", "phase", "global_step", "phase_step", "epoch", "test_set",
           "loss", "acc", "acc_color_label", "lr", "train_loss_avg", "weight_norm",
           "dist_from_init", "wall_time_s"]
CIFAR_BASE = "resnet18_a1.0_lr0.1_wd0.0005_bs128"
CMNIST_BASE = CIFAR_BASE + "_color0.90"
STEPS_PER_EPOCH = 50


def _row(run_id, experiment, phase, gstep, pstep, test_set, acc, loss=1.0, color=np.nan):
    return {"run_id": run_id, "experiment": experiment, "phase": phase, "global_step": gstep,
            "phase_step": pstep, "epoch": gstep // STEPS_PER_EPOCH, "test_set": test_set,
            "loss": loss, "acc": acc, "acc_color_label": color, "lr": 0.1,
            "train_loss_avg": 1.0, "weight_norm": 10.0, "dist_from_init": 1.0, "wall_time_s": 1.0}


def synthetic_cifar(seeds=(0, 1)):
    grid = [0] + [2 ** k for k in range(0, 11)]
    rows = []
    for seed in seeds:
        run_id = f"{CIFAR_BASE}_seed{seed}"
        for s in grid:
            for i, ts in enumerate(("real", "gaussian", "independent")):
                acc = min(0.95, 0.1 + 0.1 * math.log2(s + 1) * (1 - 0.1 * i)) + 0.01 * seed
                rows.append(_row(run_id, "cifar_fresh", "train", s, s, ts, acc,
                                 loss=2.3 - 0.2 * math.log2(s + 1)))
    return pd.DataFrame(rows, columns=COLUMNS)


def synthetic_cmnist(control_hits_target_at=100, branch_hits=None, seeds=(0,)):
    """Control final color_random acc is 0.9, so the 0.9 target is 0.81.

    A curve crosses the target exactly at its given phase_step (None = never)."""
    branch_hits = {4: 200, 16: None} if branch_hits is None else branch_hits
    hits = {0: control_hits_target_at, **branch_hits}
    target = 0.9 * 0.9
    p2_grid = sorted({0, *(2 ** k for k in range(0, 11)), *(h for h in hits.values() if h)})
    rows = []
    for seed in seeds:
        p1 = f"{CMNIST_BASE}_seed{seed}"
        p1_steps = sorted({0, *(2 ** k for k in range(0, 11)),
                           *(n * STEPS_PER_EPOCH for n in hits)})
        for g in p1_steps:
            for ts in ("color_matched", "color_random", "color_conflict", "gaussian"):
                rows.append(_row(p1, "cmnist_phase1", "phase1", g, g, ts, 0.5,
                                 color=0.7 if ts == "color_conflict" else np.nan))
        for n, hit in hits.items():
            rid = f"{p1}_N{n}ep"
            for s in p2_grid:
                for ts in ("color_matched", "color_random", "color_conflict", "gaussian"):
                    if ts == "color_random":
                        if hit is None:
                            acc = 0.05
                        elif n == 0 and s == p2_grid[-1]:
                            acc = 0.9
                        else:
                            acc = 0.05 if s < hit else (target if s == hit else 0.9)
                    else:
                        acc = 0.5
                    rows.append(_row(rid, "cmnist_phase2", "phase2", n * STEPS_PER_EPOCH + s, s,
                                     ts, acc, color=0.6 if ts == "color_conflict" else np.nan))
    return pd.DataFrame(rows, columns=COLUMNS)


def _write_logs(root, folder, df):
    for rid, g in df.groupby("run_id"):
        d = root / "logs" / folder / rid
        d.mkdir(parents=True)
        g.to_csv(d / "metrics.csv", index=False)


def test_plot_cifar_writes_png(tmp_path):
    assert plot_cifar(synthetic_cifar(), tmp_path / "a.png").stat().st_size > 0


def test_plot_cifar_single_seed(tmp_path):
    assert plot_cifar(synthetic_cifar(seeds=(0,)), tmp_path / "a.png").stat().st_size > 0


def test_plasticity_ratio(tmp_path):
    df = synthetic_cmnist(control_hits_target_at=100, branch_hits={4: 200, 16: None})
    t = plasticity_table(df).set_index("branch_n")
    assert t.loc[0, "ratio"] == 1.0 and t.loc[4, "ratio"] == 2.0
    assert t.loc[0, "steps_to_target"] == 100 and t.loc[0, "control_steps"] == 100
    assert t.loc[4, "reached"] and not t.loc[16, "reached"]
    assert pd.isna(t.loc[16, "ratio"]) and pd.isna(t.loc[16, "steps_to_target"])
    for col in ("setting", "steps_to_target", "control_steps", "ratio", "reached"):
        assert col in plasticity_table(df).columns


def test_plasticity_table_averages_seeds_and_target_frac():
    df = synthetic_cmnist(100, {4: 200}, seeds=(0, 1))
    t = plasticity_table(df).set_index("branch_n")
    assert t.loc[4, "ratio"] == 2.0
    assert plasticity_table(df).setting.nunique() == 1
    assert "seed" not in plasticity_table(df).setting.iloc[0]
    # a lower target is reached at or before the default one
    t2 = plasticity_table(df, target_frac=0.1).set_index("branch_n")
    assert t2.loc[0, "steps_to_target"] <= t.loc[0, "steps_to_target"]


def test_plasticity_table_requires_control():
    df = synthetic_cmnist(100, {4: 200})
    df = df[~df.run_id.str.endswith("_N0ep")]
    with pytest.raises(ValueError, match="resnet18_a1.0"):
        plasticity_table(df)


def test_plot_cmnist_branch_and_overlay(tmp_path):
    df = synthetic_cmnist(100, {4: 200, 16: None}, seeds=(0, 1))
    assert plot_cmnist_branch(df, 4, tmp_path / "b.png").stat().st_size > 0
    assert plot_cmnist_branch(df, 0, tmp_path / "b0.png").stat().st_size > 0
    assert plot_cmnist_overlay(df, tmp_path / "o.png").stat().st_size > 0


def test_load_runs(tmp_path):
    _write_logs(tmp_path, "cmnist", synthetic_cmnist(seeds=(0, 1)))
    df = load_runs(tmp_path / "logs", "cmnist")
    assert df.run_id.nunique() == 8  # 2 seeds x (phase 1 + N in {0, 4, 16})
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "logs", "cifar_fresh")


def test_cli_writes_all_figures(tmp_path, capsys):
    _write_logs(tmp_path, "cmnist", synthetic_cmnist(100, {4: 200, 16: None}, seeds=(0, 1)))
    main(["--root", str(tmp_path), "--experiment", "cmnist"])
    fig = tmp_path / "figures" / "cmnist"
    for name in ("overlay.png", "branch_N0.png", "branch_N4.png", "branch_N16.png",
                 "plasticity_table.csv"):
        assert (fig / name).stat().st_size > 0
    assert "ratio" in capsys.readouterr().out
    assert "ratio" in pd.read_csv(fig / "plasticity_table.csv").columns


def test_cli_cifar(tmp_path):
    _write_logs(tmp_path, "cifar_fresh", synthetic_cifar())
    main(["--root", str(tmp_path), "--experiment", "cifar_fresh"])
    assert (tmp_path / "figures" / "cifar_fresh" / "curves.png").stat().st_size > 0


def test_cli_no_metrics_exits_nonzero(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--root", str(tmp_path), "--experiment", "cmnist"])
    assert e.value.code not in (0, None)
    assert "metrics" in capsys.readouterr().err

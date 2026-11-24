"""Figures and the plasticity table, built from the per-run ``metrics.csv`` logs."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CIFAR_SETS = ("real", "gaussian", "independent")
CMNIST_SETS = ("color_matched", "color_random", "color_conflict", "gaussian")
CHANCE = 0.1
LINESTYLES = ("-", "--", ":", "-.")
# Loss before this step is a start-up instability (lr 0.05 from step 0, no warmup), not part of
# the analysis; loss panels leave it out. Phase-2 rows are always kept (they follow the switch).
STARTUP_STEPS = 64
# Steps after the switch shown in the switch figure.
SWITCH_WINDOW = 512
STARTUP_NOTE = f"steps < {STARTUP_STEPS} not shown (start-up instability)"
_BRANCH_RE = re.compile(r"_N(\d+)ep$")
_SEED_RE = re.compile(r"_seed\d+")


def _setting(run_id: str) -> str:
    return _SEED_RE.sub("", _BRANCH_RE.sub("", run_id))


def _branch_n(run_id: str) -> int | None:
    m = _BRANCH_RE.search(run_id)
    return int(m.group(1)) if m else None


def load_runs(logs_root: Path, experiment_folder: str) -> pd.DataFrame:
    """Concatenate every ``<logs_root>/<folder>/<run_id>/metrics.csv``.

    Adds ``setting`` (run_id without seed / branch suffix) and ``branch_n``
    (Phase-2 branch, NaN elsewhere). Raises FileNotFoundError if none exist."""
    files = sorted((Path(logs_root) / experiment_folder).glob("*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"no metrics.csv files under {Path(logs_root) / experiment_folder}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return _annotate(df)


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setting"] = df["run_id"].map(_setting)
    df["branch_n"] = df["run_id"].map(_branch_n)
    return df


def _x(steps) -> np.ndarray:
    """log2-axis position: step 0 is drawn at 0.5."""
    s = np.asarray(steps, dtype=float)
    return np.where(s == 0, 0.5, s)


def _style_log2(ax, max_step: float) -> None:
    ax.set_xscale("log", base=2)
    top = int(np.ceil(np.log2(max(max_step, 1))))
    ticks = [0.5] + [2.0 ** k for k in range(0, top + 1, max(1, top // 8))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{int(t)}" for t in ticks[1:]])
    ax.minorticks_off()
    ax.set_xlim(0.5 * 0.9, max(ticks[-1], max_step) * 1.1)


def _mean_std(df: pd.DataFrame, by: str, col: str):
    g = df.groupby(by)[col]
    mean, std, n = g.mean(), g.std(ddof=1), g.count()
    return mean.index.to_numpy(), mean.to_numpy(), std.to_numpy(), int(n.max()) if len(n) else 0


def _line(ax, df, xcol, ycol, label, color, ls="-", lw=1.5, band=True, log_x=True):
    xs, mean, std, n = _mean_std(df, xcol, ycol)
    x = _x(xs) if log_x else xs
    ax.plot(x, mean, color=color, ls=ls, lw=lw, label=label)
    if band and n > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)


def _drop_startup(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for loss panels: everything except pre-switch steps < STARTUP_STEPS."""
    return df[(df["phase"] == "phase2") | (df["global_step"] >= STARTUP_STEPS)]


def _style_loss(ax) -> None:
    ax.set_yscale("log")
    ax.text(0.01, 0.98, STARTUP_NOTE, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="gray")


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out


def _colors(names):
    cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {n: cyc[i % len(cyc)] for i, n in enumerate(names)}


def plot_cifar(df: pd.DataFrame, out: Path) -> Path:
    return _save(_draw_cifar(df), out)


def _draw_cifar(df: pd.DataFrame):
    """Accuracy and loss vs global step (log2 axis), one line per test set."""
    df = df if "setting" in df else _annotate(df)
    settings = sorted(df["setting"].unique())
    colors = _colors([t for t in CIFAR_SETS if t != "real"])
    colors["real"] = "black"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ("acc", "loss")):
        data = df if col == "acc" else _drop_startup(df)
        for si, setting in enumerate(settings):
            ls = LINESTYLES[si % len(LINESTYLES)]
            for ts in CIFAR_SETS:
                sub = data[(data["setting"] == setting) & (data["test_set"] == ts)]
                if sub.empty:
                    continue
                label = ts if len(settings) == 1 else f"{ts} ({setting})"
                _line(ax, sub, "global_step", col, label, colors[ts], ls, band=col == "acc")
        if col == "acc":
            ax.axhline(CHANCE, color="gray", ls="--", lw=1, label="chance")
        else:
            _style_loss(ax)
        _style_log2(ax, df["global_step"].max())
        ax.set_xlabel("global step (log2; 0 drawn at 0.5)")
        ax.set_ylabel("accuracy" if col == "acc" else "loss")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    return fig


def _switch(p2: pd.DataFrame) -> float:
    return float(p2["global_step"].min())


def plot_cmnist_branch(df: pd.DataFrame, n: int, out: Path) -> Path:
    return _save(_draw_cmnist_branch(df, n), out)


def _draw_cmnist_branch(df: pd.DataFrame, n: int):
    """Phase 1 (up to the switch) then Phase 2 of branch ``n`` on the four test sets."""
    df = df if "setting" in df else _annotate(df)
    p2_all = df[(df["phase"] == "phase2") & (df["branch_n"] == n)]
    if p2_all.empty:
        raise ValueError(f"no Phase-2 rows for branch N={n}")
    settings = sorted(p2_all["setting"].unique())
    colors = _colors(CMNIST_SETS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    switch = _switch(p2_all)
    for si, setting in enumerate(settings):
        ls = LINESTYLES[si % len(LINESTYLES)]
        p2 = p2_all[p2_all["setting"] == setting]
        p1 = df[(df["phase"] == "phase1") & (df["setting"] == setting)
                & (df["global_step"] < _switch(p2))]
        for ts in CMNIST_SETS:
            seg = pd.concat([p1[p1["test_set"] == ts], p2[p2["test_set"] == ts]])
            if seg.empty:
                continue
            label = ts if len(settings) == 1 else f"{ts} ({setting})"
            _line(axes[0], seg, "global_step", "acc", label, colors[ts], ls)
            loss_seg = _drop_startup(seg)
            if not loss_seg.empty:
                _line(axes[1], loss_seg, "global_step", "loss", label, colors[ts], ls, band=False)
        cl = pd.concat([p1, p2])
        cl = cl[cl["test_set"] == "color_conflict"].dropna(subset=["acc_color_label"])
        if not cl.empty:
            _line(axes[0], cl, "global_step", "acc_color_label",
                  "color_conflict (color label)" if len(settings) == 1 else None,
                  colors["color_conflict"], ":", 1.0)
    for ax, name in zip(axes, ("accuracy", "loss")):
        ax.axvline(_x([switch])[0], color="gray", ls="--", lw=1, label="switch")
        _style_log2(ax, df["global_step"].max())
        ax.set_xlabel("global step (log2; 0 drawn at 0.5)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    _style_loss(axes[1])
    axes[0].set_title(f"branch N = {n}")
    axes[0].legend(fontsize=8)
    return fig


def plot_cmnist_overlay(df: pd.DataFrame, out: Path) -> Path:
    return _save(_draw_cmnist_overlay(df), out)


def _draw_cmnist_overlay(df: pd.DataFrame):
    """``color_random`` accuracy vs Phase-2 step for every N (control N = 0 in black)."""
    df = df if "setting" in df else _annotate(df)
    p2 = df[(df["phase"] == "phase2") & (df["test_set"] == "color_random")]
    ns = sorted(int(n) for n in p2["branch_n"].dropna().unique())
    settings = sorted(p2["setting"].unique())
    colors = _colors([n for n in ns if n != 0])
    colors[0] = "black"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for si, setting in enumerate(settings):
        ls = LINESTYLES[si % len(LINESTYLES)]
        for n in ns:
            sub = p2[(p2["setting"] == setting) & (p2["branch_n"] == n)]
            if sub.empty:
                continue
            name = "N=0 (control)" if n == 0 else f"N={n}"
            _line(ax, sub, "phase_step", "acc", name if len(settings) == 1
                  else f"{name} ({setting})", colors[n], ls)
    _style_log2(ax, p2["phase_step"].max() if len(p2) else 1)
    ax.set_xlabel("Phase-2 step (log2; 0 drawn at 0.5)")
    ax.set_ylabel("color_random accuracy (shape learning)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_cmnist_switch(df: pd.DataFrame, out: Path) -> Path:
    return _save(_draw_cmnist_switch(df), out)


def _draw_cmnist_switch(df: pd.DataFrame):
    """The switch, close up: training loss and color_random accuracy for the first
    SWITCH_WINDOW Phase-2 steps, one line per N (seed mean; control N = 0 in black)."""
    df = df if "setting" in df else _annotate(df)
    p2 = df[(df["phase"] == "phase2") & (df["test_set"] == "color_random")
            & (df["phase_step"] <= SWITCH_WINDOW)]
    ns = sorted(int(n) for n in p2["branch_n"].dropna().unique())
    settings = sorted(p2["setting"].unique())
    colors = _colors([n for n in ns if n != 0])
    colors[0] = "black"
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for si, setting in enumerate(settings):
        ls = LINESTYLES[si % len(LINESTYLES)]
        for n in ns:
            sub = p2[(p2["setting"] == setting) & (p2["branch_n"] == n)]
            if sub.empty:
                continue
            name = "N=0 (control)" if n == 0 else f"N={n}"
            label = name if len(settings) == 1 else f"{name} ({setting})"
            _line(loss_ax, sub.dropna(subset=["train_loss_avg"]), "phase_step", "train_loss_avg",
                  label, colors[n], ls, band=False, log_x=False)
            _line(acc_ax, sub, "phase_step", "acc", label, colors[n], ls, log_x=False)
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("training loss (mean since previous check)")
    loss_ax.set_title("Loss at the switch: confidently wrong, then recovery")
    acc_ax.axhline(CHANCE, color="gray", ls="--", lw=1, label="chance")
    acc_ax.set_ylabel("color_random accuracy (shape learning)")
    acc_ax.set_title("Shape accuracy after the switch")
    for ax in (loss_ax, acc_ax):
        ax.set_xlim(0, SWITCH_WINDOW)
        ax.set_xlabel("Phase-2 step (0 = switch to random colors)")
        ax.grid(alpha=0.3)
    acc_ax.legend(fontsize=8)
    return fig


def _first_reach(steps: np.ndarray, acc: np.ndarray, target: float) -> float:
    """First step where ``acc`` reaches ``target``, interpolated between checks.

    Between the last check below the target (s0, a0) and the first at or above it (s1, a1)
    the curve is taken as linear in log(step), so the answer is not limited to the check
    grid; if s0 is step 0 (log undefined) it is linear in step. NaN if never reached."""
    steps, acc = np.asarray(steps, dtype=float), np.asarray(acc, dtype=float)
    hit = np.flatnonzero(acc >= target - 1e-12)
    if not len(hit):
        return float("nan")
    i = hit[0]
    if i == 0:
        return float(steps[0])
    s0, s1, a0, a1 = steps[i - 1], steps[i], acc[i - 1], acc[i]
    frac = min(max((target - a0) / (a1 - a0), 0.0), 1.0)
    if frac >= 1.0:  # exactly on a check: no rounding from exp/log
        return float(s1)
    if s0 <= 0:
        return float(s0 + frac * (s1 - s0))
    return float(np.exp(np.log(s0) + frac * (np.log(s1) - np.log(s0))))


def plasticity_table(df: pd.DataFrame, target_frac: float = 0.9) -> pd.DataFrame:
    """Steps for each branch to reach ``target_frac`` x the control's final color_random acc."""
    df = df if "setting" in df else _annotate(df)
    p2 = df[(df["phase"] == "phase2") & (df["test_set"] == "color_random")]
    curves = (p2.groupby(["setting", "branch_n", "phase_step"])["acc"].mean()
              .reset_index().sort_values("phase_step"))
    rows = []
    for setting, sdf in curves.groupby("setting"):
        ctrl = sdf[sdf["branch_n"] == 0]
        if ctrl.empty:
            raise ValueError(f"no N=0 control run for setting {setting!r}")
        target = target_frac * float(ctrl["acc"].iloc[-1])
        control_steps = _first_reach(ctrl["phase_step"].to_numpy(), ctrl["acc"].to_numpy(), target)
        for n, g in sdf.groupby("branch_n"):
            steps = _first_reach(g["phase_step"].to_numpy(), g["acc"].to_numpy(), target)
            ratio = (steps / control_steps
                     if control_steps and not np.isnan(control_steps) else float("nan"))
            rows.append({"setting": setting, "branch_n": int(n), "steps_to_target": steps,
                         "control_steps": control_steps, "ratio": ratio,
                         "reached": not np.isnan(steps), "target_acc": target})
    return pd.DataFrame(rows, columns=["setting", "branch_n", "steps_to_target", "control_steps",
                                       "ratio", "reached", "target_acc"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plot training curves and the plasticity table.")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--experiment", required=True, choices=("cifar_fresh", "cmnist"))
    args = ap.parse_args(argv)
    try:
        df = load_runs(args.root / "logs", args.experiment)
    except FileNotFoundError as e:
        print(f"error: no metrics files found ({e})", file=sys.stderr)
        sys.exit(1)
    fig_dir = args.root / "figures" / args.experiment
    if args.experiment == "cifar_fresh":
        print(plot_cifar(df, fig_dir / "curves.png"))
        return
    for n in sorted(int(n) for n in df["branch_n"].dropna().unique()):
        print(plot_cmnist_branch(df, n, fig_dir / f"branch_N{n}.png"))
    print(plot_cmnist_overlay(df, fig_dir / "overlay.png"))
    print(plot_cmnist_switch(df, fig_dir / "switch.png"))
    table = plasticity_table(df)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(fig_dir / "plasticity_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

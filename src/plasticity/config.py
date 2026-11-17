"""Run configuration, run ids and output directories."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path


@dataclass
class RunConfig:
    experiment: str  # "cifar_fresh" | "cmnist_phase1" | "cmnist_phase2"
    seed: int = 0
    alpha: float = 1.0
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 128
    total_steps: int = 65536  # cifar only
    phase1_epochs: int = 16
    durations: list[int] = field(default_factory=lambda: [0, 1, 2, 4, 8, 16])
    phase2_epochs: int = 20
    branch_n: int | None = None  # phase 2 only
    color_corr: float = 1.0
    eval_size: int = 10000
    save_schedule: str = "dense"  # "dense" | "sparse"
    amp: bool = True
    num_workers: int = 4
    full_state_every_s: int = 7200
    train_subset: int | None = None  # for tests
    root: str = "."

    def __post_init__(self):
        # Canonical numeric types so the same setting always yields the same run id.
        for name in ("alpha", "lr", "momentum", "weight_decay", "color_corr"):
            setattr(self, name, float(getattr(self, name)))
        for name in ("seed", "batch_size"):
            setattr(self, name, int(getattr(self, name)))
        if self.branch_n is not None:
            self.branch_n = int(self.branch_n)


def load_config(path: str, overrides: dict) -> RunConfig:
    """JSON file values, then non-None overrides. Unknown keys raise ValueError."""
    with open(path) as f:
        values = {k: v for k, v in json.load(f).items() if not k.startswith("_")}  # "_comment" etc.
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(RunConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    return RunConfig(**values)


def run_id(cfg: RunConfig) -> str:
    if cfg.experiment == "cmnist_phase2" and cfg.branch_n is None:
        raise ValueError("cmnist_phase2 requires branch_n (pass --branch-n)")
    rid = (f"resnet18_a{cfg.alpha}_lr{cfg.lr}_wd{cfg.weight_decay}"
           f"_bs{cfg.batch_size}_seed{cfg.seed}")
    if cfg.experiment.startswith("cmnist"):
        rid += f"_color{cfg.color_corr:.2f}"
    if cfg.experiment == "cmnist_phase2":
        rid += f"_N{cfg.branch_n}ep"
    return rid


def phase1_run_id(cfg: RunConfig) -> str:
    """Run id of the Phase-1 run whose checkpoints a Phase-2 run branches from."""
    return run_id(replace(cfg, experiment="cmnist_phase1", branch_n=None))


def run_dirs(cfg: RunConfig) -> dict[str, Path]:
    folder = "cmnist" if cfg.experiment.startswith("cmnist") else cfg.experiment
    root = Path(cfg.root)
    rid = run_id(cfg)
    return {
        "ckpt": root / "checkpoints" / folder / rid,
        "logs": root / "logs" / folder / rid,
        "figures": root / "figures" / folder,
        "data": root / "data",
    }

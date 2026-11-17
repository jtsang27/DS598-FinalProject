"""Training loop for the three experiments, with resume, plus the command-line entry point.

    python -m plasticity.train --config configs/cifar_fresh.json --seed 0 --root <root>

The whole uint8 training set is kept as one tensor on the device and shuffled batches are
indexed from it directly (no DataLoader, so ``num_workers`` is unused). Batch order is a
function of the seed and step count alone, so a resumed run sees the same batches; in
Phase 2 it depends on the Phase-2 step only, so every branch N sees the same data stream.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
import typing
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoints import (append_manifest, git_version, load_checkpoint, load_training_state,
                          save_checkpoint)
from .config import RunConfig, load_config, phase1_run_id, run_dirs, run_id
from .data import (COLORS, TESTSET_SEED, build_test_sets, colorize, gpu_augment_cifar,
                   load_raw, sample_color_idx, test_set_paths)
from .metrics import MetricsLogger, dist_from_init, evaluate, weight_norm
from .models import resnet18_cifar
from .schedule import check_steps, full_state_steps, phase2_check_steps, save_steps

PHASES = {"cifar_fresh": "train", "cmnist_phase1": "phase1", "cmnist_phase2": "phase2"}


# ------------------------------------------------------------------ run layout
@dataclass
class _Plan:
    phase: str
    spe: int            # steps per epoch
    total: int          # phase length in steps
    offset: int         # global_step = offset + phase_step
    color_corr: float | None  # None for CIFAR

    def global_step(self, phase_step: int) -> int:
        return self.offset + phase_step


def _plan(cfg: RunConfig, n_train: int) -> _Plan:
    if cfg.experiment not in PHASES:
        raise ValueError(f"unknown experiment {cfg.experiment!r}")
    spe = n_train // cfg.batch_size
    if spe < 1:
        raise ValueError(f"{n_train} training examples is less than one batch "
                         f"(batch_size={cfg.batch_size})")
    if cfg.experiment == "cifar_fresh":
        return _Plan("train", spe, cfg.total_steps, 0, None)
    if cfg.experiment == "cmnist_phase1":
        return _Plan("phase1", spe, cfg.phase1_epochs * spe, 0, cfg.color_corr)
    return _Plan("phase2", spe, cfg.phase2_epochs * spe, cfg.branch_n * spe, 0.0)


def _batch_position(plan: _Plan, phase_step: int) -> tuple[int, int]:
    """(shuffle epoch, batch within epoch). CIFAR and Phase 1 key the shuffle on the global
    epoch; Phase 2 on the Phase-2 epoch, so all branches share one Phase-2 data stream."""
    step = phase_step if plan.phase == "phase2" else plan.global_step(phase_step)
    return divmod(step, plan.spe)


def epoch_permutation(seed: int, epoch: int, n_train: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n_train, generator=g)


def batch_indices(seed: int, plan: _Plan, phase_step: int, n_train: int,
                  batch_size: int) -> torch.Tensor:
    """Training-set indices of the batch taken at ``phase_step`` (CPU int64 tensor)."""
    epoch, within = _batch_position(plan, phase_step)
    return epoch_permutation(seed, epoch, n_train)[within * batch_size:(within + 1) * batch_size]


def _data_recipe(cfg: RunConfig, plan: _Plan, test_names: list[str]) -> dict:
    recipe = {
        "dataset": "cifar10" if cfg.experiment == "cifar_fresh" else "mnist32_colored",
        "experiment": cfg.experiment,
        "color_corr": plan.color_corr,
        "colors": COLORS.tolist(),
        "testset_seed": TESTSET_SEED,
        "eval_size": cfg.eval_size,
        "train_subset": cfg.train_subset,
        "test_sets": test_names,
        "testset_dir": "data/testsets",
        "testset_files": {name: f"data/testsets/{p.name}"
                          for name, p in test_set_paths(cfg).items() if name in test_names},
        "steps_per_epoch": plan.spe,
    }
    if cfg.experiment == "cmnist_phase2":
        recipe["branch_n"] = cfg.branch_n
        recipe["phase1_run_id"] = phase1_run_id(cfg)
        recipe["phase1_color_corr"] = cfg.color_corr
    return recipe


def _phase1_file(cfg: RunConfig, step: int) -> Path:
    p1 = replace(cfg, experiment="cmnist_phase1", branch_n=None)
    return run_dirs(p1)["ckpt"] / f"step_{step:09d}.pt"


def _latest_full_state(ckpt_dir: Path) -> Path | None:
    for path in sorted(ckpt_dir.glob("step_*.pt"), reverse=True):
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if ckpt.get("full_state"):
            return path
    return None


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


class _TextLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, line: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a") as f:
            f.write(f"[{stamp}] {line}\n")


# -------------------------------------------------------------------- training
def train(cfg: RunConfig, raw: dict | None = None, device: str | None = None) -> Path:
    """Train (or resume) one run; returns its log directory."""
    os.umask(0o022)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.amp) and torch.device(device).type == "cuda"
    dirs = run_dirs(cfg)  # raises for phase 2 without branch_n
    rid = run_id(cfg)
    if raw is None:
        raw = load_raw(cfg)
    elif cfg.train_subset is not None:
        raw = {**raw, "x_train": raw["x_train"][: cfg.train_subset],
               "y_train": raw["y_train"][: cfg.train_subset]}
    plan = _plan(cfg, len(raw["x_train"]))

    for d in (dirs["ckpt"], dirs["logs"]):
        _write_json(asdict(cfg), d / "run_config.json")
    log = _TextLog(dirs["logs"] / "train.log")
    gpu = torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else "none"
    log(f"device {device} (GPU: {gpu}), amp {use_amp}")
    log(f"run id {rid}")
    log(f"config {json.dumps(asdict(cfg))}")
    log(f"code {json.dumps(git_version())}")

    # seeds, model, optimizer
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = resnet18_cifar(alpha=cfg.alpha)
    init_state = None
    if plan.phase == "phase2":
        branch = _phase1_file(cfg, plan.offset)
        if not branch.exists():
            raise FileNotFoundError(f"Phase-1 branch checkpoint not found: {branch}")
        model.load_state_dict(load_checkpoint(branch)[0].state_dict())
        step0 = _phase1_file(cfg, 0)
        if step0.exists():
            init_state = load_checkpoint(step0)[0].state_dict()
    model.to(device).train()
    init_state = {k: v.detach().clone().to(device)
                  for k, v in (init_state or model.state_dict()).items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    color_gen = torch.Generator().manual_seed(cfg.seed)

    # resume
    start, wall_before = 0, 0.0
    latest = _latest_full_state(dirs["ckpt"])
    if latest is not None:
        extra = load_training_state(latest, model, optimizer, device)
        start = int(extra["phase_step"])
        wall_before = float(extra.get("wall_time_s", 0.0))
        color_gen.set_state(extra["color_gen"])
        if scaler is not None and extra.get("scaler"):
            scaler.load_state_dict(extra["scaler"])
        log(f"resumed from step {plan.global_step(start)} (phase_step {start}) "
            f"using {latest.name}")
        if start >= plan.total:
            log(f"run already finished at phase_step {start} (total {plan.total}); nothing to do")
            return dirs["logs"]

    test_sets = build_test_sets(cfg, raw)
    recipe = _data_recipe(cfg, plan, list(test_sets))
    checks = set(phase2_check_steps(plan.spe, plan.total) if plan.phase == "phase2"
                 else check_steps(plan.spe, plan.total))
    saves = set(save_steps(cfg.experiment, cfg.save_schedule, plan.spe, plan.total,
                           cfg.durations))
    fulls = set(full_state_steps(cfg.experiment, plan.spe, plan.total, cfg.durations))
    logger = MetricsLogger(dirs["logs"] / "metrics.csv")
    last_logged = logger.last_step()

    x_train = raw["x_train"].to(device)
    y_train_cpu = raw["y_train"].to(torch.int64).cpu()
    y_train = y_train_cpu.to(device)
    n_train = len(x_train)
    t0 = time.time()
    last_full_save = t0
    loss_sum, loss_n = torch.zeros((), device=device), 0
    perm_epoch, perm = None, None

    def wall() -> float:
        return wall_before + time.time() - t0

    def check(ps: int) -> None:
        nonlocal loss_sum, loss_n, last_full_save
        gs = plan.global_step(ps)
        epoch = gs / plan.spe
        train_loss = loss_sum.item() / loss_n if loss_n else None
        loss_sum, loss_n = torch.zeros((), device=device), 0
        if last_logged is None or gs > last_logged:
            # evaluate()'s DataLoader draws from the global RNG; fork it so the training
            # stream (augmentation) does not depend on how many checks a run has.
            with torch.random.fork_rng(devices=[]):
                res = evaluate(model, test_sets, device)
            wn, dist, wt = weight_norm(model), dist_from_init(model, init_state), wall()
            logger.log([{
                "run_id": rid, "experiment": cfg.experiment, "phase": plan.phase,
                "global_step": gs, "phase_step": ps, "epoch": epoch, "test_set": name,
                "loss": r["loss"], "acc": r["acc"], "acc_color_label": r["acc_color_label"],
                "lr": cfg.lr, "train_loss_avg": train_loss, "weight_norm": wn,
                "dist_from_init": dist, "wall_time_s": wt,
            } for name, r in res.items()])
            accs = " ".join(f"{name} {r['acc']:.4f}" for name, r in res.items())
            tl = "-" if train_loss is None else f"{train_loss:.4f}"
            log(f"step {gs} (phase_step {ps}) epoch {epoch:.3f} train_loss {tl} | {accs} "
                f"| elapsed {wt:.0f}s")
        full = ps in fulls or time.time() - last_full_save >= cfg.full_state_every_s
        path = dirs["ckpt"] / f"step_{gs:09d}.pt"
        replayed = last_logged is not None and gs <= last_logged
        if replayed and path.exists():
            # Steps retrained after a resume: the file written before the interruption
            # matches the logged rows, so keep it (and its single manifest row).
            return
        if ps in saves or full:
            extra = {"phase_step": ps, "global_step": gs, "wall_time_s": wall(),
                     "color_gen": color_gen.get_state(),
                     "scaler": scaler.state_dict() if scaler is not None else None}
            save_checkpoint(path, model, cfg, step=gs, phase=plan.phase, phase_step=ps,
                            epoch=epoch, data_recipe=recipe,
                            optimizer=optimizer if full else None,
                            extra_state=extra if full else None)
            append_manifest(Path(cfg.root), {
                "path": os.path.relpath(path, cfg.root), "run_id": rid,
                "experiment": cfg.experiment, "phase": plan.phase, "step": gs,
                "phase_step": ps, "epoch": epoch, "full_state": full,
                "commit": git_version()["commit"],
                "saved_at": datetime.now().isoformat(timespec="seconds")})
            if full:
                last_full_save = time.time()

    if start == 0:
        check(0)
    for ps in range(start, plan.total):
        epoch_idx, within = _batch_position(plan, ps)
        if epoch_idx != perm_epoch:  # same permutation as batch_indices(), built once per epoch
            perm, perm_epoch = epoch_permutation(cfg.seed, epoch_idx, n_train), epoch_idx
        idx = perm[within * cfg.batch_size:(within + 1) * cfg.batch_size]
        idx_dev = idx.to(device)
        x = x_train[idx_dev].float() / 255
        y = y_train[idx_dev]
        if plan.color_corr is None:
            x = gpu_augment_cifar(x)
        else:
            color = sample_color_idx(y_train_cpu[idx], plan.color_corr, color_gen)
            x = colorize(x, color.to(device))
        amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp
                   else contextlib.nullcontext())
        with amp_ctx:
            loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.detach().float()
        loss_n += 1
        if ps + 1 in checks:
            check(ps + 1)
    log(f"finished at step {plan.global_step(plan.total)}; wall time {wall():.0f}s")
    return dirs["logs"]


# ------------------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m plasticity.train",
                                description="Train one plasticity run (resumes if possible).")
    p.add_argument("--config", required=True, help="JSON config, e.g. configs/cifar_fresh.json")
    hints = typing.get_type_hints(RunConfig)
    for f in fields(RunConfig):
        flag = "--" + f.name.replace("_", "-")
        hint = hints[f.name]
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if f.name == "root":
            p.add_argument(flag, default=".", help="project root (default: .)")
        elif hint is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif typing.get_origin(hint) is list:
            p.add_argument(flag, nargs="+", type=args[0], default=None)
        else:
            p.add_argument(flag, type=args[0] if args else hint, default=None)
    return p


def main(argv: list[str] | None = None) -> Path:
    args = vars(_parser().parse_args(argv))
    config = args.pop("config")
    cfg = load_config(config, {k: v for k, v in args.items() if v is not None})
    return train(cfg)


if __name__ == "__main__":
    main()

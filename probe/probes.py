from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checkpoints import load_checkpoint
from .config import RunConfig, run_dirs
from .data import COLORS, build_test_sets, colorize, load_raw
from .metrics import dist_from_init, evaluate, weight_norm

STATS = ("acc", "weights", "activations", "first_step", "overlap", "ntk", "hessian")
CHEAP = STATS[:5]
PROBE_SEED = 4321
ID_COLUMNS = ["run_id", "experiment", "phase", "step", "phase_step", "epoch"]


# ------------------------------------------------------------------ small math
def effective_rank(kernel: Tensor) -> float:
    """Entropy effective rank exp(-sum p log p), p = eigenvalues / their sum (Roy & Vetterli)."""
    eig = torch.linalg.eigvalsh(kernel.double()).clamp_min(0)
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def kernel_alignment(kernel: Tensor, y: Tensor, num_classes: int) -> float:
    """Centered kernel alignment between ``kernel`` and Y Y^T (Y = one-hot labels)."""
    k = kernel.double()
    onehot = F.one_hot(y.to(torch.int64), num_classes).double().to(k.device)
    target = onehot @ onehot.T
    n = k.shape[0]
    center = torch.eye(n, dtype=torch.float64, device=k.device) - 1.0 / n
    kc, tc = center @ k @ center, center @ target @ center
    denom = kc.norm() * tc.norm()
    if denom <= 0:
        return float("nan")
    return float((kc * tc).sum() / denom)


def spectrum_rank(features: Tensor, energy: float = 0.99) -> int:
    """Smallest k whose top-k singular values hold ``energy`` of the singular-value sum."""
    s = torch.linalg.svdvals(features.double())
    total = s.sum()
    if total <= 0:
        return 0
    cum = torch.cumsum(s, 0) / total
    return int(min((cum < energy).sum().item() + 1, len(s)))


def _grad_norm(params) -> float:
    return math.sqrt(sum(float((p.grad.detach().float() ** 2).sum())
                         for p in params if p.grad is not None))


# --------------------------------------------------------------------- weights
def conv_fc_weight_norm(model: nn.Module) -> float:
    """L2 norm of Conv2d and Linear weights only (no biases, no BatchNorm parameters)."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += float((m.weight.detach().float() ** 2).sum())
    return total ** 0.5


# ----------------------------------------------------------------- activations
def _last_linear(model: nn.Module) -> nn.Linear:
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise ValueError("model has no Linear layer to take features from")
    return layers[-1]


def activation_stats(model: nn.Module, x: Tensor, batch_size: int = 500, tau: float = 0.01,
                     energy: float = 0.99) -> dict:
    """Dormant fraction over all ReLU outputs and the rank of the penultimate features.

    A unit is one channel of one ReLU output. Its score is its mean |activation| over the
    probe inputs and positions, divided by the mean score of its layer; it is dormant when
    the score is <= ``tau``. A ReLU module that is called twice per forward (as in a
    BasicBlock) counts as two layers. Features are the inputs of the last Linear layer.
    """
    was_training = model.training
    model.eval()
    sums: dict[str, Tensor] = {}
    counts: dict[str, float] = {}
    calls: dict[str, int] = {}
    feats: list[Tensor] = []

    def relu_hook(name):
        def hook(_module, _inputs, out):
            k = calls.get(name, 0)
            calls[name] = k + 1
            key = f"{name}#{k}"
            a = out.detach().abs().float()
            dims = [0] + list(range(2, a.dim()))
            per_unit = a.sum(dim=dims).double().cpu()
            sums[key] = sums[key] + per_unit if key in sums else per_unit
            counts[key] = counts.get(key, 0.0) + a.numel() / a.shape[1]
        return hook

    def feature_hook(_module, inputs, _out):
        feats.append(inputs[0].detach().flatten(1).float().cpu())

    handles = [m.register_forward_hook(relu_hook(n)) for n, m in model.named_modules()
               if isinstance(m, nn.ReLU)]
    handles.append(_last_linear(model).register_forward_hook(feature_hook))
    try:
        with torch.no_grad():
            for s in range(0, len(x), batch_size):
                calls.clear()
                model(x[s:s + batch_size])
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    units = dormant = 0
    for key, total in sums.items():
        score = total / counts[key]
        layer_mean = score.mean()
        dead = (score <= tau * layer_mean) if layer_mean > 0 else torch.ones_like(score).bool()
        units += score.numel()
        dormant += int(dead.sum())
    return {"dormant_frac": dormant / units if units else float("nan"),
            "dormant_units": dormant,
            "feature_rank": spectrum_rank(torch.cat(feats), energy)}


# ------------------------------------------------------------------ first step
def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch_size: int = 500) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            total += F.cross_entropy(model(x[s:s + batch_size]).float(), y[s:s + batch_size],
                                     reduction="sum").item()
    model.train(was_training)
    return total / len(x)


def first_step_test(model: nn.Module, batches: list, probe_x: Tensor, probe_y: Tensor,
                    lr: float, momentum: float, weight_decay: float) -> dict:
    """Train a copy with fresh SGD on ``batches[:-1]``; report how fast the task loss moves.

    ``fs_loss_drop`` is the fall in eval-mode loss on the fixed probe set over those steps.
    ``fs_grad_norm_first`` is the gradient norm on the first batch, ``fs_grad_norm_last`` on
    the batch after the last step (no step is taken on it). The checkpoint is not modified.
    """
    if len(batches) < 2:
        raise ValueError("first_step_test needs at least 2 batches")
    m = copy.deepcopy(model)
    m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    out = {"fs_steps": len(batches) - 1, "fs_loss_before": _mean_loss(m, probe_x, probe_y)}
    for i, (x, y) in enumerate(batches):
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if i == 0:
            out["fs_train_loss_first"] = loss.item()
            out["fs_grad_norm_first"] = _grad_norm(params)
        if i == len(batches) - 1:
            out["fs_grad_norm_last"] = _grad_norm(params)
            break
        opt.step()
    out["fs_loss_after"] = _mean_loss(m, probe_x, probe_y)
    out["fs_loss_drop"] = out["fs_loss_before"] - out["fs_loss_after"]
    return out


# --------------------------------------------------------------------- overlap
def color_overlap(weight: Tensor, colors: Tensor = COLORS) -> dict:
    """How much each first-layer filter reads class color vs. plain brightness.

    For a filter w (3, kh, kw) and a unit color direction u, overlap(w, u) is the largest
    cosine between w and any patch whose every pixel has color u, i.e.
    ||u^T w|| / ||w||. ``gray`` is the mean class color (what a digit looks like on average
    once colors are random); the ``color`` directions are the class colors with their gray
    component removed, and a filter's color overlap is its maximum over the classes.
    ``chroma_energy_frac`` is the mean share of filter energy outside the gray direction
    (2/3 for random filters).
    """
    w = weight.detach().double().cpu()
    if w.dim() != 4 or w.shape[1] != 3:
        raise ValueError(f"expected first-layer weight (F, 3, kh, kw), got {tuple(w.shape)}")
    f = w.reshape(w.shape[0], 3, -1)
    filter_energy = (f ** 2).sum(dim=(1, 2)).clamp_min(1e-30)
    colors = colors.double().cpu()
    gray = colors.mean(0)
    gray = gray / gray.norm()
    chroma = colors - (colors @ gray)[:, None] * gray
    chroma = chroma / chroma.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def overlap(directions: Tensor) -> Tensor:  # (K, 3) -> (F, K)
        proj = torch.einsum("kc,fcs->fks", directions, f)
        return ((proj ** 2).sum(-1) / filter_energy[:, None]).sqrt()

    o_gray = overlap(gray[None])[:, 0]
    o_color = overlap(chroma).max(dim=1).values
    return {"overlap_color_max": float(o_color.max()),
            "overlap_color_mean": float(o_color.mean()),
            "overlap_gray_max": float(o_gray.max()),
            "overlap_gray_mean": float(o_gray.mean()),
            "chroma_energy_frac": float((1 - o_gray ** 2).mean())}


def first_conv(model: nn.Module) -> nn.Conv2d:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise ValueError("model has no Conv2d layer")


# ------------------------------------------------------------------------- NTK
def _param_blocks(params: list, max_block: int) -> list[list]:
    """Consecutive parameter tensors packed into blocks of at most ``max_block`` entries
    (a tensor larger than that gets a block of its own)."""
    blocks, cur, size = [], [], 0
    for p in params:
        if cur and size + p.numel() > max_block:
            blocks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += p.numel()
    if cur:
        blocks.append(cur)
    return blocks


def empirical_ntk(model: nn.Module, x: Tensor, output: str = "sum",
                  max_block: int = 2_400_000) -> Tensor:
    """Empirical NTK Gram matrix (n, n) on inputs ``x``, in eval mode, float64.

    ``output="sum"``: one scalar per input, the sum of its logits / sqrt(C) (the
    sum-of-logits approximation of Mohamadi et al. 2023); one backward pass per input.
    ``output="all"``: the average over classes c of the kernel of logit c; C times slower.

    Gradients are never stored for all parameters at once: parameters are split into blocks
    of about ``max_block`` entries and K is accumulated block by block, so peak memory is
    about n * max_block * 4 bytes (2.4 GB for n = 256).
    """
    if output not in ("sum", "all"):
        raise ValueError(f"unknown NTK output {output!r}; expected 'sum' or 'all'")
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    blocks = _param_blocks(params, max_block)
    n, device = len(x), x.device
    with torch.no_grad():
        num_classes = model(x[:1]).shape[1]
    heads = [None] if output == "sum" else list(range(num_classes))
    kernel = torch.zeros(n, n, dtype=torch.float64, device=device)
    try:
        for head in heads:
            for block in blocks:
                grads = torch.empty(n, sum(p.numel() for p in block), device=device)
                for i in range(n):
                    logits = model(x[i:i + 1]).float()[0]
                    out = logits.sum() / math.sqrt(num_classes) if head is None else logits[head]
                    g = torch.autograd.grad(out, block, allow_unused=True)
                    grads[i] = torch.cat([
                        (torch.zeros_like(p) if gi is None else gi).reshape(-1).float()
                        for p, gi in zip(block, g)])
                kernel += (grads @ grads.T).double()
                del grads
    finally:
        model.train(was_training)
    return kernel / len(heads)


def ntk_stats(model: nn.Module, x: Tensor, y: Tensor, num_classes: int = 10,
              output: str = "sum", max_block: int = 2_400_000) -> dict:
    kernel = empirical_ntk(model, x, output, max_block)
    return {"ntk_align": kernel_alignment(kernel, y, num_classes),
            "ntk_erank": effective_rank(kernel),
            "ntk_trace": float(kernel.diagonal().sum())}


# --------------------------------------------------------------------- Hessian
def hessian_top_eigenvalue(model: nn.Module, x: Tensor, y: Tensor, iters: int = 20,
                           tol: float = 1e-3, seed: int = PROBE_SEED) -> float:
    """Largest-magnitude eigenvalue of the Hessian of the mean cross-entropy on (x, y), in
    eval mode, by power iteration on Hessian-vector products. Stops early once the estimate
    changes by less than ``tol`` (relative). The value can be negative."""
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.Generator().manual_seed(seed)
        v = [torch.randn(p.shape, generator=g).to(device=p.device, dtype=p.dtype)
             for p in params]
        norm = math.sqrt(sum(float((t ** 2).sum()) for t in v))
        v = [t / norm for t in v]
        eig = None
        for _ in range(iters):
            hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
            new = sum(float((h * t).sum()) for h, t in zip(hv, v))  # Rayleigh quotient
            norm = math.sqrt(sum(float((h ** 2).sum()) for h in hv))
            if norm == 0:
                return 0.0
            v = [h.detach() / norm for h in hv]
            converged = eig is not None and abs(new - eig) <= tol * max(abs(new), 1e-12)
            eig = new
            if converged:
                break
        return float(eig)
    finally:
        model.train(was_training)


# ------------------------------------------------------------------ probe data
@dataclass
class ProbeData:
    cfg: RunConfig
    family: str            # "cmnist" | "cifar"
    test_sets: dict
    task_x: Tensor         # fixed probe inputs for the Phase-2 task, on the device
    task_y: Tensor
    low_x: Tensor          # fixed probe inputs carrying the low-order signal
    low_y: Tensor
    step_batches: list     # training batches of the task for first_step_test
    init_state: dict | None


def config_from_info(info: dict, root: str) -> RunConfig:
    known = {f.name for f in fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in info["config"].items() if k in known})
    return replace(cfg, root=str(root))


def _init_state(cfg: RunConfig) -> dict | None:
    """Weights of the run's step-0 save (for Phase 2: of the Phase-1 run it branched from)."""
    src = (replace(cfg, experiment="cmnist_phase1", branch_n=None)
           if cfg.experiment == "cmnist_phase2" else cfg)
    path = run_dirs(src)["ckpt"] / f"step_{0:09d}.pt"
    return load_checkpoint(path)[0].state_dict() if path.exists() else None


def build_probe_data(cfg: RunConfig, raw: dict, device: str, n_probe: int = 2000,
                     fs_steps: int = 10) -> ProbeData:
    """Fixed probe inputs and training batches; identical for every checkpoint of a run and
    for every seed (they depend on PROBE_SEED and the cached test sets only)."""
    cmnist = cfg.experiment.startswith("cmnist")
    test_sets = build_test_sets(cfg, raw)
    task_name, low_name = ("color_random", "color_matched") if cmnist else ("real", "gaussian")
    g = torch.Generator().manual_seed(PROBE_SEED)

    def subset(name):
        x, y = test_sets[name].tensors[:2]
        idx = torch.randperm(len(x), generator=g)[:n_probe]
        return x[idx].float().to(device), y[idx].to(torch.int64).to(device)

    task_x, task_y = subset(task_name)
    low_x, low_y = subset(low_name)

    bs = cfg.batch_size
    idx = torch.randperm(len(raw["x_train"]), generator=g)[:(fs_steps + 1) * bs]
    x = raw["x_train"][idx].float() / 255
    y = raw["y_train"][idx].to(torch.int64)
    if cmnist:
        x = colorize(x, torch.randint(0, 10, y.shape, generator=g))
    batches = [(x[i:i + bs].to(device), y[i:i + bs].to(device))
               for i in range(0, len(x) - bs + 1, bs)]
    return ProbeData(cfg, "cmnist" if cmnist else "cifar", test_sets, task_x, task_y,
                     low_x, low_y, batches, _init_state(cfg))


# --------------------------------------------------------------- one checkpoint
def probe_model(model: nn.Module, data: ProbeData, stats, device: str, *, act_n: int = 2000,
                fs_probe_n: int = 1000, ntk_n: int = 256, ntk_output: str = "sum",
                ntk_block: int = 2_400_000, hess_n: int = 256, hess_iters: int = 20) -> dict:
    """All requested statistics for one loaded model, as a flat dict of floats."""
    cfg, row = data.cfg, {}
    model.to(device).eval()
    if "acc" in stats:
        res = evaluate(model, data.test_sets, device)
        for name, r in res.items():
            row[f"acc_{name}"] = r["acc"]
            if r["acc_color_label"] is not None:
                row[f"acc_{name}_color_label"] = r["acc_color_label"]
        if data.family == "cmnist":
            row["gap_matched_minus_random"] = res["color_matched"]["acc"] - res["color_random"]["acc"]
            row["gap_gaussian_minus_random"] = res["gaussian"]["acc"] - res["color_random"]["acc"]
        else:
            row["gap_real_minus_gaussian"] = res["real"]["acc"] - res["gaussian"]["acc"]
            row["gap_gaussian_minus_independent"] = (res["gaussian"]["acc"]
                                                     - res["independent"]["acc"])
    if "weights" in stats:
        row["weight_norm"] = weight_norm(model)
        row["weight_norm_conv_fc"] = conv_fc_weight_norm(model)
        row["dist_from_init"] = (dist_from_init(model, data.init_state)
                                 if data.init_state is not None else float("nan"))
    if "activations" in stats:
        for tag, x in (("task", data.task_x), ("low", data.low_x)):
            for k, v in activation_stats(model, x[:act_n]).items():
                row[f"{k}_{tag}"] = v
    if "first_step" in stats:
        row.update(first_step_test(model, data.step_batches, data.task_x[:fs_probe_n],
                                   data.task_y[:fs_probe_n], cfg.lr, cfg.momentum,
                                   cfg.weight_decay))
    if "overlap" in stats and data.family == "cmnist":
        row.update(color_overlap(first_conv(model).weight))
    if "ntk" in stats:
        for tag, x, y in (("task", data.task_x, data.task_y), ("low", data.low_x, data.low_y)):
            for k, v in ntk_stats(model, x[:ntk_n], y[:ntk_n], output=ntk_output,
                                  max_block=ntk_block).items():
                row[f"{k}_{tag}"] = v
    if "hessian" in stats:
        row["hessian_top_eig"] = hessian_top_eigenvalue(
            model, data.task_x[:hess_n], data.task_y[:hess_n], iters=hess_iters)
    return row


# ---------------------------------------------------------------------- a run
def _checkpoint_steps(run_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in sorted(run_dir.glob("step_*.pt"))}


def _write_csv(rows: dict[int, dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([rows[s] for s in sorted(rows)])
    df = df[ID_COLUMNS + [c for c in df.columns if c not in ID_COLUMNS]]
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)


def probe_run(run_dir, root=".", stats=CHEAP, steps=None, branch_points: bool = False,
              raw: dict | None = None, device: str | None = None, out=None,
              n_probe: int = 2000, **probe_kwargs) -> Path:
    """Probe the checkpoints of one run directory; returns the path of its probes.csv.

    ``steps``: global steps to probe (default: every saved checkpoint).
    ``branch_points``: only the steps Phase 2 branches from (N epochs for N in durations).
    Extra keyword arguments go to :func:`probe_model`.
    """
    unknown = sorted(set(stats) - set(STATS))
    if unknown:
        raise ValueError(f"unknown stats {unknown}; expected some of {STATS}")
    run_dir = Path(run_dir)
    if not run_dir.is_dir() and (Path(root) / run_dir).is_dir():
        run_dir = Path(root) / run_dir
    saved = _checkpoint_steps(run_dir)
    if not saved:
        raise FileNotFoundError(f"no step_*.pt checkpoints in {run_dir}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = config_from_info(load_checkpoint(next(iter(saved.values())))[1], root)
    if raw is None:
        raw = load_raw(cfg)
    elif cfg.train_subset is not None:
        raw = {**raw, "x_train": raw["x_train"][: cfg.train_subset],
               "y_train": raw["y_train"][: cfg.train_subset]}
    wanted = set(saved)
    if branch_points:
        if cfg.experiment != "cmnist_phase1":
            raise ValueError("--branch-points only applies to cmnist_phase1 runs")
        spe = len(raw["x_train"]) // cfg.batch_size
        wanted &= {n * spe for n in cfg.durations}
    if steps is not None:
        missing = sorted(set(steps) - set(saved))
        if missing:
            raise FileNotFoundError(f"no checkpoint saved at steps {missing} in {run_dir}")
        wanted &= set(steps)
    if not wanted:
        raise ValueError("no checkpoints selected")

    data = build_probe_data(cfg, raw, device, n_probe=n_probe)
    out = Path(out) if out is not None else run_dirs(cfg)["logs"] / "probes.csv"
    rows: dict[int, dict] = {}
    if out.exists():
        rows = {int(r["step"]): r for r in pd.read_csv(out).to_dict("records")}
    for step in sorted(wanted):
        t0 = time.time()
        model, info = load_checkpoint(saved[step], device)
        row = {"run_id": info["run_id"], "experiment": info["config"]["experiment"],
               "phase": info["phase"], "step": info["step"], "phase_step": info["phase_step"],
               "epoch": info["epoch"]}
        row.update(probe_model(model, data, stats, device, **probe_kwargs))
        rows[step] = {**rows.get(step, {}), **row}
        _write_csv(rows, out)
        print(f"step {step}: {len(row) - len(ID_COLUMNS)} statistics in {time.time() - t0:.1f}s",
              flush=True)
    return out


# ------------------------------------------------------------------------- CLI
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m plasticity.probes",
                                description="Compute probe statistics on saved checkpoints.")
    p.add_argument("--run-dir", required=True,
                   help="checkpoint folder of one run, e.g. checkpoints/cmnist/<run_id>")
    p.add_argument("--root", default=".", help="project root (default: .)")
    p.add_argument("--stats", nargs="+", default=["cheap"],
                   choices=("cheap", "all") + STATS, help="statistic groups (default: cheap)")
    p.add_argument("--steps", nargs="+", type=int, default=None,
                   help="global steps to probe (default: every saved checkpoint)")
    p.add_argument("--branch-points", action="store_true",
                   help="only the Phase-1 steps that Phase 2 branches from")
    p.add_argument("--out", default=None, help="output CSV (default: the run's logs folder)")
    p.add_argument("--device", default=None)
    p.add_argument("--n-probe", type=int, default=2000, help="size of each fixed probe set")
    p.add_argument("--act-n", type=int, default=2000, help="inputs for dormant / feature rank")
    p.add_argument("--ntk-n", type=int, default=256, help="inputs for the NTK")
    p.add_argument("--ntk-output", choices=("sum", "all"), default="sum")
    p.add_argument("--ntk-block", type=int, default=2_400_000,
                   help="parameters per NTK block; memory is about ntk_n * this * 4 bytes")
    p.add_argument("--hess-n", type=int, default=256, help="inputs for the Hessian")
    p.add_argument("--hess-iters", type=int, default=20)
    return p


def main(argv: list[str] | None = None) -> Path:
    a = _parser().parse_args(argv)
    stats: list[str] = []
    for s in a.stats:
        for name in (CHEAP if s == "cheap" else STATS if s == "all" else (s,)):
            if name not in stats:
                stats.append(name)
    return probe_run(a.run_dir, root=a.root, stats=tuple(stats), steps=a.steps,
                     branch_points=a.branch_points, device=a.device, out=a.out,
                     n_probe=a.n_probe, act_n=a.act_n, ntk_n=a.ntk_n,
                     ntk_output=a.ntk_output, ntk_block=a.ntk_block, hess_n=a.hess_n,
                     hess_iters=a.hess_iters)


if __name__ == "__main__":
    main()

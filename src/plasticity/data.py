"""Datasets: CIFAR-10, Colored MNIST, Gaussian clone / independent-pixel test sets."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import TensorDataset

from plasticity.config import RunConfig, run_dirs

TESTSET_SEED = 1234

# red, orange, yellow, green, cyan, blue, purple, magenta, white, brown
COLORS = torch.tensor([
    [1.00, 0.00, 0.00],
    [1.00, 0.50, 0.00],
    [1.00, 1.00, 0.00],
    [0.00, 0.80, 0.00],
    [0.00, 1.00, 1.00],
    [0.00, 0.00, 1.00],
    [0.55, 0.00, 0.80],
    [1.00, 0.00, 1.00],
    [1.00, 1.00, 1.00],
    [0.55, 0.30, 0.10],
])


# ---------------------------------------------------------------- raw loading
def load_cifar(data_root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from torchvision.datasets import CIFAR10

    def conv(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,32,32)
        return x, torch.tensor(ds.targets, dtype=torch.int64)

    xtr, ytr = conv(CIFAR10(str(data_root), train=True, download=True))
    xte, yte = conv(CIFAR10(str(data_root), train=False, download=True))
    return xtr, ytr, xte, yte


def load_mnist32(data_root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from torchvision.datasets import MNIST

    def conv(ds):
        x = F.pad(ds.data.to(torch.uint8), (2, 2, 2, 2)).unsqueeze(1).contiguous()  # (N,1,32,32)
        return x, ds.targets.to(torch.int64)

    xtr, ytr = conv(MNIST(str(data_root), train=True, download=True))
    xte, yte = conv(MNIST(str(data_root), train=False, download=True))
    return xtr, ytr, xte, yte


def load_raw(cfg: RunConfig) -> dict:
    data_root = run_dirs(cfg)["data"]
    data_root.mkdir(parents=True, exist_ok=True)
    loader = load_mnist32 if cfg.experiment.startswith("cmnist") else load_cifar
    x_train, y_train, x_test, y_test = loader(data_root)
    if cfg.train_subset is not None:
        x_train, y_train = x_train[: cfg.train_subset], y_train[: cfg.train_subset]
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


# ------------------------------------------------------------------- coloring
def colorize(gray: Tensor, color_idx: Tensor) -> Tensor:
    """(B,1,32,32) grayscale in [0,1] -> (B,3,32,32), digit drawn in COLORS[color_idx]."""
    colors = COLORS.to(device=gray.device, dtype=gray.dtype)[color_idx]
    return gray * colors[:, :, None, None]


def sample_color_idx(y: Tensor, color_corr: float, generator: torch.Generator) -> Tensor:
    """With prob color_corr the color index equals the label, else uniform over 0-9."""
    keep = torch.rand(y.shape, generator=generator) < color_corr
    rand = torch.randint(0, 10, y.shape, generator=generator)
    return torch.where(keep, y.cpu(), rand).to(y.device)


# ------------------------------------------------------------ synthetic clones
def fit_gaussian_clone(x: Tensor, y: Tensor, n: int, seed: int,
                       jitter: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Per-class Gaussian fitted on flattened images; sample n images (class frequencies of y)."""
    g = torch.Generator().manual_seed(seed)
    shape = x.shape[1:]
    flat = x.reshape(len(x), -1).double().cpu()
    y = y.cpu()
    yg = y[torch.randint(0, len(y), (n,), generator=g)]
    out = torch.empty(n, flat.shape[1], dtype=torch.float64)
    eye = torch.eye(flat.shape[1], dtype=torch.float64)
    for c in yg.unique().tolist():
        xc = flat[y == c]
        mean = xc.mean(0)
        centered = xc - mean
        cov = centered.T @ centered / max(len(xc) - 1, 1) + jitter * eye
        chol = torch.linalg.cholesky(cov)
        idx = (yg == c).nonzero().squeeze(1)
        z = torch.randn(len(idx), flat.shape[1], generator=g, dtype=torch.float64)
        out[idx] = mean + z @ chol.T
    return out.clamp(0, 1).float().reshape(n, *shape), yg


def independent_sample(x: Tensor, y: Tensor, n: int, seed: int) -> tuple[Tensor, Tensor]:
    """Each pixel/channel copied from a random training image of the same class."""
    g = torch.Generator().manual_seed(seed)
    x, y = x.cpu(), y.cpu()
    yg = y[torch.randint(0, len(y), (n,), generator=g)]
    out = torch.empty(n, *x.shape[1:], dtype=x.dtype)
    for c in yg.unique().tolist():
        xc = x[y == c]
        idx = (yg == c).nonzero().squeeze(1)
        pick = torch.randint(0, len(xc), (len(idx), *x.shape[1:]), generator=g)
        out[idx] = xc.gather(0, pick)  # gather along dim 0: out[i,c,h,w] = xc[pick[i,c,h,w],c,h,w]
    return out, yg


# ------------------------------------------------------------- test-set cache
def _cmnist_images(raw: dict, split: str) -> tuple[Tensor, Tensor]:
    return raw[f"x_{split}"].float() / 255, raw[f"y_{split}"]


def _build_cifar(cfg: RunConfig, raw: dict) -> dict[str, TensorDataset]:
    n = cfg.eval_size
    x_tr, y_tr = raw["x_train"].float() / 255, raw["y_train"]
    x_te, y_te = raw["x_test"][:n].float() / 255, raw["y_test"][:n]
    xg, yg = fit_gaussian_clone(x_tr, y_tr, n, TESTSET_SEED)
    xi, yi = independent_sample(x_tr, y_tr, n, TESTSET_SEED)
    return {"real": TensorDataset(x_te, y_te),
            "gaussian": TensorDataset(xg, yg),
            "independent": TensorDataset(xi, yi)}


def _build_cmnist(cfg: RunConfig, raw: dict) -> dict[str, TensorDataset]:
    n = cfg.eval_size
    g = torch.Generator().manual_seed(TESTSET_SEED)
    gray, y = _cmnist_images(raw, "test")
    gray, y = gray[:n], y[:n]
    matched = colorize(gray, y)
    rand_idx = torch.randint(0, 10, y.shape, generator=g)
    rand = colorize(gray, rand_idx)
    y_color = (y + torch.randint(1, 10, y.shape, generator=g)) % 10
    conflict = colorize(gray, y_color)
    gray_tr, y_tr = _cmnist_images(raw, "train")
    x_tr = colorize(gray_tr, sample_color_idx(y_tr, cfg.color_corr, g))
    xg, yg = fit_gaussian_clone(x_tr, y_tr, n, TESTSET_SEED)
    return {"color_matched": TensorDataset(matched, y),
            "color_random": TensorDataset(rand, y),
            "color_conflict": TensorDataset(conflict, y, y_color),
            "gaussian": TensorDataset(xg, yg)}


def build_test_sets(cfg: RunConfig, raw: dict) -> dict[str, TensorDataset]:
    """Build (or load from data/testsets/) the fixed test sets for this experiment family."""
    cmnist = cfg.experiment.startswith("cmnist")
    key = f"cmnist_n{cfg.eval_size}_c{cfg.color_corr:.2f}" if cmnist else f"cifar_n{cfg.eval_size}"
    names = (["color_matched", "color_random", "color_conflict", "gaussian"] if cmnist
             else ["real", "gaussian", "independent"])
    folder = run_dirs(cfg)["data"] / "testsets"
    paths = {name: folder / f"{name}_{key}.pt" for name in names}
    if all(p.exists() for p in paths.values()):
        return {name: TensorDataset(*torch.load(p)) for name, p in paths.items()}
    sets = (_build_cmnist if cmnist else _build_cifar)(cfg, raw)
    folder.mkdir(parents=True, exist_ok=True)
    for name, ds in sets.items():
        torch.save(list(ds.tensors), paths[name])
    return sets


# --------------------------------------------------------------- augmentation
def gpu_augment_cifar(x: Tensor) -> Tensor:
    """Random 32x32 crop of the 4-pixel zero-padded image + random horizontal flip, on x.device."""
    b, c, h, w = x.shape
    dev = x.device
    padded = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=dev)
    ox = torch.randint(0, 9, (b,), device=dev)
    rows = (oy[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    flip = torch.rand(b, device=dev) < 0.5
    cols = torch.where(flip[:, None, None, None], 2 * ox[:, None, None, None] + w - 1 - cols, cols)
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    return padded[bi, ci, rows, cols]

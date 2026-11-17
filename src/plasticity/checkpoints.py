"""Reusable checkpoints (loadable with this package alone) and the checkpoint manifest."""
from __future__ import annotations

import csv
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import RunConfig, run_id
from .models import resnet18_cifar

MANIFEST_COLUMNS = ["path", "run_id", "experiment", "phase", "step", "phase_step", "epoch",
                    "full_state", "commit", "saved_at"]
_STATE_KEYS = ("model_state", "optimizer_state", "rng_state", "extra_state")


def git_version() -> dict:
    """Commit hash and dirty flag of the repo containing this package (not the process cwd)."""
    unknown = {"commit": "unknown", "dirty": True}
    here = str(Path(__file__).resolve().parent)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                              text=True, timeout=30)
        if head.returncode != 0 or not head.stdout.strip():
            return unknown
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=here, capture_output=True, text=True, timeout=30)
        if status.returncode != 0:
            return unknown
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return unknown


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _atomic_save(obj: dict, path: Path) -> None:
    """Write to a temp file in the same directory, then os.replace; final mode 0o666 & ~umask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(obj, tmp)
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path: Path, model: nn.Module, cfg: RunConfig, *, step: int, phase: str,
                    phase_step: int, epoch: float, data_recipe: dict, optimizer=None,
                    extra_state: dict | None = None) -> None:
    ckpt = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "step": int(step),
        "phase": phase,
        "phase_step": int(phase_step),
        "epoch": float(epoch),
        "code_version": git_version(),
        "data_recipe": data_recipe,
        "run_id": run_id(cfg),
        "full_state": optimizer is not None,
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["rng_state"] = _rng_state()
        ckpt["extra_state"] = extra_state or {}
    _atomic_save(ckpt, Path(path))


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint; returns (eval-mode model, info without state)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = resnet18_cifar(alpha=1.0)  # weights come from the file, so alpha is irrelevant here
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    info = {k: v for k, v in ckpt.items() if k not in _STATE_KEYS}
    return model, info


def load_training_state(path, model: nn.Module, optimizer, device) -> dict:
    """Restore model, optimizer and RNG states from a full-state save; returns extra_state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not ckpt.get("full_state"):
        raise ValueError(f"{path} is not a full-state checkpoint (no optimizer/RNG state)")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    rng = ckpt["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    return ckpt.get("extra_state") or {}


def append_manifest(root: Path, row: dict) -> None:
    """Append one row to root/checkpoints/manifest.csv, writing the header if the file is new."""
    path = Path(root) / "checkpoints" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(row)
    if new:
        os.chmod(path, 0o666 & ~_current_umask())

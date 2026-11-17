"""Evaluation, cheap weight statistics, and the CSV metrics logger."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

COLUMNS = [
    "run_id", "experiment", "phase", "global_step", "phase_step", "epoch", "test_set",
    "loss", "acc", "acc_color_label", "lr", "train_loss_avg", "weight_norm",
    "dist_from_init", "wall_time_s",
]


def evaluate(model, test_sets: dict, device, batch_size: int = 1000) -> dict:
    """Per set: mean CE loss, accuracy vs (shape) label, and vs color label if present."""
    was_training = model.training
    model.eval()
    results = {}
    try:
        with torch.no_grad():
            for name, ds in test_sets.items():
                n = loss_sum = correct = correct_color = 0
                has_color = len(ds.tensors) == 3
                for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    logits = model(x)
                    loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    if has_color:
                        correct_color += (pred == batch[2].to(device)).sum().item()
                    n += y.shape[0]
                results[name] = {
                    "loss": loss_sum / n,
                    "acc": correct / n,
                    "acc_color_label": correct_color / n if has_color else None,
                }
    finally:
        model.train(was_training)
    return results


def weight_norm(model) -> float:
    return torch.sqrt(sum((p.detach().float() ** 2).sum() for p in model.parameters())).item()


def dist_from_init(model, init_state: dict) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        ref = init_state[name].to(p.device)
        total += ((p.detach().float() - ref.float()) ** 2).sum().item()
    return total ** 0.5


class MetricsLogger:
    def __init__(self, path):
        self.path = Path(path)

    def log(self, rows: list[dict]) -> None:
        for row in rows:
            extra = set(row) - set(COLUMNS)
            if extra:
                raise ValueError(f"unknown metrics columns: {sorted(extra)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            for row in rows:
                w.writerow({k: ("" if row.get(k) is None else row[k]) for k in COLUMNS})

    def last_step(self) -> int | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        steps = []
        with open(self.path, newline="") as f:
            for r in csv.DictReader(f):
                if r["global_step"] != "":
                    steps.append(int(float(r["global_step"])))
        return max(steps) if steps else None

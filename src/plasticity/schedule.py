"""Step schedules: when to check (evaluate + log), save weights, save full state."""
from __future__ import annotations

EXPERIMENTS = ("cifar_fresh", "cmnist_phase1", "cmnist_phase2")
SCHEDULES = ("dense", "sparse")


def log_steps(total: int) -> list[int]:
    """Powers of two 1, 2, 4, ... not exceeding total."""
    out, s = [], 1
    while s <= total:
        out.append(s)
        s *= 2
    return out


def _half_epochs(steps_per_epoch: int, total: int) -> set[int]:
    out, k = set(), 1
    while round(k * steps_per_epoch / 2) <= total:
        out.add(round(k * steps_per_epoch / 2))
        k += 1
    return out


def _every_epochs(n: int, steps_per_epoch: int, total: int) -> set[int]:
    return set(range(n * steps_per_epoch, total + 1, n * steps_per_epoch))


def check_steps(steps_per_epoch: int, total: int) -> list[int]:
    steps = {0, total} | set(log_steps(total)) | _half_epochs(steps_per_epoch, total)
    return sorted(s for s in steps if 0 <= s <= total)


def save_steps(experiment: str, schedule: str, steps_per_epoch: int, total: int,
               durations: list[int]) -> list[int]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    spe = steps_per_epoch
    if schedule == "dense":
        if experiment == "cifar_fresh":
            steps = {0, total} | set(log_steps(total)) | _every_epochs(8, spe, total)
        elif experiment == "cmnist_phase1":
            steps = {0, total} | set(log_steps(total)) | _half_epochs(spe, total)
        else:  # phase 2: step 0 is the branch point, already saved by phase 1
            steps = {total} | set(log_steps(total)) | _every_epochs(5, spe, total)
    else:
        if experiment == "cmnist_phase2":
            steps = {total}
        else:
            steps = {0, total}
            if experiment == "cmnist_phase1":
                steps |= {n * spe for n in durations}
    return sorted(s for s in steps if 0 <= s <= total)


def full_state_steps(experiment: str, steps_per_epoch: int, total: int,
                     durations: list[int]) -> list[int]:
    steps = {total}
    if experiment == "cmnist_phase1":
        steps |= {n * steps_per_epoch for n in durations
                  if n > 0 and n * steps_per_epoch <= total}
    return sorted(steps)

# checkpoints/

Saved models, shared with the team. **Contents live on the SCC only**
(`/projectnb/ds598ns/projects/Team2project/checkpoints/`) and are excluded from git
because each file is about 45 MB.

Everyone in the `ds598ns` group can read these files; only the owner's account writes
them. Copy a checkpoint to your own folder if you want to modify it.

Layout:

```
checkpoints/
  manifest.csv                 one row per saved checkpoint
  cifar_fresh/<run_id>/        Run 1: ResNet18 on CIFAR-10
  cmnist/<run_id>/             Run 2: Colored MNIST, phase 1 and each phase-2 branch
    run_config.json            every setting used for the run
    step_000000000.pt          one file per saved step, named by the *global* step
```

## Run ids

`resnet18_a{alpha}_lr{lr}_wd{weight_decay}_bs{batch_size}_seed{seed}`, then for Colored
MNIST `_color{color_corr:.2f}` (the Phase-1 color correlation), and for a Phase-2 branch
`_N{branch_n}ep`. Examples:

- `cifar_fresh/resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0`
- `cmnist/resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0_color1.00` (Phase 1)
- `cmnist/resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0_color1.00_N4ep` (branch at epoch 4)

Run length (`total_steps`, `phase1_epochs`, `phase2_epochs`) is not part of the id.
A Phase-2 branch at epoch N starts from Phase 1's `step_{N * steps_per_epoch:09d}.pt`
(N = 0 is the untrained step-0 model); its own files continue the global step count, so
its first file is at step `N * steps_per_epoch + 1`.

## Loading one

```python
from plasticity.checkpoints import load_checkpoint

model, info = load_checkpoint("checkpoints/cmnist/<run_id>/step_000004690.pt", device="cpu")
print(info["step"], info["phase"], info["epoch"], info["config"]["lr"])
logits = model(x)            # x: float images in [0, 1], shape (B, 3, 32, 32)
```

`model` is a CIFAR-style ResNet18 in eval mode. Use `device="cuda"` on a GPU node.
To continue training exactly from a full-state file (optimizer and RNG restored):

```python
import torch
from plasticity.checkpoints import load_training_state
from plasticity.models import resnet18_cifar

model = resnet18_cifar()
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
extra = load_training_state(path, model, opt, device="cpu")   # ValueError if weights-only
```

## What each file holds

| Field | Meaning |
|---|---|
| `model_state` | weights and BatchNorm buffers (CPU tensors) |
| `config` | every `RunConfig` setting (alpha, lr, weight_decay, batch_size, seed, color_corr, branch_n, ...) |
| `run_id` | the run id above |
| `step` | global step (Phase 2 continues Phase 1's count) |
| `phase` | `train` (CIFAR), `phase1` or `phase2` |
| `phase_step` | step within the phase (Phase 2 restarts at 0) |
| `epoch` | global step / steps per epoch |
| `code_version` | `{"commit": git hash, "dirty": uncommitted changes?}` |
| `data_recipe` | dataset, experiment, `color_corr` used for training in this phase, the color table, test-set seed (1234), `eval_size`, `train_subset`, test-set names, steps per epoch (Phase 2 also: `branch_n`, Phase-1 run id) |
| `full_state` | `True` if the file also holds the fields below |
| `optimizer_state`, `rng_state`, `extra_state` | full-state files only: SGD momentum buffers, Python/NumPy/torch RNG, and the loop's bookkeeping (phase step, wall time, color generator, AMP scaler) |

Full-state files are written at the end of every run, at each Phase-1 duration N, and at
the first check after every `full_state_every_s` seconds (2 hours by default). A job that
is restarted with the same settings resumes from the latest one.

`manifest.csv` columns: `path` (relative to the project root), `run_id`, `experiment`,
`phase`, `step`, `phase_step`, `epoch`, `full_state`, `commit`, `saved_at`.

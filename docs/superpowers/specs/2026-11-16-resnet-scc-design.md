# ResNet18 on SCC: CIFAR "sudden drop" run and Colored-MNIST two-phase experiment

Date: 2026-11-16 · Status: draft for approval

## 1. Intent

The project asks whether long training on low-order statistical structure damages a
network's later ability to learn higher-order structure (see the project doc, "Research
Question" and "Colored MNIST" sections).

This first build gives the team three things:

1. **Training scripts** that run ResNet18 on BU's SCC GPUs.
2. **Comprehensive logs and plots** that show the training curves, in particular the
   "sudden drop": accuracy on a low-order-only test set rises early and then falls (or
   stalls) while accuracy on real data keeps rising.
3. **A shared, reproducible checkpoint folder** so teammates can load the model at any
   saved point and compute statistics of different orders (Hermite coefficients, NTK
   alignment, clone accuracy gaps, dormant units, …) without retraining.

## 2. Scope

### In scope (v1)

**Run 1: fresh ResNet18 on CIFAR-10.** This is the proposal's sanity check 1 and a
reproduction of the Belrose et al. vision result with ResNet18. It is tested during
training on:

| Test set | Built from | Keeps |
|---|---|---|
| `real` | CIFAR-10 test split (10k) | everything |
| `gaussian` | per-class mean + covariance of the CIFAR train set | order ≤ 2 |
| `independent` | per-class, per-pixel marginals of the train set | order 1 |

**Run 2: Colored MNIST, Phase 1 → Phase 2, with a control.**

- **Phase 1 (color phase):** each digit class is drawn in its own fixed color, and color
  predicts the label 100% of the time (configurable, e.g. 0.9). It lasts N epochs.
- **Switch:** colors become random, so color carries no label information.
- **Phase 2 (shape phase):** training continues on randomly colored digits for a fixed
  length.
- **Control:** random colors from the start (N = 0), the "never had a color phase" network.
- **Durations:** one Phase-1 run per seed, with checkpoints saved at every N in the
  duration grid. Each Phase-2 branch starts from the Phase-1 checkpoint at its N, so we
  never retrain Phase 1 for each duration.

Colored-MNIST test sets, evaluated in both phases:

| Test set | What it is | What it tells us |
|---|---|---|
| `color_matched` | test digits in their class color | color + shape |
| `color_random` | test digits in random colors | shape only |
| `color_conflict` | each digit in the color of a different class | shape vs color reliance; log accuracy against the shape label and against the color label |
| `gaussian` | per-class mean + covariance of the Phase-1 training images | order ≤ 2 |

### Out of scope for v1

- CIFAR blur → sharp validation (a later, smaller run on 2–3 settings).
- MNIST → CIFAR real-vs-GM branching (the doc's Flow section).
- Onset experiments (a color phase placed in the middle of training). The config format
  should allow adding phases later, but v1 only runs "color first, then random".
- The full α / LR / WD sweep. v1 runs one default setting; the sweep is the next step,
  using the same scripts with different arguments.
- Computing T_K, Hermite coefficients, NTK statistics, etc. These are the teammates'
  work on the saved checkpoints. v1 only logs cheap generic numbers (weight norm and
  distance from init).

### Anti-goals (would count as failure even if the code runs)

- Checkpoints that can't be loaded without reading the training code, or that don't
  record the settings, seed, data recipe and code version that produced them.
- Curves that can't be regenerated from the saved logs.
- Real and fake test sets that differ in preprocessing, so a gap could come from
  preprocessing instead of statistics.
- Filling the project storage quota.

## 3. Constraints

- **SCC access:** every new SSH connection needs the user's password and a Duo push.
  Claude can't type into the user's own terminal tab. Running jobs therefore needs
  either (a) the user running the commands Claude gives, or (b) a one-time SSH
  connection-sharing setup in `~/.ssh/config`, made only with the user's OK. This is
  decided at implementation time and does not change the code.
- **SCC is a batch system (SGE).** GPU work runs through `qsub` job files with
  `-P ds598ns` and `-l gpus=1`, never on the login node. Module names, available GPU
  types and the project's storage quota must be checked on SCC before the first job.
- **Location:** outputs go under `/projectnb/ds598ns/students/maxineyu/`. Teammates in
  the `ds598ns` group need read access to `checkpoints/`, `logs/` and `figures/`.
- **Code lives in the team GitHub repo**
  (`jtsang27/DS598-FinalProject`). SCC holds a clone of it plus the outputs.
- Dependencies stay minimal: PyTorch, torchvision, numpy, pandas, matplotlib. No
  `concept-erasure`, HuggingFace Trainer or Weights & Biases.

## 4. Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Own short PyTorch training loop + SGE job files + CSV logs + matplotlib plots** | Exact control over when we save, switch phase and test; plain files anyone can read; few dependencies. More code written from scratch. | **Chosen by the user.** |
| B. Extend the example's `train_vision.py` (HuggingFace Trainer) | Reuses more code, but the Trainer fights exact-step saves and a mid-run data switch, and needs the Colab patch fixes again. | Rejected. |
| C. PyTorch Lightning + Weights & Biases | Live dashboards, but adds accounts and dependencies; teammates would need W&B to see curves. | Rejected. |

## 5. Design

### 5.1 Folder layout on SCC

```
/projectnb/ds598ns/students/maxineyu/ds598-plasticity/
  doc/           experiment-plan.md (plain English) + technical-design.md (this file)
  code/          clone of the team repo (git pull to update)
  env/           Python virtual environment
  data/          downloads (CIFAR-10, MNIST) and generated test sets, saved once
  checkpoints/   shared, read-only for teammates
    README.md    how to load a checkpoint, folder naming, what each field means
    manifest.csv one row per saved checkpoint (path, run_id, phase, step, epoch, …)
    <experiment>/<run_id>/
      run_config.json
      step_000000000.pt …
  logs/
    <experiment>/<run_id>/metrics.csv     one row per (check, test set)
    <experiment>/<run_id>/train.log       human-readable progress
    sge/                                  job stdout/stderr
  figures/
    <experiment>/…png
```

`run_id` encodes the settings, e.g.
`resnet18_a1.0_lr0.05_wd5e-4_bs128_seed0` and, for branches,
`…_color1.00_N4ep`.

### 5.2 Code layout in the repo

```
src/plasticity/
  data.py         CIFAR-10, Colored MNIST, Gaussian clone, independent sampler, test-set builders
  models.py       CIFAR-style ResNet18 + init-scale α
  train.py        the training loop, phases, check schedule (CLI entry point)
  checkpoints.py  save / load_checkpoint() / manifest
  metrics.py      evaluation on every test set; weight norm, distance from init
  plot.py         figures from metrics.csv (CLI entry point)
configs/
  cifar_fresh.json
  cmnist_phase1.json
  cmnist_phase2.json
jobs/
  setup_env.sh    one-time environment setup on SCC
  cifar_fresh.qsub
  cmnist_phase1.qsub
  cmnist_phase2.qsub   array job, one task per duration N
tests/
  test_smoke.py   tiny CPU run of each experiment (a few steps), checks files are written and loadable
```

Run: `python -m plasticity.train --config configs/cifar_fresh.json --seed 0 --out-root <root>`.
Command-line flags override config values (for the later sweep).

### 5.3 Data

- **Images are 32×32×3 floats in [0, 1]** for every dataset, so one ResNet18 handles both.
- **CIFAR-10:** torchvision download. Training augmentation: random crop (pad 4) +
  horizontal flip. No augmentation on test sets.
- **Colored MNIST:** MNIST padded from 28×28 to 32×32. Pixel intensity multiplies a class
  RGB color on a black background. Ten fixed, well-separated colors, listed in the config.
  - Phase-1 training images: with probability `color_corr` the class color, otherwise a
    random color from the ten.
  - "Random" means uniform over the ten colors, drawn fresh each time an image is loaded.
  - No crop or flip augmentation (flips change digit identity).
- **Gaussian clone:** fit each class's mean and covariance (3,072 dimensions) on the
  training set the model trains on (for Colored MNIST, the Phase-1 colored training set).
  Add a small diagonal jitter for stability. Sample 10k test images once with a fixed
  seed, clip to [0, 1] and save to `data/`, so every run is tested on identical fakes.
- **Independent sampler (CIFAR):** for each pixel and channel, take the value from a
  random training image of the same class. Sample 10k once and save.

### 5.4 Model

- CIFAR-style ResNet18: 3×3 stride-1 stem, no max-pool, standard BatchNorm, 10 outputs.
- **Init scale α:** after standard initialization, multiply every conv and linear weight
  by α. Saved as step 0. v1 uses α = 1.

### 5.5 Training

- SGD, momentum 0.9, constant learning rate (default 0.05), weight decay (default 5e-4),
  batch size 128, mixed precision on GPU.
- Constant LR is chosen so LR and WD stay clean knobs for the later sweep.
- **Run 1 length:** 2^16 = 65,536 steps (as in Belrose et al.).
- **Colored MNIST:** Phase 1 runs to the largest N in the duration grid (default
  N ∈ {1, 2, 4, 8, 16} epochs; control N = 0). Each Phase-2 branch runs a fixed 20 epochs
  with a fresh optimizer and the same LR / WD.
- Seeds: v1 runs seed 0 to check everything, then seeds 1–4 (5 total, as the proposal asks).

### 5.6 When to test and save

- **Checks** (evaluate every test set + write metrics): step 0, then steps 1, 2, 4, 8, …
  within each phase, plus every half epoch, plus the last step.
- Checks are cheap (metrics only), so they always follow the full schedule. Weight
  files are 45 MB each for ResNet18, so saving them follows a thinner schedule:

| Where | Weights saved at | Files (seed 0) | Size |
|---|---|---|---|
| Run 1 (CIFAR, 65,536 steps ≈ 168 epochs) | step 0, log-spaced steps, every 8 epochs, end | ≈ 39 | ≈ 1.8 GB |
| Colored MNIST Phase 1 (16 epochs ≈ 7,500 steps) | step 0, log-spaced, every half epoch, end | ≈ 47 | ≈ 2.1 GB |
| Colored MNIST Phase 2 (6 branches × 20 epochs) | log-spaced from the switch, every 5 epochs, end | ≈ 6 × 19 | ≈ 5.1 GB |

- **Seed 0 saves the full schedule above (≈ 9 GB).** Seeds 1–4 save weights only at
  step 0, each duration N, and the end of each run (≈ 0.6 GB per seed). Their metrics are
  still logged on the full check schedule, so curves and the plasticity ratio cover all
  seeds. A flag (`--save-schedule dense|sparse`) switches between the two.
- **Full state** (weights + optimizer + RNG states) is saved at each Phase-1 duration N,
  at the end of every run, and every 2 hours. A teammate can continue training exactly
  from these (e.g. Thura's interventions), and a job that hits its time limit resumes
  from the latest one.
- Before the first job, check the project quota (`pquota`). If ≈ 12 GB doesn't fit,
  thin the half-epoch and every-N-epoch saves first, never the log-spaced ones.

### 5.7 Checkpoint contents (what makes them reusable)

Each `.pt` file holds:

- `model_state` (and `optimizer_state`, `rng_state` for full saves)
- `config`: every setting, including α, LR, WD, batch size, seed, dataset, color
  correlation, phase, duration N
- `step`, `phase`, `phase_step`, `epoch`
- `code_version`: git commit hash of the repo (and a flag if there were uncommitted changes)
- `data_recipe`: dataset name, color table, Gaussian-clone file and its seed

`load_checkpoint(path)` returns `(model, info)` ready to evaluate, on CPU or GPU, with
nothing else to import except this package. `checkpoints/README.md` shows a 5-line example.
After each run, the job appends to `manifest.csv` and sets group read permission.

### 5.8 Logs

`metrics.csv`, one row per test set per check:

`run_id, experiment, phase, global_step, phase_step, epoch, test_set, loss, acc,
acc_color_label, lr, train_loss_avg, weight_norm, dist_from_init, wall_time_s`

(`acc_color_label` is only filled for `color_conflict`.)

`train.log` gives a human-readable line per check, plus the GPU name, settings and timing.

### 5.9 Figures (`plot.py`)

- **Run 1:** accuracy and loss vs step (log₂ axis) on real / gaussian / independent, one
  line each, chance level marked. This is where the sudden drop should show.
- **Colored MNIST, one figure per duration N:** accuracy on all four test sets across both
  phases, with a vertical line at the switch.
- **Colored MNIST, overlay:** `color_random` accuracy in Phase 2 for every N plus the
  control, on one axis aligned at the switch. This shows whether longer color phases slow
  shape learning.
- **Summary table** (CSV + printed): for each N, the Phase-2 steps to reach 90% of the
  control's final `color_random` accuracy, divided by the control's steps (the plasticity
  ratio for this experiment).
- With several seeds, curves show the mean with a band of ± 1 standard deviation.

### 5.10 Verification

- `tests/test_smoke.py` runs each experiment for a few steps on CPU with a tiny subset and
  checks that metrics, checkpoints and the manifest are written and that
  `load_checkpoint` reproduces the same test accuracy.
- On SCC, a short interactive or 10-minute batch job runs each config before the full jobs.
- Run 1 passes when its curves show the expected pattern (gaussian accuracy rising above
  chance early, then falling behind real). If it doesn't, we stop and investigate before
  the Colored-MNIST runs.

## 6. Decisions and calls

Made in conversation by the user:

- Approach A.
- v1 covers Run 1 (CIFAR) and Run 2 (Colored MNIST color → random + control) only.
- Colored MNIST uses 100% color correlation in Phase 1, with the correlation configurable.
- Run 1 goes first.
- A separate shared checkpoint folder that teammates can reuse.

Claude's calls (correct any of these):

- Generate Colored MNIST ourselves rather than download the IEEE DataPort version, for
  exact control of color and correlation.
- The `color_conflict` test set, to show directly whether the model follows color or shape.
- Duration grid N ∈ {0, 1, 2, 4, 8, 16} epochs, fixed in epochs for v1. T_K-based branch
  points come later, once teammates compute T_K from the saved Phase-1 checkpoints.
- Phase 2 is 20 epochs.
- SGD with a constant LR (0.05) and WD 5e-4, batch size 128.
- Run 1 is 65,536 steps.

## 7. Open questions for the team (do not block v1)

- **BatchNorm and α:** with BatchNorm, scaling conv weights by α mostly changes the
  effective learning rate, not the function the network computes. For the α part of the
  study, the team may want a variant without BatchNorm, or α applied differently. v1 uses
  α = 1, so this does not block it.
- **Phase-2 target:** the plasticity ratio here uses 90% of the control's final
  `color_random` accuracy, mirroring the proposal's CIFAR definition. Confirm.

## 8. Left to the builder

- Exact SCC module versions and GPU request flags, after checking on SCC.
- Check-schedule bookkeeping, file naming details and plot styling within the rules above.
- How often to write full-state saves for resuming, within the job's time limit.

# ResNet18 SCC Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scripts that train ResNet18 on SCC GPUs for Run 1 (CIFAR-10 "sudden drop") and Run 2 (Colored MNIST color phase → random colors + control), with full metric logs, curve plots and reusable checkpoints.

**Architecture:** A small Python package `plasticity` (src layout) with one module per responsibility: config, data, model, checkpoints, metrics, training loop, plots. One CLI entry point trains any experiment from a JSON config plus flag overrides; SGE job files wrap it. Everything a teammate needs to reuse a model is inside each checkpoint file.

**Tech Stack:** Python 3.12 (SCC module `python3/3.12.4`), PyTorch + torchvision (pip wheels in a venv; SCC's `pytorch` modules top out at 1.13), numpy, pandas, matplotlib, pytest. SGE (`qsub`).

**Spec:** `doc/technical-design.md` (plain-English companion: `doc/experiment-plan.md`).

## Global Constraints

- Project root on SCC: `/projectnb/ds598ns/projects/Team2project/` (a clone of `jtsang27/DS598-FinalProject`, branch `main`).
- Outputs go in the root's `data/`, `checkpoints/`, `logs/`, `figures/`; `env/` holds the venv. Large files never enter git (`.gitignore` already covers `data/*`, `checkpoints/*`, `env/`, `*.pt`, `logs/sge/`).
- Dependencies limited to: torch, torchvision, numpy, pandas, matplotlib (+ pytest for tests). No concept-erasure, HuggingFace or W&B.
- All images are float tensors, shape `(3, 32, 32)`, values in `[0, 1]`.
- Model: CIFAR-style ResNet18 — 3×3 stride-1 stem, no max-pool, BatchNorm, 10 outputs; init scale α multiplies every conv and linear weight after default init.
- Defaults: SGD momentum 0.9, constant lr 0.05, weight decay 5e-4, batch size 128, mixed precision on GPU, α = 1.0.
- Run 1: 65,536 steps. Colored MNIST: Phase 1 to 16 epochs, durations N ∈ {0, 1, 2, 4, 8, 16}; Phase 2 = 20 epochs, fresh optimizer, same lr/wd; color correlation 1.0 (configurable).
- Test sets: 10,000 images each, generated once with fixed seed 1234 and cached in `data/testsets/`.
- Steps per epoch = `len(train_set) // batch_size` (the loader drops the last partial batch), so Phase-1 step `N · steps_per_epoch` is exactly the end of epoch N, and Phase 2 finds its branch checkpoint by that step number.
- Phase-2 check and save schedules are computed on `phase_step` (steps since the switch); file names and `global_step` use the continuing global count.
- `metrics.csv` columns, in this order: `run_id, experiment, phase, global_step, phase_step, epoch, test_set, loss, acc, acc_color_label, lr, train_loss_avg, weight_norm, dist_from_init, wall_time_s`.
- Checkpoint files named `step_{global_step:09d}.pt`; contents per design §5.7.
- SGE jobs: `-P ds598ns`, `-l gpus=1`, `-l gpu_c=7.0`, job output to `logs/sge/`.
- Teammates have read-only access: every written file must be group-readable (`umask 022`; the folder is setgid group `ds598ns`).

## Review Focus

- **Job killed at the 12 h limit mid-run** → rerunning the same command resumes from the latest full-state save, does not restart from step 0, and does not duplicate rows in `metrics.csv`. (Task 6, `test_resume_continues_without_duplicate_rows`)
- **Phase-2 job started before Phase 1 wrote checkpoint N** → fails immediately with a message naming the missing file, instead of silently training from a fresh network. (Task 6, `test_phase2_missing_branch_checkpoint_errors`)
- **Teammate loads a checkpoint on a CPU-only machine** → `load_checkpoint` works with `device="cpu"` even though it was saved from a GPU AMP run. (Task 4, `test_load_checkpoint_on_cpu`)
- **Singular class covariance in the Gaussian clone** (many constant black pixels in MNIST) → fitting still succeeds thanks to diagonal jitter; samples are finite and in [0, 1]. (Task 2, `test_gaussian_clone_handles_constant_pixels`)
- **Two runs with different settings writing to the same folders** → run ids differ for every setting that changes, so no run overwrites another. (Task 1, `test_run_id_changes_with_every_setting`)

---

## File Structure

```
pyproject.toml                 package metadata, deps, pytest config
src/plasticity/__init__.py
src/plasticity/config.py       RunConfig dataclass, load_config, run_id, paths
src/plasticity/data.py         CIFAR / MNIST loading, colorize, Gaussian clone, independent sampler, test-set cache, GPU augmentation
src/plasticity/models.py       resnet18_cifar
src/plasticity/checkpoints.py  save/load checkpoints, manifest, git version
src/plasticity/metrics.py      evaluate on test sets, weight stats, CSV logger
src/plasticity/schedule.py     check and save step schedules
src/plasticity/train.py        training loop for the 3 experiments, resume, CLI
src/plasticity/plot.py         figures and plasticity table, CLI
configs/cifar_fresh.json
configs/cmnist_phase1.json
configs/cmnist_phase2.json
jobs/setup_env.sh
jobs/smoke.qsub
jobs/cifar_fresh.qsub
jobs/cmnist_phase1.qsub
jobs/cmnist_phase2.qsub        array job, task i → N = [0,1,2,4,8,16][i-1]
tests/conftest.py              tiny fake datasets so tests never download
tests/test_config.py
tests/test_data.py
tests/test_models.py
tests/test_checkpoints.py
tests/test_metrics.py
tests/test_schedule.py
tests/test_train.py
tests/test_plot.py
checkpoints/README.md          (exists; updated in Task 6)
```

Tests run locally on CPU (`pytest -q`) and on SCC. They use tiny in-memory fake datasets via a `fake_data` fixture; no network access.

---

### Task 1: Package skeleton and config

**Files:**
- Create: `pyproject.toml`, `src/plasticity/__init__.py`, `src/plasticity/config.py`, `configs/cifar_fresh.json`, `configs/cmnist_phase1.json`, `configs/cmnist_phase2.json`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass RunConfig` with fields (defaults): `experiment: str` (`"cifar_fresh" | "cmnist_phase1" | "cmnist_phase2"`), `seed: int = 0`, `alpha: float = 1.0`, `lr: float = 0.05`, `momentum: float = 0.9`, `weight_decay: float = 5e-4`, `batch_size: int = 128`, `total_steps: int = 65536` (cifar only), `phase1_epochs: int = 16`, `durations: list[int] = [0,1,2,4,8,16]`, `phase2_epochs: int = 20`, `branch_n: int | None = None` (phase 2 only), `color_corr: float = 1.0`, `eval_size: int = 10000`, `save_schedule: str = "dense"` (`"dense" | "sparse"`), `amp: bool = True`, `num_workers: int = 4`, `full_state_every_s: int = 7200`, `train_subset: int | None = None` (tests), `root: str = "."`.
  - `load_config(path: str, overrides: dict) -> RunConfig` — JSON file values, then non-None overrides; unknown keys raise `ValueError`.
  - `run_id(cfg: RunConfig) -> str` — `resnet18_a{alpha}_lr{lr}_wd{weight_decay}_bs{batch_size}_seed{seed}`; cmnist adds `_color{color_corr:.2f}`; phase 2 adds `_N{branch_n}ep`. Phase 1 and phase 2 of the same setting share the prefix.
  - `phase1_run_id(cfg) -> str` = `run_id(replace(cfg, experiment="cmnist_phase1", branch_n=None))` — where a Phase-2 run finds its branch checkpoints.
  - `run_dirs(cfg) -> dict[str, Path]` with keys `ckpt`, `logs`, `figures`, `data` = `root/checkpoints/<experiment>/<run_id>`, `root/logs/<experiment>/<run_id>`, `root/figures/<experiment>`, `root/data`. Phase 1 and 2 both use experiment folder `cmnist`.

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`:

```python
def test_load_config_applies_overrides(tmp_path):
    p = tmp_path / "c.json"; p.write_text('{"experiment": "cifar_fresh", "lr": 0.1}')
    cfg = load_config(str(p), {"seed": 3, "lr": None})
    assert cfg.lr == 0.1 and cfg.seed == 3 and cfg.batch_size == 128

def test_load_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "c.json"; p.write_text('{"experiment": "cifar_fresh", "lrr": 0.1}')
    with pytest.raises(ValueError, match="lrr"):
        load_config(str(p), {})

def test_run_id_formats():
    assert run_id(RunConfig("cifar_fresh")) == "resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0"
    assert run_id(RunConfig("cmnist_phase2", branch_n=4)).endswith("_color1.00_N4ep")

def test_run_id_changes_with_every_setting():
    base = RunConfig("cmnist_phase2", branch_n=2)
    for field, value in [("alpha", 2.0), ("lr", 0.1), ("weight_decay", 0.0),
                         ("batch_size", 64), ("seed", 1), ("color_corr", 0.9), ("branch_n", 4)]:
        assert run_id(replace(base, **{field: value})) != run_id(base), field

def test_run_dirs_share_cmnist_folder():
    a = run_dirs(RunConfig("cmnist_phase1", root="/r"))
    b = run_dirs(RunConfig("cmnist_phase2", branch_n=1, root="/r"))
    assert a["ckpt"].parent == b["ckpt"].parent == Path("/r/checkpoints/cmnist")
```

- [ ] **Step 2: Run** `pytest tests/test_config.py -q` → FAIL (module not found).
- [ ] **Step 3: Implement** `pyproject.toml` (setuptools, `src` layout, package `plasticity`, deps from Global Constraints, `[tool.pytest.ini_options] pythonpath = ["src"]`), `config.py` as above, and the three JSON configs (each sets only `experiment`; `cmnist_phase2.json` also documents that `branch_n` comes from the command line).
- [ ] **Step 4: Run** `pytest tests/test_config.py -q` → 5 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: package skeleton and run config"`

---

### Task 2: Data — datasets, colorize, Gaussian clone, independent sampler, test-set cache

**Files:**
- Create: `src/plasticity/data.py`, `tests/conftest.py`, `tests/test_data.py`

**Interfaces:**
- Consumes: `RunConfig`, `run_dirs` (Task 1).
- Produces:
  - `COLORS: torch.Tensor` shape `(10, 3)`, ten fixed well-separated RGB colors in [0, 1] (red, orange, yellow, green, cyan, blue, purple, magenta, white, brown).
  - `load_cifar(data_root: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]` → `x_train uint8 (50000,3,32,32), y_train int64, x_test uint8 (10000,3,32,32), y_test`.
  - `load_mnist32(data_root: Path) -> same shapes but (N,1,32,32)` (28×28 padded by 2 on each side).
  - `colorize(gray: Tensor[B,1,32,32] float, color_idx: Tensor[B] long) -> Tensor[B,3,32,32]` = `gray * COLORS[color_idx][:, :, None, None]`.
  - `sample_color_idx(y: Tensor, color_corr: float, generator) -> Tensor` — with prob `color_corr` returns `y`, else uniform over 0–9.
  - `fit_gaussian_clone(x: Tensor[N,3,32,32] float, y: Tensor, n: int, seed: int, jitter: float = 1e-3) -> tuple[Tensor, Tensor]` — per-class mean + covariance on flattened images, `cov + jitter·I`, Cholesky, sample `n` images with class frequencies of `y`, clip to [0, 1].
  - `independent_sample(x, y, n, seed) -> tuple[Tensor, Tensor]` — each pixel/channel copied from a random same-class training image.
  - `build_test_sets(cfg: RunConfig, raw: dict) -> dict[str, TensorDataset]` — cifar: `real, gaussian, independent`; cmnist: `color_matched, color_random, color_conflict, gaussian`. `color_conflict` datasets hold three tensors `(x, y_shape, y_color)` with `y_color = (y + offset) % 10`, offset uniform in 1–9. The cmnist Gaussian clone is fitted on Phase-1 training images colored with `cfg.color_corr`. Seed 1234. Cached to `data/testsets/{name}_{key}.pt` where key encodes experiment family, `eval_size` and (cmnist) `color_corr`; reloaded when present.
  - `gpu_augment_cifar(x: Tensor[B,3,32,32]) -> Tensor` — random crop with 4-pixel zero padding + random horizontal flip, done on the batch's device.
  - `raw` dict format (from `load_raw(cfg) -> dict`): `{"x_train","y_train","x_test","y_test"}` uint8/int64 tensors; `train_subset` truncates train.

- [ ] **Step 1: Write the failing tests.** `tests/conftest.py` defines fixture `fake_raw(kind)` returning random uint8 tensors with the real shapes but 200 train / 100 test images, labels cycling 0–9 (MNIST-like fake: mostly zeros with a random blob). Tests in `tests/test_data.py`:

```python
def test_colorize_uses_class_color():
    gray = torch.ones(2, 1, 32, 32)
    out = colorize(gray, torch.tensor([0, 5]))
    assert out.shape == (2, 3, 32, 32)
    assert torch.allclose(out[1, :, 0, 0], COLORS[5])

def test_sample_color_idx_correlation():
    y = torch.arange(10).repeat(1000); g = torch.Generator().manual_seed(0)
    assert torch.equal(sample_color_idx(y, 1.0, g), y)
    frac = (sample_color_idx(y, 0.0, g) == y).float().mean()
    assert 0.07 < frac < 0.13                       # chance level ≈ 0.1

def test_gaussian_clone_matches_class_means(fake_raw):
    x = fake_raw("cifar")["x_train"].float() / 255; y = fake_raw("cifar")["y_train"]
    xg, yg = fit_gaussian_clone(x, y, n=2000, seed=0)
    assert xg.shape == (2000, 3, 32, 32) and xg.min() >= 0 and xg.max() <= 1
    m_real = x[y == 3].mean(0); m_fake = xg[yg == 3].mean(0)
    assert (m_real - m_fake).abs().mean() < 0.05

def test_gaussian_clone_handles_constant_pixels():
    x = torch.zeros(100, 3, 32, 32); x[:, :, 10:20, 10:20] = torch.rand(100, 3, 10, 10)
    xg, _ = fit_gaussian_clone(x, torch.arange(10).repeat(10), n=50, seed=0)
    assert torch.isfinite(xg).all()

def test_independent_sample_copies_same_class_pixels(fake_raw): ...
    # every output pixel value exists at that position in some same-class training image

def test_test_sets_cached_and_identical(tmp_path, fake_raw):
    cfg = RunConfig("cifar_fresh", root=str(tmp_path), eval_size=100)
    a = build_test_sets(cfg, fake_raw("cifar")); b = build_test_sets(cfg, fake_raw("cifar"))
    assert set(a) == {"real", "gaussian", "independent"}
    assert torch.equal(a["gaussian"].tensors[0], b["gaussian"].tensors[0])
    assert any((tmp_path / "data" / "testsets").iterdir())

def test_cmnist_test_sets(tmp_path, fake_raw):
    sets = build_test_sets(RunConfig("cmnist_phase1", root=str(tmp_path), eval_size=100), fake_raw("mnist"))
    assert set(sets) == {"color_matched", "color_random", "color_conflict", "gaussian"}
    x, ys, yc = sets["color_conflict"].tensors
    assert x.shape[1:] == (3, 32, 32) and (ys != yc).all()

def test_gpu_augment_keeps_shape_and_range():
    x = torch.rand(8, 3, 32, 32); out = gpu_augment_cifar(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
```

- [ ] **Step 2: Run** `pytest tests/test_data.py -q` → FAIL.
- [ ] **Step 3: Implement** `data.py`. `load_cifar`/`load_mnist32` use `torchvision.datasets` with `download=True` into `data_root`, converted once to tensors. Test sets for `real` / `color_*` use the first `eval_size` test images.
- [ ] **Step 4: Run** `pytest tests/test_data.py -q` → all pass.
- [ ] **Step 5: Commit** `git commit -m "feat: datasets, colored MNIST, gaussian clone and test-set cache"`

---

### Task 3: Model

**Files:**
- Create: `src/plasticity/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `resnet18_cifar(num_classes: int = 10, alpha: float = 1.0) -> nn.Module` — torchvision `resnet18(num_classes=...)` with `conv1` replaced by 3×3 stride-1 padding-1 conv (no bias), `maxpool = nn.Identity()`, then every `nn.Conv2d` / `nn.Linear` weight multiplied by `alpha`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_output_shape():
    assert resnet18_cifar()(torch.rand(4, 3, 32, 32)).shape == (4, 10)

def test_alpha_scales_weights():
    torch.manual_seed(0); a = resnet18_cifar(alpha=1.0)
    torch.manual_seed(0); b = resnet18_cifar(alpha=2.0)
    assert torch.allclose(b.fc.weight, 2 * a.fc.weight)
    assert torch.allclose(b.conv1.weight, 2 * a.conv1.weight)
    assert torch.allclose(b.bn1.weight, a.bn1.weight)      # BatchNorm untouched

def test_stem_is_cifar_style():
    m = resnet18_cifar()
    assert m.conv1.kernel_size == (3, 3) and m.conv1.stride == (1, 1)
    assert isinstance(m.maxpool, nn.Identity)
```

- [ ] **Step 2: Run** `pytest tests/test_models.py -q` → FAIL.
- [ ] **Step 3: Implement** as specified.
- [ ] **Step 4: Run** → 3 passed.
- [ ] **Step 5: Commit** `git commit -m "feat: CIFAR-style ResNet18 with init scale"`

---

### Task 4: Checkpoints and manifest

**Files:**
- Create: `src/plasticity/checkpoints.py`, `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: `RunConfig`, `run_id` (Task 1); `resnet18_cifar` (Task 3).
- Produces:
  - `git_version() -> dict` → `{"commit": str, "dirty": bool}`; `{"commit": "unknown", "dirty": True}` outside a git repo.
  - `save_checkpoint(path: Path, model, cfg: RunConfig, *, step: int, phase: str, phase_step: int, epoch: float, data_recipe: dict, optimizer=None, extra_state: dict | None = None) -> None` — writes atomically (temp file then rename). Dict keys: `model_state` (tensors moved to CPU), `config` (`asdict(cfg)`), `step`, `phase`, `phase_step`, `epoch`, `code_version`, `data_recipe`, `run_id`, `full_state: bool`; when `optimizer` is given also `optimizer_state`, `rng_state` (python, numpy, torch, cuda if available) and `extra_state`.
  - `load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]` — rebuilds `resnet18_cifar(alpha=1.0)` (weights come from the file, so α is irrelevant on load), loads `model_state`, `.eval()`, returns `(model, info)` where `info` is the dict without the state tensors.
  - `load_training_state(path, model, optimizer, device) -> dict` — restores model, optimizer and RNG states; returns `extra_state`.
  - `append_manifest(root: Path, row: dict) -> None` — appends to `root/checkpoints/manifest.csv` with columns `path, run_id, experiment, phase, step, phase_step, epoch, full_state, commit, saved_at`; writes header if new.

- [ ] **Step 1: Write the failing tests:**

```python
def test_roundtrip_same_outputs(tmp_path):
    m = resnet18_cifar().eval(); x = torch.rand(2, 3, 32, 32)
    save_checkpoint(tmp_path / "s.pt", m, RunConfig("cifar_fresh"), step=8, phase="train",
                    phase_step=8, epoch=0.02, data_recipe={"dataset": "cifar10"})
    m2, info = load_checkpoint(tmp_path / "s.pt")
    assert torch.allclose(m(x), m2(x)) and info["step"] == 8
    assert info["config"]["lr"] == 0.05 and "commit" in info["code_version"]
    assert info["full_state"] is False

def test_load_checkpoint_on_cpu(tmp_path):
    m = resnet18_cifar().to(torch.float16).to(torch.float32)   # mimics AMP-trained weights
    save_checkpoint(tmp_path / "s.pt", m, RunConfig("cifar_fresh"), step=0, phase="train",
                    phase_step=0, epoch=0.0, data_recipe={})
    model, _ = load_checkpoint(tmp_path / "s.pt", device="cpu")
    assert next(model.parameters()).device.type == "cpu"

def test_full_state_restores_optimizer_and_rng(tmp_path): ...
    # save with optimizer after one SGD step; restore into fresh model+optimizer;
    # momentum buffers equal; torch.rand(1) after restore equals torch.rand(1) after save

def test_manifest_appends_rows(tmp_path):
    append_manifest(tmp_path, {...}); append_manifest(tmp_path, {...})
    df = pd.read_csv(tmp_path / "checkpoints" / "manifest.csv")
    assert len(df) == 2 and list(df.columns)[:3] == ["path", "run_id", "experiment"]
```

- [ ] **Step 2: Run** `pytest tests/test_checkpoints.py -q` → FAIL.
- [ ] **Step 3: Implement** as specified (`torch.load(..., map_location=device, weights_only=False)`).
- [ ] **Step 4: Run** → all pass.
- [ ] **Step 5: Commit** `git commit -m "feat: reusable checkpoints and manifest"`

---

### Task 5: Metrics and schedules

**Files:**
- Create: `src/plasticity/metrics.py`, `src/plasticity/schedule.py`, `tests/test_metrics.py`, `tests/test_schedule.py`

**Interfaces:**
- Produces (`metrics.py`):
  - `evaluate(model, test_sets: dict[str, TensorDataset], device, batch_size: int = 1000) -> dict[str, dict]` → per set `{"loss": float, "acc": float, "acc_color_label": float | None}`; for 3-tensor sets, `acc` is against `y_shape` and `acc_color_label` against `y_color`. Model is put in eval mode and restored to its previous mode.
  - `weight_norm(model) -> float` (L2 over all parameters); `dist_from_init(model, init_state: dict) -> float`.
  - `class MetricsLogger(path: Path)` with `.log(rows: list[dict])` writing the Global-Constraints columns (missing → empty) and `.last_step() -> int | None` (largest `global_step` already in the file, for resume).
- Produces (`schedule.py`):
  - `log_steps(total: int) -> list[int]` → `[1, 2, 4, …]` ≤ total.
  - `check_steps(steps_per_epoch: int, total: int) -> list[int]` → sorted unique `{0} ∪ log_steps ∪ {round(k·spe/2)} ∪ {total}`.
  - `save_steps(experiment: str, schedule: str, steps_per_epoch: int, total: int, durations: list[int]) -> list[int]`:
    - dense: `cifar_fresh` → `{0} ∪ log ∪ every 8 epochs ∪ {total}`; `cmnist_phase1` → `{0} ∪ log ∪ half epochs ∪ {total}`; `cmnist_phase2` → `log ∪ every 5 epochs ∪ {total}`.
    - sparse: `{0, total}` ∪ (phase 1: `{n·spe for n in durations}`).
  - `full_state_steps(experiment, steps_per_epoch, total, durations) -> list[int]` → `{total}` ∪ (phase 1: `{n·spe for n in durations if n > 0}`).

- [ ] **Step 1: Write the failing tests:**

```python
def test_check_steps_cover_log_half_epoch_and_end():
    s = check_steps(steps_per_epoch=10, total=40)
    assert s[:5] == [0, 1, 2, 4, 5] and 32 in s and s[-1] == 40 and s == sorted(set(s))

def test_dense_vs_sparse_phase1():
    dense = save_steps("cmnist_phase1", "dense", 10, 160, [0, 1, 2, 4, 8, 16])
    sparse = save_steps("cmnist_phase1", "sparse", 10, 160, [0, 1, 2, 4, 8, 16])
    assert {0, 10, 20, 40, 80, 160} <= set(sparse) <= set(dense)
    assert len(sparse) == 6

def test_full_state_at_durations():
    assert full_state_steps("cmnist_phase1", 10, 160, [0, 1, 4, 16]) == [10, 40, 160]

def test_evaluate_conflict_set_reports_both_labels(): ...
    # a model that always predicts class 0 on a 3-tensor set with y_shape=0, y_color=1
    # → acc == 1.0, acc_color_label == 0.0

def test_logger_resume_last_step(tmp_path):
    lg = MetricsLogger(tmp_path / "m.csv"); assert lg.last_step() is None
    lg.log([{"global_step": 4, "test_set": "real"}]); assert MetricsLogger(tmp_path / "m.csv").last_step() == 4
    assert open(tmp_path / "m.csv").readline().strip().split(",")[0] == "run_id"
```

- [ ] **Step 2: Run** `pytest tests/test_metrics.py tests/test_schedule.py -q` → FAIL.
- [ ] **Step 3: Implement** as specified.
- [ ] **Step 4: Run** → all pass.
- [ ] **Step 5: Commit** `git commit -m "feat: evaluation, metrics logger and check/save schedules"`

---

### Task 6: Training loop, resume and CLI

**Files:**
- Create: `src/plasticity/train.py`, `tests/test_train.py`
- Modify: `checkpoints/README.md` (real loading example and field list)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `train(cfg: RunConfig, raw: dict | None = None, device: str | None = None) -> Path` — returns the run's log dir. `raw=None` loads real data; tests pass fake data. Device auto: CUDA if available.
  - CLI: `python -m plasticity.train --config configs/X.json [--seed N] [--branch-n N] [--lr ...] [--weight-decay ...] [--alpha ...] [--color-corr ...] [--save-schedule dense|sparse] [--root PATH]` (every `RunConfig` field available as `--kebab-case`).

Behavior the tests pin:
- Writes `run_config.json` into both the checkpoint and log dirs, plus `train.log` (one line per check: step, epoch, train loss, every test-set acc, elapsed time; first lines: GPU name, run id, config).
- Seeds python/numpy/torch from `cfg.seed` before building the model; step-0 weights are kept as `init_state` for `dist_from_init`.
- **cifar_fresh:** CIFAR train set with `gpu_augment_cifar`, `total_steps` steps (looping over shuffled epochs), test sets `real/gaussian/independent`.
- **cmnist_phase1:** MNIST train set, each batch colorized with `sample_color_idx(y, cfg.color_corr, g)`, `phase1_epochs` epochs, phase name `"phase1"`.
- **cmnist_phase2:** requires `branch_n`. Loads weights from `checkpoints/cmnist/<phase1 run_id>/step_{branch_n·spe:09d}.pt` (N = 0 → step 0); if missing, raises `FileNotFoundError` naming the path. Fresh SGD optimizer. Each batch uses random colors (`color_corr=0`). `phase2_epochs` epochs, phase `"phase2"`; `global_step` continues from the branch step, `phase_step` restarts at 0.
- At each step in `check_steps`: evaluate, log rows (one per test set), and save a checkpoint if the step is in `save_steps`; full state if in `full_state_steps` or `full_state_every_s` elapsed. Every save appends to the manifest.
- **Resume:** if the checkpoint dir already holds full-state saves, load the latest with `load_training_state`, continue from its step, and skip logging checks `≤ MetricsLogger.last_step()`.
- Files are created with `os.umask(0o022)`.

- [ ] **Step 1: Write the failing tests** (all on CPU with `fake_raw`, tiny settings: `batch_size=16`, `train_subset=200`, `eval_size=100`, `total_steps=20`, `phase1_epochs=2`, `phase2_epochs=1`, `durations=[0,1,2]`, `amp=False`):

```python
def test_cifar_smoke_writes_logs_checkpoints_manifest(tmp_path, fake_raw):
    logs = train(tiny("cifar_fresh", tmp_path), raw=fake_raw("cifar"), device="cpu")
    df = pd.read_csv(logs / "metrics.csv")
    assert set(df.test_set) == {"real", "gaussian", "independent"}
    assert df.global_step.min() == 0 and df.global_step.max() == 20
    ckpts = sorted((tmp_path / "checkpoints" / "cifar_fresh").glob("*/step_*.pt"))
    assert ckpts and (tmp_path / "checkpoints" / "manifest.csv").exists()

def test_reloaded_checkpoint_reproduces_logged_accuracy(tmp_path, fake_raw): ...
    # load the last checkpoint with load_checkpoint, evaluate on the cached test sets,
    # acc equals the logged acc at that step (abs diff < 1e-6)

def test_cmnist_phase1_then_phase2_branches(tmp_path, fake_raw):
    train(tiny("cmnist_phase1", tmp_path), raw=fake_raw("mnist"), device="cpu")
    for n in [0, 1, 2]:
        logs = train(tiny("cmnist_phase2", tmp_path, branch_n=n), raw=fake_raw("mnist"), device="cpu")
        df = pd.read_csv(logs / "metrics.csv")
        assert set(df.phase) == {"phase2"} and df.phase_step.min() == 0
        assert df.global_step.min() == n * steps_per_epoch_tiny
        assert df.loc[df.test_set == "color_conflict", "acc_color_label"].notna().all()

def test_phase2_missing_branch_checkpoint_errors(tmp_path, fake_raw):
    with pytest.raises(FileNotFoundError, match="step_"):
        train(tiny("cmnist_phase2", tmp_path, branch_n=1), raw=fake_raw("mnist"), device="cpu")

def test_resume_continues_without_duplicate_rows(tmp_path, fake_raw):
    cfg = tiny("cifar_fresh", tmp_path, full_state_every_s=0)   # full save at every check
    train(replace(cfg, total_steps=10), raw=fake_raw("cifar"), device="cpu")
    logs = train(cfg, raw=fake_raw("cifar"), device="cpu")       # same run id, longer
    df = pd.read_csv(logs / "metrics.csv")
    assert not df.duplicated(["global_step", "test_set"]).any() and df.global_step.max() == 20
```

(`tiny(...)` is a helper in `tests/test_train.py` that returns a `RunConfig` with the tiny settings; `total_steps` is excluded from `run_id`, so the resume test reuses the same folder.)

- [ ] **Step 2: Run** `pytest tests/test_train.py -q` → FAIL.
- [ ] **Step 3: Implement** `train.py`; update `checkpoints/README.md` with the real loading example, the checkpoint fields (design §5.7) and the run-id format.
- [ ] **Step 4: Run** `pytest -q` → whole suite passes.
- [ ] **Step 5: Commit** `git commit -m "feat: training loop for CIFAR and colored-MNIST phases with resume"`

---

### Task 7: Plots and plasticity table

**Files:**
- Create: `src/plasticity/plot.py`, `tests/test_plot.py`

**Interfaces:**
- Consumes: `metrics.csv` files from Task 6 (columns in Global Constraints).
- Produces:
  - `load_runs(logs_root: Path, experiment_folder: str) -> pd.DataFrame` — concatenates every `metrics.csv` under `logs_root/<folder>/*/`.
  - `plot_cifar(df, out: Path) -> Path` — 2 panels (accuracy, loss) vs `global_step` on a log₂ x-axis (step 0 drawn at x = 0.5), one line per test set, mean across seeds with ±1 std band, dashed line at chance (0.1); `real` in black.
  - `plot_cmnist_branch(df, n: int, out: Path) -> Path` — Phase 1 (up to N) + Phase 2 of branch N, four test sets, vertical line at the switch.
  - `plot_cmnist_overlay(df, out: Path) -> Path` — `color_random` accuracy vs `phase_step` in Phase 2 for every N (control N = 0 in black).
  - `plasticity_table(df, target_frac: float = 0.9) -> pd.DataFrame` — per (setting, N): `steps_to_target`, `control_steps`, `ratio`; target = `target_frac` × control's final `color_random` acc; never reaching it → `steps_to_target = NaN`, `reached = False`.
  - CLI: `python -m plasticity.plot --root PATH --experiment cifar_fresh|cmnist` writes PNGs to `figures/<folder>/` and `plasticity_table.csv` (cmnist), and prints the table.

- [ ] **Step 1: Write the failing tests** with a synthetic `metrics.csv` built in the test (no training):

```python
def test_plot_cifar_writes_png(tmp_path): assert plot_cifar(synthetic_cifar(), tmp_path / "a.png").stat().st_size > 0

def test_plasticity_ratio(tmp_path):
    df = synthetic_cmnist(control_hits_target_at=100, branch_hits={4: 200, 16: None})
    t = plasticity_table(df).set_index("branch_n")
    assert t.loc[0, "ratio"] == 1.0 and t.loc[4, "ratio"] == 2.0
    assert not t.loc[16, "reached"] and pd.isna(t.loc[16, "ratio"])

def test_cli_writes_all_figures(tmp_path): ...
    # write synthetic cmnist logs under tmp_path/logs/cmnist/<run>/metrics.csv, run main([...]),
    # expect figures/cmnist/overlay.png, branch_N{n}.png per N, plasticity_table.csv
```

- [ ] **Step 2: Run** `pytest tests/test_plot.py -q` → FAIL.
- [ ] **Step 3: Implement** with matplotlib (`Agg` backend).
- [ ] **Step 4: Run** `pytest -q` → all pass.
- [ ] **Step 5: Commit** `git commit -m "feat: training-curve figures and plasticity table"`

---

### Task 8: SCC environment, job files, smoke job, launch Run 1

**Files:**
- Create: `jobs/setup_env.sh`, `jobs/smoke.qsub`, `jobs/cifar_fresh.qsub`, `jobs/cmnist_phase1.qsub`, `jobs/cmnist_phase2.qsub`
- Modify: `README.md` (a "Run on SCC" section with the commands below)

**Interfaces:**
- Consumes: the CLIs from Tasks 6 and 7.
- `setup_env.sh`: `module load python3/3.12.4`; create `env/` venv if missing; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`; `pip install -e ".[test]"`; print `torch.__version__`.
- Every `.qsub` starts with: `#$ -P ds598ns`, `#$ -l gpus=1`, `#$ -l gpu_c=7.0`, `#$ -l h_rt=12:00:00`, `#$ -j y`, `#$ -o logs/sge/`, `#$ -N <name>`; then `cd /projectnb/ds598ns/projects/Team2project`, `module load python3/3.12.4`, `source env/bin/activate`, `umask 022`, `nvidia-smi -L`, run the CLI, then (for training jobs) the plot CLI.
- `cifar_fresh.qsub` takes `SEED` and `SCHEDULE` from `qsub -v` (defaults 0, dense).
- `cmnist_phase2.qsub` is an array job `#$ -t 1-6`; task index → N via `N=(0 1 2 4 8 16)`; submitted with `-hold_jid cmnist_p1` so it waits for Phase 1.
- `smoke.qsub`: `h_rt=00:30:00`, runs `pytest -q`, then a 200-step `cifar_fresh` run and a 1-epoch Phase 1 + Phase 2 (N = 0, 1) with `--root /scratch/$USER/smoke` so the shared folders stay clean.

- [ ] **Step 1:** On SCC, `git pull` in `Team2project` and run `bash jobs/setup_env.sh` on the login node. Expected: prints a 2.x torch version.
- [ ] **Step 2:** `mkdir -p logs/sge && qsub jobs/smoke.qsub`. Expected after it ends: `logs/sge/smoke.o<id>` shows `passed`, a GPU name, and no traceback.
- [ ] **Step 3:** `qsub -v SEED=0,SCHEDULE=dense jobs/cifar_fresh.qsub`. Expected: `qstat -u maxineyu` shows it running; `logs/cifar_fresh/<run_id>/metrics.csv` grows; `checkpoints/manifest.csv` gains rows.
- [ ] **Step 4:** After Run 1 finishes, `python -m plasticity.plot --root . --experiment cifar_fresh`; inspect `figures/cifar_fresh/curves.png` for the expected pattern (Gaussian-clone accuracy above chance early, then falling behind real). If absent, stop and investigate before Run 2.
- [ ] **Step 5:** `qsub jobs/cmnist_phase1.qsub` then `qsub -hold_jid cmnist_p1 jobs/cmnist_phase2.qsub`. Expected: six Phase-2 tasks start after Phase 1 and each writes its own `logs/cmnist/<run_id>_N{n}ep/`.
- [ ] **Step 6: Commit** job files and README; push to `main`; `git pull` on SCC.

Seeds 1–4 (`SCHEDULE=sparse`) follow once seed 0's curves look right; they reuse the same job files with `-v SEED=k,SCHEDULE=sparse`.

# DS598 Final Project

**Research question:** does prolonged training on low-order (≤ 2nd-order) statistical structure selectively impair a network's later ability to learn higher-order structure, and how do duration, onset, init scale, learning rate and weight decay modulate it?

Built on [EleutherAI/features-across-time](https://github.com/EleutherAI/features-across-time) (Belrose et al. 2024, *Neural Networks Learn Statistics of Increasing Complexity*).

## Notebooks (run on Colab GPU)

| Notebook | What it does |
|---|---|
| [00 · Reproduce features-across-time](colab/00_reproduce_features_across_time.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtsang27/DS598-FinalProject/blob/main/colab/00_reproduce_features_across_time.ipynb) | Trains ConvNeXt on real CIFAR-10 and on its Gaussian (GM) clone, and plots accuracy on real / GM / independent-pixel / CQN test sets over training. |

In Colab: **Runtime → Change runtime type → GPU**, then run all cells. Checkpoints go to `MyDrive/DS598-FinalProject/runs` when Drive is mounted.

## ResNet18 runs on the SCC

The main experiments run on BU's SCC in **`/projectnb/ds598ns/projects/Team2project/`**, which is a clone of this repo. Everyone in the class can read it there; checkpoints, datasets and the Python environment live only on the SCC and are not committed (see `.gitignore`).

- **Start here:** [doc/experiment-plan.md](doc/experiment-plan.md) — plain-English plan: what we run, why, where results go, how to load a saved model.
- **Details for builders:** [doc/technical-design.md](doc/technical-design.md).

## Run on SCC

All commands run from `/projectnb/ds598ns/projects/Team2project` on a login node (`ssh scc1.bu.edu`). Jobs request 2 GPUs (compute capability 7.0 or better), 8 cores, 32 GB, 2 hours, and run two single-GPU runs side by side.

```bash
cd /projectnb/ds598ns/projects/Team2project
git pull
bash jobs/setup_env.sh               # once: venv in env/, torch + torchvision (CUDA 12.4), the plasticity package
mkdir -p logs/sge

qsub jobs/smoke.qsub                 # ~10 min check: pytest + short runs of all three experiments in /scratch/$USER/smoke

# Real runs
qsub -v SEEDS="0 1",SCHEDULE=dense jobs/cifar_fresh.qsub      # Run 1: CIFAR-10, seeds 0 and 1 (later: SEEDS="2 3",SCHEDULE=sparse, then "4")
qsub jobs/cmnist_phase1.qsub                                  # Colored-MNIST Phase 1, seeds 0 and 1
qsub -hold_jid cmnist_p1 jobs/cmnist_phase2.qsub              # Phase 2: array of 3 tasks (N = 0 1 | 2 4 | 8 16), waits for Phase 1

bash jobs/interactive.sh             # interactive 2-GPU shell for debugging
```

- **Status:** `qstat -u $USER`. Delete a job with `qdel <job-id>`.
- **Output:** job logs in `logs/sge/<name>.o<job-id>`; per-run `metrics.csv` in `logs/<experiment>/<run_id>/`; figures in `figures/<experiment>/`; checkpoints in `checkpoints/<experiment>/<run_id>/` (listed in `checkpoints/manifest.csv`).
- **Resuming:** if a job hits its time limit or fails, submit the same `qsub` command again; each run continues from its latest checkpoint and finished runs exit immediately.
- **Plots by hand:** `python -m plasticity.plot --root . --experiment cifar_fresh` (or `cmnist`).

## Folder layout

```
README.md
doc/           experiment plan (plain English) + technical design
colab/         Colab reproduction of features-across-time
src/ configs/ jobs/ tests/   ResNet18 training code, configs, SCC job files
data/          datasets + generated test sets      (SCC only, not in git)
checkpoints/   saved models, shared with the team   (SCC only, not in git)
logs/          metrics.csv per run
figures/       training-curve plots
```

## Files

- `colab/patch_fat.py` — small patch so the example's `train_vision.py` runs on one Colab GPU (auto device, no bf16 on T4, no wandb, skips eval sets that need missing files). Configured through `FAT_*` environment variables; see the file header.

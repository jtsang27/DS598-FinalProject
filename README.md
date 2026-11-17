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

## Folder layout

```
README.md
doc/           experiment plan (plain English) + technical design
colab/         Colab reproduction of features-across-time
src/ configs/ jobs/ tests/   ResNet18 training code (coming next)
data/          datasets + generated test sets      (SCC only, not in git)
checkpoints/   saved models, shared with the team   (SCC only, not in git)
logs/          metrics.csv per run
figures/       training-curve plots
```

## Files

- `colab/patch_fat.py` — small patch so the example's `train_vision.py` runs on one Colab GPU (auto device, no bf16 on T4, no wandb, skips eval sets that need missing files). Configured through `FAT_*` environment variables; see the file header.

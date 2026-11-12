# DS598 Final Project

**Research question:** does prolonged training on low-order (≤ 2nd-order) statistical structure selectively impair a network's later ability to learn higher-order structure, and how do duration, onset, init scale, learning rate and weight decay modulate it?

Built on [EleutherAI/features-across-time](https://github.com/EleutherAI/features-across-time) (Belrose et al. 2024, *Neural Networks Learn Statistics of Increasing Complexity*).

## Notebooks (run on Colab GPU)

| Notebook | What it does |
|---|---|
| [00 · Reproduce features-across-time](colab/00_reproduce_features_across_time.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jtsang27/DS598-FinalProject/blob/colab/reproduce-fat/colab/00_reproduce_features_across_time.ipynb) | Trains ConvNeXt on real CIFAR-10 and on its Gaussian (GM) clone, and plots accuracy on real / GM / independent-pixel / CQN test sets over training. |

In Colab: **Runtime → Change runtime type → GPU**, then run all cells. Checkpoints go to `MyDrive/DS598-FinalProject/runs` when Drive is mounted.

## Files

- `colab/patch_fat.py` — small patch so the example's `train_vision.py` runs on one Colab GPU (auto device, no bf16 on T4, no wandb, skips eval sets that need missing files). Configured through `FAT_*` environment variables; see the file header.

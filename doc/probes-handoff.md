# Probes handoff (Jonathan → Thura)

*20 Nov 2026 · branch `thura-probes-tests`*

## What changed

- `probe/probes.py` moved to `src/plasticity/probes.py`. Its relative imports (`from .checkpoints import …`) only work inside the package; from `probe/` it failed with `ImportError: attempted relative import with no known parent package`. Code is unchanged.
- New `tests/test_probes.py`: 13 tests, all passing. All 99 tests in the repo pass (51 s on a MacBook Air).

## What the tests check

- Small math against known answers:
  - `effective_rank` of an identity matrix is n, and of a rank-1 matrix is 1.
  - `kernel_alignment` of the label kernel with itself is 1.
  - `spectrum_rank`.
  - `conv_fc_weight_norm` ignores biases and BatchNorm.
- `hessian_top_eigenvalue` matches the exact Hessian's top eigenvalue on a tiny linear model.
- `empirical_ntk` is symmetric and PSD, and equals J Jᵀ built by hand. Tested with tiny blocks so the blocking is exercised.
- `activation_stats` counts exactly 2 dormant units in a model built with 2 dead units.
- `first_step_test` leaves the checkpoint unchanged, and with a small learning rate the loss goes down.
- `color_overlap`: a pure-gray filter gives gray overlap 1 and chroma 0, and the wrong input shape raises an error.
- `probe_run` end to end on fake Colored MNIST and CIFAR checkpoints:
  - the CSV has no missing values, and `dist_from_init` is 0 at step 0;
  - a partial re-run keeps the earlier rows.

## Real run (local, CPU)

Colored MNIST Phase 1 on a 10k-image subset, 4 epochs, branch points N = 0, 1, 4. Training took 15 min.

| Test set | Accuracy at epoch 4 |
|---|---|
| Color matched | 100% |
| Gaussian clone | 99.9% |
| Random color | 12.9% |
| Color conflict | 3.5% |

The network locks onto color, as in Maxine's full runs. The cheap probes took about 53 s per checkpoint on CPU and produced 28 statistics each.

```bash
python -m plasticity.train --config configs/cmnist_phase1.json --root ~/ds598-mini --phase1-epochs 4 --durations 0 1 4 --train-subset 10000 --eval-size 1000 --num-workers 0
python -m plasticity.probes --root ~/ds598-mini --run-dir ~/ds598-mini/checkpoints/cmnist/resnet18_a1.0_lr0.05_wd0.0005_bs128_seed0_color1.00 --branch-points --device cpu
```

Use `--device cpu` on a Mac: the NTK and alignment math is float64, which `mps` does not support.

## Notes for review

- On random weights with fake data, `fs_loss_drop` came out negative (loss rose over 10 steps at lr 0.05). The small-lr test shows the function itself is right. Still, check whether it stays negative on real checkpoints: if it does, the probe may need a smaller lr than the run's.
- NTK and Hessian are slow on CPU (about 140 s per checkpoint even at tiny sizes). Run `--stats all` on the SCC GPUs.

## Next steps (from the project doc)

1. Run the probes on Maxine's SCC checkpoints at every branch point, seeds 0–1, with `--stats all`.
2. Add Hermite coefficients c₁, c₂, c₄. This is the one statistic from the doc's table that is not coded yet.
3. Build the interventions applied at a pause: LR up or down, smaller batches, weight decay, L2-to-init, shrink-and-perturb, re-initialize dormant or color-locked neurons. Measure each as ΔR.
4. Write the analysis: Spearman and partial ρ of each probe with R, permutation tests with Holm correction, and Wilcoxon tests of ΔR > 0.

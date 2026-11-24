# Results (first round, 17 Nov 2026)

Copies of the figures and `metrics.csv` logs from the SCC runs. The checkpoints themselves are on the SCC only, in `/projectnb/ds598ns/projects/Team2project/checkpoints/`.

All runs: ResNet18, SGD lr 0.05 (constant), weight decay 5e-4, batch size 128, α = 1, on V100 GPUs.

## Run 1: CIFAR-10, fresh network (`cifar_fresh/`)

- Seed 0, 65,536 steps, about 30 min.
- **Up to about step 128:** real, Gaussian-clone and independent-pixel accuracy rise together. Only low-order statistics are being used.
- **From step 256:** real keeps rising to 87–90%. Independent-pixel accuracy drops back to chance. The Gaussian clone peaks at about 57%, then settles around 30–36%.
- A brief start-up instability (steps 16–64, from starting at lr 0.05 with no warmup) is left out of the loss panels. It isn't part of the analysis; a short warmup will remove it next round.
- Figures:
  - `curves_annotated.png`: smoothed, with notes.
  - `curves.png`: raw accuracy, and loss on a log scale from step 64.

## Run 2: Colored MNIST, color phase → random colors (`cmnist/`)

- Seeds 0 and 1.
- **Phase 1:** 16 epochs with color always matching the digit.
- **Phase 2:** 20 epochs with random colors, branched after N = 0, 1, 2, 4, 8, 16 Phase-1 epochs. N = 0 is the control: a fresh network.
- **Phase 1:**
  - The network locks onto color within half an epoch.
  - Color-matched accuracy is about 100%; random-color accuracy stays near chance.
  - On color-conflict digits it follows the color, not the shape.
- **Phase 2:** every branch eventually learns shape (about 99%), but longer color phases learn it more slowly.

Plasticity ratio = Phase-2 steps to reach 90% of the control's final random-color accuracy (target 89.1%), divided by the control's steps. The step counts are interpolated between checks, then averaged over 2 seeds.

| Color phase N (epochs) | Steps to target | Ratio |
|---|---|---|
| 0 (control) | 46 | 1.00 |
| 1 | 59 | 1.28 |
| 2 | 75 | 1.63 |
| 4 | 65 | 1.42 |
| 8 | 73 | 1.59 |
| 16 | 176 | **3.81** |

- Figures:
  - **`switch.png`: the loss figure to present.** Training loss (log scale) and shape accuracy for the first 512 steps after the switch, one line per N. The loss starts at about 10–15: the network is still answering by color, so it is confidently wrong. Longer color phases (N = 16 most) recover more slowly.
  - `overlay.png`: shape learning in Phase 2 for every N, over the whole phase.
  - `branch_N{n}.png`: both phases for one branch, with the switch marked. The loss panel is on a log scale and leaves out the start-up steps < 64.
  - `plasticity_table.csv`: the table above.
- **Caveats:**
  - Only 2 seeds so far.
  - The slowdown is temporary: every branch reaches about 99%.
  - N = 2–8 are not ordered.

`metrics/` holds one `metrics.csv` per run, in the format described in `doc/technical-design.md` §5.8.

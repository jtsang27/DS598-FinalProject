# Experiment plan: ResNet18 runs on the SCC

*DS598 final project · draft, 16 Nov 2026*

This page explains, in plain language, what we are about to run on BU's SCC cluster,
why, where the results will be saved, and how teammates can reuse them. The technical
version (for whoever writes the code) is
`docs/superpowers/specs/2026-11-16-resnet-scc-design.md` in the team repo.

---

## 1. What we want to find out

Our project question: **if a network spends a long time learning only simple patterns,
does it get worse at learning complex patterns afterwards?**

- **Simple (low-order) patterns** are things you can get from averages, such as "frog
  photos are greener on average" or "every 3 is drawn in green".
- **Complex (higher-order) patterns** are shapes, such as the outline of a wing or the
  loop of a 6.

This first round of runs does two things:

1. **Checks that our setup can see the effect other researchers found** (Run 1).
2. **Runs our main experiment** on colored digits (Run 2).

It also saves the trained models along the way, so teammates can measure statistics on
them without training anything themselves.

---

## 2. Run 1: ResNet18 on CIFAR-10 (the "sudden drop" check)

**What happens:** we train a standard image network (ResNet18) on CIFAR-10, a dataset of
small photos in 10 classes. While it trains, we keep testing it on three versions of the
test photos:

| Test set | What's in it | Why |
|---|---|---|
| Real | the actual test photos | normal accuracy |
| Gaussian clone | fake images that only have each class's average color layout and which pixels go together (simple patterns only; they look like colored noise) | shows how much the network relies on simple patterns |
| Independent pixels | fake images where each pixel is copied from a random photo of the same class (even simpler) | the simplest possible clue |

**What we expect to see:** early in training, the network scores above chance on the
fakes, because it is using simple patterns. Later its score on the fakes **drops**, while
its score on real photos keeps going up. That drop is the moment it switches to complex
patterns. Researchers (Belrose et al. 2024) found this, and we want to see it with our own
code before trusting the main experiment.

**Length:** 65,536 training steps (about 168 passes over the data), the same as the paper.

---

## 3. Run 2: Colored MNIST (the main experiment)

**The data:** handwritten digits (MNIST) where we color each digit. Every class has its own
color: all 0s red, all 1s orange, and so on.

**Phase 1, the color phase:** the color always matches the digit. The network can get
everything right just by looking at the color, so it learns a simple shortcut. We run this
phase for N passes over the data (N = 1, 2, 4, 8 or 16).

**The switch:** suddenly the colors become random. Color is now useless.

**Phase 2, the shape phase:** the network has to learn the actual digit shapes. We
measure **how fast it learns them**.

**The control:** a network that sees random colors from the very start, so it never had a
color phase. Everything is compared with this one.

**The question in one line:** *does a longer color phase (bigger N) make the network
slower at learning shapes afterwards, compared with the control?*

We test the network throughout on four versions of the test digits:

| Test set | What's in it | What it tells us |
|---|---|---|
| Color-matched | digits in their usual color | color + shape |
| Random color | digits in random colors | shape only |
| Color conflict | each digit in **another** class's color | is it following the color or the shape? |
| Gaussian clone | fake colored digits with only simple patterns | reliance on simple patterns |

**A shortcut we use:** we don't retrain Phase 1 for every N. We train Phase 1 once (for 16
passes), save the network after 1, 2, 4, 8 and 16 passes, and start a separate Phase 2 from
each of those saves.

---

## 4. Training settings (first round)

| Setting | Value | Note |
|---|---|---|
| Network | ResNet18, adapted for 32×32 images | used in the Refinetti et al. paper too |
| Optimizer | SGD with momentum 0.9 | |
| Learning rate | 0.05, constant | kept constant so it's a clean "knob" for later experiments |
| Weight decay | 5e-4 | |
| Batch size | 128 images | |
| Initial weight size (α) | 1 (normal) | other values come in a later round |
| Seeds | seed 0 first; then seeds 1–4 | 5 repeats in total, as the proposal asks |
| Phase 2 length | 20 passes | |

The later rounds change the learning rate, weight decay and α to answer the "what makes
it better or worse" part of our question. The same scripts handle that; only the settings
change.

---

## 5. Where everything is saved

Everything lives under `/projectnb/ds598ns/students/maxineyu/ds598-plasticity/`:

```
ds598-plasticity/
  doc/           this page and the technical design
  code/          a copy of the team GitHub repo (the scripts)
  env/           the Python setup
  data/          downloaded datasets and the generated fake test images
  checkpoints/   saved models, shared with the team (read-only for others)
  logs/          the numbers recorded during training
  figures/       the curve plots
```

- **checkpoints/** has a README explaining how to load a model, plus a list
  (`manifest.csv`) of every saved model: which run, which phase, which step.
- **logs/** has a `metrics.csv` per run, with one row per test set at every check: step,
  phase, loss, accuracy. Any curve can be redrawn from it.
- **figures/** has the plots: accuracy over training for each test set, with the switch
  marked for Colored MNIST, and one chart comparing shape learning across all N.

---

## 6. For teammates: using the saved models

Every saved model records everything needed to understand or recreate it: the settings,
the seed, how the data was made, which phase and step it was saved at, and the exact
version of the code.

Loading one takes a few lines:

```python
from plasticity.checkpoints import load_checkpoint
model, info = load_checkpoint("checkpoints/cmnist/<run>/step_000004690.pt")
print(info["phase"], info["step"], info["config"]["lr"])
# model is ready to evaluate; compute your statistics on it
```

**When models are saved:**

- at the very start (untrained; the "fresh" reference)
- often at the beginning (steps 1, 2, 4, 8, …), because that's when things change fastest
- regularly during training (every half pass in Phase 1)
- at each switch point and at the end

Some saves (at each switch point and at the end) also include the optimizer's state. From
those you can **continue training exactly**, for example to try an intervention such as
raising the learning rate at a halt.

**Storage:** each saved model is about 45 MB. Seed 0 saves the full schedule (about 9 GB);
seeds 1–4 save only the key points (about 0.6 GB each). All seeds record their full
curves in the logs either way.

---

## 7. How we'll know it worked

- Before the long runs, each experiment runs for a few minutes as a test. That checks the
  logs, plots and saved models are written correctly, and that a reloaded model gives the
  same accuracy.
- **Run 1 has to show the sudden drop** before we trust Run 2. If it doesn't, we stop and
  find out why.

---

## 8. Not in this round

- CIFAR blurred → sharp (a smaller check on natural images, later).
- MNIST → CIFAR with real vs fake branches (the proposal's Flow section).
- Placing the color phase in the middle of training instead of at the start ("onset").
- The full sweep over learning rate, weight decay and initial weight size.
- Statistics such as T_K, Hermite coefficients or NTK alignment. Teammates compute these
  from the saved models.

---

## 9. Open question for the team

**BatchNorm and initial weight size (α).** ResNet18 has BatchNorm layers, which rescale
each layer's output. Because of that, making the starting weights bigger mostly changes
how big each learning step effectively is, rather than what the network computes. For the
α part of our question, we may need a version of the network without BatchNorm. This
doesn't affect the first round, which uses normal-size weights (α = 1).

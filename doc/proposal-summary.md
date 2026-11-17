# Project proposal, in short

*DS598 final project, Team 2 · plain-language summary of the project doc*

---

## The question

- If a network trains for a long time on **simple patterns only**, does it get **worse at learning complex patterns** later?
- **Simple (low-order)** = averages, such as average color or which pixels tend to be bright together (order 1 and 2).
- **Complex (higher-order)** = shapes and object parts.
- We also ask:
  - Does it matter **how long** the simple phase lasts (duration)?
  - Does it matter **when** it happens (onset)?
  - Do **initial weight size (α)**, **learning rate** and **weight decay** make it better or worse?

---

## Why we think it might happen

- **Belrose et al. 2024:** networks learn simple statistics first and complex ones later.
  - Code we build on: [EleutherAI/features-across-time](https://github.com/EleutherAI/features-across-time)
  - MNIST is mostly solvable with simple statistics alone.
- **Refinetti et al. 2023:** a network trained on fake "simple-only" images keeps up with one trained on real images for only a short time.
  - On CIFAR-10 with ResNet18, that's about 50 steps.
- **Székely et al. 2024:** explains why some models can't pick up higher-order patterns. Also gives a ready-made generator for "higher-order only" data.

---

## Datasets

**1. Colored MNIST (main experiment)**
- Each digit class gets its own color.
- **Color** = the simple clue. **Digit shape** = the complex one.
- **Phase 1:** color matches the digit for N epochs.
- **Phase 2:** colors become random, so the network has to learn shape.
- We control exactly **how long** and **when** the color phase happens.
- Fast to run, so we can try many settings.

**2. CIFAR-10 (check on natural images)**
- Small 32×32 photos in 10 classes (planes, cars, frogs…).
- **Blur test:** train on blurred images (simple patterns only), then switch to sharp images (shapes appear).
- Run only on the few best settings found with Colored MNIST.

**Skip for now:** ImageNet, Stylized-ImageNet, Waterbirds. Each run takes too long.

---

## The fake "simple-only" images (Gaussian clone)

- For each class, measure the **average image** and **which pixels move together**.
- Generate new random images with exactly those statistics.
- They look like colored noise but carry the same simple patterns, and no complex ones.
- **Doing well on the fakes** means the network is using simple patterns.

---

## How we measure damage

- **Fresh network:** an untrained copy, used as the baseline.
- **Plasticity ratio:** steps a pre-trained copy needs to reach 90% of the fresh network's final accuracy, divided by the fresh network's steps.
  - **1** = no damage.
  - **2** = twice as slow.
  - **Never reaches the target** = strongest damage.
- **t\* (t-star):** the moment the real-trained copy clearly beats the fake-trained copy.
  - "Clearly" = by more than 3× the seed noise, 3 checks in a row.
  - It marks when the network starts using complex patterns.
- **T_K:** the point where the network's "way of seeing" stops changing beyond normal training noise.
  - Used to pick the save points 0.5×, 1×, 4×, 16× and 64× T_K.

---

## Plan (Flow section of the doc)

1. Pick a setting: α, learning rate, weight decay.
2. **Stage A:** train on the simple dataset for a long time, saving copies along the way.
3. **Stage B:** from each saved copy, train two ways:
   - on **real** images (simple + complex patterns);
   - on **fake** images (simple patterns only).
   - 5 seeds each.
4. Measure the plasticity ratio and t\* for every copy.
5. Rank the settings and test whether simple statistics at the save point predict the ranking.

**Sanity checks**
- A fresh network on CIFAR shows a clear split between real and fake.
- A fresh network on MNIST shows little or no split. That gives us the noise floor.

---

## What we expect

- **Selective damage:** a longer simple phase slows the **real** branch much more than the **fake** branch.
- **Later start:** t\* moves later as the simple phase gets longer.
- **Early hurts more:** a simple phase at the start of training does more damage than the same phase later on.
- **Knobs:**
  - weight decay **reduces** the damage;
  - larger initial weights **increase** it.
- **Mechanism:** how well the network's kernel matches the labels at the save point predicts the damage, better than weight size, dead neurons or sharpness.

---

## What to save (checkpoints)

- **Step 0:** the untrained network (the fresh baseline).
- **Early, log-spaced** (steps 1, 2, 4, 8, …): things change fastest here.
- **Every save point** used for Stage B: always saved.
- **After the switch**, log-spaced again, so we can find t\*.

---

## Statistics team (Thura)

**Measured at each saved copy:**
- **Order-specific:**
  - Hermite coefficients c₁, c₂, c₄;
  - NTK–label alignment;
  - first-step gradient test;
  - clone accuracy gaps;
  - first-layer direction overlap.
- **General:**
  - NTK effective rank;
  - feature rank;
  - dormant neurons;
  - weight norm and distance from init;
  - sharpness (top Hessian eigenvalue).

**Interventions to try at a save point:**
- raise the learning rate;
- smaller batch size;
- turn on weight decay;
- pull weights back toward their initial values;
- shrink-and-perturb;
- re-initialize dead neurons, or only the color-locked neurons.

**Questions to test:**
- Does higher-order sensitivity fall during the simple phase?
- Do the order-specific statistics predict damage better than the general ones?
- Can an intervention undo the damage, and does it help only complex-pattern learning?
- Does intervening early help more than intervening late?

---

## Where we are now

- **Built:** training scripts on BU's SCC, with checkpoints, logs and plots.
  - Code: `src/`, `jobs/`
  - How to run: `README.md`
- **Run 1 (CIFAR-10, fresh ResNet18, seed 0): done.**
  - Up to about step 128: real, Gaussian-clone and order-1 fake accuracy rise together.
  - After that: real keeps climbing to 87–90%, order-1 drops to chance, and the Gaussian clone settles around 30–36%.
- **Colored MNIST (Phase 1 → Phase 2, N = 0, 1, 2, 4, 8, 16, seeds 0 and 1): submitted.**
- **Not built yet:**
  - the CIFAR blur test;
  - the MNIST → CIFAR real-vs-fake branching;
  - onset experiments;
  - the full α / learning-rate / weight-decay sweep.

---

## Open questions for the team

- **S(T):** does it mean the real ratio ÷ the fake ratio?
- **σ in t\*:** the spread of each curve across seeds, or of the real − fake difference?
- **Maximum Stage B length:** it has to be long enough for the fresh network to level off.
- **α with BatchNorm:** ResNet18's BatchNorm cancels most of the effect of larger initial weights. The α study may need a version without BatchNorm.

---

## Links

- Code: <https://github.com/jtsang27/DS598-FinalProject>
- Shared SCC folder: `/projectnb/ds598ns/projects/Team2project`
- More detail: `doc/experiment-plan.md` (plain English) and `doc/technical-design.md`

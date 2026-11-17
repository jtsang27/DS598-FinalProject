# checkpoints/

Saved models, shared with the team. **Contents live on the SCC only**
(`/projectnb/ds598ns/projects/Team2project/checkpoints/`) and are excluded from git
because each file is about 45 MB.

Everyone in the `ds598ns` group can read these files; only the owner's account writes
them. Copy a checkpoint to your own folder if you want to modify it.

Planned layout (see `doc/experiment-plan.md`, section 6):

```
checkpoints/
  manifest.csv                 one row per saved checkpoint (run, phase, step, epoch, path)
  cifar_fresh/<run_id>/        Run 1: ResNet18 on CIFAR-10
  cmnist/<run_id>/             Run 2: Colored MNIST, phase 1 and phase-2 branches
    run_config.json            every setting used for the run
    step_000000000.pt          weights + settings + data recipe + code version
```

Loading one (once the code is in `src/`):

```python
from plasticity.checkpoints import load_checkpoint
model, info = load_checkpoint("checkpoints/cmnist/<run_id>/step_000004690.pt")
```

# data/

Datasets and generated test sets. **Contents live on the SCC only**
(`/projectnb/ds598ns/projects/Team2project/data/`) and are excluded from git.

Filled by the training scripts on first run:

- CIFAR-10 and MNIST downloads (torchvision)
- Generated test sets saved once with a fixed seed, so every run is tested on identical images:
  Gaussian clone (per-class mean + covariance), independent-pixel samples (CIFAR),
  and the Colored-MNIST colour table.

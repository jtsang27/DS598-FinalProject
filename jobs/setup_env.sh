#!/bin/bash
# One-time (or repeatable) Python environment setup on the SCC login node.
# Usage: bash jobs/setup_env.sh
set -euo pipefail
cd /projectnb/ds598ns/projects/Team2project
module load python3/3.12.4
[ -d env ] || python3 -m venv env
source env/bin/activate
# Home (10 GB quota) and /tmp (2 GB) are too small for the CUDA wheels: build and cache in the project area.
mkdir -p env/tmp
export TMPDIR="$PWD/env/tmp"
export PIP_NO_CACHE_DIR=1
python -m pip install --upgrade pip
# SCC's own pytorch modules are too old, so install from pip (CUDA 12.4 wheels).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[test]"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda)
PY
rm -rf env/tmp

# Prefetch datasets into data/ (the CIFAR-10 host is slow, ~100 KB/s, so do it once here, not inside jobs).
python - <<'PY'
from torchvision.datasets import CIFAR10, MNIST
for train in (True, False):
    CIFAR10("data", train=train, download=True)
    MNIST("data", train=train, download=True)
print("datasets ready in data/")
PY

#!/bin/bash
# One-time (or repeatable) Python environment setup on the SCC login node.
# Usage: bash jobs/setup_env.sh
set -euo pipefail
cd /projectnb/ds598ns/projects/Team2project
module load python3/3.12.4
[ -d env ] || python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
# SCC's own pytorch modules are too old, so install from pip (CUDA 12.4 wheels).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[test]"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda)
PY

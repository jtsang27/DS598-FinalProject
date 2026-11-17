#!/bin/bash
# Open an interactive shell on a 2-GPU node for debugging by hand.
# Then: cd /projectnb/ds598ns/projects/Team2project; module load python3/3.12.4; source env/bin/activate
CMD="qrsh -P ds598ns -l gpus=2 -l gpu_c=7.0 -l h_rt=02:00:00 -l mem_free=32G -pe smp 8"
echo "$CMD"
exec $CMD

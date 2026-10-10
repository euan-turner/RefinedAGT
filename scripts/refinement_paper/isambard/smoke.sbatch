#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-smoke
#SBATCH --gpus=4
#SBATCH --time=00:45:00
# Checks the node before anything is submitted: GPUs, torch build, NCCL all-reduce, the test suite.
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/refinement_paper/isambard/env.sh"
nvidia-smi --list-gpus
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
torchrun --standalone --nproc-per-node=4 "$AGT_REPO/scripts/refinement_paper/isambard/nccl_check.py"
cd "$AGT_REPO" && python -m pytest -q tests

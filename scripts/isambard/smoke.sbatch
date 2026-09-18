#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-smoke
#SBATCH --gpus=4
#SBATCH --time=00:30:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
nvidia-smi --list-gpus
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
torchrun --standalone --nproc-per-node=4 "$AGT_REPO/scripts/isambard/nccl_check.py"
cd "$AGT_REPO" && python -m pytest -q tests

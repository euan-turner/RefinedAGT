#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-stage
#SBATCH --gpus=1
#SBATCH --time=02:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
mkdir -p "$AGT_ROOT/.data" "$AGT_ROOT/logs" "$AGT_ROOT/manifests"
python -c "import os, torchvision; [torchvision.datasets.CIFAR10(root=os.environ['AGT_ROOT'] + '/.data', train=t, download=True) for t in (True, False)]"
python -c "from medmnist import OCTMNIST; OCTMNIST(split='train', download=True)"   # -> ~/.medmnist/octmnist.npz (all splits)
python -c "import uci_datasets; uci_datasets.Dataset('houseelectric')"              # bundled: confirms it loads

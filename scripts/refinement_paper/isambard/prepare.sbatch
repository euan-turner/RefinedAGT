#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-prepare
#SBATCH --gpus=1
#SBATCH --time=01:00:00
# Downloads every dataset and builds the files that parallel jobs would otherwise race to write: the
# CIFAR-10 and OCT-MNIST PCA bases and pre-trained models. Run once, and wait for it, before submit.sh.
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/refinement_paper/isambard/env.sh"
cd "$AGT_REPO/scripts/refinement_paper"
for task in cifar octmnist halfmoons magic; do
    python run.py "$task" --prepare
done

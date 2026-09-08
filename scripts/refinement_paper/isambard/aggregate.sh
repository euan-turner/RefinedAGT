#!/bin/bash
# Once every submitted job has finished: the table and results file of each task's ladder, and the
# paper figure. Reads run records only (no GPU), so it runs on a login node; exits listing any run
# that is missing.
set -euo pipefail
: "${AGT_REPO:?source scripts/refinement_paper/isambard/env.sh first}"
cd "$AGT_REPO/scripts/refinement_paper"
for task in cifar octmnist halfmoons magic; do
    python aggregate.py "$task"
done
python plot_ladders.py

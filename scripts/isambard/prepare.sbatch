#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-prepare
#SBATCH --gpus=4
#SBATCH --time=06:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
python -c "import cifar_pca; cifar_pca.fit_pca()"
python -c "import octmnist_pca; octmnist_pca.fit_pca()"
python -c "import train_uci, uci_refinement as u; train_uci.get_model(u.HIDDEN_LAY, u.HIDDEN_SIZE, u.SEED)"

# E0: the 324-point selection grid per feature width, the pre-training grid split over the 4 GPUs.
# Shards touch disjoint pre-trained models; the unsharded call reads the grid back and writes the selection.
for d in 20 22 24; do
    pids=()
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i python cifar_pca_selection.py --d "$d" --shard "$i" 4 \
            > "$SLURM_SUBMIT_DIR/selection_d${d}_shard${i}_${SLURM_JOB_ID}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done   # a failed shard fails the job (set -e)
    python cifar_pca_selection.py --d "$d"
done

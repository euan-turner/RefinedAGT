#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --job-name=agt-benchmark
#SBATCH --gpus=4
#SBATCH --time=03:00:00
set -euo pipefail
: "${AGT_REPO:?source scripts/isambard/env.sh before sbatch}"
source "$AGT_REPO/scripts/isambard/env.sh"
cd "$AGT_REPO/scripts/poisoning_paper"
BENCH="$AGT_ROOT/benchmark"
mkdir -p "$BENCH/.results" "$BENCH/.models"
ln -sfn "$AGT_ROOT/.data" "$BENCH/.data"
cp "$AGT_ROOT"/.results/cifar_pca_selected_*_d20.json "$BENCH/.results/"
cp "$AGT_ROOT"/.models/cifar_pca_*n_dims=20_*.ckpt "$AGT_ROOT"/.models/uci_*.ckpt "$BENCH/.models/"
export AGT_ROOT="$BENCH"
OUT="$SLURM_SUBMIT_DIR/benchmark_${SLURM_JOB_ID}.jsonl"

CIFAR="--pca-dims 20 --k 100 --eps 0.05 --n-splits 2"
for n in 1 2 4; do                                   # 2^16 leaves: scaling across 1/2/4 GPUs
    rm -f "$BENCH"/.results/cifar_20_*
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) torchrun --standalone --nproc-per-node="$n" \
        cifar_pca_run.py $CIFAR --split-dims 16 | tee -a "$OUT"
done
rm -f "$BENCH"/.results/cifar_20_*                   # 2^20 leaves on 4 GPUs: the E1 production size
torchrun --standalone --nproc-per-node=4 cifar_pca_run.py $CIFAR --split-dims 20 | tee -a "$OUT"

for n in 1 4; do                                     # UCI 2^11 leaves on 1 and 4 GPUs
    rm -f "$BENCH"/.results/uci_refinement_*
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) torchrun --standalone --nproc-per-node="$n" \
        uci_run.py --n-splits 2 | tee -a "$OUT"
done

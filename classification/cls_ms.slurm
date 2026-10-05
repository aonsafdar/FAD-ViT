#!/bin/bash
#SBATCH --job-name=cls_ms
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --chdir=/path/to/FAD-ViT/classification
#SBATCH --output=/path/to/FAD-ViT/classification/logs/%x_%j.out
#SBATCH --error=/path/to/FAD-ViT/classification/logs/%x_%j.err

# Multi-seed MedMNIST linear probe for ONE dataset + ONE seed (locked LR).
# 3 models x 3 fractions = 9 runs. Writes into the same seed_${SEED} layout as seed_0
# so existing single-run results are extended, not overwritten.
#   Usage: DATASET=bloodmnist SEED=1 sbatch cls_ms.slurm
set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT="/path/to/FAD-ViT"
EXP_ROOT="$ROOT/Experiments/classification_linear_probe"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/breast_ultrasound/dataset}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/classification_linear_probe_locked_lr}"

DATASET="${DATASET:?set DATASET}"
SEED="${SEED:?set SEED}"
MODELS="${MODELS:-cvt13-mod vit-s16 dinov2-base}"
FRACTIONS="${FRACTIONS:-0.01 0.1 1.0}"
EPOCHS="${EPOCHS:-50}"; WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"; EARLY_STOP="${EARLY_STOP:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"; OPTIMIZER="${OPTIMIZER:-adamw}"; WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CVT13_WEIGHTS="${CVT13_WEIGHTS:-$ROOT/weights/model_best.pth}"

lr_for_frac() { case "$1" in 0.01) echo 1e-3;; 0.1) echo 1e-3;; 1.0) echo 1e-4;; *) echo 1e-4;; esac; }

DATA_PATH="$DATA_ROOT/$DATASET"
[ -d "$DATA_PATH" ] || { echo "ERROR: missing $DATA_PATH" >&2; exit 2; }
[ -f "$CVT13_WEIGHTS" ] || { echo "ERROR: missing weights $CVT13_WEIGHTS" >&2; exit 2; }
mkdir -p "$EXP_ROOT/logs"

echo "=== CLS multi-seed: dataset=$DATASET seed=$SEED ==="
for ARCH in $MODELS; do
  for FRAC in $FRACTIONS; do
    LR=$(lr_for_frac "$FRAC")
    RUN_OUT="$OUT_ROOT/$DATASET/${ARCH}/fraction_${FRAC}/lr_${LR}/seed_${SEED}"
    if [ -f "$RUN_OUT/summary.json" ]; then echo "[skip] $RUN_OUT exists"; continue; fi
    mkdir -p "$RUN_OUT"
    echo "[RUN] $DATASET arch=$ARCH frac=$FRAC lr=$LR seed=$SEED"
    "$PY" "$EXP_ROOT/run_linear_probe_classification.py" \
      --arch "$ARCH" --data-path "$DATA_PATH" --fraction "$FRAC" \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
      --optimizer "$OPTIMIZER" --weight-decay "$WEIGHT_DECAY" \
      --warmup-epochs "$WARMUP_EPOCHS" --early-stop "$EARLY_STOP" \
      --seed "$SEED" --cvt13-weights "$CVT13_WEIGHTS" --output "$RUN_OUT"
  done
done
echo "=== DONE $DATASET seed=$SEED ==="

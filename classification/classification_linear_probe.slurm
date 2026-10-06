#!/bin/bash
#SBATCH --job-name=cls_linear_probe
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --chdir=/scratch/%u/breast_ultrasound/BU-Mamba
#SBATCH --output=/scratch/%u/breast_ultrasound/Experiments/logs/%x_%j.out
#SBATCH --error=/scratch/%u/breast_ultrasound/Experiments/logs/%x_%j.err

# Classification linear-probe data-efficiency benchmark:
# - Backbones: fadvit, vit-s16, dinov2-base
# - Train fractions: 1%, 10%, 100%
# - Train only linear classifier head
#
# Usage:
#   sbatch /scratch/$USER/breast_ultrasound/Experiments/classification_linear_probe/classification_linear_probe.slurm
#   DATASETS="pathmnist dermamnist octmnist" sbatch /scratch/$USER/breast_ultrasound/Experiments/classification_linear_probe/classification_linear_probe.slurm
#   DATA_PATH=/scratch/$USER/breast_ultrasound/dataset/pathmnist sbatch /scratch/$USER/breast_ultrasound/Experiments/classification_linear_probe/classification_linear_probe.slurm

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT="/scratch/$USER/breast_ultrasound"
EXP_ROOT="$ROOT/Experiments/classification_linear_probe"
OUT_ROOT="$ROOT/outputs/classification_linear_probe"

DATA_ROOT="${DATA_ROOT:-$ROOT/dataset}"
DATA_PATH="${DATA_PATH:-}"  # Optional single-dataset override
DATASETS="${DATASETS:-pathmnist dermamnist octmnist pneumoniamnist retinamnist breastmnist bloodmnist tissuemnist organamnist organcmnist organsmnist}"
MODELS="${MODELS:-fadvit vit-s16 dinov2-base}"
FRACTIONS="${FRACTIONS:-0.01 0.1 1.0}"
LR_GRID="${LR_GRID:-1e-5 3e-5 1e-4 3e-4 1e-3}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EPOCHS="${EPOCHS:-100}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EARLY_STOP="${EARLY_STOP:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-0}"
FADVIT_WEIGHTS="${FADVIT_WEIGHTS:-$ROOT/weights/model_best.pth}"

mkdir -p "$OUT_ROOT"

echo "=== Classification Linear Probe ==="
echo "DATA_ROOT=$DATA_ROOT"
echo "DATA_PATH(single-override)=$DATA_PATH"
echo "DATASETS=$DATASETS"
echo "MODELS=$MODELS"
echo "FRACTIONS=$FRACTIONS"
echo "LR_GRID=$LR_GRID"
echo "OPTIMIZER=$OPTIMIZER WD=$WEIGHT_DECAY"
echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE SEED=$SEED"
echo "OUT_ROOT=$OUT_ROOT"
echo "=================================="

if [ -n "$DATA_PATH" ]; then
  DATASETS="$(basename "$DATA_PATH")"
  DATA_ROOT="$(dirname "$DATA_PATH")"
fi

for DS in $DATASETS; do
  DS_PATH="$DATA_ROOT/$DS"
  if [ ! -d "$DS_PATH" ]; then
    echo "[WARN] skipping missing dataset path: $DS_PATH"
    continue
  fi
  for ARCH in $MODELS; do
    for FRAC in $FRACTIONS; do
      for LR in $LR_GRID; do
        RUN_OUT="$OUT_ROOT/${DS}/${ARCH}/fraction_${FRAC}/lr_${LR}/seed_${SEED}"
        mkdir -p "$RUN_OUT"
        echo ""
        echo "[RUN] dataset=$DS arch=$ARCH frac=$FRAC lr=$LR -> $RUN_OUT"
        "$PY" "$EXP_ROOT/run_linear_probe_classification.py" \
          --arch "$ARCH" \
          --data-path "$DS_PATH" \
          --fraction "$FRAC" \
          --epochs "$EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --optimizer "$OPTIMIZER" \
          --weight-decay "$WEIGHT_DECAY" \
          --warmup-epochs "$WARMUP_EPOCHS" \
          --early-stop "$EARLY_STOP" \
          --seed "$SEED" \
          --fadvit-weights "$FADVIT_WEIGHTS" \
          --output "$RUN_OUT"
      done
    done
  done
done

echo ""
echo "Done. Results in: $OUT_ROOT"

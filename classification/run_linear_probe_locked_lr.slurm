#!/bin/bash
# Linear-probe grid with LRs locked from cvt13-mod PathMNIST best: 9 runs per dataset
# (3 models x 3 fractions, one LR per fraction). Fewer epochs and larger batch for speed.
#
# Usage:
#   Single dataset (default pathmnist):
#     sbatch run_linear_probe_locked_lr.slurm
#     sbatch run_linear_probe_locked_lr.slurm --export=DATASET=dermamnist
#   Multiple datasets (one job per dataset, submit separately or in a loop):
#     for ds in pathmnist dermamnist octmnist; do sbatch run_linear_probe_locked_lr.slurm --export=DATASET=$ds; done
#
# Locked LRs (from cvt13-mod best on PathMNIST): fraction 0.01 -> 1e-3, 0.1 -> 1e-3, 1.0 -> 1e-4
#SBATCH --job-name=cls_locked
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/path/to/FAD-ViT/classification
#SBATCH --output=/path/to/FAD-ViT/classification/logs/%x_%j.out
#SBATCH --error=/path/to/FAD-ViT/classification/logs/%x_%j.err

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT="/path/to/FAD-ViT"
EXP_ROOT="$ROOT/Experiments/classification_linear_probe"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/breast_ultrasound/dataset}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/classification_linear_probe_locked_lr}"

# Dataset to run (default pathmnist). Set via sbatch --export=DATASET=dermamnist etc.
DATASET="${DATASET:-pathmnist}"
DATA_PATH="$DATA_ROOT/$DATASET"

# 9 runs per dataset: 3 models x 3 fractions; LR locked per fraction from cvt13-mod PathMNIST best
MODELS="${MODELS:-cvt13-mod vit-s16 dinov2-base}"
FRACTIONS="${FRACTIONS:-0.01 0.1 1.0}"

# Locked LR per fraction (from cvt13-mod best on PathMNIST: 0.01/0.1 -> 1e-3, 1.0 -> 1e-4)
get_lr_for_fraction() {
  case "$1" in
    0.01) echo "1e-3" ;;
    0.1)  echo "1e-3" ;;
    1.0)  echo "1e-4" ;;
    *)    echo "1e-4" ;;
  esac
}

# Fewer epochs and larger batch for speed; early stop unchanged
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
EARLY_STOP="${EARLY_STOP:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-0}"

CVT13_WEIGHTS="${CVT13_WEIGHTS:-$ROOT/weights/model_best.pth}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
PLOT_TITLE="${PLOT_TITLE:-Linear Probe (locked LR, 9 runs/dataset)}"

mkdir -p "$OUT_ROOT" "$EXP_ROOT/logs"

if [ ! -d "$DATA_PATH" ]; then
  echo "ERROR: dataset path missing: $DATA_PATH" >&2
  exit 2
fi
if [ ! -f "$CVT13_WEIGHTS" ] && echo "$MODELS" | grep -q "cvt13-mod"; then
  echo "ERROR: CvT13-mod weights missing: $CVT13_WEIGHTS" >&2
  exit 2
fi

echo "=== Linear Probe (locked LR): $DATASET ==="
echo "DATA_PATH=$DATA_PATH"
echo "OUT_ROOT=$OUT_ROOT  (runs under $OUT_ROOT/$DATASET/...)"
echo "MODELS=$MODELS  FRACTIONS=$FRACTIONS"
echo "EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  EARLY_STOP=$EARLY_STOP"
echo "=========================================="

for ARCH in $MODELS; do
  for FRAC in $FRACTIONS; do
    LR=$(get_lr_for_fraction "$FRAC")
    RUN_OUT="$OUT_ROOT/$DATASET/${ARCH}/fraction_${FRAC}/lr_${LR}/seed_${SEED}"
    mkdir -p "$RUN_OUT"
    echo ""
    echo "[RUN] dataset=$DATASET arch=$ARCH frac=$FRAC lr=$LR -> $RUN_OUT"
    "$PY" "$EXP_ROOT/run_linear_probe_classification.py" \
      --arch "$ARCH" \
      --data-path "$DATA_PATH" \
      --fraction "$FRAC" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --optimizer "$OPTIMIZER" \
      --weight-decay "$WEIGHT_DECAY" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --early-stop "$EARLY_STOP" \
      --seed "$SEED" \
      --cvt13-weights "$CVT13_WEIGHTS" \
      --output "$RUN_OUT"
  done
done

if [ "$RUN_AGGREGATE" = "1" ]; then
  AGG_OUT="$OUT_ROOT/aggregated"
  echo ""
  echo "[AGGREGATE] root=$OUT_ROOT out=$AGG_OUT"
  "$PY" "$EXP_ROOT/aggregate_classification_linear_probe.py" \
    --root "$OUT_ROOT" \
    --out-dir "$AGG_OUT" \
    --plot-title "$PLOT_TITLE"
fi

echo ""
echo "Done. Results under: $OUT_ROOT/$DATASET"

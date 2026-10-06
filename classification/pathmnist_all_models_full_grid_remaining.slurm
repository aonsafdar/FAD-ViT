#!/bin/bash
#SBATCH --job-name=cls_path_rem
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --chdir=/path/to/FAD-ViT/classification
#SBATCH --output=/path/to/FAD-ViT/classification/logs/%x_%j.out
#SBATCH --error=/path/to/FAD-ViT/classification/logs/%x_%j.err

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT="/path/to/FAD-ViT"
EXP_ROOT="$ROOT/Experiments/classification_linear_probe"

DATA_PATH="${DATA_PATH:-/scratch/$USER/breast_ultrasound/dataset/pathmnist}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/classification_linear_probe_pathmnist_rerun}"

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

FADVIT_WEIGHTS="${FADVIT_WEIGHTS:-/path/to/FAD-ViT/weights/model_best.pth}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
PLOT_TITLE="${PLOT_TITLE:-PathMNIST Linear Probe (all models, full LR/fraction grid)}"

mkdir -p "$OUT_ROOT" "$EXP_ROOT/logs"

if [ ! -d "$DATA_PATH" ]; then
  echo "ERROR: dataset path missing: $DATA_PATH" >&2
  exit 2
fi
if [ ! -f "$FADVIT_WEIGHTS" ]; then
  echo "ERROR: FAD-ViT weights missing: $FADVIT_WEIGHTS" >&2
  exit 2
fi

echo "=== PathMNIST Linear Probe: REMAINING runs only (skip if summary.json exists) ==="
echo "OUT_ROOT=$OUT_ROOT"
echo "MODELS=$MODELS FRACTIONS=$FRACTIONS LR_GRID=$LR_GRID"
echo "Time limit: 72h"
echo "========================================================="

RUNS_DONE=0
RUNS_SKIP=0
for ARCH in $MODELS; do
  for FRAC in $FRACTIONS; do
    for LR in $LR_GRID; do
      RUN_OUT="$OUT_ROOT/pathmnist/${ARCH}/fraction_${FRAC}/lr_${LR}/seed_${SEED}"
      if [ -f "$RUN_OUT/summary.json" ]; then
        echo "[SKIP] $ARCH frac=$FRAC lr=$LR (already done)"
        ((RUNS_SKIP+=1)) || true
        continue
      fi
      mkdir -p "$RUN_OUT"
      echo ""
      echo "[RUN] dataset=pathmnist arch=$ARCH frac=$FRAC lr=$LR -> $RUN_OUT"
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
        --fadvit-weights "$FADVIT_WEIGHTS" \
        --output "$RUN_OUT"
      ((RUNS_DONE+=1)) || true
    done
  done
done

echo ""
echo "Completed $RUNS_DONE new runs (skipped $RUNS_SKIP already done)."

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
echo "Done. Results in: $OUT_ROOT"

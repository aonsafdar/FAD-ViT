#!/bin/bash
#SBATCH --job-name=axtr
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# In-place axis-swap transfer-gap runs.
# Trains fadvit_unet (FAD-ViT) with a chosen global-attention axis and init, then
# evaluates on the held-out TEST split and appends a structured result line.
#   Usage: TASK=Task02_Heart TMODE=channel INIT=pretrained sbatch seg_axis_transfer.slurm
set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d
WEIGHTS=$ROOT/weights/model_best.pth

TASK=${TASK:?set TASK}
TMODE=${TMODE:-channel}    # channel (feature-axis) | spatial (spatial-axis)
INIT=${INIT:-pretrained}  # scratch | pretrained
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
EARLY_STOP=${EARLY_STOP:-15}
SATAIL=${SATAIL:-}        # optional: 0 disables the final spatial-attention tail

if [ "$TASK" = "BUSI" ]; then DATA=$ROOT/segmentation/data/busi; fi

TAG="${TMODE}_${INIT}_s${SEED}"
TAILARG=""
if [ -n "$SATAIL" ]; then TAG="${TAG}_tail${SATAIL}"; TAILARG="--sa-tail-depth-last $SATAIL"; fi
OUT=./outputs/axis_transfer/${TASK}/${TAG}
mkdir -p logs "$OUT" ./outputs/axis_transfer

if [ "$INIT" = "scratch" ]; then
  EXTRA="--scratch"
else
  EXTRA="--fadvit-weights $WEIGHTS"
fi

echo "=== TRAIN $TASK tmode=$TMODE init=$INIT seed=$SEED tail=${SATAIL:-default} epochs=$EPOCHS early=$EARLY_STOP ==="
$PY train.py --model fadvit_unet --task "$TASK" --data "$DATA" \
  --token-mode "$TMODE" $EXTRA $TAILARG --epochs "$EPOCHS" --early-stop "$EARLY_STOP" \
  --seed "$SEED" --output "$OUT"

CKPT="$OUT/$TASK/fadvit_unet/best.pth"
echo "=== EVAL held-out TEST ==="
$PY eval_test.py --model fadvit_unet --task "$TASK" --data "$DATA" \
  --token-mode "$TMODE" $TAILARG --seed "$SEED" --checkpoint "$CKPT" | tee "$OUT/test.txt"

DICE=$(grep -oP 'Dice=\K[0-9.]+' "$OUT/test.txt" | head -1)
IOU=$(grep -oP 'IoU=\K[0-9.]+' "$OUT/test.txt" | head -1)
HD=$(grep -oP 'HD95=\K[0-9.na]+' "$OUT/test.txt" | head -1)
echo "RESULT,$TASK,$TMODE,$INIT,$SEED,${SATAIL:-def},$DICE,$IOU,$HD" >> ./outputs/axis_transfer/results_ms.csv
echo "=== DONE $TASK $TMODE $INIT seed$SEED tail=${SATAIL:-default} Dice=$DICE ==="

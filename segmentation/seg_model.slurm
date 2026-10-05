#!/bin/bash
#SBATCH --job-name=segmdl
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generic backbone-UNet transfer run on ANY task (scratch|pretrained), multi-seed.
# Used to extend the channel-hybrid comparison (XCiT / DaViT) across all Table-3 tasks.
#   Usage: MODEL=xcit_unet TASK=Task03_Liver INIT=pretrained SEED=1 sbatch seg_model.slurm
set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d

MODEL=${MODEL:?set MODEL}   # xcit_unet | davit_unet | resnet50_unet | ...
TASK=${TASK:?set TASK}
INIT=${INIT:-pretrained}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
EARLY_STOP=${EARLY_STOP:-15}

if [ "$TASK" = "BUSI" ]; then DATA=$ROOT/segmentation/data/busi; fi

TAG="${MODEL}_${INIT}_s${SEED}"
OUT=./outputs/axis_transfer/${TASK}/${TAG}
mkdir -p logs "$OUT" ./outputs/axis_transfer

if [ "$INIT" = "scratch" ]; then EXTRA="--scratch"; else EXTRA=""; fi

echo "=== TRAIN $TASK $MODEL init=$INIT seed=$SEED epochs=$EPOCHS ==="
$PY train.py --model "$MODEL" --task "$TASK" --data "$DATA" \
  $EXTRA --epochs "$EPOCHS" --early-stop "$EARLY_STOP" --seed "$SEED" --output "$OUT"

CKPT="$OUT/$TASK/$MODEL/best.pth"
echo "=== EVAL held-out TEST ==="
$PY eval_test.py --model "$MODEL" --task "$TASK" --data "$DATA" \
  --seed "$SEED" --checkpoint "$CKPT" | tee "$OUT/test.txt"

DICE=$(grep -oP 'Dice=\K[0-9.]+' "$OUT/test.txt" | head -1)
IOU=$(grep -oP 'IoU=\K[0-9.]+' "$OUT/test.txt" | head -1)
HD=$(grep -oP 'HD95=\K[0-9.na]+' "$OUT/test.txt" | head -1)
echo "RESULT,$TASK,$MODEL,$INIT,$SEED,$DICE,$IOU,$HD" >> ./outputs/axis_transfer/baselines_ms.csv
echo "=== DONE $TASK $MODEL $INIT seed$SEED Dice=$DICE ==="

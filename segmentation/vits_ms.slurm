#!/bin/bash
#SBATCH --job-name=vitsms
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Multi-seed matched-capacity spatial ViT-S/16-UNet on ANY task (scratch + pretrained).
# Generalises vits_ms_busi.slurm to the full 11-task grid for the camera-ready
# matched-capacity transfer-gap table (FAD-ViT feature-axis vs ViT-S spatial baseline).
#   Usage: TASK=Task03_Liver INIT=pretrained SEED=1 sbatch vits_ms.slurm
set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d

TASK=${TASK:?set TASK}
INIT=${INIT:-pretrained}  # scratch | pretrained
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
EARLY_STOP=${EARLY_STOP:-15}

if [ "$TASK" = "BUSI" ]; then DATA=$ROOT/segmentation/data/busi; fi

TAG="vits_${INIT}_s${SEED}"
OUT=./outputs/axis_transfer/${TASK}/${TAG}
mkdir -p logs "$OUT" ./outputs/axis_transfer

if [ "$INIT" = "scratch" ]; then EXTRA="--scratch"; else EXTRA=""; fi

echo "=== TRAIN $TASK vit_small_unet init=$INIT seed=$SEED epochs=$EPOCHS ==="
$PY train.py --model vit_small_unet --task "$TASK" --data "$DATA" \
  $EXTRA --epochs "$EPOCHS" --early-stop "$EARLY_STOP" --seed "$SEED" --output "$OUT"

CKPT="$OUT/$TASK/vit_small_unet/best.pth"
echo "=== EVAL held-out TEST ==="
$PY eval_test.py --model vit_small_unet --task "$TASK" --data "$DATA" \
  --seed "$SEED" --checkpoint "$CKPT" | tee "$OUT/test.txt"

DICE=$(grep -oP 'Dice=\K[0-9.]+' "$OUT/test.txt" | head -1)
IOU=$(grep -oP 'IoU=\K[0-9.]+' "$OUT/test.txt" | head -1)
HD=$(grep -oP 'HD95=\K[0-9.na]+' "$OUT/test.txt" | head -1)
echo "RESULT,$TASK,vits,$INIT,$SEED,$DICE,$IOU,$HD" >> ./outputs/axis_transfer/vits_ms.csv
echo "=== DONE $TASK vits $INIT seed$SEED Dice=$DICE ==="

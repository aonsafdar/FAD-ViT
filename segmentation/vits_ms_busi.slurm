#!/bin/bash
#SBATCH --job-name=vitsms
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Multi-seed matched-capacity spatial ViT-S/16-UNet on BUSI (scratch + pretrained),
# to complete the transfer-gap seed set requested by reviewers (Table 8).
#   Usage: INIT=pretrained SEED=1 sbatch vits_ms_busi.slurm
set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/busi

INIT=${INIT:-pretrained}  # scratch | pretrained
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-100}
EARLY_STOP=${EARLY_STOP:-15}

TAG="vits_${INIT}_s${SEED}"
OUT=./outputs/axis_transfer/BUSI/${TAG}
mkdir -p logs "$OUT" ./outputs/axis_transfer

if [ "$INIT" = "scratch" ]; then EXTRA="--scratch"; else EXTRA=""; fi

echo "=== TRAIN BUSI vit_small_unet init=$INIT seed=$SEED epochs=$EPOCHS ==="
$PY train.py --model vit_small_unet --task BUSI --data "$DATA" \
  $EXTRA --epochs "$EPOCHS" --early-stop "$EARLY_STOP" --seed "$SEED" --output "$OUT"

CKPT="$OUT/BUSI/vit_small_unet/best.pth"
echo "=== EVAL held-out TEST ==="
$PY eval_test.py --model vit_small_unet --task BUSI --data "$DATA" \
  --seed "$SEED" --checkpoint "$CKPT" | tee "$OUT/test.txt"

DICE=$(grep -oP 'Dice=\K[0-9.]+' "$OUT/test.txt" | head -1)
IOU=$(grep -oP 'IoU=\K[0-9.]+' "$OUT/test.txt" | head -1)
HD=$(grep -oP 'HD95=\K[0-9.na]+' "$OUT/test.txt" | head -1)
echo "RESULT,BUSI,vits,$INIT,$SEED,$DICE,$IOU,$HD" >> ./outputs/axis_transfer/vits_ms.csv
echo "=== DONE BUSI vits $INIT seed$SEED Dice=$DICE ==="

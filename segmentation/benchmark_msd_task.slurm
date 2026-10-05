#!/bin/bash
#SBATCH --job-name=seg_msd
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Benchmark: 3 backbones (CvT13, ViT-base, DINOv2) on one MSD task.
# Usage: set TASK then sbatch benchmark_msd_task.slurm
#   export TASK=Task07_Pancreas && sbatch benchmark_msd_task.slurm
#   export TASK=Task03_Liver   && sbatch benchmark_msd_task.slurm
#   export TASK=Task09_Spleen  && sbatch benchmark_msd_task.slurm
# Default TASK=Task07_Pancreas (recommended next after Task01_BrainTumour).

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d
WEIGHTS=$ROOT/weights/model_best.pth
TASK=${TASK:-Task07_Pancreas}

# num_classes by task (must match data/dataset.py TASK_NUM_CLASSES)
case "$TASK" in
  Task01_BrainTumour) NC=4 ;;
  Task02_Heart|Task09_Spleen|Task10_Colon) NC=2 ;;
  *) NC=3 ;;  # Task03,04,05,06,07,08
esac

EPOCHS=100
EARLY_STOP=20
OUT=./outputs/benchmark_msd
mkdir -p logs outputs

if [ ! -d "$ROOT/segmentation/data/msd/$TASK" ]; then
  echo "Downloading MSD task: $TASK..."
  cd $ROOT/segmentation && $PY scripts/download_msd.py --tasks $TASK --root ./data/msd --method urllib
fi
if [ ! -d "$DATA/$TASK" ]; then
  echo "Processing MSD to 2D slices: $TASK..."
  cd $ROOT/segmentation && $PY scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks $TASK
fi

echo "=== Benchmark: $TASK (num_classes=$NC) ==="
echo "Early stopping: $EARLY_STOP epochs | Max epochs: $EPOCHS"
echo ""

echo "[1/3] CvT13-UNet (modified)..."
$PY train.py --model cvt13_unet --task $TASK --data $DATA --num-classes $NC \
  --cvt13-weights $WEIGHTS --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

echo "[2/3] ViT-Base-UNet (ImageNet-1k)..."
$PY train.py --model vit_base_unet --task $TASK --data $DATA --num-classes $NC \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

echo "[3/3] DINOv2-UNet..."
$PY train.py --model dinov2_unet --task $TASK --data $DATA --num-classes $NC \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

echo ""
echo "=== Benchmark complete: $TASK ==="
echo "Results in $OUT/$TASK/{cvt13_unet,vit_base_unet,dinov2_unet}/best.pth"

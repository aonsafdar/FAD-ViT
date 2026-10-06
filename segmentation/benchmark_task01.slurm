#!/bin/bash
#SBATCH --job-name=seg_bench
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Benchmark: 3 models on Task01_BrainTumour (MSD 2D slices)
# Models: FAD-ViT, ViT-base (ImageNet-1k), DINOv2-base
# Early stopping: 20 epochs without improvement

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d
WEIGHTS=$ROOT/weights/model_best.pth
TASK=Task01_BrainTumour
NC=4
EPOCHS=100
EARLY_STOP=20
OUT=./outputs/benchmark_task01
mkdir -p logs outputs

# Ensure data exists
if [ ! -d "$ROOT/segmentation/data/msd/$TASK" ]; then
  echo "Downloading MSD..."
  cd $ROOT/segmentation && $PY scripts/download_msd.py --tasks $TASK --root ./data/msd --method urllib
fi
if [ ! -d "$DATA/$TASK" ]; then
  echo "Processing MSD to 2D slices..."
  cd $ROOT/segmentation && $PY scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks $TASK
fi

echo "=== Benchmark: $TASK (num_classes=$NC) ==="
echo "Early stopping: $EARLY_STOP epochs | Max epochs: $EPOCHS"
echo ""

# 1. FAD-ViT (our modified model, ImageNet-1k pretrained)
echo "[1/3] FADViT-UNet (modified)..."
$PY train.py --model fadvit_unet --task $TASK --data $DATA --num-classes $NC \
  --fadvit-weights $WEIGHTS --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

# 2. ViT-base (timm, ImageNet-1k pretrained)
echo "[2/3] ViT-Base-UNet (ImageNet-1k)..."
$PY train.py --model vit_base_unet --task $TASK --data $DATA --num-classes $NC \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

# 3. DINOv2-base (timm, DINOv2 pretrained)
echo "[3/3] DINOv2-UNet..."
$PY train.py --model dinov2_unet --task $TASK --data $DATA --num-classes $NC \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

echo ""
echo "=== Benchmark complete ==="
echo "Results in $OUT/$TASK/{fadvit_unet,vit_base_unet,dinov2_unet}/best.pth"

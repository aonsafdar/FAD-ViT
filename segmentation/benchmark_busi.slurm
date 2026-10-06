#!/bin/bash
#SBATCH --job-name=seg_busi
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Benchmark: 3 models on BUSI segmentation (Dataset_BUSI_with_GT: normal/benign/malignant)
# Models: FAD-ViT, ViT-base (ImageNet-1k), DINOv2-base
# Early stopping: 20 epochs without improvement

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
BUSI_ROOT=$ROOT/dataset/Dataset_BUSI_with_GT
DATA=$ROOT/segmentation/data/busi
WEIGHTS=$ROOT/weights/model_best.pth
TASK=BUSI
EPOCHS=100
EARLY_STOP=20
OUT=./outputs/benchmark_busi
mkdir -p logs outputs

# 1. Arrange dataset: Dataset_BUSI_with_GT -> data/busi train/val/test
if [ ! -f "$DATA/config.json" ]; then
  echo "Preparing BUSI segmentation data from $BUSI_ROOT..."
  $PY scripts/prepare_busi_seg.py --root "$BUSI_ROOT" --output "$DATA"
fi

echo "=== Benchmark: BUSI segmentation (num_classes=2) ==="
echo "Data: $DATA | Early stopping: $EARLY_STOP epochs"
echo ""

# 2. FAD-ViT
echo "[1/3] FADViT-UNet (modified)..."
$PY train.py --model fadvit_unet --task $TASK --data $DATA \
  --fadvit-weights $WEIGHTS --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

# 3. ViT-base
echo "[2/3] ViT-Base-UNet (ImageNet-1k)..."
$PY train.py --model vit_base_unet --task $TASK --data $DATA \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

# 4. DINOv2-base
echo "[3/3] DINOv2-UNet..."
$PY train.py --model dinov2_unet --task $TASK --data $DATA \
  --epochs $EPOCHS --early-stop $EARLY_STOP \
  --output $OUT --wandb --wandb-project Segmentation

echo ""
echo "=== Benchmark complete ==="
echo "Results in $OUT/$TASK/{fadvit_unet,vit_base_unet,dinov2_unet}/best.pth"

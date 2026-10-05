#!/bin/bash
#SBATCH --job-name=msd_seg
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# MSD 2D segmentation: CvT13-UNet + U-Net baseline
# Tasks: Task01_BrainTumour (MRI), Task07_Pancreas (CT)

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d
WEIGHTS=$ROOT/weights/model_best.pth
mkdir -p logs outputs

# 1. Download MSD (if not done)
if [ ! -d "$ROOT/segmentation/data/msd/Task01_BrainTumour" ]; then
  echo "Downloading MSD..."
  cd $ROOT/segmentation && $PY scripts/download_msd.py --tasks Task01_BrainTumour Task07_Pancreas --root ./data/msd --method urllib
fi

# 2. Process to 2D slices (if not done)
if [ ! -d "$DATA/Task01_BrainTumour" ]; then
  echo "Processing MSD to 2D..."
  $PY scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks Task01_BrainTumour Task07_Pancreas
fi

# 3. Train CvT13-UNet on Task01 (Brain, MRI)
echo "Training CvT13-UNet on Task01_BrainTumour..."
$PY train.py --model cvt13_unet --task Task01_BrainTumour --data $DATA --num-classes 4 \
  --cvt13-weights $WEIGHTS --epochs 100 --output ./outputs/seg --wandb --wandb-project Segmentation

# 4. Train CvT13-UNet on Task07 (Pancreas, CT)
echo "Training CvT13-UNet on Task07_Pancreas..."
$PY train.py --model cvt13_unet --task Task07_Pancreas --data $DATA --num-classes 3 \
  --cvt13-weights $WEIGHTS --epochs 100 --output ./outputs/seg --wandb --wandb-project Segmentation

# 5. U-Net baselines
echo "Training U-Net on Task01..."
$PY train.py --model unet --task Task01_BrainTumour --data $DATA --num-classes 4 --epochs 100 --output ./outputs/seg --wandb --wandb-project Segmentation

echo "Training U-Net on Task07..."
$PY train.py --model unet --task Task07_Pancreas --data $DATA --num-classes 3 --epochs 100 --output ./outputs/seg --wandb --wandb-project Segmentation

echo "Done. Run evaluate.py for Dice/HD95 and overlays."

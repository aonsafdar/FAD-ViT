#!/bin/bash
#SBATCH --job-name=msd_t2_t6
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --chdir=/path/to/FAD-ViT/segmentation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Benchmark: Task02_Heart + Task06_Lung, all 3 models (Ours-UNet, ViT-UNet, DINOv2-UNet).
# Same pipeline as benchmark_msd_task.slurm: download → process 2D → train each model per task.

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT=/scratch/$USER/breast_ultrasound
DATA=$ROOT/segmentation/data/msd_2d
WEIGHTS=$ROOT/weights/model_best.pth
EPOCHS=100
EARLY_STOP=20
OUT=./outputs/benchmark_msd
mkdir -p logs outputs

# num_classes by task (must match data/dataset.py)
get_nc() {
  case "$1" in
    Task01_BrainTumour) echo 4 ;;
    Task02_Heart|Task09_Spleen|Task10_Colon) echo 2 ;;
    *) echo 3 ;;
  esac
}

for TASK in Task02_Heart Task06_Lung; do
  NC=$(get_nc "$TASK")
  if [ ! -d "$ROOT/segmentation/data/msd/$TASK" ]; then
    echo "Downloading MSD task: $TASK..."
    cd $ROOT/segmentation && $PY scripts/download_msd.py --tasks $TASK --root ./data/msd --method urllib
  fi
  if [ ! -d "$DATA/$TASK" ]; then
    echo "Processing MSD to 2D slices: $TASK..."
    cd $ROOT/segmentation && $PY scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks $TASK
  fi

  echo ""
  echo "=== Benchmark: $TASK (num_classes=$NC) ==="
  echo "Early stopping: $EARLY_STOP epochs | Max epochs: $EPOCHS"
  echo ""

  echo "[1/3] Ours-UNet (FADViT)..."
  $PY train.py --model fadvit_unet --task $TASK --data $DATA --num-classes $NC \
    --fadvit-weights $WEIGHTS --epochs $EPOCHS --early-stop $EARLY_STOP \
    --output $OUT --wandb --wandb-project Segmentation

  echo "[2/3] ViT-UNet..."
  $PY train.py --model vit_base_unet --task $TASK --data $DATA --num-classes $NC \
    --epochs $EPOCHS --early-stop $EARLY_STOP \
    --output $OUT --wandb --wandb-project Segmentation

  echo "[3/3] DINOv2-UNet..."
  $PY train.py --model dinov2_unet --task $TASK --data $DATA --num-classes $NC \
    --epochs $EPOCHS --early-stop $EARLY_STOP \
    --output $OUT --wandb --wandb-project Segmentation

  echo ""
  echo "=== Done: $TASK ==="
done

echo ""
echo "=== All benchmarks complete: Task02_Heart, Task06_Lung ==="
echo "Checkpoints: $OUT/Task02_Heart/{fadvit_unet,vit_base_unet,dinov2_unet}/best.pth"
echo "             $OUT/Task06_Lung/{fadvit_unet,vit_base_unet,dinov2_unet}/best.pth"

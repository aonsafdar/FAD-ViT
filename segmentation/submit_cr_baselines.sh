#!/bin/bash
# Commitment #4: extend channel-hybrid comparison (XCiT, DaViT) across all 11
# Table-3 tasks under the identical U-Net protocol, pretrained transfer, seeds 0/1/2.
# Idempotent against baselines_ms.csv.  Usage: bash submit_cr_baselines.sh [--dry-run]
set -euo pipefail
cd /path/to/FAD-ViT/segmentation

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

TASKS=(BUSI Task01_BrainTumour Task02_Heart Task03_Liver Task04_Hippocampus \
       Task05_Prostate Task06_Lung Task07_Pancreas Task08_HepaticVessel \
       Task09_Spleen Task10_Colon)
MODELS=(xcit_unet davit_unet)
SEEDS=(0 1 2)
INIT=pretrained

CSV=./outputs/axis_transfer/baselines_ms.csv
touch "$CSV"

n=0
for m in "${MODELS[@]}"; do
  for t in "${TASKS[@]}"; do
    for s in "${SEEDS[@]}"; do
      if grep -q "^RESULT,$t,$m,$INIT,$s," "$CSV"; then continue; fi
      if [ "$DRY" = "1" ]; then
        echo "[dry] MODEL=$m TASK=$t INIT=$INIT SEED=$s sbatch seg_model.slurm"
      else
        env MODEL="$m" TASK="$t" INIT="$INIT" SEED="$s" sbatch seg_model.slurm >/dev/null \
          && echo "[submit] MODEL=$m TASK=$t SEED=$s"
      fi
      n=$((n+1))
    done
  done
done
echo "----"; echo "Total $( [ "$DRY" = 1 ] && echo would-submit || echo submitted ): $n jobs"

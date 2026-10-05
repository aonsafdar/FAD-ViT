#!/bin/bash
# Commitment #3: multi-seed MedMNIST linear-probe classification (W4).
# Extends existing seed_0 results with seeds 1 and 2 across the 9 paper datasets,
# 3 models x 3 fractions each. Idempotent (per-run summary.json skip in cls_ms.slurm;
# whole-job skip here if all 9 runs for a (dataset,seed) already exist).
#   Usage: bash submit_cls_ms.sh [--dry-run]
set -euo pipefail
cd /path/to/FAD-ViT/classification

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

# seed_0 already exists (locked-lr) for these 8 datasets -> only add seeds 1,2.
DATASETS=(dermamnist octmnist bloodmnist organamnist tissuemnist \
          retinamnist breastmnist pneumoniamnist)
# pathmnist seed_0 exists only under a different (full-grid) protocol -> run 0,1,2
# under locked-lr so all three seeds are comparable.
PATH_SEEDS=(0 1 2)
SEEDS=(1 2)
OUT_ROOT="/path/to/FAD-ViT/outputs/classification_linear_probe_locked_lr"

count_done() { # count_done <dataset> <seed>
  [ -d "$OUT_ROOT/$1" ] || { echo 0; return; }
  find "$OUT_ROOT/$1" -path "*/seed_${2}/summary.json" 2>/dev/null | wc -l
}

n=0
# pathmnist (seeds 0,1,2)
for s in "${PATH_SEEDS[@]}"; do
  have=$(count_done pathmnist "$s")
  if [ "$have" -ge 9 ]; then echo "[skip] pathmnist seed=$s ($have/9 done)"; continue; fi
  if [ "$DRY" = 1 ]; then echo "[dry] DATASET=pathmnist SEED=$s sbatch cls_ms.slurm ($have/9 present)";
  else env DATASET=pathmnist SEED="$s" sbatch cls_ms.slurm >/dev/null && echo "[submit] DATASET=pathmnist SEED=$s"; fi
  n=$((n+1))
done
for s in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    have=$(count_done "$ds" "$s")
    if [ "$have" -ge 9 ]; then echo "[skip] $ds seed=$s ($have/9 done)"; continue; fi
    if [ "$DRY" = 1 ]; then
      echo "[dry] DATASET=$ds SEED=$s sbatch cls_ms.slurm ($have/9 present)"
    else
      env DATASET="$ds" SEED="$s" sbatch cls_ms.slurm >/dev/null && echo "[submit] DATASET=$ds SEED=$s"
    fi
    n=$((n+1))
  done
done
echo "----"; echo "Total $( [ "$DRY" = 1 ] && echo would-submit || echo submitted ): $n jobs"

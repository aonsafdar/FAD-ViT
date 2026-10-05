#!/bin/bash
# Camera-ready experiment sweep for ACCV #1310 commitments.
#   #1 Matched-capacity transfer-gap grid across all 11 segmentation tasks:
#      FAD-ViT feature-axis (cvt13 channel) vs ViT-S/16 spatial baseline,
#      scratch + pretrained, seeds 0/1/2.
#   #2 2x2 spatial-extractor ablation (SATAIL=0) on representative US/MRI/CT:
#      {feature(channel), spatial} x {tail-on(done), tail-off}, scratch+pretrained, seeds 0/1/2.
#
# Idempotent: only submits (task,arm,init,seed) combos not already present in the
# result CSVs, so it is safe to re-run to fill gaps / re-queue failures.
#   Usage: bash submit_cr.sh [--dry-run]
set -euo pipefail
cd /path/to/FAD-ViT/segmentation

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

TASKS=(BUSI Task01_BrainTumour Task02_Heart Task03_Liver Task04_Hippocampus \
       Task05_Prostate Task06_Lung Task07_Pancreas Task08_HepaticVessel \
       Task09_Spleen Task10_Colon)
REP=(BUSI Task02_Heart Task09_Spleen)          # representative US/MRI/CT for 2x2
SEEDS=(0 1 2)
INITS=(scratch pretrained)

RES=./outputs/axis_transfer/results_ms.csv
VRES=./outputs/axis_transfer/vits_ms.csv
touch "$RES" "$VRES"

sub() { # sub <script> <ENV assignments...>  -> echoes/submits
  local script=$1; shift
  if [ "$DRY" = "1" ]; then
    echo "[dry] $* sbatch $script"
  else
    env "$@" sbatch "$script" >/dev/null && echo "[submit] $* $script"
  fi
}

n=0
# ---- #1a: FAD-ViT feature-axis (channel), all 11 tasks ----
for t in "${TASKS[@]}"; do
  for init in "${INITS[@]}"; do
    for s in "${SEEDS[@]}"; do
      if grep -q "^RESULT,$t,channel,$init,$s,def," "$RES"; then continue; fi
      sub seg_axis_transfer.slurm TASK="$t" TMODE=channel INIT="$init" SEED="$s"; n=$((n+1))
    done
  done
done

# ---- #1b: ViT-S/16 matched-capacity spatial baseline, all 11 tasks ----
for t in "${TASKS[@]}"; do
  for init in "${INITS[@]}"; do
    for s in "${SEEDS[@]}"; do
      if grep -q "^RESULT,$t,vits,$init,$s," "$VRES"; then continue; fi
      sub vits_ms.slurm TASK="$t" INIT="$init" SEED="$s"; n=$((n+1))
    done
  done
done

# ---- #2: 2x2 spatial-extractor ablation (tail off) on representative tasks ----
for t in "${REP[@]}"; do
  for tm in channel spatial; do
    for init in "${INITS[@]}"; do
      for s in "${SEEDS[@]}"; do
        if grep -q "^RESULT,$t,$tm,$init,$s,0," "$RES"; then continue; fi
        sub seg_axis_transfer.slurm TASK="$t" TMODE="$tm" INIT="$init" SEED="$s" SATAIL=0; n=$((n+1))
      done
    done
  done
done

echo "----"
echo "Total $( [ "$DRY" = "1" ] && echo would-submit || echo submitted ): $n jobs"

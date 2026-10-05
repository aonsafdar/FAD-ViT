#!/bin/bash
#SBATCH --job-name=cls_lp_agg
#SBATCH --partition=csgpu
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/%u/breast_ultrasound
#SBATCH --output=/scratch/%u/breast_ultrasound/Experiments/logs/%x_%j.out
#SBATCH --error=/scratch/%u/breast_ultrasound/Experiments/logs/%x_%j.err

# Aggregate classification linear-probe results and generate:
# - all_runs.csv
# - best_by_dataset_arch_fraction.csv
# - best_model_by_dataset_fraction.csv
# - global and per-dataset curves
#
# Usage:
#   sbatch Experiments/classification_linear_probe/aggregate_classification_linear_probe.slurm
#   ROOT_RESULTS=/scratch/$USER/breast_ultrasound/outputs/classification_linear_probe \
#     sbatch Experiments/classification_linear_probe/aggregate_classification_linear_probe.slurm

set -euo pipefail

module load anaconda3/2024.10-1
source /opt/software/el9/anaconda3/2024.10-1/etc/profile.d/conda.sh
PY="/scratch/$USER/conda-envs/bu-mamba/bin/python"

ROOT_RESULTS="${ROOT_RESULTS:-/scratch/$USER/breast_ultrasound/outputs/classification_linear_probe}"
OUT_DIR="${OUT_DIR:-$ROOT_RESULTS/aggregated}"
PLOT_TITLE="${PLOT_TITLE:-Classification Linear Probe Data Efficiency}"

echo "=== Aggregating classification linear-probe results ==="
echo "ROOT_RESULTS=$ROOT_RESULTS"
echo "OUT_DIR=$OUT_DIR"
echo "PLOT_TITLE=$PLOT_TITLE"

if [ ! -d "$ROOT_RESULTS" ]; then
  echo "[ERROR] ROOT_RESULTS not found: $ROOT_RESULTS"
  exit 1
fi

"$PY" /scratch/$USER/breast_ultrasound/Experiments/classification_linear_probe/aggregate_classification_linear_probe.py \
  --root "$ROOT_RESULTS" \
  --out-dir "$OUT_DIR" \
  --plot-title "$PLOT_TITLE"

echo "Done. Aggregates written to: $OUT_DIR"

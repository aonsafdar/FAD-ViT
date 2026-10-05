#!/bin/bash
# Run aggregate + plots on existing PathMNIST linear-probe results (no training).
# Use after stopping the long grid job; uses outputs/classification_linear_probe_pathmnist_rerun.
set -euo pipefail
ROOT="/path/to/FAD-ViT"
EXP_ROOT="$ROOT/Experiments/classification_linear_probe"
PATHMNIST_OUT="$ROOT/outputs/classification_linear_probe_pathmnist_rerun"
AGG_OUT="$EXP_ROOT/aggregated_pathmnist"
PY="${PY:-/scratch/$USER/conda-envs/bu-mamba/bin/python}"

if [ ! -d "$PATHMNIST_OUT" ]; then
  echo "ERROR: PathMNIST results not found: $PATHMNIST_OUT" >&2
  exit 2
fi

mkdir -p "$AGG_OUT"
echo "Aggregating PathMNIST results: $PATHMNIST_OUT -> $AGG_OUT"
"$PY" "$EXP_ROOT/aggregate_classification_linear_probe.py" \
  --root "$PATHMNIST_OUT" \
  --out-dir "$AGG_OUT" \
  --plot-title "PathMNIST Linear Probe (all models, full LR/fraction grid)" \
  --add-dinov2-pathmnist-100

echo "Building publication table..."
"$PY" "$EXP_ROOT/build_publication_table.py" \
  --best-csv "$AGG_OUT/best_by_dataset_arch_fraction.csv" \
  --out "$EXP_ROOT/docs/classification_linear_probe_pathmnist_table.md"

echo "Done. Plots: $AGG_OUT. Table: $EXP_ROOT/docs/classification_linear_probe_pathmnist_table.md"

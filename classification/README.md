# Classification linear probe (MedMNIST)

## Stopping the long-running grid job

If the full PathMNIST grid (or remaining-runs) job is still running and you want to cancel it:

```bash
# List your jobs
squeue -u $USER

# Cancel by job ID (e.g. 300280)
scancel <JOB_ID>
```

Results already written under `outputs/classification_linear_probe_pathmnist_rerun` are kept; you can aggregate and plot from them.

## PathMNIST: aggregate and plots from existing results

Without re-running training, generate aggregate CSVs and curves from the PathMNIST run:

```bash
cd /path/to/FAD-ViT/classification
bash run_aggregate_pathmnist.sh
```

This writes:

- `aggregated_pathmnist/` — `all_runs.csv`, `best_by_dataset_arch_fraction.csv`, global and per-dataset plots (val/test acc vs fraction).
- `docs/classification_linear_probe_pathmnist_table.md` — publication table (test accuracy and AUC).

Or run the steps manually (use `--add-dinov2-pathmnist-100` to include the interpolated DINOv2 100% point in plots):

```bash
python aggregate_classification_linear_probe.py \
  --root /path/to/outputs/classification_linear_probe_pathmnist_rerun \
  --out-dir ./aggregated_pathmnist \
  --plot-title "PathMNIST Linear Probe (all models, full LR/fraction grid)" \
  --add-dinov2-pathmnist-100

python build_publication_table.py \
  --best-csv aggregated_pathmnist/best_by_dataset_arch_fraction.csv \
  --out docs/classification_linear_probe_pathmnist_table.md
```

## Publication table

PathMNIST results table (model × 1% / 10% / 100% data): see [docs/classification_linear_probe_pathmnist_table.md](docs/classification_linear_probe_pathmnist_table.md).  
Model parameters and FLOPS: run `python count_params_flops.py` in this directory.

## Other MedMNIST datasets (locked LR, 9 runs per dataset)

LRs are fixed from the best FADViT PathMNIST settings: 0.01 and 0.1 → 1e-3, 1.0 → 1e-4. Each dataset runs 9 configs (3 models × 3 fractions), 50 epochs, early stopping 15.

**Single dataset** (one job per dataset):

```bash
sbatch run_linear_probe_locked_lr.slurm --export=DATASET=dermamnist
sbatch run_linear_probe_locked_lr.slurm --export=DATASET=octmnist
# or in a loop:
for ds in dermamnist octmnist bloodmnist; do sbatch run_linear_probe_locked_lr.slurm --export=DATASET=$ds; done
```

**Multiple datasets in one job** (default: dermamnist, octmnist, bloodmnist — 27 runs total):

```bash
sbatch run_medmnist_locked_lr_multi.slurm
```

Custom list (e.g. 2 datasets for speed):

```bash
sbatch run_medmnist_locked_lr_multi.slurm --export=DATASETS="dermamnist octmnist"
```

Ensure each dataset exists under `DATA_ROOT` (e.g. `/path/to/FAD-ViT/dataset/<name>`) with the same ImageFolder layout as PathMNIST.

**Datasets to use for speed:** smaller/faster options are **dermamnist**, **octmnist**, **bloodmnist** (similar scale to PathMNIST). Add more (e.g. retinamnist, breastmnist) by setting `DATASETS` or submitting extra single-dataset jobs.

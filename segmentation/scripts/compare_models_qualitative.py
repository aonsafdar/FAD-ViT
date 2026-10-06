#!/usr/bin/env python3
"""
Publication-quality comparison of the 3 models (Ours, ViT, DINOv2).

For per-dataset, per-model figures only (e.g. outputs/qualitative_msd/Task/model/sample_*.png),
use scripts/visualize_segmentation.py and run it once per (task, model).

Two modes here:
  1) Single-dataset: --single-task BUSI (or any task). Outputs per-sample figures
     and optional grid for that dataset only. Use --num-samples, --save-grid, etc.
  2) Cross-dataset: --tasks A B C (default: all). One figure with rows=Ground truth|models, cols=datasets.
     Use --num-samples 20 to loop over 20 random test samples (one figure per sample). Use --model-display, --split, --sample-indices.

Usage:
  # Single dataset (e.g. BUSI): per-sample + grid
  python scripts/compare_models_qualitative.py --single-task BUSI --busi-data ./data/busi \\
    --busi-checkpoint-dir ./outputs/benchmark_busi --output ./outputs/compare_qualitative \\
    --num-samples 12 --dpi 300 --format pdf --save-grid --grid-rows 2 --grid-cols 3

  # Cross-dataset: one figure, default test set, first sample per dataset
  python scripts/compare_models_qualitative.py --output ./outputs/compare_qualitative --dpi 300 --format pdf

  # Cross-dataset: 20 random test samples (20 figures, same index across datasets per figure)
  python scripts/compare_models_qualitative.py --num-samples 20 --output ./outputs/compare_qualitative --format pdf

  # Cross-dataset: single figure, custom sample per dataset
  python scripts/compare_models_qualitative.py --sample-indices BUSI=10 Task07_Pancreas=2 --output ./outputs/compare_qualitative
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import get_dataset
from visualization import save_datasets_comparison, save_comparison_figure, save_comparison_grid

MODELS = ["fadvit_unet", "vit_base_unet", "dinov2_unet"]
MODEL_LABELS = ["FAD-ViT-UNet", "ViT-UNet", "DINOv2-UNet"]  # display names

# All datasets we benchmark (default --tasks)
ALL_TASKS = [
    "BUSI",
    "Task01_BrainTumour",
    "Task03_Liver",
    "Task04_Hippocampus",
    "Task05_Prostate",
    "Task07_Pancreas",
    "Task09_Spleen",
    "Task10_Colon",
]

# Short names for row labels
DATASET_LABELS = {
    "BUSI": "BUSI",
    "Task01_BrainTumour": "Brain",
    "Task02_Heart": "Heart",
    "Task03_Liver": "Liver",
    "Task04_Hippocampus": "Hippocampus",
    "Task05_Prostate": "Prostate",
    "Task06_Lung": "Lung",
    "Task07_Pancreas": "Pancreas",
    "Task08_HepaticVessel": "Hepatic",
    "Task09_Spleen": "Spleen",
    "Task10_Colon": "Colon",
}


def get_model(model_name: str, in_chans: int, num_classes: int):
    if model_name == "fadvit_unet":
        from models import FADViTUNet
        return FADViTUNet(in_chans=in_chans, num_classes=num_classes, pretrained_path="")
    elif model_name == "vit_base_unet":
        from models import ViTBaseUNet
        return ViTBaseUNet(in_chans=in_chans, num_classes=num_classes, pretrained=False)
    elif model_name == "dinov2_unet":
        from models import DINOv2UNet
        return DINOv2UNet(in_chans=in_chans, num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 models: single-task (per-sample) or cross-dataset (rows=Original|GT|models|Error, cols=datasets)"
    )
    parser.add_argument("--single-task", type=str, default=None, metavar="TASK",
                        help="Single-dataset mode: compare 3 models on this task only (e.g. BUSI). Uses --data/--busi-data and checkpoint dirs. Enables --num-samples, --save-grid.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Cross-dataset mode: list of tasks (default: all). Ignored if --single-task is set.")
    parser.add_argument("--data", type=str, default="./data/msd_2d", help="Data path for MSD tasks")
    parser.add_argument("--busi-data", type=str, default="./data/busi", help="Data path for BUSI")
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/benchmark_msd",
                        help="Base dir for MSD checkpoints: <dir>/<task>/<model>/best.pth")
    parser.add_argument("--task01-checkpoint-dir", type=str, default="./outputs/benchmark_task01",
                        help="Base dir for Task01_BrainTumour (saved by benchmark_task01.slurm)")
    parser.add_argument("--busi-checkpoint-dir", type=str, default="./outputs/benchmark_busi",
                        help="Base dir for BUSI checkpoints: <dir>/BUSI/<model>/best.pth")
    parser.add_argument("--output", type=str, default="./outputs/compare_qualitative")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Single-dataset: number of samples. Cross-dataset: number of random test samples (figures); 1 = use --sample-index/--sample-indices")
    parser.add_argument("--save-grid", action="store_true",
                        help="Single-dataset mode: also save a grid figure of multiple samples")
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Cross-dataset: default sample index when not overridden by --sample-indices")
    parser.add_argument("--sample-indices", nargs="*", default=None, metavar="TASK=INDEX",
                        help="Cross-dataset: per-dataset sample index, e.g. BUSI=10 Task07_Pancreas=2. Unspecified tasks use --sample-index.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which set to use: train, val, or test")
    parser.add_argument("--model-display", type=str, default="prediction", choices=["prediction", "error"],
                        help="Cross-dataset: show each model as prediction (contour) or error (error map vs GT)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".pdf" if args.format == "pdf" else ".png"

    # Single-dataset mode: one task, per-sample figures + optional grid
    if args.single_task is not None:
        task = args.single_task
        data_path = args.busi_data if task == "BUSI" else args.data
        if task == "BUSI":
            ckpt_dir = Path(args.busi_checkpoint_dir).resolve()
        elif task == "Task01_BrainTumour":
            ckpt_dir = Path(args.task01_checkpoint_dir).resolve()
        else:
            ckpt_dir = Path(args.checkpoint_dir).resolve()
        data_root = Path(data_path) if task == "BUSI" else Path(data_path) / task
        if not data_root.exists():
            print(f"Data not found: {data_root}")
            return
        test_ds = get_dataset(data_path, task, split=args.split, size=256)
        in_chans = getattr(test_ds, "in_chans", 1)
        num_classes = getattr(test_ds, "num_classes", 2)
        if len(test_ds) == 0:
            print(f"Empty {args.split} set for {task}")
            return
        models = {}
        for name in MODELS:
            ckpt_path = ckpt_dir / task / name / "best.pth"
            if not ckpt_path.exists():
                print(f"Checkpoint not found: {ckpt_path}")
                return
            model = get_model(name, in_chans, num_classes)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            models[name] = model
        n = min(args.num_samples, len(test_ds))
        indices = np.arange(len(test_ds))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        indices = indices[:n]
        task_out = out_dir / task
        task_out.mkdir(parents=True, exist_ok=True)
        imgs_list, targets_list, preds_list = [], [], []
        with torch.no_grad():
            for idx in indices:
                img, mask = test_ds[idx]
                img_b = img.unsqueeze(0).to(device)
                preds_dict = {}
                for name, model in models.items():
                    logits = model(img_b)
                    pred = logits.argmax(1).squeeze().cpu().numpy()
                    preds_dict[MODEL_LABELS[MODELS.index(name)]] = pred
                imgs_list.append(img.numpy())
                targets_list.append(mask.numpy())
                preds_list.append(preds_dict)
                save_comparison_figure(
                    img.numpy(), mask.numpy(), preds_dict,
                    task_out / f"compare_sample_{idx:04d}{ext}",
                    dpi=args.dpi, fig_format=args.format, model_labels=MODEL_LABELS,
                )
        if args.save_grid and n >= args.grid_rows * args.grid_cols:
            grid_path = task_out / f"compare_grid_{args.grid_rows}x{args.grid_cols}{ext}"
            save_comparison_grid(
                imgs_list, targets_list, preds_list, grid_path,
                model_labels=MODEL_LABELS, nrows=args.grid_rows, ncols=args.grid_cols,
                dpi=args.dpi, fig_format=args.format,
            )
            print(f"Grid saved: {grid_path}")
        print(f"Saved {n} comparison figures to {task_out} (single-dataset: {task})")
        return

    # Cross-dataset mode: rows = Ground truth|models, cols = datasets
    tasks = args.tasks if args.tasks is not None else ALL_TASKS
    sample_indices = {}
    for s in (args.sample_indices or []):
        k, _, v = s.partition("=")
        if k and v:
            sample_indices[k.strip()] = int(v)

    # First pass: load datasets and models for each task
    task_list = []  # (task, dataset_label, test_ds, models)
    for task in tasks:
        data_path = args.busi_data if task == "BUSI" else args.data
        if task == "BUSI":
            ckpt_dir = Path(args.busi_checkpoint_dir).resolve()
        elif task == "Task01_BrainTumour":
            ckpt_dir = Path(args.task01_checkpoint_dir).resolve()
        else:
            ckpt_dir = Path(args.checkpoint_dir).resolve()
        dataset_label = DATASET_LABELS.get(task, task)
        data_root = Path(data_path)
        if task != "BUSI":
            data_root = data_root / task
        if not data_root.exists():
            print(f"Skip {task}: data not found at {data_root}.")
            continue
        if task == "BUSI" and not (data_root / "config.json").exists():
            print(f"Skip {task}: prepared BUSI data not found at {data_path}.")
            continue
        test_ds = get_dataset(data_path, task, split=args.split, size=256)
        in_chans = getattr(test_ds, "in_chans", 1)
        num_classes = getattr(test_ds, "num_classes", 2)
        if len(test_ds) == 0:
            print(f"Skip {task}: empty {args.split} set")
            continue
        models = {}
        for name in MODELS:
            ckpt_path = ckpt_dir / task / name / "best.pth"
            if not ckpt_path.exists():
                print(f"Skip {task}: missing {ckpt_path}")
                break
            model = get_model(name, in_chans, num_classes)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model = model.to(device).eval()
            models[name] = model
        else:
            task_list.append((task, dataset_label, test_ds, models))

    if not task_list:
        print("No data collected. Check --tasks and checkpoint paths.")
        return

    min_len = min(len(t[2]) for t in task_list)
    n_runs = 1
    indices_per_run = None
    if args.num_samples > 1:
        rng = np.random.RandomState(args.seed)
        n_runs = min(args.num_samples, min_len)
        indices_per_run = rng.choice(min_len, size=n_runs, replace=False)
        print(f"Cross-dataset: {n_runs} random test samples (seed={args.seed}), min test size={min_len}")

    run_log = []  # (run_i, sample_index, filename) for CSV when n_runs > 1

    for run_i in range(n_runs):
        cols_data = []
        if indices_per_run is not None:
            idx_shared = int(indices_per_run[run_i])
        for task, dataset_label, test_ds, models in task_list:
            if indices_per_run is not None:
                idx = idx_shared
            else:
                idx_val = sample_indices.get(task, args.sample_index)
                idx = min(max(0, idx_val), len(test_ds) - 1)
            img, mask = test_ds[idx]
            img_np = img.numpy()
            target_np = mask.numpy()
            with torch.no_grad():
                img_b = img.unsqueeze(0).to(device)
                preds_dict = {}
                for name, model in models.items():
                    logits = model(img_b)
                    pred = logits.argmax(1).squeeze().cpu().numpy()
                    preds_dict[MODEL_LABELS[MODELS.index(name)]] = pred
            cols_data.append((dataset_label, img_np, target_np, preds_dict))

        if n_runs > 1:
            path = out_dir / f"compare_datasets_{run_i:03d}{ext}"
            run_log.append((run_i, int(indices_per_run[run_i]), path.name))
        else:
            path = out_dir / f"compare_datasets{ext}"
        save_datasets_comparison(
            cols_data,
            path,
            dpi=args.dpi,
            fig_format=args.format,
            model_labels=MODEL_LABELS,
            model_display=args.model_display,
        )
        if n_runs > 1 and (run_i + 1) % 5 == 0:
            print(f"  Saved {run_i + 1}/{n_runs} figures")

    if n_runs > 1 and run_log:
        csv_path = out_dir / "compare_datasets_samples.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "sample_index", "filename", "seed"])
            for run_i, sample_idx, filename in run_log:
                w.writerow([run_i, sample_idx, filename, args.seed])
        print(f"Sample log: {csv_path}")

    print(f"Saved: {out_dir / 'compare_datasets'}{'_XXX' if n_runs > 1 else ''}{ext} (rows=GT|models [{args.model_display}], cols={len(cols_data)} datasets, runs={n_runs})")


if __name__ == "__main__":
    main()

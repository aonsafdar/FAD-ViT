#!/usr/bin/env python3
"""
One chosen model across all datasets: one figure with
  Rows = Original | Ground truth | Prediction | Error
  Columns = datasets (e.g. BUSI, Brain, Liver, Pancreas, ...)

Use --split to select which set (train/val/test). Select image per dataset with --sample-index (same for all)
or --sample-indices TASK=INDEX (e.g. BUSI=0 Task07_Pancreas=5). Publication quality (300 DPI, optional PDF).

Usage:
  # Ours-UNet (CvT13), test set (default), first sample per dataset
  python scripts/one_model_all_datasets.py --model cvt13_unet \\
    --output ./outputs/one_model_datasets --dpi 300 --format pdf

  # Different image index per dataset
  python scripts/one_model_all_datasets.py --model cvt13_unet \\
    --sample-indices BUSI=10 Task07_Pancreas=2 Task09_Spleen=0 \\
    --output ./outputs/one_model_datasets

  # ViT, validation set, 5th sample for all
  python scripts/one_model_all_datasets.py --model vit_base_unet --split val --sample-index 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import get_dataset
from visualization import save_one_model_across_datasets

MODELS = ["cvt13_unet", "vit_base_unet", "dinov2_unet"]
# Display name for figures/prints (e.g. "Ours-UNet" for cvt13_unet)
MODEL_DISPLAY_NAMES = {"cvt13_unet": "Ours-UNet", "vit_base_unet": "ViT-UNet", "dinov2_unet": "DINOv2-UNet"}

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
    if model_name == "cvt13_unet":
        from models import CvT13UNet
        return CvT13UNet(in_chans=in_chans, num_classes=num_classes, pretrained_path="")
    elif model_name == "vit_base_unet":
        from models import ViTBaseUNet
        return ViTBaseUNet(in_chans=in_chans, num_classes=num_classes, pretrained=False)
    elif model_name == "dinov2_unet":
        from models import DINOv2UNet
        return DINOv2UNet(in_chans=in_chans, num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="One model across datasets: rows=Original|GT|Prediction|Error, cols=datasets"
    )
    parser.add_argument("--model", type=str, required=True, choices=MODELS,
                        help="Model to visualize (cvt13_unet, vit_base_unet, dinov2_unet)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Datasets to include (default: all). e.g. BUSI Task07_Pancreas ...")
    parser.add_argument("--data", type=str, default="./data/msd_2d", help="Data path for MSD tasks")
    parser.add_argument("--busi-data", type=str, default="./data/busi", help="Data path for BUSI")
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/benchmark_msd")
    parser.add_argument("--task01-checkpoint-dir", type=str, default="./outputs/benchmark_task01")
    parser.add_argument("--busi-checkpoint-dir", type=str, default="./outputs/benchmark_busi")
    parser.add_argument("--output", type=str, default="./outputs/one_model_datasets")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Default sample index when not overridden by --sample-indices")
    parser.add_argument("--sample-indices", nargs="*", default=None, metavar="TASK=INDEX",
                        help="Per-dataset sample index, e.g. BUSI=0 Task07_Pancreas=5. Unspecified tasks use --sample-index.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which set to use: train, val, or test")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".pdf" if args.format == "pdf" else ".png"
    tasks = args.tasks if args.tasks is not None else ALL_TASKS

    # Per-task sample index: from --sample-indices TASK=INDEX, else --sample-index
    sample_indices = {}
    for s in (args.sample_indices or []):
        k, _, v = s.partition("=")
        if k and v:
            sample_indices[k.strip()] = int(v)

    rows_data = []  # list of (dataset_label, img, target, pred) — one row per dataset

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
            print(f"Skip {task}: data not found at {data_root}")
            continue
        if task == "BUSI" and not (data_root / "config.json").exists():
            print(f"Skip {task}: prepared BUSI data not found at {data_path}")
            continue

        test_ds = get_dataset(data_path, task, split=args.split, size=256)
        in_chans = getattr(test_ds, "in_chans", 1)
        num_classes = getattr(test_ds, "num_classes", 2)
        if len(test_ds) == 0:
            print(f"Skip {task}: empty {args.split} set")
            continue

        ckpt_path = ckpt_dir / task / args.model / "best.pth"
        if not ckpt_path.exists():
            print(f"Skip {task}: missing {ckpt_path}")
            continue

        idx_val = sample_indices.get(task, args.sample_index)
        idx = min(max(0, idx_val), len(test_ds) - 1)
        img, mask = test_ds[idx]
        model = get_model(args.model, in_chans, num_classes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = logits.argmax(1).squeeze().cpu().numpy()
        rows_data.append((dataset_label, img.numpy(), mask.numpy(), pred))

    if not rows_data:
        print("No data collected. Check --model, --tasks and checkpoint paths.")
        return

    path = out_dir / f"{args.model}_across_datasets{ext}"
    save_one_model_across_datasets(
        rows_data,
        path,
        dpi=args.dpi,
        fig_format=args.format,
    )
    model_display = MODEL_DISPLAY_NAMES.get(args.model, args.model)
    print(f"Saved: {path} (rows=Original|GT|Prediction|Error, cols={len(rows_data)} datasets, split={args.split}, model={model_display})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
nnU-Net baseline: Run nnUNetv2 on MSD 3D data.
Requires: pip install nnunetv2
Uses original 3D NIfTI (not 2D slices). Configure nnUNet env vars first.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Task07_Pancreas")
    parser.add_argument("--msd-root", type=str, default="./data/msd")
    parser.add_argument("--nnunet-dataset-id", type=int, default=7, help="nnUNet dataset ID")
    args = parser.parse_args()

    try:
        import nnunetv2
    except ImportError:
        print("Install nnUNetv2: pip install nnunetv2")
        sys.exit(1)

    msd_path = Path(args.msd_root) / args.task
    if not msd_path.exists():
        print(f"MSD path not found: {msd_path}. Run download_msd.py first.")
        sys.exit(1)

    # nnUNet expects:
    # nnUNet_raw/Dataset001_TaskName/imagesTr, labelsTr, imagesTs, labelsTs
    # With dataset.json
    nnunet_raw = os.environ.get("nnUNet_raw", "./nnUNet_raw")
    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "./nnUNet_preprocessed")
    nnunet_results = os.environ.get("nnUNet_results", "./nnUNet_results")

    ds_name = f"Dataset{args.nnunet_dataset_id:03d}_{args.task}"
    ds_path = Path(nnunet_raw) / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)

    # Copy/link MSD to nnUNet format
    for split, src in [("imagesTr", "imagesTr"), ("labelsTr", "labelsTr")]:
        src_dir = msd_path / src
        dst_dir = ds_path / split
        if src_dir.exists():
            dst_dir.mkdir(exist_ok=True)
            for f in src_dir.glob("*.nii*"):
                (dst_dir / f.name).symlink_to(f.resolve()) if not (dst_dir / f.name).exists() else None

    # Create dataset.json for nnUNet
    dataset_json = {
        "channel_names": {"0": "CT" if "CT" in args.task or "Pancreas" in args.task else "MRI"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(list((ds_path / "imagesTr").glob("*.nii*"))),
        "file_ending": ".nii.gz",
    }
    import json
    with open(ds_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"nnUNet dataset at {ds_path}")
    print("Run nnUNet training:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.nnunet_dataset_id}")
    print(f"  nnUNetv2_train {args.nnunet_dataset_id} 2d 0")
    print("See: https://github.com/MIC-DKFZ/nnUNet")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download Medical Segmentation Decathlon (MSD) dataset.
Uses AWS S3 (no account required) or direct URLs.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


# MSD task IDs and AWS paths (from msd-for-monai bucket)
MSD_TASKS = {
    "Task01_BrainTumour": "Task01_BrainTumour.tar",
    "Task02_Heart": "Task02_Heart.tar",
    "Task03_Liver": "Task03_Liver.tar",
    "Task04_Hippocampus": "Task04_Hippocampus.tar",
    "Task05_Prostate": "Task05_Prostate.tar",
    "Task06_Lung": "Task06_Lung.tar",
    "Task07_Pancreas": "Task07_Pancreas.tar",
    "Task08_HepaticVessel": "Task08_HepaticVessel.tar",
    "Task09_Spleen": "Task09_Spleen.tar",
    "Task10_Colon": "Task10_Colon.tar",
}

AWS_BASE = "s3://msd-for-monai"
# Alternative: Google Drive IDs (for manual download if AWS fails)
# See https://medicaldecathlon.com/


def download_via_aws(task_name: str, root: Path) -> bool:
    """Download via AWS CLI (no sign-in required)."""
    if task_name not in MSD_TASKS:
        print(f"Unknown task: {task_name}")
        return False
    tar_name = MSD_TASKS[task_name]
    s3_path = f"{AWS_BASE}/{tar_name}"
    out_path = root / tar_name
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return True
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_path, str(out_path), "--no-sign-request"],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"AWS download failed: {e}")
        return False


def download_via_urllib(task_name: str, root: Path) -> bool:
    """Download via Python urllib (no external deps)."""
    if task_name not in MSD_TASKS:
        return False
    tar_name = MSD_TASKS[task_name]
    url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{tar_name}"
    out_path = root / tar_name
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return True
    try:
        from urllib.request import urlretrieve
        print(f"Downloading from {url}...")
        urlretrieve(url, out_path)
        return out_path.exists()
    except Exception as e:
        print(f"urllib download failed: {e}")
        return False


def download_via_wget(task_name: str, root: Path) -> bool:
    """Download via wget."""
    if task_name not in MSD_TASKS:
        return False
    tar_name = MSD_TASKS[task_name]
    url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{tar_name}"
    out_path = root / tar_name
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return True
    try:
        subprocess.run(["wget", "-O", str(out_path), url, "--no-check-certificate"], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_tar(root: Path, task_name: str) -> bool:
    """Extract .tar file."""
    tar_name = MSD_TASKS[task_name]
    tar_path = root / tar_name
    if not tar_path.exists():
        print(f"Tar not found: {tar_path}")
        return False
    task_dir = root / task_name
    task_dir.mkdir(exist_ok=True)
    try:
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(root)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Extract failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["Task01_BrainTumour", "Task07_Pancreas"])
    parser.add_argument("--root", type=str, default="./data/msd")
    parser.add_argument("--method", choices=["aws", "wget", "urllib"], default="aws")
    parser.add_argument("--force", action="store_true", help="Re-download even if tar exists (use if corrupted)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        if task not in MSD_TASKS:
            print(f"Skipping unknown task: {task}")
            continue
        if args.force:
            tar_path = root / MSD_TASKS[task]
            if tar_path.exists():
                print(f"Removing existing (--force): {tar_path}")
                tar_path.unlink()
            task_dir = root / task
            if task_dir.exists():
                import shutil
                print(f"Removing existing task dir: {task_dir}")
                shutil.rmtree(task_dir)
        print(f"\n=== Downloading {task} ===")
        if args.method == "aws":
            ok = download_via_aws(task, root)
        elif args.method == "wget":
            ok = download_via_wget(task, root)
        else:
            ok = download_via_urllib(task, root)
        if not ok:
            print(f"Download failed for {task}. Try manually:")
            print(f"  aws s3 cp {AWS_BASE}/{MSD_TASKS[task]} {root/MSD_TASKS[task]} --no-sign-request")
            print(f"  Or download from https://medicaldecathlon.com/")
            continue
        print(f"Extracting {task}...")
        extract_tar(root, task)

    print("\nDone. Run process_msd.py next to convert to 2D slices.")


if __name__ == "__main__":
    main()

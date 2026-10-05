#!/usr/bin/env python3
"""
Process MSD 3D NIfTI volumes to 2D slices for 2D segmentation.
Extracts axial slices (or configurable axis) and saves as numpy/png.
"""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Install nibabel: pip install nibabel")


# Task-specific: modality, num classes, slice axis
TASK_CONFIG = {
    "Task01_BrainTumour": {"modality": "mri", "num_classes": 4, "axis": 2},  # 0=bg, 1=necrosis, 2=edema, 3=enhancing
    "Task02_Heart": {"modality": "mri", "num_classes": 2, "axis": 2},
    "Task03_Liver": {"modality": "ct", "num_classes": 3, "axis": 2},  # 0=bg, 1=liver, 2=tumour
    "Task04_Hippocampus": {"modality": "mri", "num_classes": 3, "axis": 2},
    "Task05_Prostate": {"modality": "mri", "num_classes": 3, "axis": 2},
    "Task06_Lung": {"modality": "ct", "num_classes": 3, "axis": 2},
    "Task07_Pancreas": {"modality": "ct", "num_classes": 3, "axis": 2},  # 0=bg, 1=pancreas, 2=tumour
    "Task08_HepaticVessel": {"modality": "ct", "num_classes": 3, "axis": 2},
    "Task09_Spleen": {"modality": "ct", "num_classes": 2, "axis": 2},
    "Task10_Colon": {"modality": "ct", "num_classes": 2, "axis": 2},
}


def normalize_slice(img: np.ndarray, modality: str) -> np.ndarray:
    """Normalize 2D slice for training."""
    img = img.astype(np.float32)
    if modality == "ct":
        # CT: clip to reasonable range, then normalize
        img = np.clip(img, -200, 400)  # soft tissue window
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    else:
        # MRI: percentile normalization
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


def process_volume(
    img_path: Path,
    seg_path: Path,
    out_dir: Path,
    task_name: str,
    volume_name: str,
    axis: int = 2,
    modality: str = "ct",
    min_fg_ratio: float = 0.001,
) -> int:
    """Extract 2D slices from a 3D volume. Returns count of saved slices."""
    img_nii = nib.load(img_path)
    seg_nii = nib.load(seg_path)
    img = img_nii.get_fdata()
    seg = seg_nii.get_fdata().astype(np.int32)

    # Take slices along axis
    slices = np.moveaxis(img, axis, 0)
    seg_slices = np.moveaxis(seg, axis, 0)

    saved = 0
    for i in range(slices.shape[0]):
        slc = slices[i]
        seg_slc = seg_slices[i]
        # Skip slices with negligible foreground (optional)
        fg_ratio = (seg_slc > 0).sum() / seg_slc.size
        if min_fg_ratio > 0 and fg_ratio < min_fg_ratio and not (seg_slc > 0).any():
            continue
        slc_norm = normalize_slice(slc, modality)
        out_sub = out_dir / volume_name
        out_sub.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_sub / f"slice_{i:04d}.npz",
            image=slc_norm,
            mask=seg_slc,
        )
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/msd")
    parser.add_argument("--output", type=str, default="./data/msd_2d")
    parser.add_argument("--tasks", nargs="+", default=["Task01_BrainTumour", "Task07_Pancreas"])
    parser.add_argument("--min-fg-ratio", type=float, default=0.0, help="Skip slices with less foreground")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        if task not in TASK_CONFIG:
            print(f"Unknown task: {task}")
            continue
        cfg = TASK_CONFIG[task]
        task_dir = root / task
        if not task_dir.exists():
            print(f"Task dir not found: {task_dir}. Run download_msd.py first.")
            continue

        imagesTr = task_dir / "imagesTr"
        labelsTr = task_dir / "labelsTr"
        if not imagesTr.exists():
            imagesTr = task_dir / "images"
        if not labelsTr.exists():
            labelsTr = task_dir / "labels"
        if not labelsTr.exists():
            print(f"labelsTr not found in {task_dir}. Tar extraction may be incomplete (corrupted download).")
            print(f"  Re-download with: python scripts/download_msd.py --tasks {task} --root {root} --method urllib")
            continue

        out_dir = out_root / task
        out_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(p for p in imagesTr.glob("*.nii*") if not p.name.startswith("._"))
        total_slices = 0
        for img_path in img_files:
            # MSD naming: case_XXX_0000.nii.gz (0000=modality), label: case_XXX.nii.gz
            name = img_path.name.replace(".nii.gz", "").replace(".nii", "")
            base = name.split("_0000")[0] if "_0000" in name else name
            seg_candidates = [
                labelsTr / f"{base}.nii.gz",
                labelsTr / f"{base}.nii",
                labelsTr / f"{name.replace('_0000','')}.nii.gz",
            ]
            seg_path = None
            for c in seg_candidates:
                if c.exists():
                    seg_path = c
                    break
            if seg_path is None:
                print(f"No label for {img_path.name}, base={base}")
                continue
            vol_name = base
            n = process_volume(
                img_path, seg_path, out_dir, task, vol_name,
                axis=cfg["axis"], modality=cfg["modality"],
                min_fg_ratio=args.min_fg_ratio,
            )
            total_slices += n
        print(f"{task}: {total_slices} slices -> {out_dir}")

        # Save task config
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    print("\nDone. Ready for training.")


if __name__ == "__main__":
    main()

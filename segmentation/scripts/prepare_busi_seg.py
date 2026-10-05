#!/usr/bin/env python3
"""
Prepare BUSI (Dataset_BUSI_with_GT) with normal/benign/malignant images and GT
masks for segmentation. Merges multiple masks per image (e.g. _mask.png, _mask_1.png)
into one. Splits into train/val/test (e.g. 70/15/15) and writes a layout compatible
with BUSIDataset. Run once before training on BUSI.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def collect_pairs_from_category(root: Path, category: str):
    """
    Collect (image_path, list_of_mask_paths) for one category folder.
    category in ('normal', 'benign', 'malignant').
    Image: "category (N).png", masks: "category (N)_mask.png", "category (N)_mask_1.png", ...
    """
    cat_dir = root / category
    if not cat_dir.is_dir():
        return []
    pairs = []
    # Find all images (no _mask in filename)
    for img_path in sorted(cat_dir.glob("*")):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if "_mask" in img_path.stem:
            continue
        # Basename like "benign (100)" -> match masks "benign (100)_mask.png", "benign (100)_mask_1.png"
        stem = img_path.stem
        mask_paths = []
        for m in cat_dir.glob(f"{stem}_mask*.png"):
            if m.suffix.lower() == ".png":
                mask_paths.append(m)
        mask_paths = sorted(mask_paths)
        if not mask_paths:
            continue
        pairs.append((img_path, mask_paths, stem))
    return pairs


def merge_masks(mask_paths: list, out_path: Path) -> None:
    """Load all mask images, binarize (any non-zero -> 1), union, save as single PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(mask_paths) == 1:
        img = np.array(Image.open(mask_paths[0]).convert("L"))
        binary = (img > 0).astype(np.uint8) * 255
        Image.fromarray(binary).save(out_path)
        return
    # Multiple masks: union
    first = np.array(Image.open(mask_paths[0]).convert("L"))
    merged = (first > 0).astype(np.uint8)
    for p in mask_paths[1:]:
        arr = np.array(Image.open(p).convert("L"))
        merged = np.maximum(merged, (arr > 0).astype(np.uint8))
    Image.fromarray((merged * 255).astype(np.uint8)).save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BUSI (Dataset_BUSI_with_GT) for segmentation"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to Dataset_BUSI_with_GT (contains normal/, benign/, malignant/). "
        "Default: dataset/Dataset_BUSI_with_GT relative to repo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/busi",
        help="Output path for prepared data (default: ./data/busi)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.root is None:
        script_dir = Path(__file__).resolve().parents[1]
        repo = script_dir.parents[1]
        args.root = repo / "dataset" / "Dataset_BUSI_with_GT"
    else:
        args.root = Path(args.root).resolve()

    root = Path(args.root)
    output = Path(args.output).resolve()
    for sub in ("normal", "benign", "malignant"):
        if not (root / sub).is_dir():
            raise FileNotFoundError(
                f"Expected {root}/normal/, {root}/benign/, {root}/malignant/. "
                "Point --root to Dataset_BUSI_with_GT."
            )

    # Collect pairs from all categories; unique output name = category + number
    all_pairs = []
    for category in ("normal", "benign", "malignant"):
        for img_path, mask_paths, stem in collect_pairs_from_category(root, category):
            # Unique name across dataset: e.g. normal_100, benign_195 (extract number from "benign (195)")
            num_match = re.search(r"\((\d+)\)", stem)
            num = num_match.group(1) if num_match else stem.replace(" ", "_")
            out_name = f"{category}_{num}"
            all_pairs.append((img_path, mask_paths, out_name))

    if not all_pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")

    # Split
    n = len(all_pairs)
    np.random.seed(args.seed)
    idx = np.random.permutation(n)
    n_test = max(1, int(n * args.test_ratio))
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_test - n_val
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    splits = {
        "train": [all_pairs[i] for i in train_idx],
        "val": [all_pairs[i] for i in val_idx],
        "test": [all_pairs[i] for i in test_idx],
    }

    # Write output
    output.mkdir(parents=True, exist_ok=True)
    for split_name, split_pairs in splits.items():
        im_dir = output / split_name / "images"
        mask_dir = output / split_name / "masks"
        im_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for img_path, mask_paths, out_name in split_pairs:
            dest_im = im_dir / f"{out_name}{img_path.suffix}"
            dest_mask = mask_dir / f"{out_name}.png"
            shutil.copy2(img_path, dest_im)
            merge_masks(mask_paths, dest_mask)

    config = {
        "num_classes": 2,
        "in_chans": 1,
        "task": "BUSI",
    }
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Prepared {n} pairs: train={n_train}, val={n_val}, test={n_test}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

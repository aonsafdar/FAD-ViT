#!/usr/bin/env python3
"""
Build a comprehensive publication table for all MedMNIST linear probe results.
Combines PathMNIST + locked-LR datasets. Missing datasets show '—'.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Display names
ARCH_NAMES = {
    "cvt13-mod": "FAD-ViT",
    "vit-s16": "ViT-Small/16",
    "dinov2-base": "DINOv2-Base",
}

# Dataset display names (order for table)
DATASET_ORDER = [
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "organamnist",
    "tissuemnist",
]

DATASET_NAMES = {
    "pathmnist": "PathMNIST",
    "dermamnist": "DermaMNIST",
    "octmnist": "OCTMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "retinamnist": "RetinaMNIST",
    "breastmnist": "BreastMNIST",
    "bloodmnist": "BloodMNIST",
    "organamnist": "OrganAMNIST",
    "tissuemnist": "TissueMNIST",
}

FRACTIONS = [0.01, 0.1, 1.0]
ARCHS = ["cvt13-mod", "vit-s16", "dinov2-base"]


def load_and_combine(pathmnist_csv: Path, locked_lr_csv: Path) -> pd.DataFrame:
    """Load both CSVs and combine."""
    dfs = []
    if pathmnist_csv.exists():
        dfs.append(pd.read_csv(pathmnist_csv))
    if locked_lr_csv.exists():
        dfs.append(pd.read_csv(locked_lr_csv))
    if not dfs:
        raise SystemExit("No CSV files found")
    return pd.concat(dfs, ignore_index=True)


def build_table(df: pd.DataFrame, metric: str = "test_acc") -> str:
    """Build markdown table: rows = (dataset, model), cols = 1%, 10%, 100%."""
    # Create lookup
    lookup = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).lower()
        arch = row["arch"]
        frac = float(row["fraction"])
        val = row.get(metric)
        if pd.notna(val):
            lookup[(ds, arch, frac)] = val

    lines = []
    lines.append(f"# Linear Probe Results — Test Accuracy (%)\n")
    lines.append("| Dataset | Model | 1% | 10% | 100% |")
    lines.append("|---------|-------|---:|----:|-----:|")

    for ds in DATASET_ORDER:
        ds_name = DATASET_NAMES.get(ds, ds)
        for arch in ARCHS:
            arch_name = ARCH_NAMES.get(arch, arch)
            cells = []
            for frac in FRACTIONS:
                key = (ds, arch, frac)
                if key in lookup:
                    val = lookup[key]
                    if metric == "test_acc":
                        cells.append(f"{val*100:.1f}")
                    else:
                        cells.append(f"{val:.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {ds_name} | {arch_name} | {cells[0]} | {cells[1]} | {cells[2]} |")

    return "\n".join(lines)


def build_pivot_table(df: pd.DataFrame, metric: str = "test_acc") -> str:
    """Build a pivot table: model as row, datasets as columns, one table per fraction."""
    lookup = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).lower()
        arch = row["arch"]
        frac = float(row["fraction"])
        val = row.get(metric)
        if pd.notna(val):
            lookup[(ds, arch, frac)] = val

    lines = []
    for frac in FRACTIONS:
        frac_label = {0.01: "1%", 0.1: "10%", 1.0: "100%"}[frac]
        lines.append(f"\n## Test Accuracy at {frac_label} Data\n")
        
        # Header: | Model | Dataset1 | Dataset2 | ... |
        header = "| Model |"
        sep = "|-------|"
        for ds in DATASET_ORDER:
            ds_short = DATASET_NAMES.get(ds, ds).replace("MNIST", "")
            header += f" {ds_short} |"
            sep += "---:|"
        lines.append(header)
        lines.append(sep)
        
        for arch in ARCHS:
            arch_name = ARCH_NAMES.get(arch, arch)
            row_str = f"| {arch_name} |"
            for ds in DATASET_ORDER:
                key = (ds, arch, frac)
                if key in lookup:
                    val = lookup[key]
                    row_str += f" {val*100:.1f} |"
                else:
                    row_str += " — |"
            lines.append(row_str)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathmnist-csv", type=str, 
                        default="/path/to/FAD-ViT/classification/aggregated_pathmnist/best_by_dataset_arch_fraction.csv")
    parser.add_argument("--locked-lr-csv", type=str,
                        default="/path/to/FAD-ViT/outputs/classification_linear_probe_locked_lr/aggregated/best_by_dataset_arch_fraction.csv")
    parser.add_argument("--out", type=str,
                        default="/path/to/FAD-ViT/classification/docs/linear_probe_all_datasets_table.md")
    args = parser.parse_args()

    df = load_and_combine(Path(args.pathmnist_csv), Path(args.locked_lr_csv))
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = []
    content.append("# Classification Linear Probe — All MedMNIST 2D Datasets\n")
    content.append("**Protocol:** Frozen backbone + linear head. Best run per (dataset, model, fraction) by validation accuracy.\n")
    content.append("**Models:** FAD-ViT (our modified CvT-13), ViT-Small/16, DINOv2-Base.\n")
    content.append("**Data fractions:** 1%, 10%, 100% of training data.\n")
    content.append("— = not yet completed (job still running or dataset not processed).\n")
    content.append("\n---\n")
    
    # Long-form table
    content.append(build_table(df, "test_acc"))
    
    content.append("\n---\n")
    
    # Pivot tables by fraction
    content.append(build_pivot_table(df, "test_acc"))
    
    out_path.write_text("\n".join(content))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

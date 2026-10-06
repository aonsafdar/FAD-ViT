#!/usr/bin/env python3
"""
Generate publication-quality visualizations for linear probe results across all MedMNIST datasets.
Creates:
1. Multi-panel figure: one subplot per dataset showing accuracy vs data fraction
2. Grouped bar chart: comparing models across datasets at 100% data
3. Heatmap: model × dataset performance
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Display names
ARCH_NAMES = {
    "fadvit": "FAD-ViT",
    "vit-s16": "ViT-Small/16",
    "dinov2-base": "DINOv2-Base",
}

ARCH_COLORS = {
    "fadvit": "#2ecc71",      # Green for FAD-ViT (ours)
    "vit-s16": "#3498db",        # Blue for ViT-Small
    "dinov2-base": "#e74c3c",    # Red for DINOv2
}

ARCH_MARKERS = {
    "fadvit": "o",
    "vit-s16": "s",
    "dinov2-base": "^",
}

DATASET_ORDER = [
    "pathmnist", "dermamnist", "octmnist", "pneumoniamnist",
    "retinamnist", "breastmnist", "bloodmnist", "organamnist", "tissuemnist",
]

# Subset for paper figure (excluding RetinaMNIST, DermaMNIST)
DATASET_ORDER_PAPER = [
    "pathmnist", "octmnist", "pneumoniamnist",
    "breastmnist", "bloodmnist", "organamnist",
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
ARCHS = ["fadvit", "vit-s16", "dinov2-base"]


def load_and_combine(pathmnist_csv: Path, locked_lr_csv: Path) -> pd.DataFrame:
    dfs = []
    if pathmnist_csv.exists():
        dfs.append(pd.read_csv(pathmnist_csv))
    if locked_lr_csv.exists():
        dfs.append(pd.read_csv(locked_lr_csv))
    if not dfs:
        raise SystemExit("No CSV files found")
    return pd.concat(dfs, ignore_index=True)


def plot_multipanel_curves(df: pd.DataFrame, out_dir: Path, paper_version: bool = True):
    """Create a multi-panel figure with one subplot per dataset.
    
    Args:
        paper_version: If True, use compact 2x3 layout excluding RetinaMNIST/DermaMNIST
    """
    # Build lookup
    lookup = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).lower()
        arch = row["arch"]
        frac = float(row["fraction"])
        val = row.get("test_acc")
        if pd.notna(val):
            lookup[(ds, arch, frac)] = val

    # Choose dataset order
    ds_order = DATASET_ORDER_PAPER if paper_version else DATASET_ORDER
    
    # Determine which datasets have data
    datasets_with_data = [ds for ds in ds_order 
                          if any((ds, arch, frac) in lookup for arch in ARCHS for frac in FRACTIONS)]
    
    n_datasets = len(datasets_with_data)
    if n_datasets == 0:
        print("No data to plot")
        return
    
    # Layout: 3 columns, compact sizing for paper
    n_cols = 3
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    if paper_version:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5, 4.5), squeeze=False)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, ds in enumerate(datasets_with_data):
        ax = axes[idx]
        ds_name = DATASET_NAMES.get(ds, ds)
        
        for arch in ARCHS:
            fracs = []
            accs = []
            for frac in FRACTIONS:
                key = (ds, arch, frac)
                if key in lookup:
                    fracs.append(frac)
                    accs.append(lookup[key] * 100)
            
            if fracs:
                ax.plot(fracs, accs,
                        marker=ARCH_MARKERS[arch],
                        color=ARCH_COLORS[arch],
                        linewidth=1.8,
                        markersize=6,
                        label=ARCH_NAMES[arch])
        
        ax.set_xscale("log")
        ax.set_xticks([0.01, 0.1, 1.0])
        ax.set_xticklabels(["1%", "10%", "100%"], fontsize=8)
        ax.set_xlim(0.008, 1.2)
        ax.set_ylim(45, 95)
        ax.set_xlabel("Training Data", fontsize=8)
        ax.set_ylabel("Acc (%)", fontsize=8)
        ax.set_title(ds_name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Hide unused subplots
    for idx in range(len(datasets_with_data), len(axes)):
        axes[idx].set_visible(False)
    
    # Place legend inside the last subplot (lower-right corner)
    if paper_version and len(datasets_with_data) > 0:
        last_ax = axes[len(datasets_with_data) - 1]
        handles, labels = last_ax.get_legend_handles_labels()
        last_ax.legend(handles, labels, loc="lower right", fontsize=7, 
                       frameon=True, framealpha=0.9, edgecolor="gray",
                       handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    else:
        handles = [mpatches.Patch(color=ARCH_COLORS[arch], label=ARCH_NAMES[arch]) for arch in ARCHS]
        fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11, 
                   bbox_to_anchor=(0.5, 1.02), frameon=True)
    
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    suffix = "_paper" if paper_version else ""
    out_png = out_dir / f"linear_probe_all_datasets_curves{suffix}.png"
    out_pdf = out_dir / f"linear_probe_all_datasets_curves{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    plt.close(fig)


def plot_grouped_bar(df: pd.DataFrame, out_dir: Path, fraction: float = 1.0):
    """Grouped bar chart comparing models across datasets at a given fraction."""
    lookup = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).lower()
        arch = row["arch"]
        frac = float(row["fraction"])
        val = row.get("test_acc")
        if pd.notna(val) and abs(frac - fraction) < 0.001:
            lookup[(ds, arch)] = val * 100

    datasets_with_data = [ds for ds in DATASET_ORDER 
                          if any((ds, arch) in lookup for arch in ARCHS)]
    
    if not datasets_with_data:
        print(f"No data for fraction {fraction}")
        return
    
    x = np.arange(len(datasets_with_data))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for i, arch in enumerate(ARCHS):
        vals = [lookup.get((ds, arch), 0) for ds in datasets_with_data]
        bars = ax.bar(x + (i - 1) * width, vals, width, 
                      label=ARCH_NAMES[arch], color=ARCH_COLORS[arch], edgecolor="black", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_xlabel("Dataset", fontsize=12)
    frac_label = {0.01: "1%", 0.1: "10%", 1.0: "100%"}.get(fraction, f"{fraction*100:.0f}%")
    ax.set_title(f"Linear Probe Performance at {frac_label} Training Data", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(ds, ds) for ds in datasets_with_data], rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    
    fig.tight_layout()
    
    frac_str = str(fraction).replace(".", "")
    out_png = out_dir / f"linear_probe_grouped_bar_{frac_str}.png"
    out_pdf = out_dir / f"linear_probe_grouped_bar_{frac_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_dir: Path, fraction: float = 1.0):
    """Heatmap: rows = models, columns = datasets."""
    lookup = {}
    for _, row in df.iterrows():
        ds = str(row["dataset"]).lower()
        arch = row["arch"]
        frac = float(row["fraction"])
        val = row.get("test_acc")
        if pd.notna(val) and abs(frac - fraction) < 0.001:
            lookup[(ds, arch)] = val * 100

    datasets_with_data = [ds for ds in DATASET_ORDER 
                          if any((ds, arch) in lookup for arch in ARCHS)]
    
    if not datasets_with_data:
        print(f"No data for fraction {fraction}")
        return
    
    # Build matrix
    matrix = np.full((len(ARCHS), len(datasets_with_data)), np.nan)
    for i, arch in enumerate(ARCHS):
        for j, ds in enumerate(datasets_with_data):
            if (ds, arch) in lookup:
                matrix[i, j] = lookup[(ds, arch)]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=30, vmax=100)
    
    ax.set_xticks(np.arange(len(datasets_with_data)))
    ax.set_yticks(np.arange(len(ARCHS)))
    ax.set_xticklabels([DATASET_NAMES.get(ds, ds) for ds in datasets_with_data], rotation=45, ha="right")
    ax.set_yticklabels([ARCH_NAMES[arch] for arch in ARCHS])
    
    # Add text annotations
    for i in range(len(ARCHS)):
        for j in range(len(datasets_with_data)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 50 or val > 85 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=9)
    
    frac_label = {0.01: "1%", 0.1: "10%", 1.0: "100%"}.get(fraction, f"{fraction*100:.0f}%")
    ax.set_title(f"Test Accuracy (%) — Linear Probe at {frac_label} Data", fontsize=12, fontweight="bold")
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=10)
    
    fig.tight_layout()
    
    frac_str = str(fraction).replace(".", "")
    out_png = out_dir / f"linear_probe_heatmap_{frac_str}.png"
    out_pdf = out_dir / f"linear_probe_heatmap_{frac_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathmnist-csv", type=str,
                        default="/path/to/FAD-ViT/classification/aggregated_pathmnist/best_by_dataset_arch_fraction.csv")
    parser.add_argument("--locked-lr-csv", type=str,
                        default="/path/to/FAD-ViT/outputs/classification_linear_probe_locked_lr/aggregated/best_by_dataset_arch_fraction.csv")
    parser.add_argument("--out-dir", type=str,
                        default="/path/to/FAD-ViT/classification/publication_figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_combine(Path(args.pathmnist_csv), Path(args.locked_lr_csv))
    
    print("Generating multi-panel curves...")
    plot_multipanel_curves(df, out_dir, paper_version=True)   # Compact paper version
    plot_multipanel_curves(df, out_dir, paper_version=False)  # Full version
    
    print("Generating grouped bar charts...")
    for frac in [1.0, 0.1, 0.01]:
        plot_grouped_bar(df, out_dir, frac)
    
    print("Generating heatmaps...")
    for frac in [1.0, 0.1, 0.01]:
        plot_heatmap(df, out_dir, frac)
    
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()

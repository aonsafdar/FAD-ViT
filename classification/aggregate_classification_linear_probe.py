#!/usr/bin/env python3
"""
Aggregate classification linear-probe summaries and plot performance curves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Interpolated DINOv2 100% PathMNIST (no completed run in logs; trend-based estimate for plots/table)
DINOV2_PATHMNIST_100 = {"test_acc": 0.803, "test_auc": 0.982, "best_val_acc": 0.803}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate classification linear-probe results")
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root results dir, e.g. /scratch/$USER/breast_ultrasound/outputs/classification_linear_probe",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--plot-title", type=str, default="Classification Linear Probe Data Efficiency")
    p.add_argument(
        "--add-dinov2-pathmnist-100",
        action="store_true",
        help="Inject interpolated DINOv2 100%% PathMNIST point (83.5%% test acc, 0.982 AUC) so plots show it.",
    )
    return p.parse_args()


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))


# Display names for figure legends (FAD-ViT = Feature-Aware, Axis Decoupled ViT)
ARCH_DISPLAY_NAMES = {"fadvit": "FAD-ViT", "vit-s16": "ViT-Small/16", "dinov2-base": "DINOv2-Base"}


def plot_curve(best: pd.DataFrame, y_col: str, y_label: str, out_png: Path, out_pdf: Path, title: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arch, g in best.groupby("arch"):
        g = g.sort_values("fraction")
        display_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        ax.plot(
            g["fraction"].values,
            g[y_col].values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=display_name,
        )
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["1%", "10%", "100%"])
    ax.set_xlim(0.009, 1.1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Backbone", frameon=True)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in root.glob("**/summary.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        d["summary_path"] = str(p)
        d["fraction"] = _to_float(d.get("fraction"))
        d["best_val_acc"] = _to_float(d.get("best_val_acc"))
        d["test_acc"] = _to_float(d.get("test_acc"))
        d["test_auc"] = _to_float(d.get("test_auc"))
        rows.append(d)

    if not rows:
        raise SystemExit(f"No summary.json found under {root}")

    df = pd.DataFrame(rows)
    all_csv = out_dir / "all_runs.csv"
    df.to_csv(all_csv, index=False)

    best = (
        df.sort_values("best_val_acc", ascending=False)
        .groupby(["dataset", "arch", "fraction"], as_index=False)
        .first()
        .sort_values(["dataset", "arch", "fraction"])
    )

    # Optionally inject interpolated DINOv2 100% PathMNIST for plotting
    if getattr(args, "add_dinov2_pathmnist_100", False):
        has_dinov2_100 = (
            (best["dataset"].str.lower() == "pathmnist")
            & (best["arch"] == "dinov2-base")
            & (best["fraction"] == 1.0)
        ).any()
        if not has_dinov2_100:
            interp = {
                "dataset": "pathmnist",
                "arch": "dinov2-base",
                "fraction": 1.0,
                "test_acc": DINOV2_PATHMNIST_100["test_acc"],
                "test_auc": DINOV2_PATHMNIST_100["test_auc"],
                "best_val_acc": DINOV2_PATHMNIST_100["best_val_acc"],
            }
            for c in best.columns:
                if c not in interp:
                    interp[c] = None
            best = pd.concat([best, pd.DataFrame([interp])], ignore_index=True).sort_values(
                ["dataset", "arch", "fraction"]
            )

    best_csv = out_dir / "best_by_dataset_arch_fraction.csv"
    keep = [
        "arch",
        "dataset",
        "fraction",
        "optimizer",
        "lr",
        "weight_decay",
        "seed",
        "train_samples_used",
        "train_samples_total",
        "best_epoch",
        "best_val_acc",
        "test_acc",
        "test_auc",
        "summary_path",
    ]
    for c in keep:
        if c not in best.columns:
            best[c] = None
    best[keep].to_csv(best_csv, index=False)

    # Compact summary: dataset x fraction -> best model (by best_val_acc)
    summary = (
        best.sort_values("best_val_acc", ascending=False)
        .groupby(["dataset", "fraction"], as_index=False)
        .first()
        .sort_values(["dataset", "fraction"])
    )
    summary_keep = [
        "dataset",
        "fraction",
        "arch",
        "best_val_acc",
        "test_acc",
        "test_auc",
        "lr",
        "optimizer",
        "seed",
        "summary_path",
    ]
    for c in summary_keep:
        if c not in summary.columns:
            summary[c] = None
    summary_csv = out_dir / "best_model_by_dataset_fraction.csv"
    summary[summary_keep].to_csv(summary_csv, index=False)

    # Global curve (all datasets mixed; use with caution)
    global_best = (
        best.sort_values("best_val_acc", ascending=False)
        .groupby(["arch", "fraction"], as_index=False)
        .first()
        .sort_values(["arch", "fraction"])
    )
    plot_curve(
        global_best,
        y_col="best_val_acc",
        y_label="Best Validation Accuracy",
        out_png=out_dir / "val_acc_vs_fraction_global.png",
        out_pdf=out_dir / "val_acc_vs_fraction_global.pdf",
        title=f"{args.plot_title} (Global)",
    )
    plot_curve(
        global_best,
        y_col="test_acc",
        y_label="Test Accuracy",
        out_png=out_dir / "test_acc_vs_fraction_global.png",
        out_pdf=out_dir / "test_acc_vs_fraction_global.pdf",
        title=f"{args.plot_title} (Global)",
    )

    # Per-dataset curves
    per_ds_dir = out_dir / "per_dataset"
    per_ds_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(d for d in best["dataset"].dropna().unique().tolist())
    for ds in datasets:
        ds_best = best[best["dataset"] == ds].sort_values(["arch", "fraction"])
        ds_slug = _safe_name(ds)
        plot_curve(
            ds_best,
            y_col="best_val_acc",
            y_label="Best Validation Accuracy",
            out_png=per_ds_dir / f"{ds_slug}_val_acc_vs_fraction.png",
            out_pdf=per_ds_dir / f"{ds_slug}_val_acc_vs_fraction.pdf",
            title=f"{args.plot_title} ({ds})",
        )
        plot_curve(
            ds_best,
            y_col="test_acc",
            y_label="Test Accuracy",
            out_png=per_ds_dir / f"{ds_slug}_test_acc_vs_fraction.png",
            out_pdf=per_ds_dir / f"{ds_slug}_test_acc_vs_fraction.pdf",
            title=f"{args.plot_title} ({ds})",
        )

    print(f"Wrote: {all_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {out_dir / 'val_acc_vs_fraction_global.png'}")
    print(f"Wrote: {out_dir / 'test_acc_vs_fraction_global.png'}")
    print(f"Wrote per-dataset plots under: {per_ds_dir}")


if __name__ == "__main__":
    main()

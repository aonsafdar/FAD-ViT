#!/usr/bin/env python3
"""
Build a publication-ready markdown table from aggregate best_by_dataset_arch_fraction.csv.
Table: Dataset | Model | 1% | 10% | 100% (test acc, or val acc), with optional AUC column.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Build publication table from best CSV")
    p.add_argument("--best-csv", type=str, required=True, help="Path to best_by_dataset_arch_fraction.csv")
    p.add_argument("--out", type=str, default=None, help="Output .md path (default: stdout)")
    p.add_argument("--metric", type=str, default="test_acc", choices=["test_acc", "best_val_acc", "test_auc"],
                   help="Metric for table cells")
    args = p.parse_args()

    df = pd.read_csv(args.best_csv)
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")

    # Display names (FAD-ViT = Feature-Aware, Axis Decoupled ViT)
    arch_label = {
        "fadvit": "FAD-ViT",
        "vit-s16": "ViT-Small/16",
        "dinov2-base": "DINOv2-Base",
    }
    df["model"] = df["arch"].map(lambda x: arch_label.get(x, x))

    # Pivot: rows = (dataset, model), cols = fraction (0.01, 0.1, 1.0)
    fractions = [0.01, 0.1, 1.0]
    rows = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        row = {"Dataset": dataset, "Model": model}
        for frac in fractions:
            sub = g[g["fraction"] == frac]
            if len(sub) > 0:
                val = sub[args.metric].iloc[0]
                if args.metric == "test_auc" or "auc" in args.metric:
                    row[f"{frac*100:.0f}%"] = f"{val:.4f}"
                else:
                    row[f"{frac*100:.0f}%"] = f"{val:.2%}"
            else:
                row[f"{frac*100:.0f}%"] = "—"
        rows.append(row)

    out_df = pd.DataFrame(rows)
    cols = ["Dataset", "Model"] + [f"{f*100:.0f}%" for f in fractions]
    out_df = out_df[cols]

    metric_label = {"test_acc": "Test accuracy", "best_val_acc": "Val accuracy", "test_auc": "Test AUC"}
    header = f"# Linear probe results ({metric_label[args.metric]})\n\n"
    # Build markdown table manually (no tabulate dependency)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in out_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    body = header + "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"Wrote: {args.out}")
    else:
        print(body)


if __name__ == "__main__":
    main()

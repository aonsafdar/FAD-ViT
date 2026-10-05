#!/usr/bin/env python3
"""
Aggregate multi-seed MedMNIST linear-probe results into mean +/- sd test accuracy.
Scans outputs/classification_linear_probe_locked_lr/<dataset>/<arch>/fraction_<f>/lr_*/seed_*/summary.json
Prints a per-dataset table at fraction 1.0 (headline) plus a compact all-fractions view.
"""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("/path/to/FAD-ViT/outputs/classification_linear_probe_locked_lr")
ARCHES = ["cvt13-mod", "vit-s16", "dinov2-base"]
ARCH_LBL = {"cvt13-mod": "FAD-ViT", "vit-s16": "ViT-S", "dinov2-base": "DINOv2-B"}


def load():
    # (dataset, arch, fraction) -> {seed: test_acc}
    d = defaultdict(dict)
    for f in ROOT.glob("*/*/fraction_*/lr_*/seed_*/summary.json"):
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        seed = int(f.parent.name.split("_")[1])
        d[(s["dataset"], s["arch"], float(s["fraction"]))][seed] = s["test_acc"] * 100.0
    return d


def cell(d, ds, arch, frac):
    v = list(d.get((ds, arch, frac), {}).values())
    if not v:
        return "   --   "
    return f"{mean(v):.1f}±{pstdev(v) if len(v) > 1 else 0.0:.1f}(n={len(v)})"


def main():
    d = load()
    datasets = sorted({k[0] for k in d})
    for frac in (1.0, 0.1, 0.01):
        print(f"\n== Test accuracy (%), fraction {frac} — 3-seed mean±sd ==")
        print(f"{'Dataset':16} " + " ".join(f"{ARCH_LBL[a]:16}" for a in ARCHES))
        for ds in datasets:
            row = " ".join(f"{cell(d, ds, a, frac):16}" for a in ARCHES)
            print(f"{ds:16} {row}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Count parameters and (optionally) FLOPS for segmentation models.
Use for the "Model Complexity" table in docs/segmentation_results_table.md.

  python scripts/count_seg_params_flops.py

Parameter count: sum(p.numel() for p in model.parameters()).
FLOPS: one forward pass at 1×1×256×256 (fvcore; install with: pip install fvcore).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models import FADViTUNet, ViTBaseUNet, DINOv2UNet, UNet

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

INPUT_SHAPE = (1, 1, 256, 256)  # batch, channels, H, W


def main() -> None:
    def count_params(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    def count_flops(m: torch.nn.Module) -> int | None:
        if not HAS_FVCORE:
            return None
        m.eval()
        x = torch.zeros(*INPUT_SHAPE)
        try:
            return int(FlopCountAnalysis(m, x).total())
        except Exception:
            return None

    configs = [
        ("Ours-UNet (FADViT)", FADViTUNet(in_chans=1, num_classes=2, pretrained_path="")),
        ("ViT-UNet", ViTBaseUNet(in_chans=1, num_classes=2, pretrained=False)),
        ("DINOv2-UNet", DINOv2UNet(in_chans=1, num_classes=2, pretrained=False)),
        ("U-Net (baseline)", UNet(in_chans=1, num_classes=2)),
    ]

    print("Input shape:", INPUT_SHAPE)
    print("Parameters: sum(p.numel() for p in model.parameters())")
    print("FLOPS: one forward pass (fvcore FlopCountAnalysis)" if HAS_FVCORE else "FLOPS: install fvcore")
    print("-" * 56)

    for name, model in configs:
        n_params = count_params(model)
        n_flops = count_flops(model)
        line = f"{name}: {n_params:,} params ({n_params / 1e6:.2f} M)"
        if n_flops is not None:
            line += f"  |  {n_flops:,} FLOPs ({n_flops / 1e9:.2f} GFLOPs)"
        print(line)


if __name__ == "__main__":
    main()

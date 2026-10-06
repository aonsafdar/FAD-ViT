#!/usr/bin/env python3
"""
Count parameters and FLOPS for the three models used in the linear-probe comparison.
These must match the architectures used for training (fadvit, vit-s16, dinov2-base).

Parameter count: sum(p.numel() for p in model.parameters()) — same formula everywhere.

Why "Ours" (FADViT) has different param count in classification vs segmentation:
  - Classification: FAD-ViT backbone + linear classification head → ~21.67 M.
  - Segmentation:  FAD-ViT encoder (same backbone) + U-Net decoder (conv blocks, skip fusion) → ~23.1 M.
  So the FADViT encoder is the same; the extra ~1.4 M in segmentation is the decoder.

FLOPS: one forward pass at (1, 3, 224, 224) via fvcore FlopCountAnalysis when available.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root (parent of Experiments)
ROOT = Path(__file__).resolve().parents[2]
BU_MAMBA = ROOT / "BU-Mamba"
if str(BU_MAMBA) not in sys.path:
    sys.path.insert(0, str(BU_MAMBA))

import torch

# Optional: fvcore for FLOPs (install with: pip install fvcore)
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of learnable parameters (scalars)."""
    return sum(p.numel() for p in model.parameters())


def count_flops(model: torch.nn.Module, input_shape: tuple[int, ...] = (1, 3, 224, 224)) -> int | None:
    """FLOPs for one forward pass. Returns None if fvcore not available."""
    if not HAS_FVCORE:
        return None
    model.eval()
    inp = torch.zeros(input_shape)
    try:
        flops = FlopCountAnalysis(model, inp)
        total = flops.total()
        return int(total)
    except Exception:
        return None


def main() -> None:
    from train import init_model

    class Args:
        arch: str = ""
        fadvit_module: str = ""
        fadvit_ckpt: str = ""
        cvt13_ckpt_base: str = ""
        fadvit_ckpt: str = ""
        usfm_weights: str = ""
        usfm_preprocessing: bool = False
        freeze_backbone: bool = False

    args = Args()
    ckpt_mod = ROOT / "weights" / "model_best.pth"
    if not ckpt_mod.exists():
        ckpt_mod = Path("/path/to/FAD-ViT/weights/model_best.pth")

    num_classes = 9  # e.g. PathMNIST
    input_res = 224
    input_shape = (1, 3, input_res, input_res)

    # Same models as used for linear-probe training (do not change for param/flops consistency with experiments).
    configs = [
        ("fadvit", "FAD-ViT"),  # Feature-Aware, Axis Decoupled ViT
        ("vit-s16", "ViT-Small/16"),
        ("dinov2-base", "DINOv2-Base"),
    ]

    print("Input shape:", input_shape)
    print("Parameter count: sum(p.numel() for p in model.parameters())")
    print("FLOPS: one forward pass (fvcore FlopCountAnalysis)")
    print("-" * 60)

    for arch, label in configs:
        args.arch = arch
        args.fadvit_ckpt = str(ckpt_mod) if arch == "fadvit" else ""
        model = init_model(args, "cpu", num_classes=num_classes)
        n_params = count_parameters(model)
        n_flops = count_flops(model, input_shape)

        print(f"{label} ({arch})")
        print(f"  Parameters: {n_params:,}  ({n_params / 1e6:.2f} M)")
        if n_flops is not None:
            print(f"  FLOPS:      {n_flops:,}  ({n_flops / 1e9:.3f} GFLOPs)")
        else:
            print("  FLOPS:      (install fvcore: pip install fvcore)")
        print()

    if HAS_FVCORE:
        print("(FLOPS in fvcore: one multiply-add = 2 FLOPs)")


if __name__ == "__main__":
    main()

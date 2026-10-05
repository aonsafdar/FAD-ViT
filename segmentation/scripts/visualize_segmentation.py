#!/usr/bin/env python3
"""
Generate publication-standard qualitative segmentation figures per dataset and per model.

This script produces results for ONE task and ONE model at a time. Output layout:
  <output>/<task>/<model>/sample_XXXX.<png|pdf>  (e.g. outputs/qualitative_msd/Task07_Pancreas/cvt13_unet/sample_3701.png)

To get results for each dataset and model separately (like outputs/qualitative_msd), run this
script once per (task, model) with the corresponding checkpoint — e.g. in a loop or batch.

Produces:
  - Per-sample: Input | GT contour | Pred contour | Error map (4-panel), or
    overlay-only (GT green + Pred red contours on one image), or side-by-side overlays.
  - Optional grid figure: multiple samples in one image (e.g. 2x4 for paper).

Usage:
  # BUSI, one model
  python scripts/visualize_segmentation.py --task BUSI --data ./data/busi \\
    --checkpoint ./outputs/benchmark_busi/BUSI/cvt13_unet/best.pth \\
    --model cvt13_unet --output ./outputs/qualitative_busi --num-samples 24

  # MSD task, one model (e.g. qualitative_msd-style: one dir per task per model)
  python scripts/visualize_segmentation.py --task Task07_Pancreas --data ./data/msd_2d \\
    --checkpoint ./outputs/benchmark_msd/Task07_Pancreas/cvt13_unet/best.pth \\
    --model cvt13_unet --output ./outputs/qualitative_msd --num-samples 20

  # Loop to get all tasks × models (qualitative_msd layout)
  for TASK in Task07_Pancreas Task09_Spleen; do
    for MODEL in cvt13_unet vit_base_unet dinov2_unet; do
      python scripts/visualize_segmentation.py --task $TASK --data ./data/msd_2d \\
        --checkpoint ./outputs/benchmark_msd/$TASK/$MODEL/best.pth --model $MODEL \\
        --output ./outputs/qualitative_msd --num-samples 20
    done
  done
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Run from segmentation/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import get_dataset
from visualization import save_qualitative_figure, save_qualitative_grid


def get_model(args, in_chans=1):
    if args.model == "cvt13_unet":
        from models import CvT13UNet
        # Encoder is mod_cvt (CvT13Encoder in cvt13_encoder.py). Full state loaded from --checkpoint below.
        return CvT13UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained_path=args.cvt13_weights or "")
    elif args.model == "vit_base_unet":
        from models import ViTBaseUNet
        return ViTBaseUNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=False)
    elif args.model == "dinov2_unet":
        from models import DINOv2UNet
        return DINOv2UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=False)
    elif args.model == "unet":
        from models import UNet
        return UNet(in_chans=in_chans, num_classes=args.num_classes)
    raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Publication-quality segmentation visualization")
    parser.add_argument("--model", choices=["cvt13_unet", "vit_base_unet", "dinov2_unet", "unet"], default="cvt13_unet")
    parser.add_argument("--task", type=str, default="BUSI", help="BUSI or MSD task name")
    parser.add_argument("--data", type=str, default="./data/busi", help="Path to prepared data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth (or checkpoint)")
    parser.add_argument("--num-classes", type=int, default=2, dest="num_classes")
    parser.add_argument("--cvt13-weights", type=str, default="")
    parser.add_argument("--output", type=str, default="./outputs/qualitative", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of test samples to visualize")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--layout",
        type=str,
        default="four_panel",
        choices=["four_panel", "overlay_only", "side_by_side"],
        help="four_panel=Input|GT|Pred|Error map; overlay_only=GT+Pred contours on one image",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for publication (300 for print)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="Output format")
    parser.add_argument("--save-grid", action="store_true", help="Also save a grid figure of multiple samples")
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = get_dataset(args.data, args.task, split=args.split, size=256)
    in_chans = getattr(test_ds, "in_chans", 1)
    model = get_model(args, in_chans=in_chans)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    out_dir = Path(args.output) / args.task / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_format = args.format
    ext = ".pdf" if fig_format == "pdf" else ".png"

    n = min(args.num_samples, len(test_ds))
    indices = np.arange(len(test_ds))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    indices = indices[:n]

    imgs, targets, preds = [], [], []
    with torch.no_grad():
        for idx in indices:
            img, mask = test_ds[idx]
            img_b = img.unsqueeze(0).to(device)
            logits = model(img_b)
            pred = logits.argmax(1).squeeze().cpu().numpy()
            img_np = img.numpy()
            mask_np = mask.numpy()
            imgs.append(img_np)
            targets.append(mask_np)
            preds.append(pred)
            path = out_dir / f"sample_{idx:04d}{ext}"
            save_qualitative_figure(
                img_np,
                mask_np,
                pred,
                path,
                dpi=args.dpi,
                fig_format=fig_format,
                layout=args.layout,
                num_classes=args.num_classes,
            )

    if args.save_grid and len(imgs) >= args.grid_rows * args.grid_cols:
        grid_path = out_dir / f"grid_{args.grid_rows}x{args.grid_cols}{ext}"
        save_qualitative_grid(
            imgs,
            targets,
            preds,
            grid_path,
            nrows=args.grid_rows,
            ncols=args.grid_cols,
            dpi=args.dpi,
            fig_format=fig_format,
        )
        print(f"Grid saved: {grid_path}")

    print(f"Saved {n} figures to {out_dir} (layout={args.layout}, dpi={args.dpi}, format={fig_format})")


if __name__ == "__main__":
    main()

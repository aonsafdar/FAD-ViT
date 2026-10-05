#!/usr/bin/env python3
"""
Evaluate segmentation model and save qualitative overlays.

For publication-standard figures (contour overlays, error map, high DPI), use:
  python scripts/visualize_segmentation.py --task BUSI --data ./data/busi --checkpoint <path> ...
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import get_dataset
from metrics import compute_metrics


def get_model(args, in_chans=1):
    ckpt = args.cvt13_weights or str(Path(__file__).resolve().parents[2] / "weights" / "model_best.pth")
    if args.model == "cvt13_unet":
        from models import CvT13UNet
        return CvT13UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained_path=ckpt)
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


def save_overlay(img, pred, target, path, num_classes=3):
    """Save overlay using publication-style visualization when available."""
    try:
        from visualization import save_qualitative_figure
        save_qualitative_figure(
            img, target, pred, path,
            dpi=150, fig_format="png", layout="side_by_side", num_classes=num_classes
        )
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = img.squeeze()
        if img.ndim == 3:
            img = img[0]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Input")
        axes[0].axis("off")
        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(pred, alpha=0.5, cmap="tab10", vmin=0, vmax=num_classes)
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        axes[2].imshow(img, cmap="gray")
        axes[2].imshow(target, alpha=0.5, cmap="tab10", vmin=0, vmax=num_classes)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cvt13_unet", "vit_base_unet", "dinov2_unet", "unet"], default="cvt13_unet")
    parser.add_argument("--task", type=str, default="Task01_BrainTumour", help="e.g. Task01_BrainTumour or BUSI")
    parser.add_argument("--data", type=str, default="./data/msd_2d", help="e.g. ./data/busi for BUSI")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=4, help="2 for BUSI")
    parser.add_argument("--cvt13-weights", type=str, default="")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--output", type=str, default="./outputs/seg_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = get_dataset(args.data, args.task, split="test", size=256)
    in_chans = test_ds.in_chans
    model = get_model(args, in_chans=in_chans)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    out_dir = Path(args.output) / args.task / args.model
    if args.save_overlays:
        out_dir.mkdir(parents=True, exist_ok=True)

    preds, targets, imgs = [], [], []
    with torch.no_grad():
        for i in range(min(len(test_ds), 100)):  # limit for speed
            img, mask = test_ds[i]
            img_b = img.unsqueeze(0).to(device)
            logits = model(img_b)
            pred = logits.argmax(1).squeeze().cpu().numpy()
            preds.append(pred)
            targets.append(mask.numpy())
            if args.save_overlays and i < 20:
                save_overlay(img.numpy(), pred, mask.numpy(), out_dir / f"overlay_{i:03d}.png", args.num_classes)

    preds = np.array(preds)
    targets = np.array(targets)
    metrics = compute_metrics(preds, targets, args.num_classes)
    hd95_str = f"{metrics['hd95']:.2f}" if np.isfinite(metrics["hd95"]) else "nan"
    print(f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, HD95: {hd95_str}")
    if args.save_overlays:
        print(f"Overlays saved to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train segmentation models on MSD 2D slices.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_dataset
from metrics import compute_metrics

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_model(args, in_chans=1):
    if args.model == "cvt13_unet":
        from models import CvT13UNet
        if getattr(args, "scratch", False):
            ckpt = ""  # train from scratch: no ImageNet pretraining
        else:
            ckpt = args.cvt13_weights or str(Path(__file__).resolve().parents[2] / "weights" / "model_best.pth")
        _tail = getattr(args, "sa_tail_depth_last", None)
        return CvT13UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained_path=ckpt,
                         backbone_token_mode=getattr(args, "token_mode", "channel"),
                         sa_tail_depth_last=_tail)
    elif args.model == "vit_base_unet":
        from models import ViTBaseUNet
        return ViTBaseUNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=not getattr(args, "scratch", False))
    elif args.model == "dinov2_unet":
        from models import DINOv2UNet
        return DINOv2UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=True)
    elif args.model == "davit_unet":
        from models import DaViTUNet
        return DaViTUNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=not getattr(args, "scratch", False))
    elif args.model == "xcit_unet":
        from models import XCiTUNet
        return XCiTUNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=not getattr(args, "scratch", False))
    elif args.model == "vit_small_unet":
        from models import ViTSmallUNet
        return ViTSmallUNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=not getattr(args, "scratch", False))
    elif args.model == "resnet50_unet":
        from models import ResNet50UNet
        return ResNet50UNet(in_chans=in_chans, num_classes=args.num_classes, pretrained=not getattr(args, "scratch", False))
    elif args.model == "unet":
        from models import UNet
        return UNet(in_chans=in_chans, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cvt13_unet", "vit_base_unet", "dinov2_unet", "davit_unet", "xcit_unet", "vit_small_unet", "resnet50_unet", "unet"], default="cvt13_unet")
    parser.add_argument("--task", type=str, default="Task01_BrainTumour")
    parser.add_argument("--data", type=str, default="./data/msd_2d")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=4, help="Task01=4, Task07=3")
    parser.add_argument("--cvt13-weights", type=str, default="")
    parser.add_argument("--scratch", action="store_true", help="Train FAD-ViT (cvt13_unet) from scratch, no ImageNet pretraining")
    parser.add_argument("--token-mode", type=str, default="channel", choices=["channel", "spatial", "spatial_matched"], help="Backbone global-attention axis: channel (feature-axis, default), spatial (standard spatial-token ablation), or spatial_matched (single-factor control: identical channel-mode Q/K/V, only the attention pooling axis flipped to spatial)")
    parser.add_argument("--sa-tail-depth-last", type=int, default=None, help="Override final-stage spatial-attention tail depth (0 disables the spatial head; default spec=5)")
    parser.add_argument("--output", type=str, default="./outputs/seg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="USFM_BUSI", help="W&B project name")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping: stop after N epochs without improvement (0=disabled)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    train_ds = get_dataset(args.data, args.task, split="train", val_ratio=0.15, test_ratio=0.15, seed=args.seed, size=256)
    val_ds = get_dataset(args.data, args.task, split="val", val_ratio=0.15, test_ratio=0.15, seed=args.seed, size=256)
    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset is empty. For BUSI run: python scripts/prepare_busi_seg.py --root /path/to/Dataset_BUSI_with_GT --output {args.data}. "
            f"For MSD run process_msd.py and set --data to msd_2d and --task to the task name."
        )
    in_chans = train_ds.in_chans
    num_classes = getattr(train_ds, "num_classes", args.num_classes)
    args.num_classes = num_classes
    print(f"Input channels: {in_chans}, num_classes: {num_classes}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(args, in_chans=in_chans).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.output) / args.task / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    epochs_no_improve = 0

    if args.wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args), name=f"seg-{args.task}-{args.model}")
    elif args.wandb and not HAS_WANDB:
        print("Warning: --wandb requested but wandb not installed. Run: pip install wandb")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = F.cross_entropy(logits, mask, ignore_index=255)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for img, mask in val_loader:
                img = img.to(device)
                logits = model(img)
                pred = logits.argmax(1).cpu().numpy()
                preds.append(pred)
                targets.append(mask.numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        metrics = compute_metrics(preds, targets, args.num_classes)

        hd95_val = metrics["hd95"]
        hd95_str = f"{hd95_val:.2f}" if np.isfinite(hd95_val) else "nan"
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} hd95={hd95_str}")

        if args.wandb and HAS_WANDB:
            log_d = {
                "train_loss": train_loss,
                "val_dice": metrics["dice"],
                "val_iou": metrics["iou"],
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }
            if np.isfinite(metrics["hd95"]):
                log_d["val_hd95"] = metrics["hd95"]
            wandb.log(log_d)

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "dice": best_dice}, out_dir / "best.pth")
        else:
            epochs_no_improve += 1

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {args.early_stop} epochs)")
            break

    print(f"Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()

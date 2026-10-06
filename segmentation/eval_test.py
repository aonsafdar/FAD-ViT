#!/usr/bin/env python3
"""
Evaluate a trained segmentation checkpoint on the HELD-OUT TEST split
(the 15% test partition never seen during training/validation/epoch-selection).
Reuses train.get_model so token-mode / all backbones are supported.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from data import get_dataset
from metrics import compute_metrics
from train import get_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--token-mode", type=str, default="channel")
    p.add_argument("--sa-tail-depth-last", type=int, default=None,
                   help="Must match the value used at training time so the architecture matches the checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the split used in training (same seed/ratios) so "test" is the held-out 15%.
    test_ds = get_dataset(args.data, args.task, split="test", val_ratio=0.15,
                          test_ratio=0.15, seed=args.seed, size=256)
    in_chans = test_ds.in_chans
    nc = getattr(test_ds, "num_classes", args.num_classes)
    args.num_classes = nc
    # We overwrite all weights with the finetuned checkpoint below, so skip the
    # backbone's init-time pretrained load (scratch=True) to avoid needing the
    # ImageNet weight files at eval time. Architecture is unaffected.
    args.scratch = True
    args.fadvit_weights = ""

    model = get_model(args, in_chans=in_chans).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for img, mask in loader:
            logits = model(img.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(mask.numpy())
    preds = np.concatenate(preds, 0)
    targets = np.concatenate(targets, 0)
    m = compute_metrics(preds, targets, nc)
    hd = f"{m['hd95']:.2f}" if np.isfinite(m["hd95"]) else "nan"
    print(f"TEST {args.task} {args.model}: n={len(preds)} Dice={m['dice']*100:.2f} IoU={m['iou']*100:.2f} HD95={hd}")


if __name__ == "__main__":
    main()

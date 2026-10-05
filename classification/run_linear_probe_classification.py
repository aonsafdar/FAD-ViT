#!/usr/bin/env python3
"""
Linear probing for BU-Mamba classification backbones with data-efficiency subsets.

Protocol:
  - Freeze pretrained backbone.
  - Train only final linear classifier head.
  - Evaluate on fixed val split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[2]
BU_ROOT = ROOT / "BU-Mamba"
if str(BU_ROOT) not in sys.path:
    sys.path.insert(0, str(BU_ROOT))

from data import get_transforms  # BU-Mamba/data.py
from train import init_model  # BU-Mamba/train.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probing for classification backbones")
    p.add_argument("--arch", choices=["cvt13-mod", "vit-s16", "dinov2-base"], required=True)
    p.add_argument("--data-path", type=str, required=True, help="ImageFolder dataset root")
    p.add_argument("--fraction", type=float, default=1.0, help="Train fraction in (0,1], e.g. 0.01, 0.1, 1.0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--cvt13-weights", type=str, default="")
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def make_minimal_args(args: argparse.Namespace):
    class A:
        pass

    a = A()
    a.arch = args.arch
    a.cvt13_module = str(BU_ROOT / "models" / "mod_cvt.py") if args.arch == "cvt13-mod" else str(BU_ROOT / "models" / "cls_cvt.py")
    a.cvt13_ckpt = args.cvt13_weights
    a.cvt13_ckpt_base = ""
    a.cvt13_ckpt_mod = ""
    a.usfm_weights = ""
    a.usfm_preprocessing = False
    a.freeze_backbone = False
    return a


def freeze_all_but_head(model: nn.Module) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False

    # Common final-head names across BU-Mamba architectures.
    candidate_prefixes = ["head.", "fc.", "classifier."]
    trainable = []
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in candidate_prefixes):
            p.requires_grad = True
            trainable.append(name)

    # Fallback if model uses an uncommon head name.
    if not trainable:
        named = list(model.named_parameters())
        for name, p in named[-2:]:
            p.requires_grad = True
            trainable.append(name)
    return trainable


def make_splits(labels: np.ndarray, val_size: float, test_size: float, seed: int):
    n = len(labels)
    indices = np.arange(n)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = labels[temp_idx]
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=temp_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def stratified_fraction(train_idx: np.ndarray, labels: np.ndarray, fraction: float, seed: int):
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    n = len(train_idx)
    if fraction >= 1.0:
        return train_idx
    k = max(1, int(round(n * fraction)))
    train_labels = labels[train_idx]
    # Ensure at least one sample per class where possible.
    n_classes = len(np.unique(train_labels))
    k = max(k, n_classes)
    try:
        sub_idx, _ = train_test_split(
            train_idx,
            train_size=k,
            stratify=train_labels,
            random_state=seed,
        )
    except Exception:
        rng = np.random.default_rng(seed)
        sub_idx = rng.permutation(train_idx)[:k]
    return np.array(sorted(sub_idx))


def compute_acc_auc(y_true: np.ndarray, prob: np.ndarray):
    pred = prob.argmax(axis=1)
    acc = float((pred == y_true).mean())
    auc = float("nan")
    try:
        if prob.shape[1] == 2:
            auc = float(roc_auc_score(y_true, prob[:, 1]))
        elif prob.shape[1] > 2:
            auc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return acc, auc


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train_tf = ImageFolder(args.data_path, transform=get_transforms(train=True))
    ds_eval_tf = ImageFolder(args.data_path, transform=get_transforms(train=False))
    labels = np.array(ds_train_tf.targets)
    num_classes = len(ds_train_tf.classes)

    train_idx, val_idx, test_idx = make_splits(labels, args.val_size, args.test_size, args.seed)
    train_idx_sub = stratified_fraction(train_idx, labels, args.fraction, args.seed)

    train_ds = Subset(ds_train_tf, train_idx_sub.tolist())
    val_ds = Subset(ds_eval_tf, val_idx.tolist())
    test_ds = Subset(ds_eval_tf, test_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    model = init_model(make_minimal_args(args), device, num_classes=num_classes)
    trainable_names = freeze_all_but_head(model)
    trainable = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    warmup_epochs = min(args.warmup_epochs, max(1, args.epochs // 10))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    history = []

    print(
        f"[LinearProbe-CLS] arch={args.arch} fraction={args.fraction} "
        f"train={len(train_idx_sub)}/{len(train_idx)} num_classes={num_classes}"
    )
    print(f"Trainable head params: {trainable_names}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(1, len(train_loader))

        model.eval()
        probs, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
                targets.append(y.numpy())
        prob = np.concatenate(probs, axis=0)
        y_true = np.concatenate(targets, axis=0)
        val_acc, val_auc = compute_acc_auc(y_true, prob)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} "
            f"loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"val_auc={'nan' if not np.isfinite(val_auc) else f'{val_auc:.4f}'}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics with best-val model.
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            targets.append(y.numpy())
    test_prob = np.concatenate(probs, axis=0)
    test_y = np.concatenate(targets, axis=0)
    test_acc, test_auc = compute_acc_auc(test_y, test_prob)

    ckpt_path = out_dir / "best_linear_probe_cls.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "arch": args.arch,
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "num_classes": num_classes,
            "classes": ds_train_tf.classes,
            "args": vars(args),
        },
        ckpt_path,
    )
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": args.arch,
                "task": "classification",
                "dataset": Path(args.data_path).name,
                "fraction": args.fraction,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "epochs_trained": len(history),
                "best_epoch": best_epoch,
                "best_val_acc": best_acc,
                "test_acc": test_acc,
                "test_auc": test_auc,
                "train_samples_used": int(len(train_idx_sub)),
                "train_samples_total": int(len(train_idx)),
                "num_classes": num_classes,
                "checkpoint": str(ckpt_path),
            },
            f,
            indent=2,
        )
    print(
        f"[Done] best_val_acc={best_acc:.4f} test_acc={test_acc:.4f} "
        f"test_auc={'nan' if not np.isfinite(test_auc) else f'{test_auc:.4f}'} -> {out_dir}"
    )


if __name__ == "__main__":
    main()

"""
PyTorch Datasets for segmentation: MSD 2D slices and BUSI (BUS).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Fallback when config.json missing (e.g. Task01=4, Task07=3)
TASK_NUM_CLASSES = {
    "BUSI": 2,
    "Task01_BrainTumour": 4,
    "Task02_Heart": 2,
    "Task03_Liver": 3,
    "Task04_Hippocampus": 3,
    "Task05_Prostate": 3,
    "Task06_Lung": 3,
    "Task07_Pancreas": 3,
    "Task08_HepaticVessel": 3,
    "Task09_Spleen": 2,
    "Task10_Colon": 2,
}
TASK_IN_CHANS = {
    "BUSI": 1,
    "Task01_BrainTumour": 4,
    "Task02_Heart": 1,
    "Task03_Liver": 1,
    "Task04_Hippocampus": 1,
    "Task05_Prostate": 2,  # T2 + ADC MRI
    "Task06_Lung": 1,
    "Task07_Pancreas": 1,
    "Task08_HepaticVessel": 1,
    "Task09_Spleen": 1,
    "Task10_Colon": 1,
}


class MSD2DDataset(Dataset):
    """Dataset of 2D slices from processed MSD."""

    def __init__(self, root, task, split="train", transform=None, val_ratio=0.15, test_ratio=0.15, seed=0, size=256):
        self.root = Path(root).resolve() / task
        self.task = task
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.root}\n"
                f"Run process_msd.py first: python scripts/process_msd.py --root ./data/msd --output ./data/msd_2d --tasks {task}"
            )
        config_path = self.root / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.num_classes = self.config["num_classes"]
        else:
            self.config = {}
            self.num_classes = TASK_NUM_CLASSES.get(task, 3)

        # Collect all slices (structure: task/volume_name/slice_XXXX.npz)
        self.samples = []
        for vol_dir in sorted(self.root.iterdir()):
            if not vol_dir.is_dir() or vol_dir.name.endswith(".json"):
                continue
            for npz in sorted(vol_dir.glob("*.npz")):
                self.samples.append(npz)

        # Train/val/test split by volume (not by slice)
        volumes = sorted(set(s.parent.name for s in self.samples))
        np.random.seed(seed)
        np.random.shuffle(volumes)
        n = len(volumes)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        test_vols = set(volumes[:n_test])
        val_vols = set(volumes[n_test : n_test + n_val])
        train_vols = set(volumes[n_test + n_val :])

        if split == "train":
            self.samples = [s for s in self.samples if s.parent.name in train_vols]
        elif split == "val":
            self.samples = [s for s in self.samples if s.parent.name in val_vols]
        else:
            self.samples = [s for s in self.samples if s.parent.name in test_vols]

        self.size = size
        self.in_chans = TASK_IN_CHANS.get(task, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        img = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.int64)
        if img.ndim == 2:
            img = np.expand_dims(img, 0)  # (1, H, W)
        elif img.ndim == 3:
            # (H, W, C) -> (C, H, W) for PyTorch; keep all channels (e.g. Task01 has 4 MRI modalities)
            img = np.transpose(img, (2, 0, 1))
        # Resize to fixed size for batching
        from scipy.ndimage import zoom
        h, w = img.shape[1], img.shape[2]
        if (h, w) != (self.size, self.size):
            zoom_factors = (1, self.size / h, self.size / w)
            img = zoom(img, zoom_factors, order=1)
            mask = zoom(mask, (self.size / h, self.size / w), order=0)
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


class BUSIDataset(Dataset):
    """Dataset for BUSI segmentation (prepared from BUS: original + GT). Expects root/train|val|test/images and masks."""

    def __init__(self, root, split="train", transform=None, size=256):
        self.root = Path(root).resolve()
        self.split = split
        self.transform = transform
        self.size = size
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.root}\n"
                f"Run: python scripts/prepare_busi_seg.py --root /path/to/dataset/BUS --output {self.root}"
            )
        config_path = self.root / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            self.num_classes = self.config["num_classes"]
            self.in_chans = self.config.get("in_chans", 1)
        else:
            self.config = {}
            self.num_classes = 2
            self.in_chans = 1

        im_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not im_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"Expected {im_dir} and {mask_dir}. Run prepare_busi_seg.py first.")
        self.samples = []
        for im_path in sorted(im_dir.iterdir()):
            if im_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            name = im_path.stem
            mask_path = mask_dir / f"{name}{im_path.suffix}"
            if not mask_path.exists():
                mask_path = mask_dir / f"{name}.png"
            if mask_path.exists():
                self.samples.append((im_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        im_path, mask_path = self.samples[idx]
        img = np.array(__import__("PIL.Image").Image.open(im_path).convert("L"), dtype=np.float32)  # (H, W)
        mask = np.array(__import__("PIL.Image").Image.open(mask_path).convert("L"), dtype=np.int64)
        # Binarize mask: any non-zero -> 1 (lesion)
        mask = (mask > 0).astype(np.int64)
        img = np.expand_dims(img, 0)  # (1, H, W)
        from scipy.ndimage import zoom
        h, w = img.shape[1], img.shape[2]
        if (h, w) != (self.size, self.size):
            zoom_factors = (1, self.size / h, self.size / w)
            img = zoom(img, zoom_factors, order=1)
            mask = zoom(mask, (self.size / h, self.size / w), order=0)
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return torch.from_numpy(img).float(), torch.from_numpy(mask).long()


def get_dataset(root, task, split="train", **kwargs):
    """Return the appropriate dataset. task=BUSI uses BUSIDataset; else MSD2DDataset."""
    if task == "BUSI":
        busi_kw = {k: v for k, v in kwargs.items() if k in ("transform", "size")}
        return BUSIDataset(root, split=split, **busi_kw)
    return MSD2DDataset(root, task, split=split, **kwargs)

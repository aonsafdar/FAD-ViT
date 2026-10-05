"""
Segmentation metrics: Dice, HD95, mean IoU.
HD95: 95th percentile of symmetric surface-to-surface distances (standard definition).
Uses medpy when available; otherwise a scipy-based implementation.
"""
import numpy as np
from scipy import ndimage

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False


def _hd95_binary_scipy(p: np.ndarray, r: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """
    95th percentile of symmetric surface-to-surface distances (binary masks).
    p, r: binary 2D arrays (0/1). spacing: (sy, sx) or (1,1) for voxel units.
    """
    if p.sum() == 0 and r.sum() == 0:
        return 0.0
    if p.sum() == 0 or r.sum() == 0:
        return np.nan
    # Surface = boundary: mask & ~erosion(mask)
    footprint = np.ones((3, 3), dtype=bool)
    surf_p = p.astype(bool) & ~ndimage.binary_erosion(p, structure=footprint)
    surf_r = r.astype(bool) & ~ndimage.binary_erosion(r, structure=footprint)
    if surf_p.sum() == 0 and surf_r.sum() == 0:
        return 0.0
    # Distance from every voxel to the other set (physical units if spacing given)
    try:
        dist_to_r = ndimage.distance_transform_edt(1 - r, sampling=spacing)
        dist_to_p = ndimage.distance_transform_edt(1 - p, sampling=spacing)
    except TypeError:
        dist_to_r = ndimage.distance_transform_edt(1 - r) * np.mean(spacing)
        dist_to_p = ndimage.distance_transform_edt(1 - p) * np.mean(spacing)
    # Symmetric distances: from surf_p to r, from surf_r to p
    d1 = dist_to_r[surf_p]
    d2 = dist_to_p[surf_r]
    all_d = np.concatenate([d1.ravel(), d2.ravel()])
    if all_d.size == 0:
        return np.nan
    return float(np.percentile(all_d, 95))


def dice_coef(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth=1e-6) -> np.ndarray:
    """Per-class Dice. pred/target: (H,W) int."""
    dices = []
    for c in range(1, num_classes):  # skip background 0
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        d = (2 * inter + smooth) / (union + smooth)
        dices.append(d)
    return np.array(dices) if dices else np.array([0.0])


def dice_batch(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Mean Dice over batch. pred/target: (B,H,W)."""
    dices = []
    for i in range(pred.shape[0]):
        d = dice_coef(pred[i], target[i], num_classes)
        dices.append(d.mean())
    return np.mean(dices)


def hd95_single(pred: np.ndarray, target: np.ndarray, num_classes: int, spacing=(1.0, 1.0)) -> np.ndarray:
    """Per-class HD95 for a single sample. Returns array of length num_classes-1."""
    hds = []
    for c in range(1, num_classes):
        p = (pred == c).astype(np.uint8)
        t = (target == c).astype(np.uint8)
        if p.sum() == 0 and t.sum() == 0:
            hds.append(0.0)
        elif p.sum() == 0 or t.sum() == 0:
            hds.append(np.nan)
        else:
            if HAS_MEDPY:
                try:
                    h = medpy_hd95(p, t, spacing)
                    hds.append(float(h) if np.isfinite(h) else np.nan)
                except Exception:
                    hds.append(_hd95_binary_scipy(p, t, spacing))
            else:
                hds.append(_hd95_binary_scipy(p, t, spacing))
    return np.array(hds)


def iou_coef(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth=1e-6) -> np.ndarray:
    """Per-class IoU (Jaccard). pred/target: (H,W) int."""
    ious = []
    for c in range(1, num_classes):  # skip background 0
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou = (inter + smooth) / (union + smooth)
        ious.append(iou)
    return np.array(ious) if ious else np.array([0.0])


def iou_batch(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    """Mean IoU over batch (mean over classes then over samples). pred/target: (B,H,W)."""
    ious = []
    for i in range(pred.shape[0]):
        iou = iou_coef(pred[i], target[i], num_classes)
        ious.append(iou.mean())
    return np.mean(ious)


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Compute Dice, HD95, and mean IoU. pred/target: (B,H,W) or (H,W)."""
    if pred.ndim == 2:
        pred, target = pred[np.newaxis], target[np.newaxis]
    dice = dice_batch(pred, target, num_classes)
    iou = iou_batch(pred, target, num_classes)
    # HD95: aggregate over all samples (nan for empty class pairs)
    all_hds = []
    for i in range(pred.shape[0]):
        hd = hd95_single(pred[i], target[i], num_classes)
        all_hds.append(hd)
    all_hds = np.array(all_hds)  # (N, num_classes-1)
    hd95_mean = np.nanmean(all_hds) if np.any(~np.isnan(all_hds)) else np.nan
    return {
        "dice": float(dice),
        "hd95": float(hd95_mean) if np.isfinite(hd95_mean) else float("nan"),
        "iou": float(iou),
    }

# FAD-ViT (mod-cvt)

**Feature-Aware Axis-Decoupled Vision Transformer**

This folder contains the core model architecture for FAD-ViT.

## Contents

| File | Description |
|------|-------------|
| `mod_cvt.py` | Core FAD-ViT architecture (axis-decoupled, channel-primary ViT) |
| `cvt13_encoder.py` | Encoder wrapper that loads `mod_cvt` and (optional) pretrained weights |
| `cvt13_unet.py` | FAD-ViT-UNet segmentation model (encoder + U-Net decoder) |

> **Pretrained weights.** ImageNet-1K pretrained weights (~85 MB) are released
> separately (GitHub Releases / Hugging Face); pass the checkpoint path via
> `pretrained_path=` / `--cvt13-weights`. See the top-level README.

## Architecture Overview

FAD-ViT implements **strict axis decoupling**:
- **Spatial Feature Extraction (SFE)**: Depthwise convolutions preserve and refine the 2D spatial grid
- **Feature Self-Attention (FSA)**: Self-attention over feature tokens via dimension swapping (B×N×C → B×C×N)

This makes global feature interaction—not spatial attention—the dominant modeling mechanism.

## Usage

```python
# Classification backbone
from mod_cvt import ConvolutionalVisionTransformer, get_cvt13_config
config = get_cvt13_config()
model = ConvolutionalVisionTransformer(**config)

# Load pretrained weights
import torch
ckpt = torch.load("model_best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

# Segmentation (FAD-ViT-UNet)
from cvt13_unet import CvT13UNet
seg_model = CvT13UNet(in_chans=3, num_classes=2, pretrained_path="model_best.pth")
```

## Model Specifications

| Property | Value |
|----------|-------|
| Parameters | 21.67M |
| GFLOPs (224×224) | 4.5 |
| Stages | 3 |
| Channels | 64 → 192 → 384 |
| Pretraining | ImageNet-1K |

"""
ViT-Small (ImageNet-1k) encoder + U-Net decoder for 2D medical segmentation.
Matched-capacity spatial-attention baseline (~22M encoder), the same size class
as FAD-ViT (23.1M), used to control for capacity in the transfer-gap analysis.
Uses timm vit_small_patch16_224.augreg_in1k. Single-scale encoder at 1/16 resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from .input_projection import InputProjection


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ViTSmallUNet(nn.Module):
    """ViT-Small patch16 224 (ImageNet-1k) + decoder. ~22M encoder, matched to FAD-ViT."""

    def __init__(self, in_chans=1, num_classes=3, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm")
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.img_size = 224
        self.backbone = timm.create_model(
            "vit_small_patch16_224.augreg_in1k",
            pretrained=pretrained, num_classes=0, global_pool="", in_chans=3,
        )
        self.embed_dim = 384
        self.patch_size = 16
        self.feat_size = self.img_size // self.patch_size  # 14
        self.dec1 = DecoderBlock(384, 192)
        self.dec2 = DecoderBlock(192, 96)
        self.final = nn.Conv2d(96, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input_proj(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        feat = self.backbone(x)
        feat = feat[:, 1:, :]
        feat = feat.transpose(1, 2).reshape(B, self.embed_dim, self.feat_size, self.feat_size)
        out = self.dec1(feat)
        out = self.dec2(out)
        out = self.final(out)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

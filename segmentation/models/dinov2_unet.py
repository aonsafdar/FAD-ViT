"""
DINOv2 ViT-Base encoder + U-Net decoder for 2D medical segmentation.
Uses timm vit_base_patch14_dinov2. Single-scale encoder at 1/14 resolution.
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
    """U-Net style decoder block: upsample + conv."""

    def __init__(self, in_ch, out_ch, skip_ch=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DINOv2UNet(nn.Module):
    """
    DINOv2 ViT-Base patch14 + decoder.
    Uses InputProjection when in_chans != 3 to map to 3ch for pretrained backbone.
    """

    def __init__(self, in_chans=1, num_classes=3, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required: pip install timm")
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.img_size = 224
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=224,
            in_chans=3,
        )
        self.embed_dim = 768
        self.patch_size = 14
        self.feat_size = self.img_size // self.patch_size  # 16

        # Decoder: 768 -> 384 -> 192 -> num_classes
        self.dec1 = DecoderBlock(768, 384)
        self.dec2 = DecoderBlock(384, 192)
        self.final = nn.Conv2d(192, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input_proj(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        feat = self.backbone(x)  # (B, N, 768) - includes CLS token
        feat = feat[:, 1:, :]  # drop CLS token -> (B, N-1, 768)
        feat = feat.transpose(1, 2).reshape(B, self.embed_dim, self.feat_size, self.feat_size)
        out = self.dec1(feat)
        out = self.dec2(out)
        out = self.final(out)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

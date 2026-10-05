"""
DaViT-Tiny (ImageNet-1k) hierarchical encoder + U-Net decoder for 2D medical segmentation.
DaViT interleaves spatial-window and channel-group attention (a non-decoupled channel design),
used here as a baseline for the strict feature-axis decoupling of FAD-ViT.
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


class DaViTUNet(nn.Module):
    """DaViT-Tiny features_only [96,192,384,768] @ strides [4,8,16,32] + U-Net decoder."""

    def __init__(self, in_chans=1, num_classes=3, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.backbone = timm.create_model("davit_tiny", pretrained=pretrained,
                                           features_only=True, in_chans=3)
        c = self.backbone.feature_info.channels()  # [96,192,384,768]
        self.dec3 = DecoderBlock(c[3], c[2], skip_ch=c[2])
        self.dec2 = DecoderBlock(c[2], c[1], skip_ch=c[1])
        self.dec1 = DecoderBlock(c[1], c[0], skip_ch=c[0])
        self.final = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        xi = self.input_proj(x)
        f0, f1, f2, f3 = self.backbone(xi)
        out = self.dec3(f3, f2)
        out = self.dec2(out, f1)
        out = self.dec1(out, f0)
        out = self.final(out)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


class XCiTUNet(nn.Module):
    """XCiT-S/16 (cross-covariance attention) single-scale encoder + U-Net decoder.
    Columnar backbone (stride 16); used as a non-decoupled channel-attention baseline."""

    def __init__(self, in_chans=1, num_classes=3, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.img_size = 256
        self.backbone = timm.create_model("xcit_small_12_p16_224", pretrained=pretrained,
                                           num_classes=0, img_size=self.img_size, in_chans=3)
        self.embed_dim = self.backbone.num_features  # 384
        self.patch = 16
        self.feat = self.img_size // self.patch  # 16
        self.dec1 = DecoderBlock(self.embed_dim, 192)
        self.dec2 = DecoderBlock(192, 96)
        self.dec3 = DecoderBlock(96, 48)
        self.final = nn.Conv2d(48, num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        xi = self.input_proj(x)
        xi = F.interpolate(xi, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        tok = self.backbone.forward_features(xi)  # (B, N(+cls), C)
        n = tok.shape[1]
        if n == self.feat * self.feat + 1:
            tok = tok[:, 1:, :]
        feat = tok.transpose(1, 2).reshape(B, self.embed_dim, self.feat, self.feat)
        out = self.dec1(feat)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.final(out)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

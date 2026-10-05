"""
ResNet-50 (ImageNet-1k) hierarchical encoder + U-Net decoder for 2D medical segmentation.
A purely convolutional, pretrained baseline used to test whether the transfer benefit
of FAD-ViT is due to convolutional locality alone or to the feature-axis global operator.
Uses timm resnet50 features_only.
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


class ResNet50UNet(nn.Module):
    """ResNet-50 features_only [256,512,1024,2048] @ strides [4,8,16,32] + U-Net decoder."""

    def __init__(self, in_chans=1, num_classes=3, pretrained=True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required")
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.backbone = timm.create_model("resnet50", pretrained=pretrained,
                                           features_only=True, out_indices=(1, 2, 3, 4), in_chans=3)
        c = self.backbone.feature_info.channels()  # [256,512,1024,2048]
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

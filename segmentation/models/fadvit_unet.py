"""
FAD-ViT encoder + U-Net decoder for 2D medical segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fadvit_encoder import FADViTEncoder
from .input_projection import InputProjection


class DecoderBlock(nn.Module):
    """U-Net style decoder block: upsample + conv + skip connection."""

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


class FADViTUNet(nn.Module):
    """
    FAD-ViT encoder + U-Net decoder.
    Encoder dims: [64, 192, 384] at strides [4, 8, 16]
    Uses InputProjection when in_chans != 3 to map to 3ch for pretrained backbone.
    """

    def __init__(self, in_chans=1, num_classes=3, pretrained_path="", backbone_token_mode="channel",
                 sa_tail_depth_last=None):
        super().__init__()
        self.input_proj = InputProjection(in_chans=in_chans, target_chans=3)
        self.encoder = FADViTEncoder(in_chans=3, pretrained_path=pretrained_path,
                                    backbone_token_mode=backbone_token_mode,
                                    sa_tail_depth_last=sa_tail_depth_last)
        enc_dims = self.encoder.dims  # [64, 192, 384]

        # Decoder: 384 -> 192 -> 64 -> num_classes
        self.dec2 = DecoderBlock(384, 192, skip_ch=192)
        self.dec1 = DecoderBlock(192, 64, skip_ch=64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.input_proj(x)
        feats = self.encoder(x)  # [f0, f1, f2]
        out = feats[2]  # bottleneck
        out = self.dec2(out, feats[1])
        out = self.dec1(out, feats[0])
        out = self.final(out)
        # Upsample to input resolution (encoder has stride 4)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

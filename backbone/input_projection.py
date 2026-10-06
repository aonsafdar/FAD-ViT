"""
Learnable input projection for pretrained 3-channel backbones.
Maps in_chans -> 3 when in_chans != 3 (handles both < 3 and > 3).
"""
import torch
import torch.nn as nn


class InputProjection(nn.Module):
    """
    Conv1x1 projecting input channels to 3 for pretrained RGB backbones.
    - in_chans < 3: learnable expansion (e.g. 1->3 for CT)
    - in_chans > 3: learnable reduction (e.g. 4->3 for multi-modal MRI)
    - in_chans == 3: identity (no-op)
    """

    def __init__(self, in_chans: int, target_chans: int = 3):
        super().__init__()
        self.in_chans = in_chans
        self.target_chans = target_chans
        if in_chans != target_chans:
            self.proj = nn.Conv2d(in_chans, target_chans, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return x
        return self.proj(x)

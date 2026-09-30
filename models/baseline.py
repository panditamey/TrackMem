"""B0: TrackNetV5-like baseline = motion input (MDD) + V2 U-Net, one heatmap per input frame.

No R-STR and no memory. Outputs logits; apply sigmoid for heatmaps.
"""
import torch.nn as nn

from .motion import build_motion, motion_channels
from .unet import TrackNetUNet


class TrackNetB0(nn.Module):
    def __init__(self, seq_len=3, motion="mdd", width_mult=1.0):
        super().__init__()
        self.motion = build_motion(motion)
        self.net = TrackNetUNet(motion_channels(motion, seq_len), seq_len, width_mult)

    def forward(self, frames):
        """frames: (B, T, 3, H, W) float in [0, 1] -> logits (B, T, H, W)."""
        return self.net(self.motion(frames))


def build_model(cfg):
    m = cfg["model"]
    if m["name"] == "b0":
        return TrackNetB0(cfg["data"]["seq_len"], m["motion"], m["width_mult"])
    raise ValueError(f"unknown model {m['name']}")

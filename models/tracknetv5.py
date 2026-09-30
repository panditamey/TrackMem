"""TrackNetV5 (arXiv 2512.02789), ported from the official implementation
(github.com/thaonan/TrackNetV5-SDK: models_factory/models/tracknetv5.py, heads/r_strhead.py).

    frames -> MDD maps -> interleaved 13-ch input -> V2 U-Net -> draft logits (3 maps)
           -> [R-STR] motion fusion + transformer residual -> refined logits

Where the SDK and the paper differ, this follows the SDK:
  - R-STR runs full joint attention over all 3 x (H/16 x W/16) patch tokens (the paper
    describes factorized spatial/temporal attention).
  - Motion fusion multiplies the frame-1 and frame-2 drafts by the *brighten* maps of the
    (0,1) and (1,2) intervals; frame 0 passes through unchanged.
  - In training, the residual is added to the dropout-masked draft (stochastic context masking).
Outputs logits; the SDK applies the final sigmoid inside the head, we apply it in the loss.
rstr=False gives TrackNetV5 without R-STR (MDD + U-Net only).
"""
import torch
import torch.nn as nn

from .motion import MDD, mdd_channels
from .unet import TrackNetUNet


class RSTRHead(nn.Module):
    """Residual-driven Spatio-Temporal Refinement over the 3 draft maps."""

    def __init__(self, img_size=(288, 512), patch_size=16, embed_dim=256, num_layers=4,
                 num_heads=2, context_dropout=0.1):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        h, w = img_size[0] // patch_size, img_size[1] // patch_size
        self.embed_conv = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, h * w, embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, 3, embed_dim))
        self.context_dropout = nn.Dropout(context_dropout)
        layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim * 4,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.decoder = nn.Sequential(nn.Conv2d(embed_dim, patch_size ** 2, kernel_size=1),
                                     nn.PixelShuffle(patch_size))

    def forward(self, draft, maps):
        """draft: (B, 3, H, W) logits; maps: (B, 4, H, W) MDD maps -> (B, 3, H, W) logits."""
        fused = torch.stack([draft[:, 0], draft[:, 1] * maps[:, 0], draft[:, 2] * maps[:, 2]], dim=1)
        ctx = self.context_dropout(fused)  # identity in eval mode
        b, t, h, w = ctx.shape
        hf, wf = h // self.patch_size, w // self.patch_size

        tokens = self.embed_conv(ctx.reshape(b * t, 1, h, w))                 # (B*T, D, hf, wf)
        tokens = tokens.flatten(2).transpose(1, 2).reshape(b, t, hf * wf, self.embed_dim)
        tokens = tokens + self.spatial_pos_embed.unsqueeze(1) + self.time_embed.unsqueeze(2)
        tokens = self.transformer(tokens.reshape(b, t * hf * wf, self.embed_dim))

        feat = tokens.reshape(b * t, hf, wf, self.embed_dim).permute(0, 3, 1, 2)
        residual = self.decoder(feat).reshape(b, t, h, w)
        return ctx + residual


class TrackNetV5(nn.Module):
    def __init__(self, seq_len=3, img_size=(288, 512), rstr=True, width_mult=1.0,
                 conv_order="conv_relu_bn", upsample="bilinear", head_bias=False,
                 mdd_init=(0.2, 0.15), mdd_eps=1e-6):
        super().__init__()
        if rstr and seq_len != 3:
            raise ValueError("R-STR motion fusion is defined for 3-frame windows")
        self.motion = MDD(*mdd_init, eps=mdd_eps)
        self.net = TrackNetUNet(mdd_channels(seq_len), seq_len, width_mult, conv_order,
                                upsample, head_bias)
        self.rstr = RSTRHead(img_size) if rstr else None

    def forward(self, frames):
        """frames: (B, T, 3, H, W) float in [0, 1] -> logits (B, T, H, W)."""
        maps = self.motion.maps(frames)
        draft = self.net(self.motion.interleave(frames, maps))
        return self.rstr(draft, maps) if self.rstr is not None else draft

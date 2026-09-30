"""Motion Direction Decoupling (MDD), following TrackNetV5 (arXiv 2512.02789) and its
official implementation (github.com/thaonan/TrackNetV5-SDK, models_factory/basic/mdd.py).

Signed luminance differences are split into brighten/darken polarity fields and mapped by
a learnable sigmoid:

    A = 1 / (1 + exp(-k(a) * (|x| - m(b)))),
    k(a) = 5 / (0.45 * |tanh(a)| + eps),   m(b) = 0.6 * tanh(b)

For 3 frames the backbone input interleaves RGB and attention maps:
[I_0, A+_01, A-_01, I_1, A+_12, A-_12, I_2] -> 13 channels.
"""
import torch
import torch.nn as nn

GRAY = (0.299, 0.587, 0.114)


def to_gray(x):
    """x: (..., 3, H, W) in [0, 1] -> (..., H, W)."""
    w = x.new_tensor(GRAY).view(3, 1, 1)
    return (x * w).sum(dim=-3)


class MDD(nn.Module):
    def __init__(self, alpha=0.2, beta=0.15, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.eps = eps

    def attention(self, p):
        k = 5.0 / (0.45 * torch.tanh(self.alpha).abs() + self.eps)
        m = 0.6 * torch.tanh(self.beta)
        return torch.sigmoid(k * (p.abs() - m))

    def maps(self, frames):
        """frames: (B, T, 3, H, W) in [0, 1] -> (B, 2(T-1), H, W) ordered
        [brighten_01, darken_01, brighten_12, darken_12, ...]."""
        gray = to_gray(frames)
        delta = gray[:, 1:] - gray[:, :-1]
        pos = self.attention(torch.relu(delta))
        neg = self.attention(torch.relu(-delta))
        return torch.stack([pos, neg], dim=2).flatten(1, 2)

    def interleave(self, frames, maps):
        """Backbone input [I_0, A_01, I_1, A_12, I_2, ...] -> (B, 3T + 2(T-1), H, W)."""
        chunks = [frames[:, 0]]
        for t in range(frames.shape[1] - 1):
            chunks += [maps[:, 2 * t:2 * t + 2], frames[:, t + 1]]
        return torch.cat(chunks, dim=1)


def mdd_channels(seq_len):
    return 3 * seq_len + 2 * (seq_len - 1)

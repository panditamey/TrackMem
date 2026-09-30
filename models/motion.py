"""Motion inputs.

MDD follows TrackNetV5 (arXiv 2512.02789): signed frame differences split into
positive/negative polarity fields, mapped by a learnable sigmoid

    A = 1 / (1 + exp(-k(a) * (|x| - m(b)))),
    k(a) = 5 / (0.45 * |tanh(a)| + eps),   m(b) = 0.6 * tanh(b)

and interleaved with RGB: [I_0, A_01, I_1, A_12, I_2] -> 13 channels for 3 frames.
The paper does not state initial values for a, b; ours give k ~ 14.6, m ~ 0.1.
"""
import torch
import torch.nn as nn

GRAY = (0.299, 0.587, 0.114)


def to_gray(x):
    """x: (..., 3, H, W) in [0, 1] -> (..., H, W)."""
    w = x.new_tensor(GRAY).view(3, 1, 1)
    return (x * w).sum(dim=-3)


class MDD(nn.Module):
    def __init__(self, alpha=1.0, beta=0.17, eps=1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.eps = eps

    def attention(self, p):
        k = 5.0 / (0.45 * torch.tanh(self.alpha).abs() + self.eps)
        m = 0.6 * torch.tanh(self.beta)
        return torch.sigmoid(k * (p.abs() - m))

    def forward(self, frames):
        """frames: (B, T, 3, H, W) in [0, 1] -> (B, 3T + 2(T-1), H, W)."""
        gray = to_gray(frames)                      # (B, T, H, W)
        delta = gray[:, 1:] - gray[:, :-1]          # (B, T-1, H, W)
        pos = self.attention(torch.relu(delta))
        neg = self.attention(torch.relu(-delta))
        chunks = [frames[:, 0]]
        for t in range(delta.shape[1]):
            chunks += [pos[:, t:t + 1], neg[:, t:t + 1], frames[:, t + 1]]
        return torch.cat(chunks, dim=1)


class NoMotion(nn.Module):
    """Plain RGB stack (TrackNetV2-style input)."""

    def forward(self, frames):
        return frames.flatten(1, 2)


def build_motion(name):
    return {"mdd": MDD, "none": NoMotion}[name]()


def motion_channels(name, seq_len):
    return 3 * seq_len + (2 * (seq_len - 1) if name == "mdd" else 0)

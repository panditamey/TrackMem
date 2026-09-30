"""Heatmap losses."""
import torch


def wbce_loss(logits, target, eps=1e-6):
    """Weighted BCE from TrackNetV2/V5:
    L = -mean[(1-p)^2 * y * log p + p^2 * (1-y) * log(1-p)].
    Computed in fp32 for AMP stability.
    """
    p = torch.sigmoid(logits.float()).clamp(eps, 1 - eps)
    y = target.float()
    loss = (1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p)
    return -loss.mean()

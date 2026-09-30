"""TrackMem losses for one recurrent step (3-frame window)."""
import torch
import torch.nn.functional as F


def trackmem_step_loss(out, heatmaps, vis, xy, cfg, eps=1e-6):
    """out: TrackMem outputs; heatmaps (B, 3, H, W); vis (B, 3); xy (B, 3, 2) input space.

    heatmap:    TrackNet WBCE per frame; invisible frames weighted by invisible_heatmap_weight,
                since absence is handled by the visibility head (presence-conditioned).
    visibility: BCE on the visibility logits.
    offset:     L1 on sub-pixel offsets (GT - pixel) at target-disk pixels of visible frames.
    """
    p = torch.sigmoid(out["logits"].float()).clamp(eps, 1 - eps)
    y = heatmaps.float()
    wbce = -((1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p)).mean((2, 3))  # (B, 3)
    fw = torch.where(vis.bool(), torch.ones_like(wbce), torch.full_like(wbce, cfg["invisible_heatmap_weight"]))
    l_hm = (wbce * fw).sum() / fw.sum()

    l_vis = F.binary_cross_entropy_with_logits(out["vis_logit"].float(), vis.float())

    h, w = heatmaps.shape[-2:]
    xs = torch.arange(w, device=xy.device, dtype=torch.float32).view(1, 1, 1, w)
    ys = torch.arange(h, device=xy.device, dtype=torch.float32).view(1, 1, h, 1)
    tx = xy[..., 0, None, None] - xs
    ty = xy[..., 1, None, None] - ys
    mask = (y > 0) & vis.bool()[..., None, None]
    off = out["offset"].float()
    l1 = (off[:, :, 0] - tx).abs() + (off[:, :, 1] - ty).abs()
    l_off = (l1 * mask).sum() / mask.sum().clamp(min=1)

    total = l_hm + cfg["vis_weight"] * l_vis + cfg["offset_weight"] * l_off
    return total, {"hm": l_hm.detach(), "vis": l_vis.detach(), "off": l_off.detach()}

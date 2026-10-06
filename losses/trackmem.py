"""TrackMem losses for one recurrent step (3-frame window)."""
import torch
import torch.nn.functional as F


def trackmem_step_loss(out, heatmaps, vis, xy, cfg, eps=1e-6):
    """out: TrackMem outputs; heatmaps (B, 3, H, W); vis (B, 3); xy (B, 3, 2) input space.

    heatmap:    TrackNet WBCE per frame; invisible frames weighted by invisible_heatmap_weight,
                since absence is handled by the visibility head (presence-conditioned).
    evidence:   same WBCE on the prior-free evidence map (late fusion), weight evidence_weight.
    visibility: BCE on the visibility logits; invisible frames weighted by vis_invisible_weight.
    offset:     L1 on sub-pixel offsets (GT - pixel) at target-disk pixels of visible frames.
    """
    y = heatmaps.float()
    fw = torch.where(vis.bool(), 1.0, cfg["invisible_heatmap_weight"]).float()

    def heatmap_loss(logits):
        p = torch.sigmoid(logits.float()).clamp(eps, 1 - eps)
        wbce = -((1 - p) ** 2 * y * torch.log(p) + p ** 2 * (1 - y) * torch.log(1 - p)).mean((2, 3))  # (B, 3)
        return (wbce * fw).sum() / fw.sum()

    l_hm = heatmap_loss(out["logits"])

    vw = torch.where(vis.bool(), 1.0, cfg.get("vis_invisible_weight", 1.0)).float()
    l_vis = (F.binary_cross_entropy_with_logits(out["vis_logit"].float(), vis.float(), reduction="none") * vw
             ).sum() / vw.sum()

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
    parts = {"hm": l_hm.detach(), "vis": l_vis.detach(), "off": l_off.detach()}
    if "evidence" in out:
        l_ev = heatmap_loss(out["evidence"])
        total = total + cfg.get("evidence_weight", 1.0) * l_ev
        parts["ev"] = l_ev.detach()
    return total, parts

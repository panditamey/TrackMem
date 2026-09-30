"""TrackMem: TrackNetV5 + learned persistent kinematic memory fed back into detection.

Per step (window t-1, t, t+1, memory state at t-1):
    prior  = memory.render(state)                       3 Gaussian prior maps
    input  = [MDD interleaved frames (13 ch), prior (3 ch)]
    U-Net  -> features -> draft -> R-STR -> heatmap logits (3 maps)
    heads  -> visibility logits (3), sub-pixel offsets (3 x 2 maps)
    state  = memory.update(state, centre-frame outputs)

Trained recurrently on short sequences (see train.py); inference runs the same step loop
over whole rallies.
"""
import torch
import torch.nn as nn

from .memory import KinematicMemory
from .motion import MDD, mdd_channels
from .tracknetv5 import RSTRHead
from .unet import TrackNetUNet


class TrackMem(nn.Module):
    recurrent = True

    def __init__(self, seq_len=3, img_size=(288, 512), rstr=True, latent_dim=64, prior_sigma_min=1.5,
                 mdd_init=(0.2, 0.15)):
        super().__init__()
        if seq_len != 3:
            raise ValueError("TrackMem uses 3-frame windows")
        self.h, self.w = img_size
        self.motion = MDD(*mdd_init)
        self.net = TrackNetUNet(mdd_channels(3) + 3, 3)
        c = self.net.head.in_channels
        self.rstr = RSTRHead(img_size) if rstr else None
        self.offset_head = nn.Conv2d(c, 6, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        self.vis_head = nn.Sequential(nn.Linear(c + 2 + 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.memory = KinematicMemory(feat_dim=c, latent_dim=latent_dim, sigma_min=prior_sigma_min)

    def forward(self, frames, prior):
        """frames: (B, 3, 3, H, W) in [0, 1]; prior: (B, 3, H, W) -> dict of outputs."""
        b = frames.shape[0]
        maps = self.motion.maps(frames)
        x = torch.cat([self.motion.interleave(frames, maps), prior.to(frames.dtype)], dim=1)
        feats = self.net.features(x)
        draft = self.net.head(feats)
        logits = self.rstr(draft, maps) if self.rstr is not None else draft
        offset = self.offset_head(feats).view(b, 3, 2, self.h, self.w)

        # Visibility (presence): features pooled where each frame's heatmap points,
        # plus that frame's peak logit and prior peak.
        wts = torch.softmax(logits.float().flatten(2), dim=-1)                      # (B, 3, HW)
        pooled = torch.einsum("btn,bcn->btc", wts, feats.float().flatten(2))         # (B, 3, C)
        peak = logits.float().flatten(2).amax(-1, keepdim=True)
        prior_peak = prior.float().flatten(2).amax(-1, keepdim=True)
        frame_id = torch.eye(3, device=frames.device).expand(b, 3, 3)
        vis_logit = self.vis_head(torch.cat([pooled, peak / 10, prior_peak, frame_id], -1)).squeeze(-1)
        return {"logits": logits, "vis_logit": vis_logit, "offset": offset, "pooled": pooled}

    def step(self, frames, state, dt):
        """One recurrent step. dt: (B, 1) = 30 / fps. Returns (outputs, new_state, prior)."""
        prior = self.memory.render(state, dt, self.h, self.w)
        out = self(frames, prior)
        new_state, _ = self.memory.update(state, out, dt, self.w, self.h)
        return out, new_state, prior

    @staticmethod
    @torch.no_grad()
    def readout(out, state):
        """Centre-frame predictions: visibility prob, argmax + offset position, peak, latent position."""
        lc = out["logits"][:, 1].float()
        b, h, w = lc.shape
        idx = lc.reshape(b, -1).argmax(1)
        iy, ix = idx // w, idx % w
        off = out["offset"][:, 1].float()[torch.arange(b), :, iy, ix]                # (B, 2)
        return {"vis_prob": torch.sigmoid(out["vis_logit"][:, 1].float()),
                "x": ix.float() + off[:, 0], "y": iy.float() + off[:, 1],
                "peak": torch.sigmoid(lc.reshape(b, -1).max(1).values),
                "latent_x": state["p"][:, 0], "latent_y": state["p"][:, 1]}

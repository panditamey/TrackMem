"""TrackMem: TrackNetV5 + learned persistent kinematic memory fed back into detection.

Per step (window t-1, t, t+1, memory state at t-1):
    prior  = memory.render(state)                       3 Gaussian prior maps
    fusion='early': input = [MDD frames (13 ch), prior (3 ch)] -> U-Net -> features
    fusion='late':  MDD frames (13 ch) -> U-Net -> prior-free features -> evidence logits (3 maps);
                    [features, prior] -> fusion conv -> features
    features -> draft -> R-STR -> heatmap logits (3 maps)
    heads  -> visibility logits (3), sub-pixel offsets (3 x 2 maps)
    state  = memory.update(state, centre-frame outputs; evidence map when present)

Late fusion keeps an image-only measurement for the memory: with early fusion the memory
measured a heatmap computed from its own prior and could confirm a wrong position.
vis_memory feeds memory cues (speed, confidence, predicted position outside the frame) to the
visibility head.

Trained recurrently on short sequences (see train.py); inference runs the same step loop
over whole rallies.
"""
import math

import torch
import torch.nn as nn

from .memory import KinematicMemory
from .motion import MDD, mdd_channels
from .tracknetv5 import RSTRHead
from .unet import TrackNetUNet, conv_block


class TrackMem(nn.Module):
    recurrent = True

    def __init__(self, seq_len=3, img_size=(288, 512), rstr=True, latent_dim=64, prior_sigma_min=1.5,
                 mdd_init=(0.2, 0.15), fusion="early", vis_memory=False):
        super().__init__()
        if seq_len != 3:
            raise ValueError("TrackMem uses 3-frame windows")
        if fusion not in ("early", "late"):
            raise ValueError(f"fusion must be 'early' or 'late', got {fusion!r}")
        self.h, self.w = img_size
        self.fusion, self.vis_memory = fusion, vis_memory
        self.motion = MDD(*mdd_init)
        self.net = TrackNetUNet(mdd_channels(3) + (3 if fusion == "early" else 0), 3, head_bias=fusion == "late")
        c = self.net.head.in_channels
        if fusion == "late":
            # The evidence map is used directly (no R-STR), so start it at p = 0.01 (RetinaNet prior).
            nn.init.constant_(self.net.head.bias, -math.log(99.0))
            self.fuse = conv_block(c + 3, c, 1, "conv_relu_bn")
            self.fuse_head = nn.Conv2d(c, 3, 1, bias=False)
        self.rstr = RSTRHead(img_size) if rstr else None
        self.offset_head = nn.Conv2d(c, 6, 1)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        n_cues = 2 + 3 + (3 if vis_memory else 0)
        self.vis_head = nn.Sequential(nn.Linear(c + n_cues, 64), nn.ReLU(), nn.Linear(64, 1))
        self.memory = KinematicMemory(feat_dim=c, latent_dim=latent_dim, sigma_min=prior_sigma_min)

    def forward(self, frames, prior, mem_feats=None):
        """frames: (B, 3, 3, H, W) in [0, 1]; prior: (B, 3, H, W); mem_feats: (B, 3, 3) memory cues
        (required when vis_memory) -> dict of outputs."""
        b = frames.shape[0]
        maps = self.motion.maps(frames)
        x = self.motion.interleave(frames, maps)
        out = {}
        if self.fusion == "early":
            feats = self.net.features(torch.cat([x, prior.to(x.dtype)], dim=1))
            draft = self.net.head(feats)
        else:
            feats = self.net.features(x)
            out["evidence"] = self.net.head(feats)
            feats = self.fuse(torch.cat([feats, prior.to(feats.dtype)], dim=1))
            draft = self.fuse_head(feats)
        logits = self.rstr(draft, maps) if self.rstr is not None else draft
        offset = self.offset_head(feats).view(b, 3, 2, self.h, self.w)

        # Visibility (presence): features pooled where each frame's heatmap points,
        # plus that frame's peak logit and prior peak.
        wts = torch.softmax(logits.float().flatten(2), dim=-1)                      # (B, 3, HW)
        pooled = torch.einsum("btn,bcn->btc", wts, feats.float().flatten(2))         # (B, 3, C)
        peak = logits.float().flatten(2).amax(-1, keepdim=True)
        prior_peak = prior.float().flatten(2).amax(-1, keepdim=True)
        frame_id = torch.eye(3, device=frames.device).expand(b, 3, 3)
        cues = [pooled, peak / 10, prior_peak, frame_id]
        if self.vis_memory:
            cues.append(mem_feats.float())
        vis_logit = self.vis_head(torch.cat(cues, -1)).squeeze(-1)
        out.update(logits=logits, vis_logit=vis_logit, offset=offset, pooled=pooled)
        return out

    def step(self, frames, state, dt, prior_dropout=0.0):
        """One recurrent step. dt: (B, 1) = 30 / fps. prior_dropout (training): chance per
        sample of hiding the prior, so detection cannot rely on it.
        Returns (outputs, new_state, prior)."""
        prior = self.memory.render(state, dt, self.h, self.w)
        if prior_dropout > 0:
            keep = (torch.rand(prior.shape[0], 1, 1, 1, device=prior.device) >= prior_dropout).to(prior.dtype)
            prior = prior * keep
        mem_feats = self.memory.vis_features(state, dt, self.w, self.h) if self.vis_memory else None
        out = self(frames, prior, mem_feats)
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
        ro = {"vis_prob": torch.sigmoid(out["vis_logit"][:, 1].float()),
              "x": ix.float() + off[:, 0], "y": iy.float() + off[:, 1],
              "peak": torch.sigmoid(lc.reshape(b, -1).max(1).values),
              "latent_x": state["p"][:, 0], "latent_y": state["p"][:, 1]}
        if "evidence" in out:   # prior-free detection, for analysing what the prior changes
            ec = out["evidence"][:, 1].float().reshape(b, -1)
            eidx = ec.argmax(1)
            ro.update(ev_x=(eidx % w).float(), ev_y=(eidx // w).float(), ev_peak=torch.sigmoid(ec.max(1).values))
        return ro

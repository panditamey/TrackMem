"""Learned kinematic memory for TrackMem.

Per rally (batch element) the memory holds

    p, v, a   position (px, 512x288 input space), velocity, acceleration (per 1/30 s)
    var       position variance per axis (px^2)
    conf      memory confidence in [0, 1] (EMA of accepted updates)
    age       soft frames since the last accepted update
    h         learned latent vector (GRU state)

Time unit is 1/30 s, so dt = 30 / fps (1.0 at 30 fps, 1.2 at 25 fps).

Each step the memory (1) renders a prior heatmap for the next window, (2) measures the
model's centre-frame output with a differentiable local soft-argmax, and (3) applies a
learned gated update. Everything is differentiable and trained in the recurrent loop;
the constant-acceleration model is the only fixed part, and a learned correction adjusts it.
All maths runs in fp32.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CTRL = 8          # scalar control features fed to the gate and GRU
CORR_SCALE = 8.0    # max learned position correction (px)
MAX_V, MAX_A = 60.0, 30.0


class KinematicMemory(nn.Module):
    def __init__(self, feat_dim=64, latent_dim=64, sigma_min=1.5, local=3):
        super().__init__()
        self.latent_dim, self.sigma_min, self.local = latent_dim, sigma_min, local
        self.dyn = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 5))
        self.gate = nn.Sequential(nn.Linear(N_CTRL + latent_dim, 64), nn.ReLU(), nn.Linear(64, 4))
        self.gru = nn.GRUCell(N_CTRL + feat_dim, latent_dim)
        self.conf_keep = nn.Parameter(torch.tensor(0.85))  # sigmoid -> ~0.70
        # Start as a plain physics filter whose gate follows visibility; learning adds residuals.
        for head in (self.dyn[-1], self.gate[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    # ----------------------------------------------------------------------------- state
    def init_cold(self, batch, device, width, height):
        z = lambda *s: torch.zeros(batch, *s, device=device)
        p = torch.tensor([width / 2, height / 2], device=device).expand(batch, 2).clone()
        return {"p": p, "v": z(2), "a": z(2), "var": torch.full((batch, 2), 50.0 ** 2, device=device),
                "conf": z(1), "age": torch.full((batch, 1), 10.0, device=device), "h": z(self.latent_dim)}

    @staticmethod
    def detach(state):
        return {k: v.detach() for k, v in state.items()}

    # ------------------------------------------------------------------------ prediction
    def _dynamics(self, state):
        out = self.dyn(state["h"])
        corr = torch.tanh(out[:, :4]).view(-1, 2, 2) * CORR_SCALE     # corrections for k = 1, 2
        step_sigma = 1.0 + 4.0 * torch.sigmoid(out[:, 4:5])            # per-step spread growth (px)
        return corr, step_sigma

    def _predict(self, state, k, dt, corr, step_sigma):
        tau = k * dt
        p = state["p"] + state["v"] * tau + 0.5 * state["a"] * tau ** 2
        if k > 0:
            p = p + corr[:, k - 1]
        var = state["var"] + (k * dt * step_sigma) ** 2
        return p, var

    def render(self, state, dt, height, width):
        """Prior heatmaps for the window (state time + 0, 1, 2 steps) -> (B, 3, H, W)."""
        with torch.autocast(state["p"].device.type, enabled=False):
            corr, step_sigma = self._dynamics(state)
            ys = torch.arange(height, device=dt.device, dtype=torch.float32).view(1, height, 1)
            xs = torch.arange(width, device=dt.device, dtype=torch.float32).view(1, 1, width)
            maps = []
            for k in range(3):
                p, var = self._predict(state, k, dt, corr, step_sigma)
                sig = var.sqrt().clamp(self.sigma_min, 60.0)
                g = ((xs - p[:, 0, None, None]) / sig[:, 0, None, None]) ** 2 + \
                    ((ys - p[:, 1, None, None]) / sig[:, 1, None, None]) ** 2
                maps.append(state["conf"][:, :, None] * torch.exp(-0.5 * g))
            return torch.stack(maps, dim=1)

    def vis_features(self, state, dt, width, height):
        """Memory cues for the visibility head, per window frame -> (B, 3, 3):
        distance of the predicted position outside the frame, memory speed, memory confidence.
        A shuttle the memory sees standing still (held before a serve, on the floor after a
        rally) is not in play and is labelled invisible."""
        with torch.autocast(state["p"].device.type, enabled=False):
            corr, step_sigma = self._dynamics(state)
            speed = (state["v"].norm(dim=1, keepdim=True) / 10.0).clamp(max=3.0)
            feats = []
            for k in range(3):
                p, _ = self._predict(state, k, dt, corr, step_sigma)
                outside = (F.relu(-p[:, 0]) + F.relu(p[:, 0] - width) +
                           F.relu(-p[:, 1]) + F.relu(p[:, 1] - height)) / 32.0
                feats.append(torch.cat([outside.clamp(max=2.0)[:, None], speed, state["conf"]], dim=1))
            return torch.stack(feats, dim=1)

    # ----------------------------------------------------------------------- measurement
    def measure(self, logits_c):
        """Local soft-argmax around the peak of the centre-frame logits (B, H, W).
        Returns position z (B, 2), peak probability (B, 1), normalised local entropy (B, 1)."""
        b, h, w = logits_c.shape
        flat = logits_c.reshape(b, -1)
        idx = flat.argmax(1).detach()
        iy, ix = idx // w, idx % w
        r = torch.arange(-self.local, self.local + 1, device=logits_c.device)
        yy = (iy[:, None, None] + r[None, :, None]).clamp(0, h - 1).expand(b, len(r), len(r))
        xx = (ix[:, None, None] + r[None, None, :]).clamp(0, w - 1).expand(b, len(r), len(r))
        vals = flat.gather(1, (yy * w + xx).reshape(b, -1))
        wts = torch.softmax(vals, dim=1)
        z = torch.stack([(wts * xx.reshape(b, -1)).sum(1), (wts * yy.reshape(b, -1)).sum(1)], dim=1)
        peak = torch.sigmoid(vals.max(1, keepdim=True).values)
        ent = -(wts * torch.log(wts + 1e-9)).sum(1, keepdim=True) / math.log(wts.shape[1])
        return z, peak, ent

    # ---------------------------------------------------------------------------- update
    def update(self, state, out, dt, width, height):
        """Advance the state by one step using the model output for the window centre frame."""
        with torch.autocast(state["p"].device.type, enabled=False):
            corr, step_sigma = self._dynamics(state)
            p1, var1 = self._predict(state, 1, dt, corr, step_sigma)
            v1, a1 = state["v"] + state["a"] * dt, state["a"]

            # Measure from the prior-free evidence map when the model has one, so the memory
            # never reads back its own prior as confirmation.
            z, peak, ent = self.measure(out.get("evidence", out["logits"])[:, 1].float())
            lv = out["vis_logit"][:, 1:2].float().clamp(-10, 10)
            innov = z - p1
            sd = var1.mean(1, keepdim=True).sqrt().clamp(min=1.0)
            nin = (innov.norm(dim=1, keepdim=True) / sd).clamp(max=20.0)
            ctrl = torch.cat([lv / 5, peak, ent, nin / 5, torch.log1p(state["age"]) / 3,
                              state["conf"], dt - 1.0, sd / 20], dim=1)

            g = self.gate(torch.cat([ctrl, state["h"]], dim=1))
            g_p = torch.sigmoid(lv + g[:, 0:1])
            # Velocity/acceleration corrections need a trusted previous position.
            g_v = torch.sigmoid(lv + g[:, 1:2]) * state["conf"]
            g_a = torch.sigmoid(lv - 1.0 + g[:, 2:3]) * state["conf"]
            meas_sigma = 1.0 + F.softplus(g[:, 3:4])

            p = p1 + g_p * innov
            v_obs = (z - state["p"]) / dt
            v = (v1 + g_v * (v_obs - v1)).clamp(-MAX_V, MAX_V)
            a = (a1 + g_a * ((v - state["v"]) / dt - a1)).clamp(-MAX_A, MAX_A)
            keep = torch.sigmoid(self.conf_keep)
            new = {
                "p": torch.stack([p[:, 0].clamp(-64, width + 64), p[:, 1].clamp(-64, height + 64)], 1),
                "v": v, "a": a,
                "var": ((1 - g_p) * var1 + g_p * meas_sigma ** 2).clamp(max=200.0 ** 2),
                "conf": keep * state["conf"] + (1 - keep) * g_p,
                "age": (1 - g_p) * (state["age"] + dt),
                "h": self.gru(torch.cat([ctrl, out["pooled"][:, 1].float()], dim=1), state["h"]),
            }
            return new, {"z": z, "gate": g_p}


def training_init_state(memory, batch, width, height, cfg):
    """Initial memory for a training sequence: noisy warm start from GT history, cold start,
    or a deliberately wrong prior (CenterTrack-style robustness).
    batch: init_p/v/a (B, 2), m_p/m_v/m_a (B,) masks, all on device."""
    device = batch["init_p"].device
    b = batch["init_p"].shape[0]
    state = memory.init_cold(b, device, width, height)
    rnd = lambda *s: torch.rand(*s, device=device)
    warm = batch["m_p"].bool() & (rnd(b) < cfg["warm_start_p"])
    false = warm & (rnd(b) < cfg["false_prior_p"])
    jitter = torch.randn(b, 2, device=device) * cfg["jitter_px"]
    p = batch["init_p"] + jitter
    p_false = rnd(b, 2) * torch.tensor([width, height], device=device)
    p = torch.where(false[:, None], p_false, p)
    mv = (batch["m_v"].bool() & warm & ~false)[:, None]
    ma = (batch["m_a"].bool() & warm & ~false)[:, None]
    w = warm[:, None]
    state["p"] = torch.where(w, p, state["p"])
    state["v"] = torch.where(mv, (batch["init_v"] + torch.randn(b, 2, device=device)).clamp(-MAX_V, MAX_V),
                             state["v"])
    state["a"] = torch.where(ma, batch["init_a"].clamp(-MAX_A, MAX_A), state["a"])
    sig0 = 2.0 + 6.0 * rnd(b, 1)
    state["var"] = torch.where(w, (sig0 ** 2).expand(b, 2), state["var"])
    state["conf"] = torch.where(w, 0.3 + 0.7 * rnd(b, 1), state["conf"])
    state["age"] = torch.where(w, torch.zeros_like(state["age"]), state["age"])
    return state


def corrupt_state(state, prob, px):
    """Randomly shift memory positions mid-sequence so the model learns to recover."""
    b = state["p"].shape[0]
    hit = (torch.rand(b, 1, device=state["p"].device) < prob).float()
    state = dict(state)
    state["p"] = state["p"] + hit * torch.randn_like(state["p"]) * px
    return state

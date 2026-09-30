"""Shuttlecock Trajectory Dataset (TrackNetV2) loading.

A Rally holds labels (scaled to the 512x288 input space) and a lazily opened
mmap of its decoded frames. Label frame i is assumed to match video frame i;
rallies whose video is longer than the CSV are truncated to the label count.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ORIG_W, ORIG_H = 1280, 720


class Rally:
    def __init__(self, meta, width, height):
        self.split, self.match, self.name = meta["split"], meta["match"], meta["rally"]
        self.key = f"{self.split}/{self.match}/{self.name}"
        self.frames_path = meta["frames"]
        self.fps = meta["fps"]
        self.scale = width / meta.get("src_w", ORIG_W)

        df = pd.read_csv(meta["csv"])
        n = min(len(df), meta["n_decoded"])
        df = df.iloc[:n]
        self.vis = df["Visibility"].to_numpy().astype(np.int64)
        # Official labels (used for evaluation), in input space.
        self.xy = df[["X", "Y"]].to_numpy().astype(np.float32) * self.scale
        self.xy[self.vis == 0] = np.nan
        # Training labels: visible-with-zero-coordinate rows are label errors.
        bad = (self.vis == 1) & ((df["X"].to_numpy() <= 0) | (df["Y"].to_numpy() <= 0))
        self.train_vis = self.vis.copy()
        self.train_vis[bad] = 0
        self._frames = None

    def __len__(self):
        return len(self.vis)

    @property
    def frames(self):
        # Opened lazily so each DataLoader worker gets its own mmap handle.
        if self._frames is None:
            self._frames = np.load(self.frames_path, mmap_mode="r")
        return self._frames

    def window(self, center, seq_len):
        """Frame indices for a window centered at `center`, clamped at rally edges."""
        half = seq_len // 2
        return np.clip(np.arange(center - half, center - half + seq_len), 0, len(self) - 1)


def load_rallies(cfg, splits=None, matches=None, exclude_matches=None):
    """Load rallies from the frame cache index, filtered by split or split/match."""
    d = cfg["data"]
    with open(os.path.join(d["cache"], "index.json")) as f:
        index = json.load(f)
    rallies, per_split = [], {}
    for meta in index:
        sm = f"{meta['split']}/{meta['match']}"
        if splits is not None and meta["split"] not in splits:
            continue
        if matches is not None and sm not in matches:
            continue
        if exclude_matches and sm in exclude_matches:
            continue
        per_split[meta["split"]] = per_split.get(meta["split"], 0) + 1
        if d.get("limit_rallies") and per_split[meta["split"]] > d["limit_rallies"]:
            continue
        rallies.append(Rally(meta, d["width"], d["height"]))
    return rallies


def disk_heatmap(xy, vis, height, width, radius):
    """Binary disk targets. xy: (T, 2) in input space, vis: (T,). Returns (T, H, W) float32."""
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    out = np.zeros((len(vis), height, width), dtype=np.float32)
    for i, (v, (x, y)) in enumerate(zip(vis, xy)):
        if v:
            out[i] = ((xs - x) ** 2 + (ys - y) ** 2) <= radius ** 2
    return out


class WindowDataset(Dataset):
    """Training samples: seq_len-frame windows with per-frame heatmaps and labels."""

    def __init__(self, rallies, cfg, train=True):
        d = cfg["data"]
        self.rallies = rallies
        self.seq_len = d["seq_len"]
        self.h, self.w = d["height"], d["width"]
        self.radius = d["heatmap_radius"]
        self.hflip = d["hflip"] if train else 0.0
        stride = d["train_stride"] if train else 1
        self.samples = [(ri, c) for ri, r in enumerate(rallies) for c in range(0, len(r), stride)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ri, center = self.samples[i]
        r = self.rallies[ri]
        idx = r.window(center, self.seq_len)
        frames = np.ascontiguousarray(r.frames[idx])       # (T, H, W, 3) uint8
        vis = r.train_vis[idx].copy()
        xy = np.nan_to_num(r.xy[idx].copy(), nan=0.0)

        if self.hflip and np.random.rand() < self.hflip:
            frames = frames[:, :, ::-1].copy()
            xy[:, 0] = np.where(vis == 1, self.w - 1 - xy[:, 0], 0.0)

        heatmaps = disk_heatmap(xy, vis, self.h, self.w, self.radius)
        return {
            "frames": torch.from_numpy(frames).permute(0, 3, 1, 2),   # (T, 3, H, W) uint8
            "heatmaps": torch.from_numpy(heatmaps),                    # (T, H, W)
            "vis": torch.from_numpy(vis),
            "xy": torch.from_numpy(xy),
        }

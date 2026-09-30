"""Sliding-window inference over whole rallies and split-level evaluation."""
import numpy as np
import pandas as pd
import torch

from .metrics import ORIG_SCALE, classify, heatmap_to_point, summarize
from .subsets import frame_tags

SUBSET_ORDER = ("normal", "fast", "hit", "reappear", "occl_short", "occl_long",
                "out_of_frame", "edge_invisible", "visible", "invisible")


@torch.no_grad()
def predict_rally(model, rally, seq_len, device, threshold, batch_size=16, ensemble="average", amp=None):
    """Run stride-1 windows over a rally.

    ensemble='center': frame t uses the center output of the window centered at t.
    ensemble='average': frame t averages every window output that maps to t.
    Returns dict of per-frame arrays: vis, x, y, peak (input space).
    """
    n = len(rally)
    half = seq_len // 2
    sums, counts = {}, {}
    out = {"vis": np.zeros(n, np.int64), "x": np.full(n, np.nan), "y": np.full(n, np.nan),
           "peak": np.zeros(n)}

    def finalize(f):
        hm = sums.pop(f) / counts.pop(f)
        out["vis"][f], out["x"][f], out["y"][f], out["peak"][f] = heatmap_to_point(hm, threshold)

    for start in range(0, n, batch_size):
        centers = list(range(start, min(start + batch_size, n)))
        idx = np.stack([rally.window(c, seq_len) for c in centers])
        frames = torch.from_numpy(np.ascontiguousarray(rally.frames[idx.reshape(-1)]))
        frames = frames.view(len(centers), seq_len, *frames.shape[1:]).permute(0, 1, 4, 2, 3)
        frames = frames.to(device, non_blocking=True).float() / 255.0
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(frames)
        heat = torch.sigmoid(logits.float()).cpu().numpy()

        for b, c in enumerate(centers):
            ks = [half] if ensemble == "center" else range(seq_len)
            for k in ks:
                f = int(idx[b, k])
                if ensemble == "average" and c - half + k != f:
                    continue  # clamped duplicate at rally edge
                sums[f] = sums.get(f, 0) + heat[b, k]
                counts[f] = counts.get(f, 0) + 1
        # Frames below this can no longer receive outputs from later windows.
        next_center = centers[-1] + 1
        for f in sorted(k for k in sums if k < next_center - half or next_center >= n):
            finalize(f)
    return out


def evaluate_rallies(model, rallies, cfg, device, amp=None, progress=False):
    """Returns (per-frame DataFrame, metrics dict keyed by space -> subset -> summary)."""
    e, d = cfg["eval"], cfg["data"]
    model.eval()
    rows = []
    it = rallies
    if progress:
        from tqdm import tqdm
        it = tqdm(rallies, desc="eval")
    for r in it:
        p = predict_rally(model, r, d["seq_len"], device, e["threshold"], e["batch_size"],
                          e["ensemble"], amp)
        tags = frame_tags(r.vis, r.xy, d["height"], cfg)
        df = pd.DataFrame({"rally": r.key, "frame": np.arange(len(r)), "fps": r.fps,
                           "gt_vis": r.vis, "gt_x": r.xy[:, 0], "gt_y": r.xy[:, 1],
                           "pred_vis": p["vis"], "pred_x": p["x"], "pred_y": p["y"],
                           "peak": p["peak"]})
        for k, v in tags.items():
            df[f"tag_{k}"] = v
        rows.append(df)
    frames = pd.concat(rows, ignore_index=True)
    return frames, compute_metrics(frames, e["tolerance"])


def compute_metrics(frames, tolerance):
    pv = frames["pred_vis"].to_numpy()
    pxy = frames[["pred_x", "pred_y"]].to_numpy()
    gv = frames["gt_vis"].to_numpy()
    gxy = frames[["gt_x", "gt_y"]].to_numpy()
    metrics = {}
    for space, scale in (("input", 1.0), ("orig", ORIG_SCALE)):
        outcomes, dist = classify(pv, pxy, gv, gxy, tolerance, scale)
        m = {"all": summarize(outcomes, dist, gv)}
        for s in SUBSET_ORDER:
            mask = frames[f"tag_{s}"].to_numpy()
            if mask.any():
                m[s] = summarize(outcomes[mask], dist[mask], gv[mask])
        metrics[space] = m
    return metrics


def format_metrics(metrics, space):
    lines = [f"[{space} space] {'subset':<15}{'n':>7}{'F1':>8}{'prec':>8}{'rec':>8}{'acc':>8}"
             f"{'FP2':>6}{'FN':>6}{'err50':>7}{'err90':>7}"]
    for name, m in metrics[space].items():
        lines.append(f"{'':<{len(space) + 9}}{name:<15}{m['n']:>7}{m['f1']:>8.4f}{m['precision']:>8.4f}"
                     f"{m['recall']:>8.4f}{m['accuracy']:>8.4f}{m['FP2']:>6}{m['FN']:>6}"
                     f"{m.get('loc_err_median', float('nan')):>7.2f}{m.get('loc_err_p90', float('nan')):>7.2f}")
    return "\n".join(lines)

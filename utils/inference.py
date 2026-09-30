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


def detect(vis_prob, peak, threshold, rule="product"):
    """TrackMem detection decision. 'product': P(visible) x heatmap peak > threshold (default);
    'vis': P(visible) > threshold alone."""
    score = vis_prob * peak if rule == "product" else vis_prob
    return (score > threshold).astype(np.int64)


@torch.no_grad()
def predict_rallies_recurrent(model, rallies, device, amp=None, batch_size=16, memory_mode="normal",
                              threshold=0.5, progress=False, rule="product"):
    """Run a recurrent model (TrackMem) frame by frame over whole rallies, several rallies in
    parallel. memory_mode='reset' clears the memory every step (memory-use test).
    Returns {rally key: dict of per-frame arrays} in input space."""
    results = {}
    chunks = [rallies[i:i + batch_size] for i in range(0, len(rallies), batch_size)]
    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=sum(len(r) for r in rallies), unit="frame", desc="eval")
    for chunk in chunks:
        b, lens = len(chunk), [len(r) for r in chunk]
        dt = torch.tensor([[30.0 / r.fps] for r in chunk], device=device)
        state = model.memory.init_cold(b, device, model.w, model.h)
        rec = {r.key: {k: np.zeros(len(r)) for k in ("vis_prob", "x", "y", "peak", "latent_x", "latent_y")}
               for r in chunk}
        for c in range(max(lens)):
            batch = np.stack([r.frames[r.window(min(c, len(r) - 1), 3)] for r in chunk])
            frames = torch.from_numpy(batch).to(device).permute(0, 1, 4, 2, 3).float() / 255.0
            if memory_mode == "reset":
                state = model.memory.init_cold(b, device, model.w, model.h)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                out, state, _ = model.step(frames, state, dt)
            ro = {k: v.float().cpu().numpy() for k, v in model.readout(out, state).items()}
            for i, r in enumerate(chunk):
                if c < lens[i]:
                    for k in rec[r.key]:
                        rec[r.key][k][c] = ro[k][i]
            if bar:
                bar.update(sum(c < n for n in lens))
        for r in chunk:
            p = rec[r.key]
            p["vis"] = detect(p["vis_prob"], p["peak"], threshold, rule)
            results[r.key] = p
    if bar:
        bar.close()
    return results


def evaluate_rallies(model, rallies, cfg, device, amp=None, progress=False):
    """Returns (per-frame DataFrame, metrics dict keyed by space -> subset -> summary)."""
    e, d = cfg["eval"], cfg["data"]
    model.eval()
    recurrent = getattr(model, "recurrent", False)
    if recurrent:
        preds = predict_rallies_recurrent(model, rallies, device, amp, e["batch_size"],
                                          e.get("memory_mode", "normal"), e["threshold"], progress,
                                          e.get("detect", "product"))
    rows = []
    it = rallies
    if progress and not recurrent:
        from tqdm import tqdm
        it = tqdm(rallies, desc="eval")
    for r in it:
        if recurrent:
            p = preds[r.key]
        else:
            p = predict_rally(model, r, d["seq_len"], device, e["threshold"], e["batch_size"],
                              e["ensemble"], amp)
        tags = frame_tags(r.vis, r.xy, d["height"], cfg)
        df = pd.DataFrame({"rally": r.key, "frame": np.arange(len(r)), "fps": r.fps,
                           "gt_vis": r.vis, "gt_x": r.xy[:, 0], "gt_y": r.xy[:, 1],
                           "pred_vis": p["vis"], "pred_x": np.where(p["vis"] == 1, p["x"], np.nan),
                           "pred_y": np.where(p["vis"] == 1, p["y"], np.nan), "peak": p["peak"]})
        if recurrent:
            df["vis_prob"], df["latent_x"], df["latent_y"] = p["vis_prob"], p["latent_x"], p["latent_y"]
            df["raw_x"], df["raw_y"] = p["x"], p["y"]   # position for every frame (offline threshold sweeps)
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

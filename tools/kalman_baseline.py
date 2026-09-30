"""Open-loop control: post-hoc constant-acceleration Kalman filter on a model's saved predictions.

    python tools/kalman_baseline.py --frames runs/tracknetv5/eval_val/frames.csv            # grid search
    python tools/kalman_baseline.py --frames runs/tracknetv5/eval_test/frames.csv --coast 2 --gate 9

Per rally, detections (pred_vis=1) update a CA Kalman filter (dt = 30 / fps). A detection whose
Mahalanobis distance exceeds `gate` (chi^2, 2 dof) is rejected as an outlier; a missed frame
within `coast` frames of the last accepted detection is filled with the filter prediction.
Detection-level output only (the filter never sees heatmaps), so this is what memory can add
*without* feeding back into detection. Pick coast/gate on val, then apply once to test.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import compute_metrics, format_metrics  # noqa: E402


def ca_matrices(dt, q, r):
    F = np.eye(6)
    for i in range(2):
        F[i, i + 2], F[i, i + 4], F[i + 2, i + 4] = dt, 0.5 * dt ** 2, dt
    G = np.array([0.5 * dt ** 2, 0.5 * dt ** 2, dt, dt, 1.0, 1.0])
    Q = np.diag(G ** 2) * q ** 2
    H = np.zeros((2, 6))
    H[0, 0] = H[1, 1] = 1.0
    return F, Q, H, np.eye(2) * r ** 2


def filter_rally(df, coast, gate, q=3.0, r=1.5, max_miss=8):
    dt = 30.0 / float(df.fps.iloc[0])
    F, Q, H, R = ca_matrices(dt, q, r)
    vis = df.pred_vis.to_numpy().copy()
    xs, ys = df.pred_x.to_numpy().copy(), df.pred_y.to_numpy().copy()
    x, P, miss = None, None, 0
    for t in range(len(df)):
        if x is not None:
            x, P = F @ x, F @ P @ F.T + Q
        det = vis[t] == 1
        if det and x is not None:
            y = np.array([xs[t], ys[t]]) - H @ x
            S = H @ P @ H.T + R
            if y @ np.linalg.solve(S, y) > gate:
                det = False           # outlier: reject detection
                vis[t] = 0
        if det:
            z = np.array([xs[t], ys[t]])
            if x is None:
                x, P = np.r_[z, np.zeros(4)], np.diag([r ** 2, r ** 2, 100, 100, 50, 50])
            else:
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x, P = x + K @ (z - H @ x), (np.eye(6) - K @ H) @ P
            miss = 0
        elif x is not None:
            miss += 1
            if miss <= coast:
                vis[t], xs[t], ys[t] = 1, x[0], x[1]   # fill gap with prediction
            if miss > max_miss:
                x, P = None, None
    out = df.copy()
    out["pred_vis"], out["pred_x"], out["pred_y"] = vis, np.where(vis == 1, xs, np.nan), np.where(vis == 1, ys, np.nan)
    return out


def run(frames, coast, gate, tolerance):
    out = pd.concat([filter_rally(g, coast, gate) for _, g in frames.groupby("rally", sort=False)],
                    ignore_index=True)
    return out, compute_metrics(out, tolerance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frames.csv written by evaluate.py")
    ap.add_argument("--coast", type=int, default=None)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--tolerance", type=float, default=4)
    ap.add_argument("--space", default="orig", choices=["orig", "input"])
    args = ap.parse_args()
    frames = pd.read_csv(args.frames)

    base = compute_metrics(frames, args.tolerance)
    print("model alone:\n" + format_metrics(base, args.space))
    if args.coast is None or args.gate is None:
        print(f"\ngrid search ({args.space} F1, all frames):")
        best = None
        for coast, gate in itertools.product([0, 1, 2, 3, 5], [6.0, 9.2, 13.8, 1e9]):
            f1 = run(frames, coast, gate, args.tolerance)[1][args.space]["all"]["f1"]
            print(f"  coast={coast} gate={gate:g}  F1={f1:.4f}")
            best = max(best or (f1, coast, gate), (f1, coast, gate))
        _, args.coast, args.gate = best
        print(f"best: coast={args.coast} gate={args.gate:g} (choose on val, apply to test)")
    out, m = run(frames, args.coast, args.gate, args.tolerance)
    print(f"\nwith Kalman filter (coast={args.coast}, gate={args.gate:g}):\n" + format_metrics(m, args.space))
    path = os.path.splitext(args.frames)[0] + f"_kalman_c{args.coast}_g{args.gate:g}.csv"
    out.to_csv(path, index=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

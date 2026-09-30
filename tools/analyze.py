"""Compare two evaluated runs and diagnose errors, from saved frames.csv files (no GPU).

    python tools/analyze.py --a runs/trackmem/eval_val/frames.csv --b runs/tracknetv5/eval_val/frames.csv
    python tools/analyze.py --a runs/trackmem/eval_val/frames.csv --sweep     # visibility threshold sweep

Reports, for both tolerance spaces:
  - F1 per subset for each run
  - error breakdown: misses (FN), false detections on invisible frames (FP2), wrong-position
    detections (FP1) split into near (tolerance..3x) and gross (>3x tolerance), per subset
  - frames one run gets right and the other wrong, per subset
With --sweep (TrackMem, needs vis_prob): F1 for detection rules 'vis' (P(visible) > thr) and
'product' (P(visible) x peak > thr) over a threshold grid.
Tune thresholds on val only, then apply the chosen one to test.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import SUBSET_ORDER, compute_metrics  # noqa: E402
from utils.metrics import ORIG_SCALE, classify  # noqa: E402

SPACES = {"orig": ORIG_SCALE, "input": 1.0}
VISIBLE_SUBSETS = [s for s in SUBSET_ORDER if s not in ("occl_short", "occl_long", "out_of_frame",
                                                      "edge_invisible", "invisible")]


def outcomes(df, tol, scale):
    out, dist = classify(df.pred_vis.to_numpy(), df[["pred_x", "pred_y"]].to_numpy(), df.gt_vis.to_numpy(),
                         df[["gt_x", "gt_y"]].to_numpy(), tol, scale)
    return out, dist


def breakdown(df, name, tol):
    print(f"\n=== error breakdown: {name}")
    for space, scale in SPACES.items():
        out, dist = outcomes(df, tol, scale)
        print(f"[{space}] {'subset':<15}{'n':>7}{'FN':>7}{'FP2':>7}{'FP1 near':>10}{'FP1 gross':>11}")
        for s in SUBSET_ORDER:
            m = df[f"tag_{s}"].to_numpy()
            if not m.any():
                continue
            o, d = out[m], dist[m]
            near = ((o == "FP1") & (d <= 3 * tol)).sum()
            gross = ((o == "FP1") & (d > 3 * tol)).sum()
            print(f"        {s:<15}{m.sum():>7}{(o == 'FN').sum():>7}{(o == 'FP2').sum():>7}{near:>10}{gross:>11}")


def compare(a, b, na, nb, tol):
    keys = ["rally", "frame"]
    m = a.merge(b[keys + ["pred_vis", "pred_x", "pred_y"]], on=keys, suffixes=("", "_b"))
    mb = m.copy()
    mb["pred_vis"], mb["pred_x"], mb["pred_y"] = m.pred_vis_b, m.pred_x_b, m.pred_y_b
    for space, scale in SPACES.items():
        ma, mbm = compute_metrics(m, tol)[space], compute_metrics(mb, tol)[space]
        oa, _ = outcomes(m, tol, scale)
        ob, _ = outcomes(mb, tol, scale)
        ok_a, ok_b = np.isin(oa, ["TP", "TN"]), np.isin(ob, ["TP", "TN"])
        print(f"\n=== [{space}] F1 per subset: {na} vs {nb}")
        print(f"{'subset':<15}{'n':>7}{na[:10]:>11}{nb[:10]:>11}{'diff':>8}"
              f"{'only ' + na[:6] + ' ok':>16}{'only ' + nb[:6] + ' ok':>16}")
        for s in SUBSET_ORDER:
            if s not in ma:
                continue
            t = m[f"tag_{s}"].to_numpy()
            fa = ma[s]["f1"] if s in VISIBLE_SUBSETS else ma[s]["accuracy"]
            fb = mbm[s]["f1"] if s in VISIBLE_SUBSETS else mbm[s]["accuracy"]
            label = s if s in VISIBLE_SUBSETS else s + " (acc)"
            print(f"{label:<15}{t.sum():>7}{fa:>11.4f}{fb:>11.4f}{fa - fb:>+8.4f}"
                  f"{(ok_a & ~ok_b & t).sum():>16}{(ok_b & ~ok_a & t).sum():>16}")
        print(f"{'all':<15}{len(m):>7}{ma['all']['f1']:>11.4f}{mbm['all']['f1']:>11.4f}"
              f"{ma['all']['f1'] - mbm['all']['f1']:>+8.4f}{(ok_a & ~ok_b).sum():>16}{(ok_b & ~ok_a).sum():>16}")


def sweep(df, tol):
    if "vis_prob" not in df:
        sys.exit("--sweep needs vis_prob (TrackMem frames.csv)")
    has_raw = "raw_x" in df
    if not has_raw:
        print("\nnote: frames.csv has no raw_x/raw_y (older evaluate.py); only thresholds stricter than the "
              "original decision can be evaluated. Re-run evaluate.py for a full sweep.")
    print("\n=== detection-rule sweep (tune on val only)")
    print(f"{'rule':<9}{'thr':>6}{'F1 orig':>10}{'F1 input':>10}{'prec in':>9}{'rec in':>8}{'FP2':>6}{'FN':>6}")
    base_vis = df.pred_vis.to_numpy() == 1
    for rule in ("vis", "product"):
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            score = df.vis_prob * df.peak if rule == "product" else df.vis_prob
            keep = (score > thr).to_numpy()
            if not has_raw:
                keep &= base_vis
            d = df.copy()
            d["pred_vis"] = keep.astype(int)
            xs, ys = (d.raw_x, d.raw_y) if has_raw else (d.pred_x, d.pred_y)
            d["pred_x"], d["pred_y"] = np.where(keep, xs, np.nan), np.where(keep, ys, np.nan)
            m = compute_metrics(d, tol)
            a = m["input"]["all"]
            print(f"{rule:<9}{thr:>6.2f}{m['orig']['all']['f1']:>10.4f}{a['f1']:>10.4f}{a['precision']:>9.4f}"
                  f"{a['recall']:>8.4f}{a['FP2']:>6}{a['FN']:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="frames.csv of run A (e.g. TrackMem)")
    ap.add_argument("--b", default=None, help="frames.csv of run B (e.g. TrackNetV5)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--tolerance", type=float, default=4)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    a = pd.read_csv(args.a)
    if args.sweep:
        sweep(a, args.tolerance)
    breakdown(a, args.name_a, args.tolerance)
    if args.b:
        b = pd.read_csv(args.b)
        breakdown(b, args.name_b, args.tolerance)
        compare(a, b, args.name_a, args.name_b, args.tolerance)


if __name__ == "__main__":
    main()

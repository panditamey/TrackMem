"""Evaluate a checkpoint on a split, overall and per subset.

    python evaluate.py --ckpt runs/tracknetv5/best.pt --split test
Writes <ckpt_dir>/eval_<split>/{frames.csv, metrics.json}. frames.csv holds per-frame
predictions, reusable by post-hoc baselines (e.g. Kalman filter) without rerunning the model.
"""
import argparse
import json
import os

import torch

from datasets.tracknet_dataset import load_rallies
from models import build_model
from utils.common import amp_dtype, apply_overrides, load_config
from utils.inference import evaluate_rallies, format_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = load_config(args.config, args.set) if args.config else apply_overrides(ck["cfg"], args.set)

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    amp = amp_dtype(cfg["train"]["amp"], device)

    d = cfg["data"]
    rallies = (load_rallies(cfg, splits=d["test_splits"]) if args.split == "test"
               else load_rallies(cfg, matches=d["val_matches"]))
    frames, metrics = evaluate_rallies(model, rallies, cfg, device, amp, progress=True)

    for space in ("orig", "input"):
        print(format_metrics(metrics, space))
    out = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out, exist_ok=True)
    frames.to_csv(os.path.join(out, "frames.csv"), index=False)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()

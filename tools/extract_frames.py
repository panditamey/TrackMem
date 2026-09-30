"""Decode rally videos once into per-rally uint8 arrays at 512x288.

Output layout (under --out):
    <split>/<match>/<rally>.npy     uint8 (N, 288, 512, 3), RGB
    index.json                      per-rally metadata

Arrays are loaded with mmap so multi-frame windows need no per-sample image
decoding. INTER_AREA resizing averages the tiny shuttle instead of aliasing it.
Full dataset is ~40 GB uncompressed: put --out on local disk (e.g. Colab /content).
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

W, H = 512, 288


def extract(job):
    video, out_path = job
    cap = cv2.VideoCapture(video)
    n_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np.stack(frames))
    return dict(n_decoded=len(frames), n_header=n_header, fps=fps, src_w=src_w, src_h=src_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/TracknetV2")
    ap.add_argument("--out", default="data/cache/frames_512x288")
    ap.add_argument("--splits", nargs="*", default=None, help="e.g. Professional Test")
    ap.add_argument("--limit", type=int, default=None, help="max rallies per split (smoke tests)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    jobs, keys, per_split = [], [], {}
    for csv in sorted(glob.glob(os.path.join(args.root, "*", "*", "csv", "*_ball.csv"))):
        parts = csv.split(os.sep)
        split, match = parts[-4], parts[-3]
        if args.splits and split not in args.splits:
            continue
        per_split[split] = per_split.get(split, 0) + 1
        if args.limit and per_split[split] > args.limit:
            continue
        rally = os.path.basename(csv).replace("_ball.csv", "")
        video = os.path.join(os.path.dirname(os.path.dirname(csv)), "video", rally + ".mp4")
        out_path = os.path.join(args.out, split, match, rally + ".npy")
        jobs.append((video, out_path))
        keys.append(dict(split=split, match=match, rally=rally, csv=csv, video=video, frames=out_path))

    index, total = [], 0
    with ProcessPoolExecutor(args.workers) as pool:
        bar = tqdm(zip(keys, pool.map(extract, jobs)), total=len(jobs), unit="rally", desc="decode")
        for key, meta in bar:
            n_labels = sum(1 for _ in open(key["csv"])) - 1
            index.append({**key, **meta, "n_labels": n_labels})
            total += meta["n_decoded"]
            bar.set_postfix(frames=total, last=f"{key['match']}/{key['rally']}")
    mismatched = sum(m["n_decoded"] != m["n_labels"] for m in index)
    print(f"decoded {len(index)} rallies, {total} frames; {mismatched} rallies with frame/label count mismatch")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()

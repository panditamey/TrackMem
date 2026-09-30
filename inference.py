"""Run a trained checkpoint on a video and save an annotated copy.

    python inference.py                                   # best ckpt, first Test rally video
    python inference.py --video my_match.mp4 --out out/my_match_pred.mp4
    python inference.py --ckpt runs/tracknetv5/best.pt --video data/TracknetV2/Test/match1/video/1_05_02.mp4

Outputs:
    <out>.mp4     annotated video: prediction (red) + trail, GT (green) if a dataset CSV exists,
                  TrackMem memory estimate while the shuttle is not detected (yellow)
    <out>.csv     predictions in the dataset label format: Frame,Visibility,X,Y (+ Peak; TrackMem also
                  VisProb, LatentX, LatentY), video resolution

The model type comes from the checkpoint. TrackNetV5 runs overlapping chunks; TrackMem streams
the video frame by frame with its memory carried across the whole video.
Frames are resized to 512x288 for the model; non-16:9 videos get stretched (a warning is printed).
"""
import argparse
import glob
import os
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models import build_model
from utils.common import amp_dtype
from utils.inference import detect, predict_rally


class FrameClip:
    """In-memory frame block exposing the interface predict_rally expects."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def window(self, center, seq_len):
        half = seq_len // 2
        return np.clip(np.arange(center - half, center - half + seq_len), 0, len(self) - 1)


def predict_video(model, path, cfg, device, amp, chunk=1024):
    """Stream-decode `path` and predict every frame. Returns dict of arrays in 512x288 space."""
    d, e = cfg["data"], cfg["eval"]
    seq_len, W, H = d["seq_len"], d["width"], d["height"]
    ctx = seq_len  # context frames kept on each side of a chunk (>= seq_len - 1 for averaging)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {k: [] for k in ("vis", "x", "y", "peak")}
    buf, start, emitted = [], 0, 0
    bar = tqdm(total=total, unit="frame", desc="infer")
    eof = False
    while not eof:
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            eof = True
        if buf and (eof or len(buf) >= chunk + 2 * ctx):
            p = predict_rally(model, FrameClip(np.stack(buf)), seq_len, device, e["threshold"],
                              e["batch_size"], e["ensemble"], amp)
            end = start + len(buf) if eof else start + len(buf) - ctx
            for k in out:
                out[k].append(p[k][emitted - start:end - start])
            bar.update(end - emitted)
            emitted = end
            keep_from = max(end - ctx, start)
            buf, start = buf[keep_from - start:], keep_from
    cap.release()
    bar.close()
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


@torch.no_grad()
def predict_video_recurrent(model, path, cfg, device, amp):
    """Stream a video through a recurrent model (TrackMem), carrying memory across the whole video.
    Frame c is processed once frame c+1 is decoded (one-frame lookahead, as in training)."""
    W, H = cfg["data"]["width"], cfg["data"]["height"]
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = torch.tensor([[30.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0)]], device=device)
    state = model.memory.init_cold(1, device, W, H)
    buf, out, n, c, eof = {}, {k: [] for k in ("vis_prob", "x", "y", "peak", "latent_x", "latent_y")}, 0, 0, False
    bar = tqdm(total=total, unit="frame", desc="infer")
    while not eof:
        ok, frame = cap.read()
        if ok:
            buf[n] = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            n += 1
        eof = not ok
        while c < n and (c + 1 < n or eof):
            idx = np.clip([c - 1, c, c + 1], 0, n - 1)
            x = torch.from_numpy(np.stack([buf[i] for i in idx])[None]).to(device).permute(0, 1, 4, 2, 3)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                o, state, _ = model.step(x.float() / 255.0, state, dt)
            for k, v in model.readout(o, state).items():
                out[k].append(float(v[0]))
            buf.pop(c - 1, None)
            c += 1
            bar.update(1)
    cap.release()
    bar.close()
    res = {k: np.array(v) for k, v in out.items()}
    res["vis"] = detect(res["vis_prob"], res["peak"], cfg["eval"]["threshold"], cfg["eval"].get("detect", "product"))
    return res


def load_gt(video):
    """Dataset layout: .../video/<rally>.mp4 -> .../csv/<rally>_ball.csv."""
    csv = os.path.join(os.path.dirname(os.path.dirname(video)), "csv",
                       os.path.splitext(os.path.basename(video))[0] + "_ball.csv")
    return pd.read_csv(csv) if os.path.exists(csv) else None


def draw_video(video, out_path, pred, gt, trail):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    history = []
    for i in tqdm(range(len(pred)), unit="frame", desc="draw"):
        ok, frame = cap.read()
        if not ok:
            break
        if gt is not None and i < len(gt) and gt.Visibility[i]:
            cv2.circle(frame, (int(gt.X[i]), int(gt.Y[i])), 10, (0, 255, 0), 2)
        visible = bool(pred.Visibility[i])
        history.append((pred.X[i], pred.Y[i]) if visible else None)
        history = history[-trail:]
        pts = [p for p in history if p is not None]
        for j in range(1, len(pts)):
            cv2.line(frame, tuple(map(int, pts[j - 1])), tuple(map(int, pts[j])), (0, 0, 255), 2)
        if visible:
            cv2.circle(frame, (int(pred.X[i]), int(pred.Y[i])), 6, (0, 0, 255), -1)
        elif "LatentX" in pred:
            cv2.circle(frame, (int(pred.LatentX[i]), int(pred.LatentY[i])), 8, (0, 255, 255), 2)
        lines = [f"frame {i}  {'visible' if visible else 'not detected'}  peak {pred.Peak[i]:.2f}"]
        legend = "red: prediction" + ("   yellow: memory estimate" if "LatentX" in pred else "")
        lines.append(legend + ("   green: ground truth" if gt is not None else ""))
        cv2.rectangle(frame, (10, 10), (900, 20 + 36 * len(lines)), (0, 0, 0), -1)
        for k, text in enumerate(lines):
            cv2.putText(frame, text, (20, 42 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    cap.release()
    writer.release()


def to_h264(path):
    """Re-encode for browser/Colab playback when ffmpeg is available (OpenCV writes mp4v)."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path + ".tmp.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/trackmem/best.pt")
    ap.add_argument("--video", default=None, help="defaults to the first video of the Test split")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: outputs/<video>_pred.mp4)")
    ap.add_argument("--trail", type=int, default=8, help="frames of predicted trajectory to draw")
    ap.add_argument("--no-gt", action="store_true", help="don't draw ground truth even if available")
    ap.add_argument("--batch-size", type=int, default=None, help="windows per forward pass (default: from ckpt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    if args.batch_size:
        cfg["eval"]["batch_size"] = args.batch_size
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()
    amp = amp_dtype(cfg["train"]["amp"], device)

    video = args.video
    if video is None:
        root = cfg["data"]["root"]
        videos = sorted(glob.glob(os.path.join(root, *cfg["data"]["test_splits"][:1], "*", "video", "*.mp4")))
        if not videos:
            raise SystemExit(f"no --video given and no Test videos found under {root}")
        video = videos[0]
    out = args.out or os.path.join("outputs", os.path.splitext(os.path.basename(video))[0] + "_pred.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"checkpoint {args.ckpt} (epoch {ck.get('epoch')}, val F1 {ck.get('f1', float('nan')):.4f})")
    print(f"video {video} -> {out}")

    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if abs(w / h - cfg["data"]["width"] / cfg["data"]["height"]) > 0.01:
        print(f"warning: video is {w}x{h}, not 16:9; frames are stretched to the model input size")

    recurrent = getattr(model, "recurrent", False)
    p = (predict_video_recurrent if recurrent else predict_video)(model, video, cfg, device, amp)
    sx, sy = w / cfg["data"]["width"], h / cfg["data"]["height"]
    pred = pd.DataFrame({"Frame": np.arange(len(p["vis"])), "Visibility": p["vis"],
                         "X": np.where(p["vis"] == 1, np.round(p["x"] * sx), 0).astype(int),
                         "Y": np.where(p["vis"] == 1, np.round(p["y"] * sy), 0).astype(int),
                         "Peak": np.round(p["peak"], 4)})
    if recurrent:
        pred["VisProb"] = np.round(p["vis_prob"], 4)
        pred["LatentX"] = np.round(p["latent_x"] * sx).astype(int)
        pred["LatentY"] = np.round(p["latent_y"] * sy).astype(int)
    pred.to_csv(os.path.splitext(out)[0] + ".csv", index=False)

    gt = None if args.no_gt else load_gt(video)
    draw_video(video, out, pred, gt, args.trail)
    to_h264(out)
    print(f"detected in {pred.Visibility.mean():.1%} of {len(pred)} frames; saved {out} and "
          f"{os.path.splitext(out)[0]}.csv")


if __name__ == "__main__":
    main()

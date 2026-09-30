"""Train a TrackNet model.

    python train.py --config config.yaml [--set train.batch_size=4 data.limit_rallies=2 ...]
    python train.py --config config.yaml --resume runs/b0_v5like/last.pt
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from datasets.tracknet_dataset import WindowDataset, load_rallies
from losses.heatmap import wbce_loss
from models.baseline import build_model
from utils.common import amp_dtype, load_config, seed_everything
from utils.inference import evaluate_rallies, format_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=1")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d, t = cfg["data"], cfg["train"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp_dtype(t["amp"], device)
    out_dir = os.path.join(t["out_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)

    train_rallies = load_rallies(cfg, splits=d["train_splits"], exclude_matches=d["val_matches"])
    val_rallies = load_rallies(cfg, matches=d["val_matches"])
    train_set = WindowDataset(train_rallies, cfg, train=True)
    loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                        num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                        persistent_workers=t["num_workers"] > 0)
    print(f"train rallies={len(train_rallies)} windows={len(train_set)} | val rallies={len(val_rallies)} "
          f"| device={device} amp={amp}")

    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, t["milestones"], t["gamma"])
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)
    fwd = torch.compile(model) if t["compile"] else model

    start_epoch, best_f1 = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader):
            if t["max_steps_per_epoch"] and step >= t["max_steps_per_epoch"]:
                break
            frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
            target = batch["heatmaps"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = fwd(frames)
            loss = wbce_loss(logits, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if (step + 1) % t["log_every"] == 0:
                rate = (step + 1) * t["batch_size"] / (time.time() - t0)
                print(f"epoch {epoch} step {step + 1}/{len(loader)} loss {running / t['log_every']:.5f} "
                      f"lr {scheduler.get_last_lr()[0]:.1e} {rate:.1f} samples/s", flush=True)
                log.write(json.dumps({"epoch": epoch, "step": step + 1, "loss": running / t["log_every"]}) + "\n")
                running = 0.0
        scheduler.step()

        record = {"epoch": epoch, "epoch_time_s": time.time() - t0}
        if val_rallies and ((epoch + 1) % t["val_every"] == 0 or epoch + 1 == t["epochs"]):
            _, metrics = evaluate_rallies(model, val_rallies, cfg, device, amp)
            print(format_metrics(metrics, cfg["eval"]["primary_space"]), flush=True)
            f1 = metrics[cfg["eval"]["primary_space"]]["all"]["f1"]
            record["val"] = {space: m["all"] for space, m in metrics.items()}
            if f1 > best_f1:
                best_f1 = f1
                torch.save({"model": model.state_dict(), "epoch": epoch, "f1": f1, "cfg": cfg},
                           os.path.join(out_dir, "best.pt"))
                print(f"new best val F1 {f1:.4f}")
        log.write(json.dumps(record) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "cfg": cfg}, os.path.join(out_dir, "last.pt"))


if __name__ == "__main__":
    main()

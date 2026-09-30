"""Train a model.

    python train.py --model tracknetv5 [--set train.batch_size=4 train.epochs=10 ...]
    python train.py --model trackmem
    python train.py --model trackmem --resume runs/trackmem/last.pt

tracknetv5 trains on independent 3-frame windows. trackmem trains recurrently on sequences
of `train.trackmem.seq_steps` windows with the memory in the loop (after single-step warm-up
epochs); train.batch_size counts windows, so a batch holds batch_size // seq_steps sequences.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from datasets.tracknet_dataset import SequenceDataset, WindowDataset, disk_heatmap_torch, load_rallies
from losses.heatmap import wbce_loss
from losses.trackmem import trackmem_step_loss
from models import MODELS, build_model
from models.memory import corrupt_state, training_init_state
from utils.common import amp_dtype, load_config, seed_everything
from utils.inference import evaluate_rallies, format_metrics


def make_loader(dataset, batch_size, t, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=t["num_workers"], pin_memory=device.type == "cuda",
                      persistent_workers=t["num_workers"] > 0)


def window_loss(model, batch, cfg, device, amp):
    d = cfg["data"]
    frames = batch["frames"].to(device, non_blocking=True).float() / 255.0
    target = disk_heatmap_torch(batch["xy"].to(device, non_blocking=True), batch["vis"].to(device, non_blocking=True),
                                d["height"], d["width"], d["heatmap_radius"])
    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        logits = model(frames)
    loss = wbce_loss(logits, target)
    return loss, {"loss": loss.detach()}


def sequence_loss(model, batch, cfg, device, amp, train=True):
    """Unroll the memory over the sequence; loss averaged over steps."""
    d, tm = cfg["data"], cfg["train"]["trackmem"]
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    frames = b["frames"].float() / 255.0                                   # (B, S + 2, 3, H, W)
    steps = frames.shape[1] - 2
    heat = disk_heatmap_torch(b["xy"], b["vis"], d["height"], d["width"], d["heatmap_radius"])
    state = training_init_state(model.memory, b, d["width"], d["height"], tm)
    total, parts = 0.0, {}
    for k in range(steps):
        if train and k > 0 and tm["corrupt_p"] > 0:
            state = corrupt_state(state, tm["corrupt_p"], tm["corrupt_px"])
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            out, state, _ = model.step(frames[:, k:k + 3], state, b["dt"])
        loss, p = trackmem_step_loss(out, heat[:, k:k + 3], b["vis"][:, k:k + 3], b["xy"][:, k:k + 3], tm)
        total = total + loss / steps
        for key, v in p.items():
            parts[key] = parts.get(key, 0.0) + v / steps
    return total, {"loss": total.detach(), **parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", choices=MODELS, default=None, help="overrides model.name in the config")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=1")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.model:
        cfg["model"]["name"] = args.model
    cfg["experiment"] = cfg.get("experiment") or cfg["model"]["name"]
    d, t = cfg["data"], cfg["train"]
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = amp_dtype(t["amp"], device)
    out_dir = os.path.join(t["out_dir"], cfg["experiment"])
    os.makedirs(out_dir, exist_ok=True)

    train_rallies = load_rallies(cfg, splits=d["train_splits"], exclude_matches=d["val_matches"])
    val_rallies = load_rallies(cfg, matches=d["val_matches"])
    model = build_model(cfg).to(device).to(memory_format=torch.channels_last)
    recurrent = getattr(model, "recurrent", False)

    if recurrent:
        tm = t["trackmem"]
        steps = tm["seq_steps"]
        warm_loader = make_loader(SequenceDataset(train_rallies, cfg, 1), t["batch_size"], t, device)
        seq_loader = make_loader(SequenceDataset(train_rallies, cfg, steps), max(t["batch_size"] // steps, 1),
                                 t, device)
        print(f"train rallies={len(train_rallies)} warm-up windows={len(warm_loader.dataset)} "
              f"sequences={len(seq_loader.dataset)}x{steps} | val rallies={len(val_rallies)}")
    else:
        loader = make_loader(WindowDataset(train_rallies, cfg, train=True), t["batch_size"], t, device)
        print(f"train rallies={len(train_rallies)} windows={len(loader.dataset)} | val rallies={len(val_rallies)}")
    print(f"model={cfg['model']['name']} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"device={device} amp={amp} -> {out_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, t["milestones"], t["gamma"])
    scaler = torch.amp.GradScaler(enabled=amp == torch.float16)
    fwd = torch.compile(model) if t["compile"] and not recurrent else model

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
        if recurrent:
            warmup = epoch < tm["warmup_epochs"]
            loader = warm_loader if warmup else seq_loader
            windows_per_batch = loader.batch_size * (1 if warmup else steps)
            print(f"epoch {epoch}: {'warm-up (single step)' if warmup else f'recurrent ({steps} steps)'}")
        else:
            windows_per_batch = loader.batch_size
        t0, running = time.time(), {}
        for step, batch in enumerate(loader):
            if t["max_steps_per_epoch"] and step >= t["max_steps_per_epoch"]:
                break
            if recurrent:
                loss, parts = sequence_loss(model, batch, cfg, device, amp)
            else:
                loss, parts = window_loss(fwd, batch, cfg, device, amp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v)
            if (step + 1) % t["log_every"] == 0:
                rate = (step + 1) * windows_per_batch / (time.time() - t0)
                avg = {k: v / t["log_every"] for k, v in running.items()}
                shown = " ".join(f"{k} {v:.5f}" for k, v in avg.items())
                print(f"epoch {epoch} step {step + 1}/{len(loader)} {shown} "
                      f"lr {scheduler.get_last_lr()[0]:.1e} {rate:.1f} windows/s", flush=True)
                log.write(json.dumps({"epoch": epoch, "step": step + 1, **avg}) + "\n")
                running = {}
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

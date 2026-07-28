#!/usr/bin/env python
"""Train the learned ADT (drum transcriber).

    .venv/bin/python scripts/prepare_adt.py            # first, build the clips
    .venv/bin/python scripts/train_adt.py --config configs/adt.yaml
    .venv/bin/python scripts/train_adt.py --config configs/adt.yaml --overfit-one-batch

Maps per-step mel features of a drum signal -> the 9-class drum grid. Selects
runs/adt_best.pt on per-class hit-F1 over all nine classes.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, "src")
from musibot.config import CH_HIT, CH_VELOCITY, CHECKPOINT_DIR, DRUM_CLASSES  # noqa: E402
from musibot.data.adt_dataset import make_adt_loaders  # noqa: E402
from musibot.models.adt import DrumTranscriber  # noqa: E402
from musibot.train import load_config, resolve_device  # noqa: E402


def loss_fn(hit_logit, vel, target, pos_weight):
    t_hit, t_vel = target[..., CH_HIT], target[..., CH_VELOCITY]
    hit = F.binary_cross_entropy_with_logits(hit_logit, t_hit, pos_weight=pos_weight)
    denom = t_hit.sum().clamp(min=1.0)
    vel_l = (((vel - t_vel) ** 2) * t_hit).sum() / denom
    return hit + vel_l, hit, vel_l


def run_epoch(model, loader, device, pw, optimizer=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    tot = hl = vl = 0.0; n = 0
    for batch in loader:
        mel = batch["mel"].to(device); drums = batch["drums"].to(device)
        with torch.set_grad_enabled(training):
            hit_logit, vel = model(mel)
            loss, h, v = loss_fn(hit_logit, vel, drums, pw)
        if training:
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip); optimizer.step()
        tot += float(loss.detach()); hl += float(h.detach()); vl += float(v.detach()); n += 1
    return {"total": tot / max(1, n), "hit": hl / max(1, n), "vel": vl / max(1, n)}


@torch.no_grad()
def per_class_f1(model, loader, device, threshold=0.4):
    model.eval()
    C = len(DRUM_CLASSES)
    tp = torch.zeros(C); fp = torch.zeros(C); fn = torch.zeros(C)
    for batch in loader:
        mel = batch["mel"].to(device); drums = batch["drums"].to(device)
        hit_logit, _ = model(mel)
        p = (torch.sigmoid(hit_logit) > threshold).cpu()
        t = drums[..., CH_HIT].bool().cpu()
        tp += (p & t).sum((0, 1)); fp += (p & ~t).sum((0, 1)); fn += (~p & t).sum((0, 1))
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
    return float(f1.mean()), {DRUM_CLASSES[c]: round(float(f1[c]), 2) for c in range(C)}


def main():
    ap = argparse.ArgumentParser(description="Train the learned ADT")
    ap.add_argument("--config", default="configs/adt.yaml")
    ap.add_argument("--overfit-one-batch", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    win = cfg.extra.get("window", 64)
    torch.manual_seed(cfg.seed)
    loaders = make_adt_loaders(cfg.batch_size, win=win, num_workers=cfg.num_workers)
    print(f"device={device}  ADT  win={win}  train={len(loaders['train'].dataset)}  "
          f"val={len(loaders['val'].dataset)}")

    model = DrumTranscriber(hidden=cfg.hidden_dim, n_layers=cfg.n_layers, dropout=cfg.dropout).to(device)
    pw = torch.tensor(cfg.hit_pos_weight, device=device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if args.overfit_one_batch:
        batch = next(iter(loaders["train"]))
        mel = batch["mel"].to(device); drums = batch["drums"].to(device)
        model.train()
        for step in range(300):
            hit_logit, vel = model(mel)
            loss, _, _ = loss_fn(hit_logit, vel, drums, pw)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 50 == 0 or step == 299:
                print(f"  step {step:4d}  total={float(loss):.4f}")
        print("Overfit check done — loss should approach zero above.")
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best, no_improve = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, loaders["train"], device, pw, opt, cfg.grad_clip)
        va = run_epoch(model, loaders["val"], device, pw)
        f1, per = per_class_f1(model, loaders["val"], device)
        print(f"epoch {epoch:3d}  train={tr['total']:.4f}  val={va['total']:.4f}  "
              f"hit_f1={f1:.4f}  {per}")
        if f1 > best:
            best, no_improve = f1, 0
            ckpt = CHECKPOINT_DIR / "adt_best.pt"
            torch.save({"model": model.state_dict(), "config": vars(cfg),
                        "epoch": epoch, "hit_f1": f1}, ckpt)
            print(f"  saved {ckpt}  (hit_f1={f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop: no improvement in {cfg.patience} epochs.")
                break


if __name__ == "__main__":
    main()

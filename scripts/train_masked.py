#!/usr/bin/env python
"""Train the masked drum-infilling transformer (Phase 2).

    .venv/bin/python scripts/train_masked.py --config configs/model_masked.yaml
    .venv/bin/python scripts/train_masked.py --config configs/model_masked.yaml --overfit-one-batch

Bidirectional masked reconstruction (MAE/BERT-style): each batch masks a random
fraction of whole bars and the model reconstructs *only* the masked steps from
the surrounding drums + audio conditioning. Reuses the same width-21 clips and
DataLoaders as the autoregressive model; writes runs/masked_best.pt selected on
masked-reconstruction F1 over the backbone classes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.insert(0, "src")
from musibot.config import CHECKPOINT_DIR, DRUM_CLASSES  # noqa: E402
from musibot.data.dataset import make_loaders  # noqa: E402
from musibot.models.losses import MaskedDrumLoss  # noqa: E402
from musibot.models.masked_transformer import MaskedDrumTransformer, make_span_mask  # noqa: E402
from musibot.train import load_config, resolve_device  # noqa: E402

BACKBONE = ["kick", "snare", "hat_closed", "hat_open"]


def build(cfg):
    return MaskedDrumTransformer(cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.dropout)


def run_epoch(model, loader, loss_fn, device, cfg, optimizer=None):
    training = optimizer is not None
    model.train(training)
    lo, hi = cfg.extra.get("mask_min_ratio", 0.15), cfg.extra.get("mask_max_ratio", 0.65)
    totals = {"total": 0.0, "hit": 0.0, "velocity": 0.0, "offset": 0.0}
    n = 0
    for batch in loader:
        cond = batch["conditioning"].to(device)
        drums = batch["drums"].to(device)
        style = batch["style"].to(device)
        mask = make_span_mask(drums.size(0), drums.size(1), device, lo, hi)
        with torch.set_grad_enabled(training):
            pred = model(cond, drums, mask, style)
            losses = loss_fn(pred, drums, mask)
        if training:
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        for k in totals:
            totals[k] += float(losses[k].detach())
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def recon_f1(model, loader, device, max_batches=6, ratio=0.5, threshold=0.5):
    """Masked-reconstruction F1 over backbone classes, with fixed eval masks."""
    model.eval()
    bb = [DRUM_CLASSES.index(c) for c in BACKBONE]
    tp = fp = fn = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        torch.manual_seed(1000 + i)  # deterministic eval masks
        cond = batch["conditioning"].to(device)
        drums = batch["drums"].to(device)
        style = batch["style"].to(device)
        mask = make_span_mask(drums.size(0), drums.size(1), device, ratio, ratio)
        pred = model(cond, drums, mask, style)
        p = (torch.sigmoid(pred[..., 0]) > threshold)          # [B,T,9]
        t = drums[..., 0].bool()
        sel = mask.unsqueeze(-1)                               # only masked steps
        pb, tb = p[..., bb] & sel, t[..., bb] & sel
        tp += int((pb & tb).sum()); fp += int((pb & ~tb).sum()); fn += int((~pb & tb).sum())
    return tp / (tp + 0.5 * (fp + fn) + 1e-9)


def overfit_one_batch(model, loader, loss_fn, device, cfg, steps=300):
    batch = next(iter(loader))
    cond, drums = batch["conditioning"].to(device), batch["drums"].to(device)
    style = batch["style"].to(device)
    torch.manual_seed(0)
    mask = make_span_mask(drums.size(0), drums.size(1), device, 0.5, 0.5)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    for step in range(steps):
        pred = model(cond, drums, mask, style)
        losses = loss_fn(pred, drums, mask)
        opt.zero_grad(); losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}  total={float(losses['total']):.4f}")
    print("Overfit check done — masked-reconstruction loss should approach zero above.")


def main():
    ap = argparse.ArgumentParser(description="Train the masked drum-infilling model")
    ap.add_argument("--config", default="configs/model_masked.yaml")
    ap.add_argument("--sources", nargs="*", default=["lmd", "user"])
    ap.add_argument("--overfit-one-batch", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    print(f"device={device}  model=masked  mask=[{cfg.extra.get('mask_min_ratio')},"
          f"{cfg.extra.get('mask_max_ratio')}]  sources={args.sources}")

    torch.manual_seed(cfg.seed)
    loaders = make_loaders(cfg.batch_size, sources=args.sources, num_workers=cfg.num_workers,
                           max_train_windows=cfg.max_train_windows, seed=cfg.seed)
    print(f"train windows={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}")

    model = build(cfg).to(device)
    loss_fn = MaskedDrumLoss(cfg.velocity_loss_weight, cfg.offset_loss_weight,
                             cfg.hit_pos_weight).to(device)

    if args.overfit_one_batch:
        overfit_one_batch(model, loaders["train"], loss_fn, device, cfg)
        return

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best, no_improve = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, loaders["train"], loss_fn, device, cfg, optimizer)
        va = run_epoch(model, loaders["val"], loss_fn, device, cfg)
        f1 = recon_f1(model, loaders["val"], device)
        print(f"epoch {epoch:3d}  train={tr['total']:.4f}  val={va['total']:.4f}  "
              f"(hit={va['hit']:.3f} vel={va['velocity']:.3f} off={va['offset']:.3f})  "
              f"recon_f1={f1:.4f}")
        if f1 > best:
            best, no_improve = f1, 0
            ckpt = CHECKPOINT_DIR / "masked_best.pt"
            torch.save({"model": model.state_dict(), "config": vars(cfg),
                        "epoch": epoch, "recon_f1": f1}, ckpt)
            print(f"  saved {ckpt}  (recon_f1={f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop: no improvement in {cfg.patience} epochs.")
                break


if __name__ == "__main__":
    main()

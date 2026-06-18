"""Training entrypoint for both the LSTM baseline and the Transformer.

    python -m musibot.train --config configs/model_transformer.yaml
    python -m musibot.train --config configs/model_lstm.yaml --overfit-one-batch

``--overfit-one-batch`` is the Phase 3 sanity check: a correct data/model/loss
wiring drives a single batch's loss to ~0.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW

from .config import CHECKPOINT_DIR, TrainConfig
from .data.dataset import make_loaders
from .models.baseline_lstm import BaselineLSTM
from .models.losses import DrumLoss


def load_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(TrainConfig)}
    kwargs = {k: v for k, v in raw.items() if k in known}
    kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}
    return TrainConfig(**kwargs)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.model == "lstm":
        return BaselineLSTM(cfg.hidden_dim, cfg.n_layers, cfg.dropout)
    if cfg.model == "transformer":
        from .models.transformer import ConditionalDrumTransformer

        return ConditionalDrumTransformer(
            cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.dropout
        )
    raise ValueError(f"unknown model: {cfg.model}")


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=1.0):
    """One pass; trains when ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "hit": 0.0, "velocity": 0.0, "offset": 0.0}
    n = 0

    for batch in loader:
        cond = batch["conditioning"].to(device)
        drums = batch["drums"].to(device)
        style = batch["style"].to(device)
        with torch.set_grad_enabled(training):
            pred = model(cond, drums, teacher_forcing=True, style=style)
            losses = loss_fn(pred, drums)
        if training:
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for k in totals:
            totals[k] += float(losses[k].detach())
        n += 1

    return {k: v / max(1, n) for k, v in totals.items()}


def overfit_one_batch(model, loader, loss_fn, device, optimizer, grad_clip, steps=300):
    batch = next(iter(loader))
    cond = batch["conditioning"].to(device)
    drums = batch["drums"].to(device)
    style = batch["style"].to(device)
    model.train()
    for step in range(steps):
        pred = model(cond, drums, teacher_forcing=True, style=style)
        losses = loss_fn(pred, drums)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}  total={float(losses['total']):.4f}")
    print("Overfit check done — loss should be near zero above.")


def train(cfg: TrainConfig, sources: list[str] | None = None):
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"device={device}  model={cfg.model}")

    loaders = make_loaders(cfg.batch_size, sources=sources, num_workers=0)
    print(f"train windows={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}")

    model = build_model(cfg).to(device)
    loss_fn = DrumLoss(cfg.velocity_loss_weight, cfg.offset_loss_weight).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.overfit_one_batch:
        overfit_one_batch(model, loaders["train"], loss_fn, device, optimizer, cfg.grad_clip)
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer, cfg.grad_clip)
        va = run_epoch(model, loaders["val"], loss_fn, device)
        print(
            f"epoch {epoch:3d}  train={tr['total']:.4f}  val={va['total']:.4f}  "
            f"(hit={va['hit']:.3f} vel={va['velocity']:.3f} off={va['offset']:.3f})"
        )
        if va["total"] < best_val:
            best_val = va["total"]
            no_improve = 0
            ckpt = CHECKPOINT_DIR / f"{cfg.model}_best.pt"
            torch.save({"model": model.state_dict(), "config": vars(cfg), "epoch": epoch}, ckpt)
            print(f"  saved {ckpt}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop: val loss has not improved in {cfg.patience} epochs.")
                break


def main():
    parser = argparse.ArgumentParser(description="Train a MusiBot drum model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--sources", nargs="*", default=None, help="restrict to these tiers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overfit_one_batch:
        cfg.overfit_one_batch = True
    train(cfg, sources=args.sources)


if __name__ == "__main__":
    main()

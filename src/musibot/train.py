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

import numpy as np
import torch
import yaml
from torch.optim import AdamW

from .config import CHECKPOINT_DIR, DRUM_CLASSES, STEPS_PER_BAR, TrainConfig
from .data.dataset import make_loaders
from .models.baseline_lstm import BaselineLSTM
from .models.losses import DrumLoss

# The classes that carry the groove; toms/crash/ride are rare in the corpus and
# noisy to score, so checkpoint selection tracks F1 over these four.
BACKBONE_CLASSES = ["kick", "snare", "hat_closed", "hat_open"]


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


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=1.0, ss_prob=0.0):
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
            pred = model(
                cond, drums, teacher_forcing=True, style=style,
                ss_prob=ss_prob if training else 0.0,
            )
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


def ss_probability(cfg: TrainConfig, epoch: int) -> float:
    """Scheduled-sampling probability for this epoch: 0 until ``ss_start_epoch``,
    then a linear ramp to ``ss_max_prob`` over ``ss_ramp_epochs``."""
    if not cfg.scheduled_sampling or epoch <= cfg.ss_start_epoch:
        return 0.0
    ramp = (epoch - cfg.ss_start_epoch) / max(1, cfg.ss_ramp_epochs)
    return round(cfg.ss_max_prob * min(1.0, ramp), 3)


@torch.no_grad()
def gen_eval(model, loader, device, max_batches: int = 6, threshold: float = 0.5) -> dict:
    """Autoregressive quality on a capped val subset — the metric that actually
    reflects inference. Reports micro hit-F1 over the backbone classes and the
    share of generated snare hits landing on beats 2 & 4 (a flat, structureless
    groove scores ~0.25 here; a real backbeat approaches 1.0)."""
    model.eval()
    bb = [DRUM_CLASSES.index(c) for c in BACKBONE_CLASSES]
    snare = DRUM_CLASSES.index("snare")
    tp = fp = fn = 0
    pos = np.zeros(STEPS_PER_BAR)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        cond = batch["conditioning"].to(device)
        style = batch["style"].to(device)
        drums = batch["drums"].to(device)
        pred = model.generate(cond, threshold=threshold, style=style)
        p = torch.sigmoid(pred[..., 0]) > threshold          # [B, T, 9]
        t = drums[..., 0].bool()
        pb, tb = p[..., bb], t[..., bb]
        tp += int((pb & tb).sum()); fp += int((pb & ~tb).sum()); fn += int((~pb & tb).sum())
        ps = p[..., snare].cpu().numpy()
        nb = ps.shape[1] // STEPS_PER_BAR * STEPS_PER_BAR
        if nb:
            pos += ps[:, :nb].reshape(-1, STEPS_PER_BAR).sum(0)
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
    beat24 = pos[[4, 12]].sum() / (pos.sum() + 1e-9)
    return {"backbone_f1": round(f1, 4), "snare_beat24": round(float(beat24), 3)}


def train(cfg: TrainConfig, sources: list[str] | None = None, resume: str | None = None):
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"device={device}  model={cfg.model}  select_by={cfg.select_by}  "
          f"scheduled_sampling={cfg.scheduled_sampling}")

    loaders = make_loaders(
        cfg.batch_size, sources=sources, num_workers=cfg.num_workers,
        max_train_windows=cfg.max_train_windows, seed=cfg.seed,
    )
    print(f"train windows={len(loaders['train'].dataset)}  val={len(loaders['val'].dataset)}")

    model = build_model(cfg).to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device)["model"])
        print(f"resumed weights from {resume}")
    loss_fn = DrumLoss(
        cfg.velocity_loss_weight, cfg.offset_loss_weight, cfg.hit_pos_weight
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.overfit_one_batch:
        overfit_one_batch(model, loaders["train"], loss_fn, device, optimizer, cfg.grad_clip)
        return

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    by_gen = cfg.select_by == "gen"
    best_score = -float("inf")   # normalized so higher is always better
    no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        ssp = ss_probability(cfg, epoch)
        tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer, cfg.grad_clip, ssp)
        va = run_epoch(model, loaders["val"], loss_fn, device)
        gen = gen_eval(model, loaders["val"], device, cfg.gen_eval_batches)
        print(
            f"epoch {epoch:3d}  ss={ssp:.2f}  train={tr['total']:.4f}  val={va['total']:.4f}  "
            f"(hit={va['hit']:.3f} vel={va['velocity']:.3f} off={va['offset']:.3f})  "
            f"gen_f1={gen['backbone_f1']:.4f} snare2&4={gen['snare_beat24']:.2f}"
        )

        # Higher is better: backbone gen-F1 when select_by="gen", else negative loss.
        score = gen["backbone_f1"] if by_gen else -va["total"]
        if score > best_score:
            best_score = score
            no_improve = 0
            ckpt = CHECKPOINT_DIR / f"{cfg.model}_best.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch,
                 "val": va["total"], "gen": gen},
                ckpt,
            )
            print(f"  saved {ckpt}  (gen_f1={gen['backbone_f1']:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stop: no improvement in {cfg.patience} epochs.")
                break


def main():
    parser = argparse.ArgumentParser(description="Train a MusiBot drum model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--sources", nargs="*", default=None, help="restrict to these tiers")
    parser.add_argument("--resume", default=None, help="warm-start weights from a checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overfit_one_batch:
        cfg.overfit_one_batch = True
    train(cfg, sources=args.sources, resume=args.resume)


if __name__ == "__main__":
    main()

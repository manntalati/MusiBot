"""Objective evaluation of a trained drum model on the held-out test split.

Metrics are computed from *autoregressive* generation (no teacher forcing), so
they reflect real inference behaviour:
    hit F1        per-class and macro — did we place hits correctly?
    velocity MAE  loudness error on correctly placed hits
    offset MAE    microtiming error on correctly placed hits
    groove sim    cosine similarity of per-step hit-density curves
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score

from .config import CH_HIT, CH_OFFSET, CH_VELOCITY, DRUM_CLASSES
from .data.dataset import make_loaders
from .train import build_model, load_config, resolve_device


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    model.eval()
    pred_hits, true_hits = [], []
    vel_err, off_err, n_hits = 0.0, 0.0, 0
    groove_sims = []

    for batch in loader:
        cond = batch["conditioning"].to(device)
        drums = batch["drums"].to(device)
        style = batch["style"].to(device)
        pred = model.generate(cond, threshold=threshold, style=style)

        p_hit = (torch.sigmoid(pred[..., CH_HIT]) > threshold).float()
        t_hit = drums[..., CH_HIT]
        pred_hits.append(p_hit.cpu().numpy())
        true_hits.append(t_hit.cpu().numpy())

        mask = t_hit.bool()
        if mask.any():
            vel_err += (pred[..., CH_VELOCITY] - drums[..., CH_VELOCITY]).abs()[mask].sum().item()
            off_err += (pred[..., CH_OFFSET] - drums[..., CH_OFFSET]).abs()[mask].sum().item()
            n_hits += int(mask.sum())

        p_density = p_hit.sum(-1)          # [B, T]
        t_density = t_hit.sum(-1)
        cos = torch.nn.functional.cosine_similarity(p_density, t_density, dim=-1)
        groove_sims.extend(cos.cpu().tolist())

    p = np.concatenate(pred_hits).reshape(-1, len(DRUM_CLASSES))
    t = np.concatenate(true_hits).reshape(-1, len(DRUM_CLASSES))

    per_class = {
        cls: round(f1_score(t[:, i], p[:, i], zero_division=0), 4)
        for i, cls in enumerate(DRUM_CLASSES)
    }
    return {
        "hit_f1_macro": round(f1_score(t, p, average="macro", zero_division=0), 4),
        "hit_f1_per_class": per_class,
        "velocity_mae": round(vel_err / max(1, n_hits), 4),
        "offset_mae": round(off_err / max(1, n_hits), 4),
        "groove_similarity": round(float(np.mean(groove_sims)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a MusiBot drum model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="hit probability cutoff"
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="report F1 across thresholds 0.2..0.6 to find the operating point",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    loaders = make_loaders(cfg.batch_size, sources=args.sources, num_workers=0)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6] if args.threshold_sweep else [args.threshold]
    for t in thresholds:
        metrics = evaluate(model, loaders["test"], device, threshold=t)
        print(f"Test metrics @ threshold={t}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

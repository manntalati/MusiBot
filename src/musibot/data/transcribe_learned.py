"""Inference for the learned ADT: a drum signal -> a 9-class drum grid.

Loads ``runs/adt_best.pt`` lazily. ``adt_available()`` lets callers fall back to
the heuristic ``transcribe_drums`` when the model hasn't been trained yet.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from ..config import SAMPLE_RATE
from .adt_features import drum_mel_features

ADT_CKPT = os.environ.get("MUSIBOT_ADT_CKPT", "runs/adt_best.pt")
ADT_CONFIG = os.environ.get("MUSIBOT_ADT_CONFIG", "configs/adt.yaml")
_ADT: dict = {}


def adt_available() -> bool:
    return Path(ADT_CKPT).exists()


def _load() -> dict:
    if "model" not in _ADT:
        from ..models.adt import DrumTranscriber
        from ..train import load_config, resolve_device

        cfg = load_config(ADT_CONFIG)
        device = resolve_device(cfg.device)
        model = DrumTranscriber(hidden=cfg.hidden_dim, n_layers=cfg.n_layers, dropout=cfg.dropout)
        model.load_state_dict(torch.load(ADT_CKPT, map_location=device)["model"])
        model.to(device).eval()
        _ADT.update(model=model, device=device)
    return _ADT


@torch.no_grad()
def transcribe_drums_learned(
    audio: np.ndarray, sr: int = SAMPLE_RATE, tempo: float = 120.0,
    n_steps: int | None = None, threshold: float = 0.4,
) -> np.ndarray:
    """Drum signal -> grid ``[T, 9, 3]`` via the learned transcriber."""
    eng = _load()
    mel = drum_mel_features(audio, sr, tempo, n_steps)          # [T, n_mels]
    x = torch.from_numpy(mel[None]).to(eng["device"])
    return eng["model"].transcribe(x, threshold=threshold)[0].cpu().numpy()


__all__ = ["transcribe_drums_learned", "adt_available"]

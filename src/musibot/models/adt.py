"""Learned Automatic Drum Transcription (ADT).

A small conv + bidirectional-GRU network that maps per-step log-mel features of a
drum signal to the 9-class drum grid (hit + velocity per step). It replaces the
hand-written onset/spectral-band ``transcribe_drums`` — crucially it can predict
**all nine classes** (incl. toms and ride, which the heuristic never emits), so
the drum context fed to the masked infill model is no longer impoverished.

Trained on FluidSynth-rendered GMD drums (ground-truth grids); see
``scripts/prepare_adt.py`` and ``scripts/train_adt.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CH_HIT, CH_VELOCITY, N_CHANNELS, N_DRUM_CLASSES
from ..data.adt_features import N_MELS


class DrumTranscriber(nn.Module):
    def __init__(self, n_mels: int = N_MELS, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.inorm = nn.BatchNorm1d(n_mels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden // 2, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.hit = nn.Linear(hidden, N_DRUM_CLASSES)   # per-class hit logits
        self.vel = nn.Linear(hidden, N_DRUM_CLASSES)   # per-class velocity (0..1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``mel`` [B, T, n_mels] -> (hit_logits [B,T,9], velocity [B,T,9])."""
        x = self.inorm(mel.transpose(1, 2))            # normalise over mel bins
        x = self.conv(x).transpose(1, 2)               # [B, T, hidden]
        x, _ = self.gru(x)
        return self.hit(x), torch.sigmoid(self.vel(x))

    @torch.no_grad()
    def transcribe(self, mel: torch.Tensor, threshold: float | torch.Tensor = 0.4) -> torch.Tensor:
        """``mel`` [B,T,n_mels] -> drum grid [B,T,9,3] (offset channel is 0)."""
        self.eval()
        hit_logit, vel = self.forward(mel)
        thr = threshold if torch.is_tensor(threshold) else torch.tensor(float(threshold))
        thr = thr.to(device=mel.device, dtype=torch.float32)
        hit = (torch.sigmoid(hit_logit) > thr).float()
        grid = torch.zeros(*hit.shape, N_CHANNELS, device=mel.device)
        grid[..., CH_HIT] = hit
        grid[..., CH_VELOCITY] = vel * hit             # velocity only where a hit fired
        return grid


__all__ = ["DrumTranscriber"]

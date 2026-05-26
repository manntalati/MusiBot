"""Phase 3 baseline: an encoder-decoder LSTM with attention.

A bidirectional LSTM encodes the conditioning sequence; a causal LSTM decoder
produces the drum grid one step at a time, attending over the encoder outputs
and consuming its own previous step. Causality means the same module trains
(teacher forcing) and generates (autoregressive) with one code path.

Output is ``[B, T, 9, 3]``: channel 0 raw hit logit, channel 1 sigmoid
velocity, channel 2 tanh offset — the contract ``DrumLoss`` expects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FEATURE_DIM, N_CHANNELS, N_DRUM_CLASSES

_FLAT = N_DRUM_CLASSES * N_CHANNELS  # 27


def activate(raw: torch.Tensor) -> torch.Tensor:
    """Apply per-channel output activations to a raw ``[..., 9, 3]`` tensor."""
    hit = raw[..., 0:1]                       # kept as logit
    vel = torch.sigmoid(raw[..., 1:2])        # 0..1
    off = 0.5 * torch.tanh(raw[..., 2:3])     # -0.5..0.5
    return torch.cat([hit, vel, off], dim=-1)


def binarize(step: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Turn a generated step into autoregressive feedback (hard hit decisions).

    Lower thresholds make the model more "willing" to place hits — useful when
    a sparsely trained model collapses to silence at the default 0.5.
    """
    hit = (torch.sigmoid(step[..., 0:1]) > threshold).float()
    return torch.cat([hit, step[..., 1:2], step[..., 2:3]], dim=-1)


class BaselineLSTM(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            FEATURE_DIM,
            hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.prev_proj = nn.Linear(_FLAT, hidden_dim)
        self.cell = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, _FLAT),
        )

    def forward(
        self,
        conditioning: torch.Tensor,
        drums: torch.Tensor | None = None,
        teacher_forcing: bool = True,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        B, T, _ = conditioning.shape
        device = conditioning.device
        enc, _ = self.encoder(conditioning)               # [B, T, hidden]

        h = torch.zeros(B, self.hidden_dim, device=device)
        c = torch.zeros(B, self.hidden_dim, device=device)
        prev = torch.zeros(B, N_DRUM_CLASSES, N_CHANNELS, device=device)

        steps = []
        for t in range(T):
            prev_emb = self.prev_proj(prev.reshape(B, _FLAT))
            h, c = self.cell(torch.cat([enc[:, t], prev_emb], dim=-1), (h, c))

            # Dot-product attention of the decoder state over the encoder.
            scores = torch.bmm(enc, self.attn(h).unsqueeze(-1)).squeeze(-1)
            context = torch.bmm(F.softmax(scores, dim=1).unsqueeze(1), enc).squeeze(1)

            raw = self.head(torch.cat([h, context], dim=-1)).reshape(B, N_DRUM_CLASSES, N_CHANNELS)
            step = activate(raw)
            steps.append(step)

            if teacher_forcing and drums is not None:
                prev = drums[:, t]
            else:
                prev = binarize(step, threshold)

        return torch.stack(steps, dim=1)                  # [B, T, 9, 3]

    @torch.no_grad()
    def generate(self, conditioning: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        return self.forward(conditioning, drums=None, teacher_forcing=False, threshold=threshold)

"""Phase 4 main model: a conditional Transformer.

A Transformer encoder reads the conditioning sequence; a causal Transformer
decoder autoregressively produces the drum grid, cross-attending to the encoded
conditioning. This is the current SOTA pattern for symbolic music and scales
with data and compute far better than the LSTM baseline.

Shares the output contract and activations with ``baseline_lstm``:
``[B, T, 9, 3]`` — channel 0 raw hit logit, 1 sigmoid velocity, 2 tanh offset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import FEATURE_DIM, N_CHANNELS, N_DRUM_CLASSES, WINDOW_STEPS
from .baseline_lstm import activate, binarize

_FLAT = N_DRUM_CLASSES * N_CHANNELS  # 27


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConditionalDrumTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim)

        self.cond_proj = nn.Linear(FEATURE_DIM, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, n_heads, hidden_dim * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

        self.drum_proj = nn.Linear(_FLAT, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            hidden_dim, n_heads, hidden_dim * 4, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_layers)
        self.head = nn.Linear(hidden_dim, _FLAT)

    def encode(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos(self.cond_proj(conditioning)))

    def _decode(self, tgt_steps: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Run the decoder over teacher/autoregressive input ``[B, L, 9, 3]``."""
        B, L, _, _ = tgt_steps.shape
        tgt = self.pos(self.drum_proj(tgt_steps.reshape(B, L, _FLAT)))
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=mask)
        return activate(self.head(out).reshape(B, L, N_DRUM_CLASSES, N_CHANNELS))

    def forward(
        self,
        conditioning: torch.Tensor,
        drums: torch.Tensor | None = None,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        memory = self.encode(conditioning)
        B, T, _ = conditioning.shape

        if teacher_forcing and drums is not None:
            # Shift right: position t sees ground-truth steps 0..t-1.
            start = torch.zeros(B, 1, N_DRUM_CLASSES, N_CHANNELS, device=drums.device)
            tgt_in = torch.cat([start, drums[:, :-1]], dim=1)
            return self._decode(tgt_in, memory)

        return self._generate(memory, T)

    def _generate(self, memory: torch.Tensor, T: int) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device
        generated = torch.zeros(B, 1, N_DRUM_CLASSES, N_CHANNELS, device=device)
        outputs = []
        for _ in range(T):
            step = self._decode(generated, memory)[:, -1]   # [B, 9, 3]
            outputs.append(step)
            generated = torch.cat([generated, binarize(step).unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, conditioning: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(conditioning, drums=None, teacher_forcing=False)


__all__ = ["ConditionalDrumTransformer", "WINDOW_STEPS"]

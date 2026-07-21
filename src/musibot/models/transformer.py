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

from ..config import (
    FEATURE_DIM,
    N_CHANNELS,
    N_DRUM_CLASSES,
    N_STYLES,
    WINDOW_STEPS,
)
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
        self.style_emb = nn.Embedding(N_STYLES, hidden_dim)
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

    def _style_bias(self, style: torch.Tensor | None, B: int, device) -> torch.Tensor:
        """Per-clip style embedding ``[B, 1, hidden]``, broadcast over time."""
        if style is None:
            style = torch.zeros(B, dtype=torch.long, device=device)  # unknown
        return self.style_emb(style).unsqueeze(1)

    def encode(self, conditioning: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos(self.cond_proj(conditioning))
        x = x + self._style_bias(style, x.size(0), x.device)
        return self.encoder(x)

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
        threshold: float = 0.5,
        style: torch.Tensor | None = None,
        ss_prob: float = 0.0,
    ) -> torch.Tensor:
        memory = self.encode(conditioning, style)
        B, T, _ = conditioning.shape

        if teacher_forcing and drums is not None:
            # Shift right: position t sees ground-truth steps 0..t-1.
            start = torch.zeros(B, 1, N_DRUM_CLASSES, N_CHANNELS, device=drums.device)
            tgt_in = torch.cat([start, drums[:, :-1]], dim=1)
            if ss_prob > 0.0:
                tgt_in = self._mix_own(tgt_in, memory, threshold, ss_prob)
            return self._decode(tgt_in, memory)

        return self._generate(memory, T, threshold)

    def _mix_own(
        self, tgt_tf: torch.Tensor, memory: torch.Tensor, threshold: float, ss_prob: float
    ) -> torch.Tensor:
        """Scheduled sampling: swap a fraction ``ss_prob`` of the teacher-forced
        decoder inputs for the model's own (detached) hard predictions.

        A pure teacher-forced decoder learns to copy the ground-truth previous
        bar rather than read the conditioning, then collapses to a flat groove at
        inference (it never has ground truth to copy). Feeding it its own errors
        during training closes that gap. Two parallel passes (the first detached)
        approximate a full 128-step rollout at ~1.2x the teacher-forced cost.
        """
        with torch.no_grad():
            own = binarize(self._decode(tgt_tf, memory), threshold)
            own_in = torch.cat([tgt_tf[:, :1], own[:, :-1]], dim=1)  # re-shift right
        use_own = (
            torch.rand(tgt_tf.size(0), tgt_tf.size(1), 1, 1, device=tgt_tf.device) < ss_prob
        ).float()
        return use_own * own_in + (1.0 - use_own) * tgt_tf

    def _generate(self, memory: torch.Tensor, T: int, threshold: float = 0.5) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device
        generated = torch.zeros(B, 1, N_DRUM_CLASSES, N_CHANNELS, device=device)
        outputs = []
        for _ in range(T):
            step = self._decode(generated, memory)[:, -1]   # [B, 9, 3]
            outputs.append(step)
            generated = torch.cat([generated, binarize(step, threshold).unsqueeze(1)], dim=1)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(
        self,
        conditioning: torch.Tensor,
        threshold: float = 0.5,
        style: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        return self.forward(
            conditioning, drums=None, teacher_forcing=False, threshold=threshold, style=style
        )


__all__ = ["ConditionalDrumTransformer", "WINDOW_STEPS"]

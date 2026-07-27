"""Masked drum-infilling transformer (Phase 2 — the MAE/BERT analogy for drums).

A **bidirectional** transformer that reconstructs *masked* steps of the drum grid
from the surrounding (unmasked) drums plus the full audio conditioning. Unlike
the causal ``ConditionalDrumTransformer`` (which only sees the past), this attends
in both directions, so it can fill a gap in the MIDDLE of a song using what comes
*after* it — the capability behind "replace a break / fill a missing section".

- Training: mask whole bars at a random ratio; reconstruct only the masked steps
  (``MaskedDrumLoss``). Masking everything ≈ generation from scratch; masking a
  span ≈ infilling.
- Inference: mark a gap → mask those steps → ``infill`` predicts them from both
  sides + audio. (A single bidirectional pass; MaskGIT-style iterative decoding is
  a drop-in enhancement later.)

Shares the output contract with the other models: ``[B, T, 9, 3]`` — channel 0 is
a raw hit logit, 1 sigmoid velocity, 2 tanh offset.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import (
    CH_HIT,
    FEATURE_DIM,
    N_CHANNELS,
    N_DRUM_CLASSES,
    N_STYLES,
    STEPS_PER_BAR,
)
from .baseline_lstm import activate
from .transformer import PositionalEncoding

_FLAT = N_DRUM_CLASSES * N_CHANNELS  # 27


def make_span_mask(
    B: int, T: int, device, min_ratio: float = 0.15, max_ratio: float = 0.6,
    bar: int = STEPS_PER_BAR,
) -> torch.Tensor:
    """Per-sample mask over whole bars. Returns ``[B, T]`` bool (True = masked).

    Each sample masks a random fraction (in ``[min_ratio, max_ratio]``) of its
    bars, so the model learns to reconstruct gaps of varying size — from a single
    dropped bar up to most of the clip.
    """
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    n_bars = max(1, T // bar)
    for b in range(B):
        ratio = torch.empty(1, device=device).uniform_(min_ratio, max_ratio).item()
        n = max(1, round(ratio * n_bars))
        bars = torch.randperm(n_bars, device=device)[:n]
        for bi in bars.tolist():
            mask[b, bi * bar:(bi + 1) * bar] = True
    return mask


class MaskedDrumTransformer(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_layers: int = 4, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.pos = PositionalEncoding(hidden_dim)
        self.cond_proj = nn.Linear(FEATURE_DIM, hidden_dim)
        self.drum_proj = nn.Linear(_FLAT, hidden_dim)
        self.style_emb = nn.Embedding(N_STYLES, hidden_dim)
        # Learned embedding substituted at masked steps (the "[MASK]" token).
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        enc = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)         # over audio conditioning
        dec = nn.TransformerDecoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout,
                                         batch_first=True)
        self.decoder = nn.TransformerDecoder(dec, n_layers)         # bidirectional (no causal mask)
        self.head = nn.Linear(hidden_dim, _FLAT)

    def _style_bias(self, style: torch.Tensor | None, B: int, device) -> torch.Tensor:
        if style is None:
            style = torch.zeros(B, dtype=torch.long, device=device)
        return self.style_emb(style).unsqueeze(1)

    def encode(self, conditioning: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos(self.cond_proj(conditioning))
        x = x + self._style_bias(style, x.size(0), x.device)
        return self.encoder(x)

    def forward(self, conditioning: torch.Tensor, drums: torch.Tensor,
                mask: torch.Tensor, style: torch.Tensor | None = None) -> torch.Tensor:
        """``conditioning`` [B,T,F], ``drums`` [B,T,9,3], ``mask`` [B,T] bool."""
        B, T = drums.shape[:2]
        memory = self.encode(conditioning, style)
        d = self.drum_proj(drums.reshape(B, T, _FLAT))              # [B,T,H]
        tok = self.mask_token.view(1, 1, -1).expand(B, T, -1)
        d = torch.where(mask.unsqueeze(-1), tok, d)                 # swap in [MASK] at gaps
        d = self.pos(d) + self._style_bias(style, B, d.device)
        out = self.decoder(d, memory)                              # no tgt_mask -> bidirectional
        return activate(self.head(out).reshape(B, T, N_DRUM_CLASSES, N_CHANNELS))

    @torch.no_grad()
    def infill(self, conditioning: torch.Tensor, drums: torch.Tensor, mask: torch.Tensor,
               style: torch.Tensor | None = None, threshold: "float | torch.Tensor" = 0.5,
               iters: int = 8) -> torch.Tensor:
        """Reconstruct masked steps; keep known steps untouched. Returns [B,T,9,3]
        with a binarised hit channel.

        ``threshold`` may be a scalar or a per-class ``[9]`` vector (lower the bar
        for classes the model is shy about, e.g. hats/cymbals). ``iters > 1`` runs
        **MaskGIT-style iterative decoding**: each round predicts the still-masked
        steps, commits the most *confident* ones (they become context for the rest),
        and re-masks the remainder on a cosine schedule — sharper, more coherent
        fills, because low-confidence steps condition on already-committed structure.
        """
        self.eval()
        thr = threshold if torch.is_tensor(threshold) else torch.tensor(float(threshold))
        thr = thr.to(device=drums.device, dtype=torch.float32)     # scalar or [9], broadcasts over classes
        out = drums.clone()
        out[..., CH_HIT] = (out[..., CH_HIT] > 0.5).float()   # binarise known hits
        remaining = mask.clone()                              # [B, T] still-masked

        def commit(pred: torch.Tensor, idx: torch.Tensor, b: int) -> None:
            step = pred[b, idx].clone()                       # [k, 9, 3]
            step[..., CH_HIT] = (torch.sigmoid(pred[b, idx, :, CH_HIT]) > thr).float()
            out[b, idx] = step
            remaining[b, idx] = False

        if iters <= 1:
            pred = self.forward(conditioning, out, remaining, style)
            for b in range(out.size(0)):
                idx = remaining[b].nonzero(as_tuple=True)[0]
                if len(idx):
                    commit(pred, idx, b)
            return out

        total = mask.sum(1)                                   # [B] original masked count
        for it in range(iters):
            if not bool(remaining.any()):
                break
            pred = self.forward(conditioning, out, remaining, style)
            probs = torch.sigmoid(pred[..., CH_HIT])          # [B, T, 9]
            # Confidence = strongest hit probability at the step, so the groove
            # skeleton (the steps most likely to contain a hit) is committed first
            # and conditions the rest — rather than committing confident *silence*.
            conf = probs.amax(-1)                              # [B, T]
            conf = conf.masked_fill(~remaining, -1.0)
            keep_ratio = math.cos(math.pi / 2 * (it + 1) / iters)  # 1 -> 0 over the run
            for b in range(out.size(0)):
                nrem = int(remaining[b].sum())
                if nrem == 0:
                    continue
                keep = int(math.floor(keep_ratio * float(total[b])))
                n_commit = nrem if it == iters - 1 else max(1, nrem - keep)
                idx = torch.topk(conf[b], min(n_commit, nrem)).indices
                commit(pred, idx, b)
        return out


__all__ = ["MaskedDrumTransformer", "make_span_mask"]

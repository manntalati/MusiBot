"""Training loss for the drum grid.

Model outputs are ``[B, T, 9, 3]`` where channel 0 is a raw **hit logit** and
channels 1-2 are already-activated **velocity** (0..1) and **offset**
(-0.5..0.5). Loss = hit BCE + velocity/offset MSE, with the regression terms
masked to steps that actually contain a hit (predicting velocity for a silent
cell is meaningless).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CH_HIT, CH_OFFSET, CH_VELOCITY


class DrumLoss(nn.Module):
    def __init__(
        self,
        velocity_weight: float = 1.0,
        offset_weight: float = 1.0,
        hit_pos_weight: float = 1.5,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.offset_weight = offset_weight
        # Drum grids are sparse — most cells are silent — so up-weight hits.
        self.register_buffer("pos_weight", torch.tensor(hit_pos_weight))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        hit_logit = pred[..., CH_HIT]
        hit_target = target[..., CH_HIT]

        hit_loss = F.binary_cross_entropy_with_logits(
            hit_logit, hit_target, pos_weight=self.pos_weight
        )

        mask = hit_target  # only score velocity/offset where a hit exists
        denom = mask.sum().clamp(min=1.0)
        vel_loss = (((pred[..., CH_VELOCITY] - target[..., CH_VELOCITY]) ** 2) * mask).sum() / denom
        off_loss = (((pred[..., CH_OFFSET] - target[..., CH_OFFSET]) ** 2) * mask).sum() / denom

        total = hit_loss + self.velocity_weight * vel_loss + self.offset_weight * off_loss
        return {"total": total, "hit": hit_loss, "velocity": vel_loss, "offset": off_loss}

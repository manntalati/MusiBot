"""Both models must honour the [B, T, 9, 3] I/O contract and reduce the loss."""

import torch

from musibot.config import FEATURE_DIM, N_CHANNELS, N_DRUM_CLASSES, WINDOW_STEPS
from musibot.models.baseline_lstm import BaselineLSTM
from musibot.models.losses import DrumLoss
from musibot.models.transformer import ConditionalDrumTransformer

B, T = 2, WINDOW_STEPS


def _batch():
    torch.manual_seed(444)
    cond = torch.randn(B, T, FEATURE_DIM)
    drums = torch.zeros(B, T, N_DRUM_CLASSES, N_CHANNELS)
    drums[..., 0] = (torch.rand(B, T, N_DRUM_CLASSES) < 0.25).float()
    drums[..., 1] = torch.rand(B, T, N_DRUM_CLASSES) * drums[..., 0]
    return cond, drums


def _check_models():
    return [
        BaselineLSTM(hidden_dim=32, n_layers=1, dropout=0.0),
        ConditionalDrumTransformer(hidden_dim=32, n_layers=2, n_heads=4, dropout=0.0),
    ]


def test_teacher_forced_output_shape():
    cond, drums = _batch()
    for model in _check_models():
        out = model(cond, drums, teacher_forcing=True)
        assert out.shape == (B, T, N_DRUM_CLASSES, N_CHANNELS)


def test_autoregressive_generation_shape():
    cond, _ = _batch()
    for model in _check_models():
        out = model.generate(cond)
        assert out.shape == (B, T, N_DRUM_CLASSES, N_CHANNELS)


def test_loss_decreases_on_one_batch():
    cond, drums = _batch()
    for model in _check_models():
        loss_fn = DrumLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        first = None
        for _ in range(15):
            loss = loss_fn(model(cond, drums, teacher_forcing=True), drums)["total"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.detach())
        assert float(loss.detach()) < first

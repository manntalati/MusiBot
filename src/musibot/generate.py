"""Generate a drum MIDI track for a drumless song.

    python -m musibot.generate --song drumless.wav --checkpoint runs/transformer_best.pt \
        --config configs/model_transformer.yaml --out drums.mid

The song is windowed into 8-bar chunks (the training window length); each chunk
is generated autoregressively and the results are concatenated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch

from .config import SAMPLE_RATE, WINDOW_STEPS
from .data.features import extract_features
from .data.representation import grid_to_midi
from .train import build_model, load_config, resolve_device


def load_model(checkpoint: str, cfg, device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _window(features: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad and reshape ``[T, F]`` features into ``[n_windows, WINDOW_STEPS, F]``."""
    T, F = features.shape
    n_win = int(np.ceil(T / WINDOW_STEPS))
    padded = np.zeros((n_win * WINDOW_STEPS, F), dtype=np.float32)
    padded[:T] = features
    return padded.reshape(n_win, WINDOW_STEPS, F), T


@torch.no_grad()
def song_to_grid(model, audio: np.ndarray, sr: int, device) -> tuple[np.ndarray, float]:
    """Drumless audio -> generated drum grid ``[T, 9, 3]`` and its tempo."""
    features, tempo, _ = extract_features(audio, sr)
    windows, T = _window(features)
    cond = torch.from_numpy(windows).to(device)
    grid = model.generate(cond)                       # [n_win, WINDOW_STEPS, 9, 3]
    grid = grid.cpu().numpy().reshape(-1, grid.shape[-2], grid.shape[-1])
    return grid[:T], tempo


def generate_drums(
    song_path: str | Path,
    checkpoint: str,
    config: str,
    out_midi: str | Path = "drums.mid",
) -> Path:
    cfg = load_config(config)
    device = resolve_device(cfg.device)
    model = load_model(checkpoint, cfg, device)

    audio, sr = librosa.load(str(song_path), sr=SAMPLE_RATE, mono=True)
    grid, tempo = song_to_grid(model, audio, sr, device)

    pm = grid_to_midi(grid, tempo)
    out_midi = Path(out_midi)
    pm.write(str(out_midi))
    print(f"tempo={tempo:.1f} BPM  steps={len(grid)}  -> {out_midi}")
    return out_midi


def main():
    parser = argparse.ArgumentParser(description="Generate a drum track")
    parser.add_argument("--song", required=True, help="drumless song (wav/mp3)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="drums.mid")
    args = parser.parse_args()
    generate_drums(args.song, args.checkpoint, args.config, args.out)


if __name__ == "__main__":
    main()

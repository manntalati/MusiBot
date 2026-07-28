"""Per-step log-mel features for the learned ADT (drum transcription).

An isolated drum signal -> ``[n_steps, N_MELS]`` log-mel, max-pooled within each
16th-note step so a transient onset in a step survives the pooling. This is the
input to ``models.adt.DrumTranscriber``; the target is the aligned drum grid.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..config import HOP_LENGTH, SAMPLE_RATE
from .representation import step_duration

N_MELS = 64


def drum_mel_features(
    audio: np.ndarray, sr: int = SAMPLE_RATE, tempo: float = 120.0,
    n_steps: int | None = None, n_mels: int = N_MELS,
) -> np.ndarray:
    """Return ``[n_steps, n_mels]`` log-mel features aligned to the step grid."""
    dt = step_duration(tempo)
    if n_steps is None:
        n_steps = max(1, int(len(audio) / sr / dt))
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=HOP_LENGTH, fmax=sr / 2
    )
    mel = librosa.power_to_db(mel, ref=np.max)          # [n_mels, F], ~[-80, 0] dB
    frame_times = librosa.frames_to_time(np.arange(mel.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    feats = np.full((n_steps, n_mels), -80.0, np.float32)
    step_times = np.arange(n_steps) * dt
    for i, t in enumerate(step_times):
        lo = int(np.searchsorted(frame_times, t))
        hi = int(np.searchsorted(frame_times, t + dt))
        if hi > lo:
            feats[i] = mel[:, lo:hi].max(axis=1)
        elif lo < mel.shape[1]:
            feats[i] = mel[:, lo]
    return feats


__all__ = ["drum_mel_features", "N_MELS"]

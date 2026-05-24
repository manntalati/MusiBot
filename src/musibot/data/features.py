"""Per-step conditioning features extracted from a drumless song.

These features are the model's *input*: they tell it where the beats are, how
loud/busy the song is, what the harmony is doing, and where structural
sections change. The output is a ``[T, FEATURE_DIM]`` array aligned step-for-
step with the drum grid that ``representation.py`` produces.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..config import (
    BEATS_PER_BAR,
    FEATURE_DIM,
    HOP_LENGTH,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
)
from .representation import step_duration


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _to_steps(values: np.ndarray, frame_times: np.ndarray, step_times: np.ndarray) -> np.ndarray:
    """Interpolate a frame-rate feature onto the 16th-note step grid."""
    return np.interp(step_times, frame_times, values)


def estimate_tempo(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, np.ndarray]:
    """Return (tempo BPM, beat times in seconds)."""
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    return float(np.atleast_1d(tempo)[0]), beats


def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    tempo: float | None = None,
    n_sections: int = 8,
) -> tuple[np.ndarray, float, int]:
    """Extract conditioning features from a drumless mix.

    Returns ``(features [T, FEATURE_DIM], tempo, n_steps)``. The 16th-note grid
    is anchored at t=0 with step size derived from ``tempo``.
    """
    if tempo is None:
        tempo, _ = estimate_tempo(audio, sr)

    dt = step_duration(tempo)
    duration = len(audio) / sr
    n_steps = max(1, int(duration / dt))
    step_times = np.arange(n_steps) * dt

    # Frame-rate features.
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    onset = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH).mean(axis=0)
    frame_times = librosa.frames_to_time(
        np.arange(rms.shape[0]), sr=sr, hop_length=HOP_LENGTH
    )

    features = np.zeros((n_steps, FEATURE_DIM), dtype=np.float32)
    steps = np.arange(n_steps)

    # 0: beat phase, 1: downbeat flag — purely positional.
    features[:, 0] = (steps % STEPS_PER_BEAT) / STEPS_PER_BEAT
    features[:, 1] = (steps % STEPS_PER_BAR == 0).astype(np.float32)

    # 2-4: energy / onset / spectral contrast.
    features[:, 2] = _minmax(_to_steps(rms, frame_times, step_times))
    features[:, 3] = _minmax(_to_steps(onset, frame_times, step_times))
    features[:, 4] = _minmax(_to_steps(contrast, frame_times, step_times))

    # 5-16: 12-d chroma.
    for c in range(12):
        features[:, 5 + c] = _to_steps(chroma[c], frame_times, step_times)

    # 17th feature: structural section id, normalized.
    features[:, -1] = _section_ids(chroma, frame_times, step_times, n_sections)

    return features, tempo, n_steps


def _section_ids(
    chroma: np.ndarray, frame_times: np.ndarray, step_times: np.ndarray, k: int
) -> np.ndarray:
    """Assign each step a normalized structural-section index (verse/chorus...)."""
    if chroma.shape[1] < k:
        return np.zeros(len(step_times), dtype=np.float32)
    bounds = librosa.segment.agglomerative(chroma, k)
    bound_times = librosa.frames_to_time(bounds, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    section = np.searchsorted(bound_times, step_times)
    return (section / max(1, k)).astype(np.float32)


__all__ = ["estimate_tempo", "extract_features", "BEATS_PER_BAR"]

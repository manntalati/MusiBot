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
    N_BASE_FEATURES,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
)
from .mood import compute_mood
from .normalize import load_stats
from .representation import step_duration

# Quarter-note tempo band for PLP beat tracking (excludes the common 8th-note
# octave error for typical pop/rock; out-of-band songs fall back to beat_track).
BEAT_TEMPO_MIN = 70.0
BEAT_TEMPO_MAX = 180.0


def _to_steps(values: np.ndarray, frame_times: np.ndarray, step_times: np.ndarray) -> np.ndarray:
    """Interpolate a frame-rate feature onto the 16th-note step grid."""
    return np.interp(step_times, frame_times, values)


def estimate_tempo(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[float, np.ndarray]:
    """Return (tempo BPM, beat times in seconds)."""
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    return float(np.atleast_1d(tempo)[0]), beats


def beat_sync_grid(audio: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray | None, float]:
    """Beat-synchronous step times (seconds) following the song's tempo curve.

    Tracks beats (which speed up / slow down with the music) and subdivides each
    beat interval into ``STEPS_PER_BEAT`` 16th-note steps, so the grid stays
    locked to the music even when the tempo drifts. Step 0 is the first beat.
    Returns ``(None, tempo)`` when too few beats are found (caller falls back to a
    uniform grid). Works on drumless audio — beat tracking uses the onset envelope
    (bass / chords / vocals), not just drums.

    Uses **predominant local pulse** (``librosa.beat.plp``), which follows tempo
    changes far better than the dynamic-programming beat tracker (that one assumes a
    roughly constant tempo); falls back to ``beat_track`` then to a uniform grid.
    """
    # PLP constrained to a typical quarter-note tempo band, so it locks onto the
    # beat (not an 8th/16th subdivision — the free tracker hits the octave) while
    # still following tempo changes. The band excludes the common 2x octave error
    # for most pop/rock; genuinely fast/slow songs fall back to the DP tracker.
    pulse = librosa.beat.plp(y=audio, sr=sr, tempo_min=BEAT_TEMPO_MIN, tempo_max=BEAT_TEMPO_MAX)
    beat_frames = np.flatnonzero(librosa.util.localmax(pulse))
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) >= 2:
        tempo = 60.0 / float(np.median(np.diff(beats)))
    else:                                          # fall back to the DP tracker's beats
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, units="time")
        tempo = float(np.atleast_1d(tempo)[0])
        beats = np.atleast_1d(beats).astype(float)
    if len(beats) < 2:
        return None, tempo

    duration = len(audio) / sr
    times: list[float] = []
    for i in range(len(beats) - 1):
        b0, b1 = beats[i], beats[i + 1]
        s = (b1 - b0) / STEPS_PER_BEAT
        times.extend(b0 + k * s for k in range(STEPS_PER_BEAT))
    # Extend past the last beat at the final local rate, to the end of the audio.
    s = (beats[-1] - beats[-2]) / STEPS_PER_BEAT
    t = beats[-1]
    while t < duration:
        times.append(t)
        t += s
    return np.asarray(times, dtype=np.float32), tempo


def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    tempo: float | None = None,
    n_sections: int = 8,
    beat_sync: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Extract conditioning features from a drumless mix.

    Returns ``(features [T, FEATURE_DIM], tempo, step_times)`` where ``step_times``
    are the seconds of each 16th-note step. With ``beat_sync=False`` (training
    prep, where the MIDI tempo is exact) the grid is **uniform**. With
    ``beat_sync=True`` (inference) the grid follows the tracked beats, so the drums
    stay locked to a song that speeds up or slows down.
    """
    step_times = None
    if beat_sync:
        step_times, tempo = beat_sync_grid(audio, sr)

    if tempo is None:
        tempo, _ = estimate_tempo(audio, sr)
    if step_times is None:                         # uniform grid (default / fallback)
        dt = step_duration(tempo)
        n_steps = max(1, int(len(audio) / sr / dt))
        step_times = np.arange(n_steps) * dt
    n_steps = len(step_times)

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

    # 2-4: energy / onset / spectral contrast, in raw librosa units. These are
    # z-scored later against corpus statistics (data/normalize.py) rather than
    # min-maxed per clip, so the same physical loudness maps to the same feature
    # value at train and inference time.
    features[:, 2] = _to_steps(rms, frame_times, step_times)
    features[:, 3] = _to_steps(onset, frame_times, step_times)
    features[:, 4] = _to_steps(contrast, frame_times, step_times)

    # 5-16: 12-d chroma.
    for c in range(12):
        features[:, 5 + c] = _to_steps(chroma[c], frame_times, step_times)

    # Last base feature: structural section id, normalized.
    features[:, N_BASE_FEATURES - 1] = _section_ids(
        chroma, frame_times, step_times, n_sections
    )

    # Song-global mood scalars, computed from the base columns (pre-normalization)
    # and broadcast across every step. Same function + corpus stats used when
    # migrating already-prepared clips, so train/inference values line up.
    features[:, N_BASE_FEATURES:] = compute_mood(
        features[:, :N_BASE_FEATURES], tempo, load_stats()
    )

    return features, tempo, step_times


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

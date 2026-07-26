"""Song-global "Spotify-style" mood descriptors, derived locally from audio.

Spotify's ``audio-features`` endpoint (energy / valence / danceability / tempo)
was deprecated for new apps on 2024-11-27, so we compute analogous descriptors
ourselves from the same librosa features the model already conditions on. Three
bounded 0..1 scalars are produced:

    energy        loudness + onset activity, calibrated against the corpus
    valence       major/bright (high) vs. minor/dark (low), from chroma + contrast
    danceability  rhythmic drive and a danceable tempo

``compute_mood`` operates on the **base** conditioning columns (the first
``N_BASE_FEATURES`` produced by :func:`features.extract_features`), *before*
normalization, so the exact same numbers come out whether we are:
  - extracting features at inference (``features.py``),
  - or migrating an already-prepared clip (``scripts/add_mood_features.py``).

Both paths pass the corpus stats (``feature_stats.npz``) so the loudness/onset
z-scores mean the same thing at train and inference time.
"""

from __future__ import annotations

import numpy as np

from ..config import MOOD_NEUTRAL, NORM_FEATURE_INDICES

# Krumhansl-Kessler key profiles; correlating the mean chroma against all 12
# rotations of each tells us whether the song leans major (brighter -> higher
# valence) or minor (darker -> lower valence).
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Column layout within the base features (see config.FEATURE_NAMES).
_RMS, _ONSET, _CONTRAST = NORM_FEATURE_INDICES  # (2, 3, 4)
_CHROMA = slice(5, 17)  # 12 chroma dims


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _best_corr(chroma_centered: np.ndarray, profile: np.ndarray) -> float:
    """Best correlation of the mean chroma with a key profile over 12 rotations."""
    p = profile - profile.mean()
    best = -1.0
    for k in range(12):
        c = np.corrcoef(np.roll(chroma_centered, -k), p)[0, 1]
        if np.isfinite(c) and c > best:
            best = c
    return best


def _mode_score(mean_chroma: np.ndarray) -> float:
    """(major_fit - minor_fit) in ~[-1, 1]; >0 = major/brighter."""
    c = mean_chroma - mean_chroma.mean()
    if not np.any(c):
        return 0.0
    return _best_corr(c, _KS_MAJOR) - _best_corr(c, _KS_MINOR)


_NOTES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]


def detect_key(mean_chroma: np.ndarray) -> tuple[str, bool]:
    """Estimate musical key from a mean chroma vector -> (e.g. "A minor", is_major).

    Correlates the chroma against all 12 rotations of the major and minor
    Krumhansl-Kessler profiles and returns the best-fitting tonic + mode.
    """
    c = mean_chroma - mean_chroma.mean()
    if not np.any(c):
        return ("—", True)

    def scan(profile: np.ndarray) -> tuple[int, float]:
        p = profile - profile.mean()
        cors = [np.corrcoef(np.roll(c, -k), p)[0, 1] for k in range(12)]
        cors = np.nan_to_num(cors, nan=-1.0)
        k = int(np.argmax(cors))
        return k, float(cors[k])

    maj_k, maj_c = scan(_KS_MAJOR)
    min_k, min_c = scan(_KS_MINOR)
    if maj_c >= min_c:
        return (f"{_NOTES[maj_k]} major", True)
    return (f"{_NOTES[min_k]} minor", False)


def compute_mood(
    base_features: np.ndarray, tempo: float, stats: dict | None = None
) -> np.ndarray:
    """Map base conditioning ``[T, >=N_BASE_FEATURES]`` + tempo -> ``[energy, valence,
    danceability]`` as a float32 array of three 0..1 scalars.

    ``stats`` is the corpus normalization dict (``normalize.load_stats()``); when
    absent the loudness/onset terms fall back to uncalibrated raw values (fine for
    unit tests, but pass real stats for train/inference consistency). Clips with
    no audio content (drum-only / positional) return the neutral value.
    """
    cb = np.asarray(base_features, dtype=np.float64)
    rms = float(cb[:, _RMS].mean())
    onset = float(cb[:, _ONSET].mean())
    contrast = float(cb[:, _CONTRAST].mean())

    # No audio signal (positional/drum-only clip) -> neutral, don't invent a mood.
    if abs(rms) < 1e-8 and abs(onset) < 1e-8 and abs(contrast) < 1e-8:
        return np.full(3, MOOD_NEUTRAL, dtype=np.float32)

    if stats is not None:
        mean, std = stats["mean"], stats["std"]  # aligned to NORM_FEATURE_INDICES
        z_rms = (rms - mean[0]) / std[0]
        z_onset = (onset - mean[1]) / std[1]
        z_contrast = (contrast - mean[2]) / std[2]
    else:
        z_rms, z_onset, z_contrast = rms, onset, contrast

    energy = _sigmoid(0.6 * z_rms + 0.4 * z_onset)

    # Danceable tempo peaks around 120 BPM; combine with onset drive.
    tempo_bell = float(np.exp(-(((float(tempo) - 120.0) / 40.0) ** 2)))
    danceability = _sigmoid(0.5 * z_onset + 1.2 * (2.0 * tempo_bell - 1.0))

    valence = _sigmoid(2.5 * _mode_score(cb[:, _CHROMA].mean(axis=0)) + 0.4 * z_contrast)

    return np.array([energy, valence, danceability], dtype=np.float32)


__all__ = ["compute_mood", "detect_key"]

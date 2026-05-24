"""Automatic drum transcription: an isolated drum stem -> drum MIDI.

This is a deliberately simple onset-detection + spectral-band classifier. It is
used only for the *fine-tuning* tier (the user's own songs); the bulk of
training comes from already-symbolic MIDI (GMD / LMD), so transcription error
here does not pollute the main training signal.

A learned ADT model can be dropped in behind ``transcribe_drums`` later.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pretty_midi

from ..config import CLASS_TO_GM_NOTE, SAMPLE_RATE

# Frequency band edges (Hz) used to fingerprint each onset.
_LOW = (20, 150)        # kick energy
_MID = (150, 2000)      # snare / tom body
_HIGH = (6000, 11000)   # hi-hat / cymbal shimmer


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(spectrum[mask].sum())


def _classify_onset(window: np.ndarray, sr: int) -> int:
    """Return a drum class index for a short audio window around an onset."""
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    freqs = np.fft.rfftfreq(len(window), 1 / sr)

    low = _band_energy(spectrum, freqs, *_LOW)
    mid = _band_energy(spectrum, freqs, *_MID)
    high = _band_energy(spectrum, freqs, *_HIGH)
    total = low + mid + high + 1e-9

    # Sustain: cymbals ring out, kicks/snares decay fast.
    half = len(window) // 2
    sustain = float(np.abs(window[half:]).mean() / (np.abs(window[:half]).mean() + 1e-9))

    if low / total > 0.5:
        return 0  # kick
    if high / total > 0.55:
        return 5 if sustain > 0.5 else 6  # open vs closed hi-hat
    if high / total > 0.35 and sustain > 0.7:
        return 7  # crash
    return 1  # snare (mid-dominant default)


def transcribe_drums(audio: np.ndarray, sr: int = SAMPLE_RATE) -> pretty_midi.PrettyMIDI:
    """Transcribe a mono drum-stem signal into a single-instrument drum MIDI."""
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, backtrack=True, units="frames"
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)

    win = int(0.05 * sr)  # 50 ms classification window
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")

    for t in onset_times:
        start = int(t * sr)
        window = audio[start : start + win]
        if len(window) < win // 2:
            continue
        cls = _classify_onset(window, sr)

        frame = min(int(t * sr / 512), len(onset_env) - 1)
        strength = onset_env[frame] / (onset_env.max() + 1e-9)
        velocity = int(np.clip(40 + strength * 87, 1, 127))

        drum.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=CLASS_TO_GM_NOTE[cls],
                start=float(t),
                end=float(t) + 0.1,
            )
        )

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(drum)
    return pm


def transcribe_file(path: str | Path, out_midi: str | Path) -> Path:
    """Transcribe a drum-stem audio file and write the resulting MIDI."""
    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    pm = transcribe_drums(audio, sr)
    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_midi))
    return out_midi


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drum-stem transcription")
    parser.add_argument("drum_stem", help="isolated drum audio file")
    parser.add_argument("--out", default="drums.mid", help="output MIDI path")
    args = parser.parse_args()
    print("wrote", transcribe_file(args.drum_stem, args.out))

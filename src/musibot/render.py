"""Render a generated drum MIDI to audio and mix it onto the drumless song.

Requires a General-MIDI ``.sf2`` soundfont and FluidSynth. Point to a soundfont
with --soundfont or the ``MUSIBOT_SOUNDFONT`` environment variable.

    python -m musibot.render --song drumless.wav --drums drums.mid --out final.wav
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from .config import SAMPLE_RATE


def _soundfont(explicit: str | None) -> str | None:
    return explicit or os.environ.get("MUSIBOT_SOUNDFONT")


def render_midi(midi_path: str | Path, sr: int = SAMPLE_RATE, soundfont: str | None = None) -> np.ndarray:
    """Synthesize a MIDI file to a mono float32 array via FluidSynth."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    sf2 = _soundfont(soundfont)
    audio = pm.fluidsynth(fs=sr, sf2_path=sf2) if sf2 else pm.fluidsynth(fs=sr)
    return audio.astype(np.float32)


def overlay(song: np.ndarray, drums: np.ndarray, drum_gain: float = 0.8) -> np.ndarray:
    """Sum drum audio onto the song, length-matched, with light limiting."""
    n = max(len(song), len(drums))
    song = np.pad(song, (0, n - len(song)))
    drums = np.pad(drums, (0, n - len(drums)))
    mix = song + drum_gain * drums
    peak = np.abs(mix).max()
    return mix / peak if peak > 1.0 else mix


def render(
    song_path: str | Path,
    drums_midi: str | Path,
    out_wav: str | Path = "final.wav",
    soundfont: str | None = None,
    drum_gain: float = 0.8,
) -> Path:
    song, _ = librosa.load(str(song_path), sr=SAMPLE_RATE, mono=True)
    drums = render_midi(drums_midi, SAMPLE_RATE, soundfont)
    mix = overlay(song, drums, drum_gain)

    out_wav = Path(out_wav)
    sf.write(out_wav, mix, SAMPLE_RATE)
    print(f"wrote {out_wav}  ({len(mix) / SAMPLE_RATE:.1f}s)")
    return out_wav


def main():
    parser = argparse.ArgumentParser(description="Render + overlay drums")
    parser.add_argument("--song", required=True, help="drumless song")
    parser.add_argument("--drums", required=True, help="generated drum MIDI")
    parser.add_argument("--out", default="final.wav")
    parser.add_argument("--soundfont", default=None, help="path to a .sf2 soundfont")
    parser.add_argument("--drum-gain", type=float, default=0.8)
    args = parser.parse_args()
    render(args.song, args.drums, args.out, args.soundfont, args.drum_gain)


if __name__ == "__main__":
    main()

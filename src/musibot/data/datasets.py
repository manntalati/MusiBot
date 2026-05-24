"""Prepare every training tier into a uniform set of ``.npz`` clips.

Each processed clip stores:
    conditioning  float32 [T, FEATURE_DIM]   model input
    drums         float32 [T, 9, 3]          model target
    tempo         float                      BPM
    source        str                        which tier it came from

Tiers (see the implementation plan):
    gmd          Magenta Groove MIDI       drum-language pretraining
    lmd          Lakh MIDI multi-track     conditional supervision
    musdb18      MUSDB18 isolated stems    conditional, real audio
    user         the 5 songs in data/raw   fine-tuning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import yaml
from tqdm import tqdm

from ..config import (
    EXTERNAL_DIR,
    FEATURE_DIM,
    PROCESSED_DIR,
    RAW_SONGS_DIR,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
)
from .features import extract_features
from .representation import midi_to_grid, step_duration
from .separate import separate_song
from .transcribe import transcribe_drums


# --- helpers ----------------------------------------------------------------

def midi_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Best-effort tempo (BPM) for a MIDI file."""
    _, tempi = pm.get_tempo_changes()
    if len(tempi):
        return float(tempi[0])
    return float(pm.estimate_tempo() or 120.0)


def save_clip(name: str, source: str, conditioning: np.ndarray, drums: np.ndarray, tempo: float):
    """Trim both arrays to a common length and write one ``.npz`` clip."""
    T = min(len(conditioning), len(drums))
    if T < STEPS_PER_BAR:
        return
    out = PROCESSED_DIR / source
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{name}.npz",
        conditioning=conditioning[:T].astype(np.float32),
        drums=drums[:T].astype(np.float32),
        tempo=np.float32(tempo),
        source=source,
    )


def positional_conditioning(n_steps: int) -> np.ndarray:
    """Minimal conditioning (beat phase + downbeat) for drum-only data."""
    feats = np.zeros((n_steps, FEATURE_DIM), dtype=np.float32)
    steps = np.arange(n_steps)
    feats[:, 0] = (steps % STEPS_PER_BEAT) / STEPS_PER_BEAT
    feats[:, 1] = (steps % STEPS_PER_BAR == 0).astype(np.float32)
    return feats


def symbolic_conditioning(pm: pretty_midi.PrettyMIDI, tempo: float, n_steps: int) -> np.ndarray:
    """Conditioning features derived directly from non-drum MIDI tracks."""
    feats = positional_conditioning(n_steps)
    dt = step_duration(tempo)
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if not notes:
        return feats

    for note in notes:
        step = int(round(note.start / dt))
        end = int(round(note.end / dt))
        if not 0 <= step < n_steps:
            continue
        feats[step, 3] += 1.0                              # onset density
        pc = note.pitch % 12
        for s in range(step, min(end + 1, n_steps)):
            feats[s, 2] += note.velocity / 127.0           # energy
            feats[s, 5 + pc] += 1.0                        # chroma

    for col in (2, 3):
        m = feats[:, col].max()
        if m > 0:
            feats[:, col] /= m
    chroma = feats[:, 5:17]
    norm = chroma.sum(axis=1, keepdims=True)
    feats[:, 5:17] = np.divide(chroma, norm, out=np.zeros_like(chroma), where=norm > 0)
    return feats


# --- per-tier preparation ---------------------------------------------------

def prepare_gmd(root: Path) -> int:
    """Groove MIDI: every ``.mid`` becomes a drum-only clip (weak conditioning)."""
    count = 0
    for mid in tqdm(sorted(root.rglob("*.mid")), desc="gmd"):
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
            tempo = midi_tempo(pm)
            drums = midi_to_grid(pm, tempo)
            cond = positional_conditioning(len(drums))
            save_clip(mid.stem + f"_{count}", "gmd", cond, drums, tempo)
            count += 1
        except Exception:
            continue
    return count


def prepare_lmd(root: Path, limit: int | None = None) -> int:
    """Lakh MIDI: split drum vs non-drum tracks for clean conditional pairs."""
    count = 0
    files = sorted(root.rglob("*.mid"))
    for mid in tqdm(files[:limit], desc="lmd"):
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
            if not any(inst.is_drum and inst.notes for inst in pm.instruments):
                continue
            tempo = midi_tempo(pm)
            drums = midi_to_grid(pm, tempo)
            cond = symbolic_conditioning(pm, tempo, len(drums))
            save_clip(mid.stem + f"_{count}", "lmd", cond, drums, tempo)
            count += 1
        except Exception:
            continue
    return count


def prepare_musdb18(root: Path) -> int:
    """MUSDB18: use the supplied drum/accompaniment stems (no separation)."""
    count = 0
    for drum_wav in tqdm(sorted(root.rglob("drums.wav")), desc="musdb18"):
        try:
            song_dir = drum_wav.parent
            drum_audio, sr = librosa.load(str(drum_wav), sr=SAMPLE_RATE, mono=True)
            acc_audio = _load_accompaniment(song_dir, sr)
            _prepare_audio_pair(song_dir.name, "musdb18", acc_audio, drum_audio, sr)
            count += 1
        except Exception:
            continue
    return count


def prepare_user_songs(root: Path = RAW_SONGS_DIR) -> int:
    """The 5 source songs: Demucs-separate, then transcribe the drum stem."""
    count = 0
    for song in tqdm(sorted(root.glob("*.mp3")), desc="user"):
        try:
            stems = separate_song(song)
            sr = 44100
            drum_audio = librosa.resample(stems["drums"], orig_sr=sr, target_sr=SAMPLE_RATE)
            acc_audio = librosa.resample(stems["accompaniment"], orig_sr=sr, target_sr=SAMPLE_RATE)
            _prepare_audio_pair(song.stem, "user", acc_audio, drum_audio, SAMPLE_RATE)
            count += 1
        except Exception:
            continue
    return count


def _load_accompaniment(song_dir: Path, sr: int) -> np.ndarray:
    """Sum MUSDB18's non-drum stems into a single drumless mix."""
    mix = None
    for stem in ("bass.wav", "vocals.wav", "other.wav"):
        path = song_dir / stem
        if path.exists():
            audio, _ = librosa.load(str(path), sr=sr, mono=True)
            mix = audio if mix is None else mix[: len(audio)] + audio[: len(mix)]
    return mix if mix is not None else np.zeros(sr, dtype=np.float32)


def _prepare_audio_pair(name: str, source: str, acc: np.ndarray, drums: np.ndarray, sr: int):
    """Shared path: transcribe the drum stem, extract features from the mix."""
    cond, tempo, _ = extract_features(acc, sr)
    drum_pm = transcribe_drums(drums, sr)
    drum_grid = midi_to_grid(drum_pm, tempo, n_steps=len(cond))
    save_clip(name, source, cond, drum_grid, tempo)


# --- entrypoint -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare MusiBot training clips")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--lmd-limit", type=int, default=None, help="cap LMD files")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["datasets"]
    totals: dict[str, int] = {}

    if cfg.get("gmd"):
        totals["gmd"] = prepare_gmd(EXTERNAL_DIR / "gmd")
    if cfg.get("lmd"):
        totals["lmd"] = prepare_lmd(EXTERNAL_DIR / "lmd", limit=args.lmd_limit)
    if cfg.get("musdb18"):
        totals["musdb18"] = prepare_musdb18(EXTERNAL_DIR / "musdb18")
    if cfg.get("user_songs"):
        totals["user"] = prepare_user_songs()

    print("Prepared clips:", totals)
    print("Output:", PROCESSED_DIR)


if __name__ == "__main__":
    main()

"""Prepare every training tier into a uniform set of ``.npz`` clips.

Each processed clip stores:
    conditioning  float32 [T, FEATURE_DIM]   model input (raw magnitude dims;
                                             standardized at load by normalize.py)
    drums         float32 [T, 9, 3]          model target
    tempo         float                      BPM
    source        str                        which tier it came from
    style_id      int                        coarse genre token (config.STYLE_VOCAB)

Tiers (see the implementation plan):
    gmd          Magenta Groove MIDI       drum-language pretraining (positional cond.)
    lmd          Lakh MIDI, rendered       conditional supervision; drums read from the
                                           MIDI (ground-truth) and the drumless mix is
                                           synthesized in-memory for real audio features
                                           — no audio is written to disk
    musdb18      MUSDB18 isolated stems    conditional, real audio
    user         the 5 songs in data/raw   fine-tuning
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import yaml
from tqdm import tqdm

log = logging.getLogger("musibot.datasets")

# Audio file extensions accepted by prepare_song_dir.
AUDIO_EXTS = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg")

# LMD curation thresholds (see prepare_lmd_rendered).
MIN_DRUM_NOTES = 50          # must carry a real groove, not a stray hit
MIN_PITCHED_INSTRUMENTS = 2  # need harmony/rhythm to condition on
MIN_SECONDS = 30.0
MAX_SECONDS = 360.0

from ..config import (
    EXTERNAL_DIR,
    FEATURE_DIM,
    MOOD_NEUTRAL,
    N_BASE_FEATURES,
    PROCESSED_DIR,
    RAW_SONGS_DIR,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    UNKNOWN_STYLE_ID,
    canonical_style,
)
from ..render import _soundfont
from .features import extract_features
from .normalize import fit_stats, save_stats
from .representation import midi_to_grid
from .separate import separate_song
from .transcribe import transcribe_drums


# --- helpers ----------------------------------------------------------------

def midi_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Best-effort tempo (BPM) for a MIDI file."""
    _, tempi = pm.get_tempo_changes()
    if len(tempi):
        return float(tempi[0])
    return float(pm.estimate_tempo() or 120.0)


def save_clip(
    name: str,
    source: str,
    conditioning: np.ndarray,
    drums: np.ndarray,
    tempo: float,
    style_id: int = UNKNOWN_STYLE_ID,
):
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
        style_id=np.int64(style_id),
    )


def positional_conditioning(n_steps: int) -> np.ndarray:
    """Minimal conditioning (beat phase + downbeat) for drum-only data."""
    feats = np.zeros((n_steps, FEATURE_DIM), dtype=np.float32)
    steps = np.arange(n_steps)
    feats[:, 0] = (steps % STEPS_PER_BEAT) / STEPS_PER_BEAT
    feats[:, 1] = (steps % STEPS_PER_BAR == 0).astype(np.float32)
    # No audio to derive mood from -> neutral, matching mood.compute_mood.
    feats[:, N_BASE_FEATURES:] = MOOD_NEUTRAL
    return feats


def gmd_style_map(root: Path) -> dict[str, int]:
    """Map each GMD ``.mid`` (relative path) to a style id from ``info.csv``."""
    info = root / "info.csv"
    if not info.exists():
        return {}
    out: dict[str, int] = {}
    with info.open(newline="") as f:
        for row in csv.DictReader(f):
            mid = row.get("midi_filename")
            if mid:
                out[mid] = canonical_style(row.get("style"))
    return out


def render_non_drums(pm: pretty_midi.PrettyMIDI, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthesize only the non-drum instruments to a mono array, in memory.

    This is the disk-free substitute for downloading rendered-audio corpora: the
    drumless mix never touches disk — we extract features from it and let it be
    garbage-collected. Requires FluidSynth + a soundfont (``MUSIBOT_SOUNDFONT``).
    """
    non_drum = pretty_midi.PrettyMIDI()
    non_drum.instruments = [
        inst for inst in pm.instruments if not inst.is_drum and inst.notes
    ]
    sf2 = _soundfont(None)
    audio = non_drum.fluidsynth(fs=sr, sf2_path=sf2) if sf2 else non_drum.fluidsynth(fs=sr)
    return audio.astype(np.float32)


# --- per-tier preparation ---------------------------------------------------

def prepare_gmd(root: Path) -> int:
    """Groove MIDI: every ``.mid`` becomes a drum-only clip (weak conditioning).

    Drum-isolated, so conditioning is positional only — use for unconditional
    decoder pretraining. The clip still carries its real style id (from
    ``info.csv``) so the style embedding is exercised during pretraining too.
    """
    styles = gmd_style_map(root)
    count = 0
    for mid in tqdm(sorted(root.rglob("*.mid")), desc="gmd"):
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
            tempo = midi_tempo(pm)
            drums = midi_to_grid(pm, tempo)
            cond = positional_conditioning(len(drums))
            rel = str(mid.relative_to(root))
            save_clip(mid.stem + f"_{count}", "gmd", cond, drums, tempo, styles.get(rel, 0))
            count += 1
        except Exception as e:
            log.warning("gmd: skipped %s (%s)", mid, e)
    return count


def _lmd_is_usable(pm: pretty_midi.PrettyMIDI) -> bool:
    """Curate: a real groove, enough harmony to condition on, sane length."""
    drum_notes = sum(len(i.notes) for i in pm.instruments if i.is_drum)
    pitched = sum(1 for i in pm.instruments if not i.is_drum and i.notes)
    if drum_notes < MIN_DRUM_NOTES or pitched < MIN_PITCHED_INSTRUMENTS:
        return False
    return MIN_SECONDS <= pm.get_end_time() <= MAX_SECONDS


def prepare_lmd_rendered(root: Path, limit: int | None = None) -> int:
    """Lakh MIDI -> audio-consistent conditional clips, with no audio on disk.

    For each curated MIDI we (1) read the drum grid straight from the file —
    ground-truth targets, no transcription error — and (2) render *only* the
    non-drum tracks to an in-memory mix and run the **same** ``extract_features``
    pipeline used at inference. The MIDI tempo drives both the grid and the
    feature grid, so they stay aligned without relying on beat tracking.

    Near-duplicate arrangements (LMD is full of them) are dropped by content
    hash. The rendered audio is never written out, so the disk cost is just the
    ~10 kB feature/target tensors.
    """
    if _soundfont(None) is None:
        log.warning(
            "lmd: MUSIBOT_SOUNDFONT not set — relying on a system default soundfont; "
            "set it to a .sf2 (e.g. FluidR3_GM) if rendering fails."
        )
    seen: set[str] = set()
    count = 0
    files = sorted(root.rglob("*.mid"))
    for mid in tqdm(files, desc="lmd"):
        if limit is not None and count >= limit:
            break
        try:
            digest = hashlib.md5(mid.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)

            pm = pretty_midi.PrettyMIDI(str(mid))
            if not _lmd_is_usable(pm):
                continue

            tempo = midi_tempo(pm)
            audio = render_non_drums(pm, SAMPLE_RATE)
            if audio.size < SAMPLE_RATE:  # < 1s of sound -> nothing to condition on
                continue
            cond, _, _ = extract_features(audio, SAMPLE_RATE, tempo=tempo)
            drums = midi_to_grid(pm, tempo, n_steps=len(cond))
            save_clip(f"{mid.stem}_{count}", "lmd", cond, drums, tempo)
            count += 1
        except Exception as e:
            log.warning("lmd: skipped %s (%s)", mid, e)
    return count


def prepare_musdb18(root: Path) -> int:
    """MUSDB18: use the supplied drum/accompaniment stems (no separation)."""
    count = 0
    drum_files = sorted(root.rglob("drums.wav"))
    if not drum_files:
        log.warning("musdb18: no drums.wav under %s — download MUSDB18-HQ first", root)
    for drum_wav in tqdm(drum_files, desc="musdb18"):
        try:
            song_dir = drum_wav.parent
            drum_audio, sr = librosa.load(str(drum_wav), sr=SAMPLE_RATE, mono=True)
            acc_audio = _load_accompaniment(song_dir, sr)
            _prepare_audio_pair(song_dir.name, "musdb18", acc_audio, drum_audio, sr)
            count += 1
        except Exception as e:
            log.warning("musdb18: skipped %s (%s)", drum_wav.parent, e)
    return count


def prepare_song_dir(root: Path, source_name: str, style: str | None = None) -> int:
    """Demucs-separate every audio file under ``root`` and prepare a clip.

    Use this to grow the training set: drop a folder of MP3s/WAVs anywhere and
    point this at it, e.g. ``--song-dir ~/Music/rock --song-dir-name rock
    --style rock`` to tag the whole folder with a style.
    """
    files = sorted(p for ext in AUDIO_EXTS for p in root.glob(ext))
    if not files:
        log.warning("%s: no audio files in %s (looked for %s)", source_name, root, AUDIO_EXTS)
        return 0
    style_id = canonical_style(style)
    count = 0
    for song in tqdm(files, desc=source_name):
        try:
            stems = separate_song(song)
            drum_audio = librosa.resample(stems["drums"], orig_sr=44100, target_sr=SAMPLE_RATE)
            acc_audio = librosa.resample(stems["accompaniment"], orig_sr=44100, target_sr=SAMPLE_RATE)
            _prepare_audio_pair(song.stem, source_name, acc_audio, drum_audio, SAMPLE_RATE, style_id)
            count += 1
        except Exception as e:
            log.warning("%s: skipped %s (%s)", source_name, song, e)
    return count


def prepare_user_songs(root: Path = RAW_SONGS_DIR) -> int:
    """The 5 source songs: Demucs-separate, then transcribe the drum stem."""
    return prepare_song_dir(root, "user")


def _load_accompaniment(song_dir: Path, sr: int) -> np.ndarray:
    """Sum MUSDB18's non-drum stems into a single drumless mix."""
    mix = None
    for stem in ("bass.wav", "vocals.wav", "other.wav"):
        path = song_dir / stem
        if path.exists():
            audio, _ = librosa.load(str(path), sr=sr, mono=True)
            mix = audio if mix is None else mix[: len(audio)] + audio[: len(mix)]
    return mix if mix is not None else np.zeros(sr, dtype=np.float32)


def _prepare_audio_pair(
    name: str,
    source: str,
    acc: np.ndarray,
    drums: np.ndarray,
    sr: int,
    style_id: int = UNKNOWN_STYLE_ID,
):
    """Shared path: transcribe the drum stem, extract features from the mix."""
    cond, tempo, _ = extract_features(acc, sr)
    drum_pm = transcribe_drums(drums, sr)
    drum_grid = midi_to_grid(drum_pm, tempo, n_steps=len(cond))
    save_clip(name, source, cond, drum_grid, tempo, style_id)


# --- entrypoint -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare MusiBot training clips")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--lmd-limit", type=int, default=None, help="cap LMD files")
    parser.add_argument(
        "--song-dir",
        type=Path,
        default=None,
        help="additional folder of MP3/WAV songs to Demucs-separate and prepare",
    )
    parser.add_argument(
        "--song-dir-name",
        default="extra",
        help="tier name under data/processed/ for --song-dir output",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="style tag for --song-dir clips (e.g. rock, funk, jazz)",
    )
    parser.add_argument(
        "--norm-sources",
        nargs="*",
        default=None,
        help="tiers to fit feature-normalization stats over "
        "(default: the audio-consistent tiers lmd/musdb18/user)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())["datasets"]
    totals: dict[str, int] = {}

    if cfg.get("gmd"):
        totals["gmd"] = prepare_gmd(EXTERNAL_DIR / "gmd")
    if cfg.get("lmd"):
        totals["lmd"] = prepare_lmd_rendered(EXTERNAL_DIR / "lmd", limit=args.lmd_limit)
    if cfg.get("musdb18"):
        totals["musdb18"] = prepare_musdb18(EXTERNAL_DIR / "musdb18")
    if cfg.get("user_songs"):
        totals["user"] = prepare_user_songs()
    if args.song_dir:
        totals[args.song_dir_name] = prepare_song_dir(
            args.song_dir, args.song_dir_name, args.style
        )

    print("Prepared clips:", totals)

    # Fit corpus normalization over the audio-consistent tiers (the ones whose
    # magnitude features carry real signal). GMD's positional-only conditioning
    # is excluded so its zeros don't skew the statistics.
    norm_sources = args.norm_sources or [
        s for s in ("lmd", "musdb18", "user", args.song_dir_name)
        if (PROCESSED_DIR / s).exists() and any((PROCESSED_DIR / s).glob("*.npz"))
    ]
    if norm_sources:
        stats = fit_stats(norm_sources)
        path = save_stats(stats)
        print(f"Feature stats over {norm_sources} -> {path}")
        print(f"  mean={stats['mean']}  std={stats['std']}")
    else:
        print("No audio-consistent tiers present; skipped normalization stats.")
    print("Output:", PROCESSED_DIR)


if __name__ == "__main__":
    main()

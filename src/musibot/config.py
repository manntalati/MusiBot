"""Central configuration: paths, the rhythmic grid, and the drum class map.

Every module imports from here so the representation stays consistent across
data preparation, training, generation, and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_SONGS_DIR = DATA_DIR / "raw" / "songs"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "runs"

# --- Audio ------------------------------------------------------------------

SAMPLE_RATE = 22050           # feature extraction rate (librosa)
SEPARATION_SAMPLE_RATE = 44100  # Demucs operates at 44.1 kHz
HOP_LENGTH = 512              # ~23 ms frames at 22050 Hz

# --- Rhythmic grid ----------------------------------------------------------

STEPS_PER_BEAT = 4            # 16th-note resolution
BEATS_PER_BAR = 4             # assume 4/4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR  # 16
WINDOW_BARS = 8               # training window length
WINDOW_STEPS = STEPS_PER_BAR * WINDOW_BARS      # 128

# Per-step drum tensor channels.
CH_HIT = 0       # 0/1 — was the drum struck this step
CH_VELOCITY = 1  # 0..1 — normalized strike loudness
CH_OFFSET = 2    # -0.5..0.5 — microtiming, fraction of a step early/late
N_CHANNELS = 3

# --- Drum classes -----------------------------------------------------------

# Nine canonical classes (the Magenta Groove MIDI reduced taxonomy).
DRUM_CLASSES = [
    "kick",
    "snare",
    "tom_high",
    "tom_mid",
    "tom_low",
    "hat_open",
    "hat_closed",
    "crash",
    "ride",
]
N_DRUM_CLASSES = len(DRUM_CLASSES)

# General-MIDI percussion note -> class index. Used when reading drum MIDI.
GM_NOTE_TO_CLASS: dict[int, int] = {
    36: 0,  # kick
    35: 0,  # acoustic bass drum
    38: 1, 40: 1, 37: 1,            # snare head / rim / x-stick
    48: 2, 50: 2,                   # high tom
    45: 3, 47: 3,                   # mid tom
    43: 4, 58: 4, 41: 4,            # low / floor tom
    46: 5, 26: 5,                   # open hi-hat
    42: 6, 22: 6, 44: 6,            # closed hi-hat / pedal
    49: 7, 55: 7, 57: 7, 52: 7,     # crash / splash / china
    51: 8, 59: 8, 53: 8,            # ride / bell
}

# Class index -> representative GM note used when rendering generated drums.
CLASS_TO_GM_NOTE: list[int] = [36, 38, 48, 45, 43, 46, 42, 49, 51]

# --- Conditioning features --------------------------------------------------

# Order of per-step conditioning features produced by data/features.py.
FEATURE_NAMES = [
    "beat_phase",       # position within the beat, 0..1
    "is_downbeat",      # 1 on the first step of a bar
    "rms_energy",       # normalized loudness
    "onset_density",    # onset strength of the drumless mix
    "spectral_contrast",
    *[f"chroma_{i}" for i in range(12)],
    "section_id",       # normalized structural-section index
]
FEATURE_DIM = len(FEATURE_NAMES)  # 18


@dataclass
class TrainConfig:
    """Hyperparameters; overridable from configs/*.yaml."""

    model: str = "transformer"          # "transformer" or "lstm"
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-2
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    velocity_loss_weight: float = 1.0
    offset_loss_weight: float = 1.0
    grad_clip: float = 1.0
    seed: int = 444
    device: str = "cuda"
    overfit_one_batch: bool = False     # Phase 3 pipeline sanity check
    extra: dict = field(default_factory=dict)

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
    "rms_energy",       # loudness (raw; standardized by data/normalize.py)
    "onset_density",    # onset strength of the drumless mix (raw)
    "spectral_contrast",  # (raw)
    *[f"chroma_{i}" for i in range(12)],
    "section_id",       # normalized structural-section index
]
FEATURE_DIM = len(FEATURE_NAMES)  # 18

# The unbounded "magnitude" features whose scale is clip-dependent. These are
# emitted in raw librosa units and z-scored with corpus-level statistics
# (data/normalize.py) so the train and inference distributions line up. The
# remaining dims are already bounded (phase/downbeat/chroma/section) and pass
# through untouched.
NORM_FEATURE_INDICES = (2, 3, 4)
FEATURE_STATS_FILE = PROCESSED_DIR / "feature_stats.npz"

# --- Style conditioning -----------------------------------------------------

# Coarse genre/style token. A global per-clip label, embedded and broadcast
# across the conditioning sequence, so the model can pick a kit and groove
# idiom (a rock song and a bossa nova want very different drums). Index 0 is
# the catch-all for clips with no known style (e.g. rendered LMD, user songs).
STYLE_VOCAB = [
    "unknown",
    "rock",
    "pop",
    "punk",
    "metal",
    "funk",
    "soul",
    "hiphop",
    "jazz",
    "blues",
    "country",
    "reggae",
    "latin",
    "afrocuban",
    "neworleans",
    "gospel",
    "dance",
    "electronic",
    "world",
]
STYLE_TO_ID: dict[str, int] = {s: i for i, s in enumerate(STYLE_VOCAB)}
N_STYLES = len(STYLE_VOCAB)
UNKNOWN_STYLE_ID = STYLE_TO_ID["unknown"]

# Raw style strings that don't match a vocab entry are remapped here before
# falling back to "unknown".
_STYLE_ALIASES = {
    "highlife": "world",
    "middleeastern": "world",
    "afrobeat": "world",
    "samba": "latin",
    "brazilian": "latin",
    "bossanova": "latin",
    "hip-hop": "hiphop",
    "rnb": "soul",
    "r&b": "soul",
}


def canonical_style(raw: str | None) -> int:
    """Map a free-form style string (e.g. ``"neworleans/funk"``) to a style id.

    Uses the top-level token before any ``/`` and falls back to ``unknown``.
    """
    if not raw:
        return UNKNOWN_STYLE_ID
    top = raw.strip().lower().split("/")[0].replace(" ", "")
    top = _STYLE_ALIASES.get(top, top)
    return STYLE_TO_ID.get(top, UNKNOWN_STYLE_ID)


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
    # BCE positive-class weight for hits. The grid is sparse, so this trades
    # recall for precision. Keep it near 1-2: high values (the old default was
    # 8, tuned for a near-empty corpus) make the model fire common drums on
    # every step — recall ~1, precision in the floor, 5x too dense.
    hit_pos_weight: float = 1.5
    grad_clip: float = 1.0
    seed: int = 444
    device: str = "cuda"
    overfit_one_batch: bool = False     # Phase 3 pipeline sanity check
    patience: int = 15                  # epochs with no val improvement -> early stop
    extra: dict = field(default_factory=dict)

"""End-to-end pipeline: uploaded song -> generated drums you can actually hear.

This is the engine behind the Gradio app (``app.py``). Unlike
``scripts/render_all_songs.py`` (which conditions on the *full* song and layers
drums on top of the original ones), this does the musically correct thing:

  1. **Separate** the song with Demucs and keep the drumless *accompaniment*
     -- the exact conditioning distribution the model was trained on.
  2. **Analyze** the accompaniment into a "Spotify-style" song profile
     (tempo, energy, valence, danceability, key, instrumentalness) via
     ``data/mood.py`` -- the same descriptors the model now conditions on.
  3. **Generate** a drum grid conditioned on the accompaniment, optionally with
     the mood scalars overridden by the user (the mood sliders in the UI), using
     the same feedback sweep + collapse-guard as the batch script.
  4. **Mix** the synthesized drums back onto the drumless accompaniment.

The model + Demucs weights load once and cache at module scope; each uploaded
file's separation (the slow step) is cached too, so re-generating with different
mood/feedback settings is fast and reproducible.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .config import (
    CH_HIT,
    DRUM_CLASSES,
    N_BASE_FEATURES,
    N_CHANNELS,
    N_DRUM_CLASSES,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    WINDOW_STEPS,
    canonical_style,
)
from .data.features import extract_features
from .data.mood import detect_key
from .data.normalize import apply_stats, load_stats
from .data.representation import grid_to_midi
from .data.separate import SEPARATION_SR, separate_song
from .render import overlay, render_midi
from .train import build_model, load_config, resolve_device

CKPT = os.environ.get("MUSIBOT_CKPT", "runs/transformer_best.pt")
CONFIG = os.environ.get("MUSIBOT_CONFIG", "configs/model_transformer.yaml")
SF2 = os.environ.get("MUSIBOT_SOUNDFONT", "data/soundfonts/MuseScore_General.sf3")

# Autoregressive feedback thresholds swept in "auto" mode, least-aggressive
# first. See scripts/render_all_songs.py for the rationale.
FEEDBACK_SWEEP = [0.5, 0.35, 0.25]
RENDER_THR = 0.5      # output read-out threshold (fixed, matches training)
MIN_DENSITY = 0.20    # hits/step below this counts as a collapse

_snare_i = DRUM_CLASSES.index("snare")
_kick_i = DRUM_CLASSES.index("kick")
_CHROMA = slice(5, 17)  # 12 chroma dims within the base features

_STATE: dict = {}
# Demucs on MPS is non-deterministic and slow; cache each uploaded file's
# drumless accompaniment (+ its vocal-derived instrumentalness) so analysis and
# repeat generations are reproducible and instant. Bounded to a few uploads.
_SEP_CACHE: dict = {}
_SEP_CACHE_MAX = 4


def _engine() -> dict:
    """Lazily load + cache the model, device and corpus stats."""
    if "model" not in _STATE:
        cfg = load_config(CONFIG)
        device = resolve_device(cfg.device)
        model = build_model(cfg).to(device)
        ckpt = torch.load(CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _STATE.update(
            model=model, device=device, stats=load_stats(), epoch=ckpt.get("epoch", "?")
        )
    return _STATE


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _separate(song_path: str | Path) -> dict:
    """Drumless accompaniment (22.05 kHz, 0.97 peak) + instrumentalness, cached.

    Keyed by path + size + mtime so re-uploading a changed file re-separates.
    """
    p = Path(song_path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, int(st.st_mtime))
    if key not in _SEP_CACHE:
        stems = separate_song(song_path)
        accomp = librosa.resample(
            stems["accompaniment"], orig_sr=SEPARATION_SR, target_sr=SAMPLE_RATE
        ).astype(np.float32)
        # Summing non-drum stems can overshoot [-1, 1] by an arbitrary amount;
        # peak-normalise so ``drum_gain`` means the same thing across songs.
        peak = float(np.abs(accomp).max())
        if peak > 0:
            accomp = accomp * (0.97 / peak)
        # Instrumentalness: how little of the mix is vocals (1 = instrumental).
        v, a = _rms(stems.get("vocals", np.zeros(1))), _rms(stems["accompaniment"])
        instrumentalness = 1.0 - v / (v + a + 1e-9)
        if len(_SEP_CACHE) >= _SEP_CACHE_MAX:
            _SEP_CACHE.pop(next(iter(_SEP_CACHE)))
        _SEP_CACHE[key] = {"accomp": accomp, "instrumentalness": float(instrumentalness)}
    return _SEP_CACHE[key]


def _window(features: np.ndarray) -> tuple[torch.Tensor, int]:
    T, F = features.shape
    n = int(np.ceil(T / WINDOW_STEPS))
    padded = np.zeros((n * WINDOW_STEPS, F), np.float32)
    padded[:T] = features
    return torch.from_numpy(padded.reshape(n, WINDOW_STEPS, F)), T


def _snare_beat24(hits: np.ndarray, T: int) -> float:
    nb = (T // STEPS_PER_BAR) * STEPS_PER_BAR
    if not nb:
        return 0.0
    pos = hits[:nb, _snare_i].reshape(-1, STEPS_PER_BAR).sum(0)
    return float(pos[[4, 12]].sum() / (pos.sum() + 1e-9))


# Backbone rows shown in the UI step-sequencer (label -> drum class).
_SEQ_ROWS = [("Kick", "kick"), ("Snare", "snare"), ("Hat", "hat_closed"),
             ("Tom", "tom_mid"), ("Ride", "ride")]


def _display_pattern(grid: np.ndarray) -> list[tuple[str, list[bool]]]:
    """Reduce a generated grid ``[T, 9, 3]`` to one representative 16-step bar
    of backbone rows, for the UI sequencer. Picks the busiest bar."""
    hits = grid[:, :, CH_HIT] > 0.5
    T = len(hits)
    nb = T // STEPS_PER_BAR
    if nb == 0:
        bar = np.zeros((STEPS_PER_BAR, hits.shape[1]), bool)
        bar[: len(hits)] = hits[:STEPS_PER_BAR]
    else:
        acts = [hits[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR].sum() for b in range(nb)]
        b = int(np.argmax(acts))
        bar = hits[b * STEPS_PER_BAR:(b + 1) * STEPS_PER_BAR]
    out = []
    for label, cls in _SEQ_ROWS:
        ci = DRUM_CLASSES.index(cls)
        out.append((label, [bool(bar[s, ci]) for s in range(STEPS_PER_BAR)]))
    return out


def analyze_song(song_path: str | Path) -> dict:
    """Separate (cached) + return the song's mood/energy profile for the UI."""
    accomp = _separate(song_path)["accomp"]
    feats, tempo, T = extract_features(accomp, SAMPLE_RATE)  # base + auto mood
    energy, valence, danceability = (float(x) for x in feats[0, N_BASE_FEATURES:])
    key_name, is_major = detect_key(feats[:, _CHROMA].mean(axis=0))
    return {
        "tempo": float(tempo),
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "key": key_name,
        "is_major": is_major,
        "instrumentalness": _separate(song_path)["instrumentalness"],
        "steps": int(T),
    }


@torch.no_grad()
def _generate_grid(
    accomp_22k: np.ndarray,
    feedback: str | float,
    style: str | None,
    mood_override: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, float, str]:
    """Drumless mix (22.05 kHz) -> binarised drum grid ``[T, 9, 3]``, tempo, report.

    ``mood_override`` replaces the auto-computed (energy, valence, danceability)
    scalars, letting the UI sliders steer the groove. ``feedback`` is ``"auto"``
    (sweep + collapse-guard) or a fixed float threshold.
    """
    eng = _engine()
    model, device = eng["model"], eng["device"]

    feats, tempo, _ = extract_features(accomp_22k, SAMPLE_RATE)
    if mood_override is not None:
        feats[:, N_BASE_FEATURES:] = np.asarray(mood_override, np.float32)
    feats = apply_stats(feats, eng["stats"])
    cond, T = _window(feats)
    cond = cond.to(device)
    style_t = torch.full((cond.size(0),), canonical_style(style), dtype=torch.long, device=device)

    sweep = FEEDBACK_SWEEP if feedback == "auto" else [float(feedback)]
    takes = []  # (fb, density, snare24, logit_grid, hits)
    lines = [f"tempo={tempo:.0f} BPM   steps={T}"]
    if mood_override is not None:
        lines.append("mood [E,V,D]=" + str(tuple(round(m, 2) for m in mood_override)))
    for fb in sweep:
        g = model.generate(cond, threshold=fb, style=style_t)
        g = g.cpu().numpy().reshape(-1, N_DRUM_CLASSES, N_CHANNELS)[:T]
        probs = 1.0 / (1.0 + np.exp(-g[:, :, CH_HIT]))
        hits = probs > RENDER_THR
        dens = hits.sum() / max(1, T)
        s24 = _snare_beat24(hits, T)
        takes.append((fb, dens, s24, g, hits))
        lines.append(
            f"feedback={fb}: density={dens:.2f} hits/step  "
            f"snare2&4={s24:.2f}  kick/step={hits[:, _kick_i].mean():.2f}  "
            f"total={int(hits.sum())}"
        )

    ok = [t for t in takes if t[1] >= MIN_DENSITY]
    fb, dens, s24, g, hits = ok[0] if ok else takes[-1]
    flag = "" if ok else "  (still collapsed -> likely out-of-distribution)"
    lines.append(f"-> chosen feedback={fb}  density={dens:.2f}  snare2&4={s24:.2f}{flag}")

    out_grid = g.copy()
    out_grid[:, :, CH_HIT] = hits.astype(np.float32)
    return out_grid, tempo, "\n".join(lines)


def process_song(
    song_path: str | Path,
    mood: tuple[float, float, float] | None = None,
    feedback: str | float = "auto",
    drum_gain: float = 1.0,
    style: str | None = None,
    workdir: str | Path | None = None,
) -> dict:
    """Run the full upload -> separate -> generate -> mix pipeline.

    ``mood`` is an optional (energy, valence, danceability) override for the
    generation; ``None`` uses the values auto-detected from the audio.
    """
    if not Path(SF2).exists():
        raise FileNotFoundError(f"soundfont not found: {SF2} (set MUSIBOT_SOUNDFONT)")
    work = Path(workdir or tempfile.mkdtemp(prefix="musibot_"))
    work.mkdir(parents=True, exist_ok=True)
    name = Path(song_path).stem

    accomp = _separate(song_path)["accomp"]
    grid, tempo, report = _generate_grid(accomp, feedback, style, mood_override=mood)

    midi_path = work / f"drums_{name}.mid"
    grid_to_midi(grid, tempo).write(str(midi_path))
    drums = render_midi(str(midi_path), SAMPLE_RATE, SF2)
    mix = overlay(accomp, drums, drum_gain=drum_gain)

    return {
        "mix": (SAMPLE_RATE, mix),
        "drums": (SAMPLE_RATE, drums),
        "accompaniment": (SAMPLE_RATE, accomp),
        "midi_path": str(midi_path),
        "tempo": tempo,
        "report": report,
        "pattern": _display_pattern(grid),
    }

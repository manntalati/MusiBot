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
    CH_VELOCITY,
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
from .data.representation import grid_to_midi, midi_to_grid, step_duration
from .data.separate import SEPARATION_SR, separate_song
from .data.transcribe import transcribe_drums
from .data.transcribe_learned import adt_available, transcribe_drums_learned
from .models.masked_transformer import MaskedDrumTransformer
from .render import overlay, render_midi
from .train import build_model, load_config, resolve_device

CKPT = os.environ.get("MUSIBOT_CKPT", "runs/transformer_best.pt")
CONFIG = os.environ.get("MUSIBOT_CONFIG", "configs/model_transformer.yaml")
MASKED_CKPT = os.environ.get("MUSIBOT_MASKED_CKPT", "runs/masked_best.pt")
MASKED_CONFIG = os.environ.get("MUSIBOT_MASKED_CONFIG", "configs/model_masked.yaml")
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
_MASKED: dict = {}     # lazily-loaded masked-infill model (Phase 2)
# Demucs on MPS is non-deterministic and slow; cache each uploaded file's
# drumless accompaniment (+ its vocal-derived instrumentalness) so analysis and
# repeat generations are reproducible and instant. Bounded to a few uploads.
_SEP_CACHE: dict = {}
_SEP_CACHE_MAX = 4
# Phase-1 style imitation: transcribed fingerprint of the song's OWN drums,
# cached per (file, tempo) so analysis and generation share one transcription.
_REF_CACHE: dict = {}


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
        # Keep the original drum stem (22.05 kHz) for Phase-1 style imitation.
        drums = librosa.resample(
            stems["drums"], orig_sr=SEPARATION_SR, target_sr=SAMPLE_RATE
        ).astype(np.float32)
        if len(_SEP_CACHE) >= _SEP_CACHE_MAX:
            _SEP_CACHE.pop(next(iter(_SEP_CACHE)))
        _SEP_CACHE[key] = {
            "accomp": accomp,
            "drums": drums,
            "instrumentalness": float(instrumentalness),
        }
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


# UI step-sequencer rows -> the drum classes each one aggregates (a row lights
# if ANY of its classes fire, so e.g. open+closed hats both show on the Hat row).
_SEQ_ROWS = [("Kick", ["kick"]), ("Snare", ["snare"]),
             ("Hat", ["hat_closed", "hat_open"]),
             ("Tom", ["tom_high", "tom_mid", "tom_low"]),
             ("Ride", ["ride", "crash"])]


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
    for label, classes in _SEQ_ROWS:
        cis = [DRUM_CLASSES.index(c) for c in classes]
        out.append((label, [bool(bar[s, cis].any()) for s in range(STEPS_PER_BAR)]))
    return out


def _fingerprint(grid: np.ndarray) -> dict:
    """Summarize a drum grid ``[T, 9, 3]`` into a style fingerprint."""
    hits = grid[:, :, CH_HIT] > 0.5
    T = max(1, len(hits))
    active = int(hits.sum())
    mean_vel = float(grid[:, :, CH_VELOCITY][hits].mean()) if active else 0.0
    class_density = {DRUM_CLASSES[c]: float(hits[:, c].sum() / T) for c in range(N_DRUM_CLASSES)}
    kit = [k for k, d in sorted(class_density.items(), key=lambda kv: -kv[1]) if d > 0.02][:4]
    return {
        "density": float(active / T),
        "snare_beat24": _snare_beat24(hits, len(hits)),
        "kick_per_step": float(hits[:, _kick_i].mean()),
        "mean_velocity": mean_vel,
        "class_density": class_density,
        "kit": kit,
        "pattern": _display_pattern(grid),
        # Enough hits to be a real groove worth imitating (not silence/artifacts).
        "has_drums": bool(active >= max(4, T * 0.02)),
    }


def _transcribe_grid(drum_audio: np.ndarray, tempo: float, n_steps: int | None,
                     step_times: np.ndarray | None = None) -> np.ndarray:
    """Drum signal -> grid ``[T, 9, 3]``. Uses the learned ADT when trained
    (all 9 classes, honours ``step_times``), else the heuristic transcriber."""
    if adt_available():
        try:
            return transcribe_drums_learned(
                drum_audio, SAMPLE_RATE, tempo, n_steps, step_times=step_times
            )
        except Exception:  # noqa: BLE001 — never let the ADT break the pipeline
            pass
    return midi_to_grid(transcribe_drums(drum_audio, SAMPLE_RATE), tempo, n_steps=n_steps)


def reference_fingerprint(song_path: str | Path, tempo: float) -> dict:
    """Transcribe the song's OWN drum stem into a style fingerprint (cached).

    This is the Phase-1 "take ideas from the current song" signal: the original
    drummer's density, backbone placement, kit and velocity feel.
    """
    p = Path(song_path)
    st = p.stat()
    key = (str(p.resolve()), st.st_size, int(st.st_mtime), round(float(tempo), 2))
    if key not in _REF_CACHE:
        grid = _transcribe_grid(_separate(song_path)["drums"], tempo, None)
        _REF_CACHE[key] = _fingerprint(grid)
    return _REF_CACHE[key]


def suggested_controls(fp: dict) -> tuple[float, float, float]:
    """Fingerprint -> (energy, danceability, target_density) to imitate the song.

    Valence stays audio-derived (drums don't carry it). Energy blends the
    original's busyness + how hard it's hit; danceability tracks how squarely it
    lands on the backbeat; target_density is the original's density, which the
    generator matches directly (the reliable knob for "as busy as the original").
    """
    dens = fp["density"]
    energy = float(np.clip(0.6 * min(dens / 1.2, 1.0) + 0.4 * fp["mean_velocity"], 0.15, 0.95))
    danceability = float(np.clip(0.35 + 0.6 * fp["snare_beat24"], 0.2, 0.95))
    target_density = float(np.clip(dens, 0.1, 2.0))
    return energy, danceability, target_density


def analyze_song(song_path: str | Path) -> dict:
    """Separate (cached) + return the song's mood/energy profile for the UI."""
    entry = _separate(song_path)
    feats, tempo, step_times = extract_features(entry["accomp"], SAMPLE_RATE)  # base + auto mood
    energy, valence, danceability = (float(x) for x in feats[0, N_BASE_FEATURES:])
    key_name, is_major = detect_key(feats[:, _CHROMA].mean(axis=0))
    return {
        "tempo": float(tempo),
        "energy": energy,
        "valence": valence,
        "danceability": danceability,
        "key": key_name,
        "is_major": is_major,
        "instrumentalness": entry["instrumentalness"],
        "steps": len(step_times),
        "reference": reference_fingerprint(song_path, tempo),
    }


@torch.no_grad()
def _generate_grid(
    accomp_22k: np.ndarray,
    feedback: str | float,
    style: str | None,
    mood_override: tuple[float, float, float] | None = None,
    target_density: float | None = None,
) -> tuple[np.ndarray, float, str, np.ndarray]:
    """Drumless mix (22.05 kHz) -> binarised drum grid ``[T, 9, 3]``, tempo, report,
    step_times.

    ``mood_override`` replaces the auto-computed (energy, valence, danceability)
    scalars, letting the UI sliders steer the groove. ``feedback`` is ``"auto"``
    (sweep + collapse-guard) or a fixed float threshold. ``target_density`` (Phase-1
    style imitation) picks the swept take whose density best matches the original.
    Features use a **beat-synchronous** grid so the drums stay locked to a
    tempo-varying song; ``step_times`` maps each step back to seconds for rendering.
    """
    eng = _engine()
    model, device = eng["model"], eng["device"]

    feats, tempo, step_times = extract_features(accomp_22k, SAMPLE_RATE, beat_sync=True)
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

    if target_density is not None:
        # Style imitation: match the original drummer's busyness.
        fb, dens, s24, g, hits = min(takes, key=lambda t: abs(t[1] - target_density))
        lines.append(f"-> imitate: target density~{target_density:.2f}  chosen feedback={fb}  "
                     f"density={dens:.2f}  snare2&4={s24:.2f}")
    else:
        ok = [t for t in takes if t[1] >= MIN_DENSITY]
        fb, dens, s24, g, hits = ok[0] if ok else takes[-1]
        flag = "" if ok else "  (still collapsed -> likely out-of-distribution)"
        lines.append(f"-> chosen feedback={fb}  density={dens:.2f}  snare2&4={s24:.2f}{flag}")

    out_grid = g.copy()
    out_grid[:, :, CH_HIT] = hits.astype(np.float32)
    return out_grid, tempo, "\n".join(lines), step_times[:len(out_grid)]


def process_song(
    song_path: str | Path,
    mood: tuple[float, float, float] | None = None,
    feedback: str | float = "auto",
    drum_gain: float = 1.0,
    style: str | None = None,
    target_density: float | None = None,
    workdir: str | Path | None = None,
) -> dict:
    """Run the full upload -> separate -> generate -> mix pipeline.

    ``mood`` is an optional (energy, valence, danceability) override for the
    generation; ``None`` uses the values auto-detected from the audio.
    ``target_density`` (Phase-1 imitation) matches the original drummer's busyness.
    """
    if not Path(SF2).exists():
        raise FileNotFoundError(f"soundfont not found: {SF2} (set MUSIBOT_SOUNDFONT)")
    work = Path(workdir or tempfile.mkdtemp(prefix="musibot_"))
    work.mkdir(parents=True, exist_ok=True)
    name = Path(song_path).stem

    accomp = _separate(song_path)["accomp"]
    grid, tempo, report, step_times = _generate_grid(
        accomp, feedback, style, mood_override=mood, target_density=target_density
    )

    midi_path = work / f"drums_{name}.mid"
    grid_to_midi(grid, tempo, step_times=step_times).write(str(midi_path))
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


def _masked_engine() -> dict:
    """Lazily load + cache the masked drum-infilling model (Phase 2)."""
    if "model" not in _MASKED:
        cfg = load_config(MASKED_CONFIG)
        device = resolve_device(cfg.device)
        model = MaskedDrumTransformer(cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.dropout)
        ckpt = torch.load(MASKED_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        _MASKED.update(model=model, device=device, epoch=ckpt.get("epoch", "?"))
    return _MASKED


def masked_available() -> bool:
    return Path(MASKED_CKPT).exists()


# The model under-fires on non-backbone classes; lower their hit threshold so
# hats / cymbals / toms actually appear in fills (subtracted from the base
# threshold, floored at 0.08). Kick + snare keep the base threshold.
_CLASS_THR_DELTA = {
    "hat_closed": 0.10, "hat_open": 0.10, "ride": 0.10, "crash": 0.08,
    "tom_high": 0.06, "tom_mid": 0.06, "tom_low": 0.06,
}


def _class_thresholds(base: float) -> np.ndarray:
    """Per-class hit thresholds ``[9]`` from a base (backbone) threshold."""
    thr = np.full(N_DRUM_CLASSES, base, np.float32)
    for cls, delta in _CLASS_THR_DELTA.items():
        thr[DRUM_CLASSES.index(cls)] = max(0.08, base - delta)
    return thr


@torch.no_grad()
def infill_break(
    song_path: str | Path,
    start_sec: float,
    end_sec: float,
    style: str | None = None,
    drum_gain: float = 1.0,
    threshold: float = 0.55,
    iters: int = 8,
    workdir: str | Path | None = None,
) -> dict:
    """Reconstruct the drums in ``[start_sec, end_sec)`` with the masked model.

    The song's own drums (transcribed) provide the surrounding context; the model
    fills the masked span from both sides + the audio. Output = the original song
    with our generated drums laid into the marked range only.
    """
    if not masked_available():
        raise FileNotFoundError(f"masked model not found: {MASKED_CKPT} — train it first")
    if not Path(SF2).exists():
        raise FileNotFoundError(f"soundfont not found: {SF2}")
    eng = _masked_engine()
    model, device = eng["model"], eng["device"]
    work = Path(workdir or tempfile.mkdtemp(prefix="musibot_infill_"))
    work.mkdir(parents=True, exist_ok=True)
    name = Path(song_path).stem

    entry = _separate(song_path)
    feats, tempo, step_times = extract_features(entry["accomp"], SAMPLE_RATE, beat_sync=True)
    T = len(step_times)
    known = _transcribe_grid(entry["drums"], tempo, T, step_times=step_times)

    # Map the requested seconds to step indices via the (possibly non-uniform)
    # beat-synchronous step times.
    s = max(0, int(np.searchsorted(step_times, start_sec)))
    e = min(T, int(np.searchsorted(step_times, end_sec)))
    if e <= s:
        raise ValueError("infill range is empty (end must be after start, within the song)")

    mask = np.zeros(T, dtype=bool)
    mask[s:e] = True
    feats = apply_stats(feats, load_stats())

    # Window into WINDOW_STEPS chunks and infill each (each window keeps its own
    # surrounding drums + audio as context).
    n = int(np.ceil(T / WINDOW_STEPS))
    P = n * WINDOW_STEPS
    fp = np.zeros((P, feats.shape[1]), np.float32); fp[:T] = feats
    kp = np.zeros((P, N_DRUM_CLASSES, N_CHANNELS), np.float32); kp[:T] = known
    mp = np.zeros(P, dtype=bool); mp[:T] = mask
    cond = torch.from_numpy(fp.reshape(n, WINDOW_STEPS, -1)).to(device)
    dgrid = torch.from_numpy(kp.reshape(n, WINDOW_STEPS, N_DRUM_CLASSES, N_CHANNELS)).to(device)
    mten = torch.from_numpy(mp.reshape(n, WINDOW_STEPS)).to(device)
    style_t = torch.full((n,), canonical_style(style), dtype=torch.long, device=device)

    thr = torch.from_numpy(_class_thresholds(threshold)).to(device)   # per-class [9]
    filled = model.infill(cond, dgrid, mten, style_t, threshold=thr, iters=iters)
    filled = filled.cpu().numpy().reshape(-1, N_DRUM_CLASSES, N_CHANNELS)[:T]

    # Render ONLY the reconstructed span; lay it over the original song audio.
    gap = np.zeros_like(filled)
    gap[s:e] = filled[s:e]
    midi_path = work / f"infill_{name}.mid"
    grid_to_midi(gap, tempo, step_times=step_times).write(str(midi_path))
    gap_audio = render_midi(str(midi_path), SAMPLE_RATE, SF2)
    original, _ = librosa.load(str(song_path), sr=SAMPLE_RATE, mono=True)
    mix = overlay(original, gap_audio, drum_gain=drum_gain)

    gap_hits = int((filled[s:e, :, CH_HIT] > 0.5).sum())
    t0, t1 = float(step_times[s]), float(step_times[min(e, T - 1)])
    return {
        "mix": (SAMPLE_RATE, mix),
        "gap_drums": (SAMPLE_RATE, gap_audio),
        "midi_path": str(midi_path),
        "pattern": _display_pattern(filled[s:e]),
        "range": (round(t0, 2), round(t1, 2)),
        "report": f"filled {e - s} steps ({t0:.1f}-{t1:.1f}s)  hits={gap_hits}  tempo={tempo:.0f}",
    }

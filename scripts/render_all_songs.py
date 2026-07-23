#!/usr/bin/env python
"""Generate + render drum tracks for every song in data/raw/songs using the
retrained transformer, in one shot.

    .venv/bin/python scripts/render_all_songs.py

Writes drums_<song>.mid and final_<song>.wav into renders/. Prints per-song
diagnostics (density + snare-on-beat share across a few output thresholds) so we
can judge groove quality and tune without a second round-trip.

Notes on the operating point: the model was retrained with scheduled sampling
whose own-prediction feedback was binarised at 0.5, so 0.5 is the matched
inference threshold for BOTH the autoregressive feedback and the final read-out
(this is the point where training measured gen_f1 ~0.5). We still sweep the
output threshold for reporting.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "src")
from musibot.config import (  # noqa: E402
    CH_HIT,
    CH_OFFSET,
    CH_VELOCITY,
    DRUM_CLASSES,
    N_CHANNELS,
    N_DRUM_CLASSES,
    SAMPLE_RATE,
    STEPS_PER_BAR,
    WINDOW_STEPS,
    canonical_style,
)
from musibot.data.features import extract_features  # noqa: E402
from musibot.data.normalize import apply_stats, load_stats  # noqa: E402
from musibot.data.representation import grid_to_midi  # noqa: E402
from musibot.render import overlay, render_midi  # noqa: E402
from musibot.train import build_model, load_config, resolve_device  # noqa: E402

CKPT = "runs/transformer_best.pt"
CONFIG = "configs/model_transformer.yaml"
SONGS = sorted(glob.glob("data/raw/songs/*.mp3"))
SF2 = os.environ.get("MUSIBOT_SOUNDFONT", "data/soundfonts/MuseScore_General.sf3")
OUT = Path("renders")
# The autoregressive FEEDBACK threshold is the main knob for collapse: a high
# value starves an underconfident (out-of-distribution / fast-tempo) song back
# toward silence. Sweep it per song and pick the least-aggressive value that
# still avoids collapse. Output read-out stays fixed at 0.5.
FEEDBACK_SWEEP = [0.5, 0.35, 0.25]
RENDER_THR = 0.5         # output threshold used for the rendered take
MIN_DENSITY = 0.20       # hits/step below this is treated as a collapse
STYLE = None             # None -> "unknown" (how the LMD corpus was conditioned)

snare_i = DRUM_CLASSES.index("snare")
kick_i = DRUM_CLASSES.index("kick")


def window(features: np.ndarray):
    T, F = features.shape
    n = int(np.ceil(T / WINDOW_STEPS))
    pad = np.zeros((n * WINDOW_STEPS, F), np.float32)
    pad[:T] = features
    return torch.from_numpy(pad.reshape(n, WINDOW_STEPS, F)), T


def snare_beat24(hits: np.ndarray, T: int) -> float:
    nb = (T // STEPS_PER_BAR) * STEPS_PER_BAR
    if not nb:
        return 0.0
    pos = hits[:nb, snare_i].reshape(-1, STEPS_PER_BAR).sum(0)
    return float(pos[[4, 12]].sum() / (pos.sum() + 1e-9))


def main() -> None:
    if not SONGS:
        sys.exit("no songs found in data/raw/songs/*.mp3")
    if not Path(SF2).exists():
        sys.exit(f"soundfont not found: {SF2} (set MUSIBOT_SOUNDFONT)")
    OUT.mkdir(exist_ok=True)

    cfg = load_config(CONFIG)
    device = resolve_device(cfg.device)
    model = build_model(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = load_stats()
    print(f"loaded {CKPT} (epoch {ckpt.get('epoch','?')})  device={device}  soundfont={SF2}")

    for song in SONGS:
        name = Path(song).stem
        audio, sr = librosa.load(song, sr=SAMPLE_RATE, mono=True)
        feats, tempo, _ = extract_features(audio, sr)
        feats = apply_stats(feats, stats)
        cond, T = window(feats)
        cond = cond.to(device)
        style_t = torch.full(
            (cond.size(0),), canonical_style(STYLE), dtype=torch.long, device=device
        )
        print(f"\n=== {name}  tempo={tempo:.0f} BPM  steps={T} ===")
        takes = []  # (fb, density, snare24, grid[T,9,3], hits[T,9])
        for fb in FEEDBACK_SWEEP:
            with torch.no_grad():
                g = model.generate(cond, threshold=fb, style=style_t)
            g = g.cpu().numpy().reshape(-1, N_DRUM_CLASSES, N_CHANNELS)[:T]
            probs = 1.0 / (1.0 + np.exp(-g[:, :, CH_HIT]))  # sigmoid of hit logit
            hits = probs > RENDER_THR
            dens = hits.sum() / max(1, T)
            s24 = snare_beat24(hits, T)
            takes.append((fb, dens, s24, g, hits))
            print(
                f"  feedback={fb}: density={dens:.2f} hits/step  snare2&4={s24:.2f}  "
                f"kick/step={hits[:, kick_i].mean():.2f}  total={int(hits.sum())}"
            )

        # Pick the least-aggressive feedback (highest fb) that clears MIN_DENSITY;
        # if every take collapses, take the most-aggressive (lowest fb) as best rescue.
        ok = [t for t in takes if t[1] >= MIN_DENSITY]
        fb, dens, s24, g, hits = ok[0] if ok else takes[-1]
        flag = "" if ok else "  (still collapsed -> likely out-of-distribution)"
        print(f"  -> chosen feedback={fb}  density={dens:.2f}  snare2&4={s24:.2f}{flag}")

        # Keep the model's velocity/microtiming, binarise the hit channel.
        out_grid = g.copy()
        out_grid[:, :, CH_HIT] = hits.astype(np.float32)
        pm = grid_to_midi(out_grid, tempo)
        midi_path = OUT / f"drums_{name}.mid"
        pm.write(str(midi_path))

        drums = render_midi(str(midi_path), SAMPLE_RATE, SF2)
        mix = overlay(audio, drums, drum_gain=0.8)
        wav_path = OUT / f"final_{name}.wav"
        sf.write(wav_path, mix, SAMPLE_RATE)
        print(f"  rendered {midi_path.name} + {wav_path.name}  ({len(mix)/SAMPLE_RATE:.0f}s)")

    print(f"\nDone. Listen to {OUT}/final_*.wav")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare ADT training clips: FluidSynth-render GMD drum MIDI -> (mel, grid).

    .venv/bin/python scripts/prepare_adt.py                # all GMD
    .venv/bin/python scripts/prepare_adt.py --limit 40     # a subset (smoke test)

GMD is drum-only and uses the full GM percussion map, so rendering each file
gives clean drum audio paired with a ground-truth 9-class grid. Writes
``data/processed/adt/*.npz`` with ``mel [T, N_MELS]`` + ``drums [T, 9, 3]``.
Needs a soundfont (``MUSIBOT_SOUNDFONT`` or the bundled MuseScore .sf3).
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pretty_midi
from tqdm import tqdm

sys.path.insert(0, "src")
from musibot.config import PROCESSED_DIR, SAMPLE_RATE, STEPS_PER_BAR  # noqa: E402
from musibot.data.adt_features import drum_mel_features  # noqa: E402
from musibot.data.datasets import midi_tempo  # noqa: E402
from musibot.data.representation import midi_to_grid  # noqa: E402

log = logging.getLogger("musibot.adt")
SF2 = os.environ.get("MUSIBOT_SOUNDFONT", "data/soundfonts/MuseScore_General.sf3")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ADT (mel, grid) clips from GMD")
    ap.add_argument("--gmd", default="data/external/gmd")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not Path(SF2).exists():
        sys.exit(f"soundfont not found: {SF2} (set MUSIBOT_SOUNDFONT)")
    out = PROCESSED_DIR / "adt"
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{args.gmd}/**/*.mid", recursive=True))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"no GMD midis under {args.gmd}")
    print(f"rendering {len(files)} GMD drum files -> {out}")

    count = 0
    for mid in tqdm(files, desc="adt"):
        try:
            pm = pretty_midi.PrettyMIDI(mid)
            if not any(i.is_drum and i.notes for i in pm.instruments):
                continue
            tempo = midi_tempo(pm)
            audio = pm.fluidsynth(fs=SAMPLE_RATE, sf2_path=SF2).astype(np.float32)
            if audio.size < SAMPLE_RATE // 2:
                continue
            grid = midi_to_grid(pm, tempo)
            mel = drum_mel_features(audio, SAMPLE_RATE, tempo, n_steps=len(grid))
            T = min(len(mel), len(grid))
            if T < STEPS_PER_BAR:
                continue
            np.savez_compressed(
                out / f"{Path(mid).stem}_{count}.npz",
                mel=mel[:T].astype(np.float32),
                drums=grid[:T].astype(np.float32),
                tempo=np.float32(tempo),
            )
            count += 1
        except Exception as e:  # noqa: BLE001
            log.warning("skipped %s (%s)", mid, e)
    print(f"wrote {count} ADT clips to {out}")


if __name__ == "__main__":
    main()

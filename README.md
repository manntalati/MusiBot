# MusiBot

MusiBot generates a realistic AI drum track for a song that has **no drums**,
then overlays it back onto the song.

Drums are modeled **symbolically** — discrete hits on a 16th-note grid — rather
than as raw audio. A sequence model conditioned on musical features of the
drumless song predicts a drum grid, which is rendered to audio with a drum
soundfont and mixed in.

## Pipeline

```
drumless song (mp3/wav)
  -> beat / tempo tracking            -> 16th-note grid
  -> per-step conditioning features   (energy, onsets, chroma, downbeats, section)
  -> conditional sequence model       -> drum grid [T, 9 classes, 3 channels]
  -> MIDI -> soundfont render         -> beat-aligned overlay -> final WAV
```

The 3 channels per drum class are **hit** (0/1), **velocity** (0..1), and
**microtiming offset** (-0.5..0.5 of a step) — velocity and microtiming are
what make the output groove like a human instead of a quantized machine.

## Layout

```
configs/         training + data YAML configs
data/raw/songs/  the source MP3s
data/external/   downloaded datasets (GMD / LMD / MUSDB18)
data/processed/  cached feature/grid tensors
src/musibot/     the package
  data/          separation, transcription, features, representation, datasets
  models/        baseline LSTM, conditional Transformer, losses
  train.py generate.py render.py evaluate.py
app/demo.py      Gradio upload-and-listen demo
notebooks/       exploration + Colab GPU training
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
```

`fluidsynth` and `ffmpeg` must be installed on the system (e.g.
`brew install fluid-synth ffmpeg`).

## Usage

The recommended workflow uses **audio-consistent** training data only —
MUSDB18 (150 songs with isolated stems) + your own songs. Train and inference
share the same librosa feature pipeline, so the model generalizes.

```bash
# 1. Datasets. GMD is auto-downloaded; MUSDB18-HQ must be fetched manually
#    from Zenodo (see scripts/download_data.sh) and unpacked into
#    data/external/musdb18/ so each song dir has drums.wav / bass.wav / etc.
bash scripts/download_data.sh

# 2. Prepare clips (configs/data.yaml has LMD off by default).
python -m musibot.data.datasets --config configs/data.yaml

# Optional: drop any folder of MP3s in for more training data.
python -m musibot.data.datasets --song-dir ~/Music/rock --song-dir-name rock

# 3. Train on audio-consistent tiers only (skip gmd/lmd).
python -m musibot.train --config configs/model_transformer.yaml --sources musdb18 user

# 4. Evaluate — sweep thresholds to find the operating point.
python -m musibot.evaluate \
  --checkpoint runs/transformer_best.pt --config configs/model_transformer.yaml \
  --sources musdb18 user --threshold-sweep

# 5. Inference on a drumless song. Use whichever threshold gave the best F1.
python -m musibot.generate --song path/to/drumless.wav --checkpoint runs/transformer_best.pt \
  --config configs/model_transformer.yaml --threshold 0.3 --out drums.mid
python -m musibot.render --song path/to/drumless.wav --drums drums.mid --out final.wav

# 6. Demo.
python app/demo.py --checkpoint runs/transformer_best.pt --config configs/model_transformer.yaml
```

## Status

Under active development — see the implementation plan for phased milestones.

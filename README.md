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
what make the output groove like a human instead of a quantized machine. A
coarse **style token** (rock / funk / jazz / …) conditions the whole clip so
the model picks an idiomatic kit and groove. See [DATA.md](DATA.md) for how the
training corpus is built and why.

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

Every conditional tier is **audio-consistent**: features come from real audio
through the same librosa pipeline used at inference. The primary source is
**Lakh MIDI rendered on the fly** — the non-drum tracks are synthesized to an
in-memory mix for features while the drum grid is read straight from the MIDI
(ground-truth targets, no transcription). No rendered audio is written to disk,
so a 178k-file corpus costs only the ~10 kB tensors. MUSDB18 + your own songs
anchor it with real recordings. See [DATA.md](DATA.md) for the rationale.

```bash
# 0. A General-MIDI soundfont is required to render LMD. Point to it:
export MUSIBOT_SOUNDFONT=/path/to/FluidR3_GM.sf2

# 1. Datasets. GMD is auto-downloaded; MUSDB18-HQ is optional (fetch from
#    Zenodo per scripts/download_data.sh into data/external/musdb18/).
bash scripts/download_data.sh

# 2. Prepare clips. --lmd-limit caps how many LMD files to render (rendering all
#    178k is unnecessary; the curation filters already drop junk/duplicates).
#    This also fits + saves corpus feature-normalization stats.
python -m musibot.data.datasets --config configs/data.yaml --lmd-limit 20000

# Optional: drop any folder of MP3s in for more training data, tagged by style.
python -m musibot.data.datasets --song-dir ~/Music/rock --song-dir-name rock --style rock

# 3. Pretrain the drum prior on GMD (decoder), then train conditionally on the
#    audio-consistent tiers (rendered LMD + MUSDB18 + your songs).
python -m musibot.train --config configs/model_transformer.yaml --sources lmd musdb18 user

# 4. Evaluate — sweep thresholds to find the operating point.
python -m musibot.evaluate \
  --checkpoint runs/transformer_best.pt --config configs/model_transformer.yaml \
  --sources lmd musdb18 user --threshold-sweep

# 5. Inference on a drumless song. --style is optional; threshold from step 4.
python -m musibot.generate --song path/to/drumless.wav --checkpoint runs/transformer_best.pt \
  --config configs/model_transformer.yaml --threshold 0.3 --style rock --out drums.mid
python -m musibot.render --song path/to/drumless.wav --drums drums.mid --out final.wav

# 6. Demo.
python app/demo.py --checkpoint runs/transformer_best.pt --config configs/model_transformer.yaml
```

> Retrain after this upgrade: the checkpoints in `runs/` predate the style
> embedding and raw/normalized features, so they won't load or generalize.

## Status

Under active development — see the implementation plan for phased milestones.

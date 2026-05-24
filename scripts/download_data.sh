#!/usr/bin/env bash
# Fetch the external training datasets into data/external/.
# Datasets are large; comment out tiers you do not need.
set -euo pipefail

EXTERNAL="$(cd "$(dirname "$0")/.." && pwd)/data/external"
mkdir -p "$EXTERNAL"

# --- Groove MIDI Dataset (drum-language pretraining) ------------------------
if [ ! -d "$EXTERNAL/gmd" ]; then
  echo "Downloading Groove MIDI Dataset..."
  curl -L -o "$EXTERNAL/gmd.zip" \
    "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
  unzip -q "$EXTERNAL/gmd.zip" -d "$EXTERNAL"
  mv "$EXTERNAL/groove" "$EXTERNAL/gmd"
  rm "$EXTERNAL/gmd.zip"
fi

# --- Lakh MIDI Dataset (conditional supervision) ----------------------------
if [ ! -d "$EXTERNAL/lmd" ]; then
  echo "Downloading Lakh MIDI Dataset (lmd_full, ~1.6 GB)..."
  curl -L -o "$EXTERNAL/lmd.tar.gz" \
    "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
  mkdir -p "$EXTERNAL/lmd"
  tar -xzf "$EXTERNAL/lmd.tar.gz" -C "$EXTERNAL/lmd" --strip-components=1
  rm "$EXTERNAL/lmd.tar.gz"
fi

# --- MUSDB18 (real-audio stems) ---------------------------------------------
# MUSDB18 requires accepting a license on Zenodo. Download manually from
#   https://zenodo.org/record/3338373  (MUSDB18-HQ, wav stems)
# and extract so that each song folder under data/external/musdb18/ contains
# drums.wav / bass.wav / vocals.wav / other.wav.
if [ ! -d "$EXTERNAL/musdb18" ]; then
  echo "MUSDB18: download manually — see comment in this script."
fi

echo "Done. Datasets in $EXTERNAL"

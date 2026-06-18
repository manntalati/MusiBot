# Data pipeline

This document explains how MusiBot's training data is built and why — i.e. what
it takes to make the model generate *good* drums rather than generic or empty
patterns. The model architecture was never the bottleneck; the data was.

## Why the old setup generated poorly

The task is a mapping: **drumless-audio features → drum grid**. Learning it
needs many examples where (a) the input features come from the *same* pipeline
used at inference and (b) the target drums are accurate and beat-aligned. The
old corpus had almost none of that:

| Tier | Conditioning | Target | Problem |
|---|---|---|---|
| `gmd` | positional only (beat phase + downbeat) | real MIDI | carries no info about any song → trains an *unconditional* prior |
| `lmd` | **symbolic** (MIDI velocity/pitch sums) | real MIDI | feature distribution ≠ librosa inference features; was disabled |
| `musdb18` | real librosa audio | toy transcription | never downloaded (0 clips) |
| `user` | real librosa audio | toy transcription | only 5 songs |

So the conditional model effectively trained on ~5 real pairs plus a large pile
of null-conditioned drum loops. With no usable input→output signal it learned to
ignore the conditioning and emit a generic groove or collapse to silence (hence
the threshold-sweep hack).

Four concrete defects:

1. **No real conditional data at scale.** 5 usable songs.
2. **Wrong targets on audio tiers.** The band-energy `transcribe_drums` only
   emits ~5 of 9 classes, fakes velocity, and mistimes hits.
3. **Train/inference feature mismatch.** `symbolic_conditioning` (MIDI) and
   `extract_features` (librosa) produced incomparable vectors for the same 18
   slots; and per-clip min-max rescaled every clip against its own range, so the
   same loudness mapped to different values at train vs. inference.
4. **Unreliable beat grid.** A single global `librosa.beat_track` tempo anchored
   at t=0, so `is_downbeat` was rarely the real downbeat and microtiming was
   meaningless.

## What this pipeline does instead

### Rendered LMD as the primary conditional corpus (disk-free)

Slakh2100 — the obvious "stems + ground-truth MIDI" dataset — is ~100 GB and is
itself just Lakh MIDI rendered to audio. We already have all 178k LMD MIDI files
locally, so we render them ourselves, transiently:

For each curated MIDI (`prepare_lmd_rendered`):
- read the **drum grid straight from the MIDI** → ground-truth targets, no
  transcription error (fixes defect #2 for this tier);
- render **only the non-drum instruments** to an in-memory mix with FluidSynth
  (`render_non_drums`) and run the **same `extract_features`** used at inference
  → matched feature distribution (fixes #3);
- drive both the feature grid and the drum grid from the **MIDI tempo**, so they
  stay aligned without beat tracking (sidesteps #4 for this tier).

The rendered audio is never written to disk — only the ~10 kB feature/target
tensors are. A 20k-clip corpus is ~250 MB. Curation drops junk: ≥50 drum notes,
≥2 pitched instruments, 30 s–6 min, and content-hash de-duplication (LMD is full
of near-identical arrangements).

Caveat: FluidSynth audio has a timbre gap to real recordings (the same gap
Slakh has). Close it by anchoring with real audio — MUSDB18 and your own
songs — and fine-tuning on those last.

### Corpus-level feature normalization

`extract_features` now emits the magnitude features (`rms_energy`,
`onset_density`, `spectral_contrast`) in **raw librosa units**. `normalize.py`
fits per-dimension mean/std **once over the training split** (saved to
`data/processed/feature_stats.npz`) and `DrumDataset` / `generate` apply the
exact same stats. Identical physical loudness now maps to the same feature value
at train and inference time. The bounded dims (phase / downbeat / chroma /
section) pass through untouched.

### Style conditioning

A coarse genre token (`config.STYLE_VOCAB`) is stored per clip and embedded as a
global bias broadcast across the conditioning sequence (`style_emb` in both
models). GMD's `info.csv` provides real style labels for free; `--song-dir
--style` tags your own folders; rendered LMD and transcribed songs default to
`unknown`. This lets the model pick an idiomatic kit/groove — a rock song and a
bossa nova want very different drums.

## Tiers today

| Tier | Role | Conditioning | Target |
|---|---|---|---|
| `gmd` | unconditional decoder pretraining | positional + style | real MIDI |
| `lmd` | **primary conditional supervision** | rendered-audio librosa | ground-truth MIDI |
| `musdb18` | real-audio anchor | real librosa | transcription* |
| `user` | fine-tuning | real librosa | transcription* |

\* still uses the heuristic transcriber — the next upgrade is a learned ADT
(train on E-GMD, which ships paired audio+MIDI) to fix the real-audio targets.

## Known follow-ups (not yet implemented)

- **Learned drum transcription** to replace `transcribe_drums` for the
  `musdb18`/`user` tiers.
- **Downbeat/meter tracking** (madmom/BeatNet) for the real-audio tiers and at
  inference, so the grid origin is a true downbeat.
- **Richer conditioning**: bass onsets/pitch, real structure segmentation,
  vocal-presence flag.

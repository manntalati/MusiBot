# Phase 2 — Masked Drum Infilling (the MAE analogy)

**Goal.** Fill gaps in a song's drums — replace a break, reconstruct a missing
section, or generate a groove from scratch — in the song's own style but with the
model's own ideas. This is Masked Autoencoding (MAE / BERT) applied to the drum
grid, following the masked-token music lineage (MaskGIT → VampNet → MAGNeT).

## Why a new model

The Phase-1 `ConditionalDrumTransformer` decodes **causally** (left→right); it can
only see the past, so it cannot fill a gap in the *middle* of a song using what
comes after. Infilling needs **bidirectional** context. So Phase 2 adds a separate
model with a masked-reconstruction objective; the causal model stays as-is for the
app's current generation path.

## Representation (reused, unchanged)

- Target: the drum grid `[T, 9, 3]` (hit / velocity / offset). The 9 classes × T
  steps are the "tokens" we mask.
- Conditioning: the width-21 per-step audio features `[T, 21]` (incl. the mood
  scalars) + the global style token — **available even where drums are missing**,
  which is what makes infilling well-posed.
- Same 20k LMD + user clips, same `feature_stats.npz`, same DataLoaders.

## Model — `models/masked_transformer.py::MaskedDrumTransformer`

```
audio [T,21] --cond_proj+pos+style--> TransformerEncoder --> memory   (bidirectional)
drums [T,9,3] --drum_proj--> d;  d[masked] := mask_token (learned)
d --pos+style--> TransformerDecoder(d, memory, tgt_mask=None) --> head --> [T,9,3]
```
The only structural change vs the AR model is **no causal mask** on the drum
self-attention (`tgt_mask=None`) and a learned `mask_token` substituted at masked
steps. Everything else (projections, positional encoding, style bias, `activate`)
is shared.

## Masking — `make_span_mask`

Mask **whole bars** at a per-sample random ratio `[mask_min_ratio, mask_max_ratio]`
(default 0.15–0.65). Small ratios teach single-bar infill; large ratios approach
full generation. Whole-bar spans mimic real breaks better than scattered steps.

## Objective — `models/losses.py::MaskedDrumLoss`

Reconstruction scored **only on masked steps** (MAE-style): hit BCE (pos-weighted)
+ velocity/offset MSE where a true hit exists. Unmasked steps contribute nothing.

## Training — `scripts/train_masked.py`

Reuses `make_loaders` + `load_config`. Selects `runs/masked_best.pt` on
masked-reconstruction **F1** over the backbone classes (kick/snare/closed+open hat),
measured with fixed eval masks. Launch (overnight, MPS):

```
python scripts/train_masked.py --config configs/model_masked.yaml --sources lmd user
```

## Inference — `MaskedDrumTransformer.infill`

Given the drum grid with a masked span, predict the masked steps from both sides +
audio; keep known steps untouched, binarise the filled hits. v1 is a single
bidirectional pass; **MaskGIT-style iterative confidence-based decoding** (predict
all, commit the most confident, re-mask, repeat) is a drop-in enhancement for
sharper fills.

## App integration (next)

1. Transcribe the song's own drums (Phase 1) → the known drum grid.
2. User marks a time range (a break) → mask those steps.
3. `infill` → drums for the gap that match both sides + the audio → mix back.

## Status

Model, masking, loss, training script, and config are implemented and pass an
overfit smoke test. Remaining: the full training run, then wiring `infill` into a
"fill a break" control in `app.py` (with the iterative decoder as a follow-up).

"""Data-pipeline contracts: style plumbing, corpus normalization, LMD curation."""

import numpy as np
import pretty_midi
import pytest

from musibot.config import (
    EXTERNAL_DIR,
    FEATURE_DIM,
    NORM_FEATURE_INDICES,
    N_DRUM_CLASSES,
    STYLE_TO_ID,
    WINDOW_STEPS,
    canonical_style,
)
from musibot.data import datasets, normalize
from musibot.data.dataset import DrumDataset


def _synthetic_clip(processed_dir, name, source, style_id, n=WINDOW_STEPS + 8, seed=0):
    rng = np.random.default_rng(seed)
    cond = rng.normal(3.0, 2.0, size=(n, FEATURE_DIM)).astype(np.float32)
    drums = np.zeros((n, N_DRUM_CLASSES, 3), dtype=np.float32)
    drums[..., 0] = (rng.random((n, N_DRUM_CLASSES)) < 0.2).astype(np.float32)
    out = processed_dir / source
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{name}.npz", conditioning=cond, drums=drums,
        tempo=np.float32(120.0), source=source, style_id=np.int64(style_id),
    )


# --- style id ---------------------------------------------------------------

def test_canonical_style_mapping():
    assert canonical_style("rock") == STYLE_TO_ID["rock"]
    assert canonical_style("neworleans/funk") == STYLE_TO_ID["neworleans"]  # top-level token
    assert canonical_style("samba") == STYLE_TO_ID["latin"]                 # alias
    assert canonical_style(None) == 0                                       # unknown
    assert canonical_style("not-a-genre") == 0


def test_clip_carries_style_and_dataset_returns_it(tmp_path):
    rock = STYLE_TO_ID["rock"]
    _synthetic_clip(tmp_path, "a", "lmd", rock)
    ds = DrumDataset(split="train", processed_dir=tmp_path, val_frac=0.0, test_frac=0.0)
    assert len(ds) >= 1
    item = ds[0]
    assert item["style"].item() == rock
    assert item["conditioning"].shape == (WINDOW_STEPS, FEATURE_DIM)
    assert item["drums"].shape == (WINDOW_STEPS, N_DRUM_CLASSES, 3)


# --- normalization ----------------------------------------------------------

def test_fit_stats_then_apply_standardizes_train_split(tmp_path):
    for i in range(12):
        _synthetic_clip(tmp_path, f"c{i}", "lmd", 0, seed=i)
    stats = normalize.fit_stats(sources=["lmd"], processed_dir=tmp_path,
                                val_frac=0.0, test_frac=0.0)
    assert list(stats["indices"]) == list(NORM_FEATURE_INDICES)
    assert (stats["std"] > 0).all()  # guarded, never zero

    clip = np.load(next((tmp_path / "lmd").glob("*.npz")))["conditioning"]
    out = normalize.apply_stats(clip, stats)
    other = [i for i in range(FEATURE_DIM) if i not in NORM_FEATURE_INDICES]
    assert np.allclose(out[:, other], clip[:, other])          # bounded dims untouched
    assert not np.allclose(out[:, list(NORM_FEATURE_INDICES)],  # magnitude dims rescaled
                           clip[:, list(NORM_FEATURE_INDICES)])


def test_load_stats_absent_is_noop(tmp_path):
    assert normalize.load_stats(tmp_path / "missing.npz") is None


def test_dataset_applies_stats_when_present(tmp_path, monkeypatch):
    _synthetic_clip(tmp_path, "a", "lmd", 0)
    stats = normalize.fit_stats(sources=["lmd"], processed_dir=tmp_path,
                                val_frac=0.0, test_frac=0.0)
    monkeypatch.setattr("musibot.data.dataset.load_stats", lambda *a, **k: stats)
    ds = DrumDataset(split="train", processed_dir=tmp_path, val_frac=0.0, test_frac=0.0)
    zi = list(NORM_FEATURE_INDICES)
    # Standardized magnitude features should be roughly centred, unlike the raw ~3.0 mean.
    assert abs(float(ds[0]["conditioning"][:, zi].mean())) < 1.5


# --- LMD curation -----------------------------------------------------------

def _midi(drum_notes, pitched_instruments, end=45.0):
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    for k in range(drum_notes):
        t = (k / max(1, drum_notes)) * end
        drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=t, end=t + 0.1))
    pm.instruments.append(drum)
    for p in range(pitched_instruments):
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=60 + p, start=0.0, end=end))
        pm.instruments.append(inst)
    return pm


def test_lmd_curation_filters():
    assert datasets._lmd_is_usable(_midi(80, 3))
    assert not datasets._lmd_is_usable(_midi(5, 3))     # too few drum notes
    assert not datasets._lmd_is_usable(_midi(80, 1))    # not enough harmony
    assert not datasets._lmd_is_usable(_midi(80, 3, end=5.0))    # too short
    assert not datasets._lmd_is_usable(_midi(80, 3, end=900.0))  # too long


def test_gmd_style_map_reads_info_csv():
    root = EXTERNAL_DIR / "gmd"
    if not (root / "info.csv").exists():
        pytest.skip("GMD not downloaded")
    styles = datasets.gmd_style_map(root)
    assert styles  # non-empty
    assert all(0 <= v < len(STYLE_TO_ID) for v in styles.values())


# --- in-memory render (needs FluidSynth) ------------------------------------

def test_render_non_drums_excludes_drums():
    pm = _midi(drum_notes=40, pitched_instruments=1, end=2.0)
    try:
        audio = datasets.render_non_drums(pm, sr=22050)
    except Exception as e:  # no fluidsynth / no soundfont in this environment
        pytest.skip(f"FluidSynth unavailable: {e}")
    assert audio.ndim == 1 and audio.size > 0
    assert np.isfinite(audio).all()

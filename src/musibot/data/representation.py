"""Conversion between drum MIDI and the ``[T, 9, 3]`` grid tensor.

The grid is the model's target. Time is quantized to 16th-note steps; the
microtiming channel records how far each hit deviates from its quantized step,
so the discrete grid still captures human feel.

``grid -> midi -> grid`` round-trips exactly (hits) and within quantization
tolerance (velocity/offset) — see tests/test_representation.py.
"""

from __future__ import annotations

import numpy as np
import pretty_midi

from ..config import (
    CH_HIT,
    CH_OFFSET,
    CH_VELOCITY,
    CLASS_TO_GM_NOTE,
    GM_NOTE_TO_CLASS,
    N_CHANNELS,
    N_DRUM_CLASSES,
    STEPS_PER_BEAT,
)


def step_duration(tempo: float) -> float:
    """Seconds per 16th-note step at the given tempo (BPM)."""
    return 60.0 / tempo / STEPS_PER_BEAT


def midi_to_grid(
    pm: pretty_midi.PrettyMIDI, tempo: float, n_steps: int | None = None
) -> np.ndarray:
    """Convert a drum ``PrettyMIDI`` to a ``[T, 9, 3]`` grid.

    ``tempo`` (BPM) defines the step grid. ``n_steps`` fixes the length; if
    ``None`` it is inferred from the last note.
    """
    dt = step_duration(tempo)
    notes = [n for inst in pm.instruments if inst.is_drum for n in inst.notes]

    if n_steps is None:
        last = max((n.start for n in notes), default=0.0)
        n_steps = int(round(last / dt)) + 1

    grid = np.zeros((n_steps, N_DRUM_CLASSES, N_CHANNELS), dtype=np.float32)

    for note in notes:
        cls = GM_NOTE_TO_CLASS.get(note.pitch)
        if cls is None:
            continue
        step_f = note.start / dt
        step = int(round(step_f))
        if not 0 <= step < n_steps:
            continue
        offset = float(np.clip(step_f - step, -0.5, 0.5))
        velocity = note.velocity / 127.0

        # On a collision keep the louder hit.
        if grid[step, cls, CH_HIT] and velocity <= grid[step, cls, CH_VELOCITY]:
            continue
        grid[step, cls, CH_HIT] = 1.0
        grid[step, cls, CH_VELOCITY] = velocity
        grid[step, cls, CH_OFFSET] = offset

    return grid


def grid_to_midi(
    grid: np.ndarray, tempo: float, note_length: float = 0.1,
    step_times: np.ndarray | None = None,
) -> pretty_midi.PrettyMIDI:
    """Convert a ``[T, 9, 3]`` grid back to a single-instrument drum MIDI.

    ``step_times`` (seconds per step, e.g. from a beat-synchronous grid) places
    each hit at its real time so the drums track a tempo-varying song; without it
    the grid is uniform at ``tempo``.
    """
    dt = step_duration(tempo)
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    idx = None if step_times is None else np.arange(len(step_times))

    steps, classes = np.where(grid[:, :, CH_HIT] > 0.5)
    for step, cls in zip(steps, classes):
        offset = float(np.clip(grid[step, cls, CH_OFFSET], -0.5, 0.5))
        if step_times is None:
            start = max(0.0, (step + offset) * dt)
        else:                                       # interpolate the fractional step to seconds
            start = max(0.0, float(np.interp(step + offset, idx, step_times)))
        velocity = int(np.clip(round(grid[step, cls, CH_VELOCITY] * 127), 1, 127))
        drum.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=CLASS_TO_GM_NOTE[int(cls)],
                start=start,
                end=start + note_length,
            )
        )

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm.instruments.append(drum)
    return pm


def drum_density(grid: np.ndarray) -> float:
    """Mean number of drum hits per step — a quick groove-busyness summary."""
    return float(grid[:, :, CH_HIT].sum() / max(1, grid.shape[0]))

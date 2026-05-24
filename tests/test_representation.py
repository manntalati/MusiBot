"""The drum grid must survive a grid -> MIDI -> grid round-trip."""

import numpy as np

from musibot.config import CH_HIT, CH_OFFSET, CH_VELOCITY, N_CHANNELS, N_DRUM_CLASSES
from musibot.data.representation import drum_density, grid_to_midi, midi_to_grid


def _random_grid(n_steps: int = 64, seed: int = 444) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.zeros((n_steps, N_DRUM_CLASSES, N_CHANNELS), dtype=np.float32)
    hits = rng.random((n_steps, N_DRUM_CLASSES)) < 0.25
    grid[..., CH_HIT] = hits.astype(np.float32)
    grid[..., CH_VELOCITY] = np.where(hits, rng.uniform(0.2, 1.0, hits.shape), 0.0)
    grid[..., CH_OFFSET] = np.where(hits, rng.uniform(-0.45, 0.45, hits.shape), 0.0)
    return grid


def test_round_trip_preserves_hits():
    grid = _random_grid()
    back = midi_to_grid(grid_to_midi(grid, tempo=120.0), tempo=120.0, n_steps=len(grid))
    assert np.array_equal(grid[..., CH_HIT], back[..., CH_HIT])


def test_round_trip_velocity_and_offset_within_tolerance():
    grid = _random_grid()
    back = midi_to_grid(grid_to_midi(grid, tempo=120.0), tempo=120.0, n_steps=len(grid))
    hit = grid[..., CH_HIT] > 0.5
    assert np.allclose(grid[..., CH_VELOCITY][hit], back[..., CH_VELOCITY][hit], atol=0.01)
    assert np.allclose(grid[..., CH_OFFSET][hit], back[..., CH_OFFSET][hit], atol=1e-3)


def test_round_trip_holds_across_tempos():
    grid = _random_grid(n_steps=48)
    for tempo in (90.0, 128.0, 174.0):
        back = midi_to_grid(grid_to_midi(grid, tempo), tempo, n_steps=len(grid))
        assert np.array_equal(grid[..., CH_HIT], back[..., CH_HIT])


def test_density_matches_hit_count():
    grid = _random_grid()
    expected = grid[..., CH_HIT].sum() / grid.shape[0]
    assert abs(drum_density(grid) - expected) < 1e-6

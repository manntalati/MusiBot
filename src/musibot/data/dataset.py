"""PyTorch ``Dataset`` over the prepared ``.npz`` clips.

Long clips are sliced into fixed-length 8-bar windows. Splits are assigned per
*clip* (by deterministic hash of the filename) so windows from one song never
leak across the train/val/test boundary. The magnitude conditioning features
are standardized on load with the corpus stats fitted by ``normalize.py``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import PROCESSED_DIR, UNKNOWN_STYLE_ID, WINDOW_STEPS
from .normalize import apply_stats, load_stats


def split_of(name: str, val_frac: float, test_frac: float) -> str:
    """Deterministically bucket a clip into train/val/test by name hash."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000 / 1000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


@lru_cache(maxsize=256)
def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.load(path)
    style_id = int(data["style_id"]) if "style_id" in data else UNKNOWN_STYLE_ID
    return data["conditioning"], data["drums"], style_id


class DrumDataset(Dataset):
    """Yields ``conditioning [W, F]``, ``drums [W, 9, 3]``, ``style`` tensors."""

    def __init__(
        self,
        split: str = "train",
        sources: list[str] | None = None,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        processed_dir: Path = PROCESSED_DIR,
    ):
        self.split = split
        self.stats = load_stats()  # None when no stats file -> features pass through
        hop = WINDOW_STEPS // 2 if split == "train" else WINDOW_STEPS
        self.index: list[tuple[str, int]] = []

        search = (
            [processed_dir / s for s in sources]
            if sources
            else [processed_dir]
        )
        for root in search:
            for npz in sorted(Path(root).rglob("*.npz")):
                if split_of(npz.stem, val_frac, test_frac) != split:
                    continue
                cond, _, _ = _load_npz(str(npz))
                T = len(cond)
                if T < WINDOW_STEPS:
                    continue
                for start in range(0, T - WINDOW_STEPS + 1, hop):
                    self.index.append((str(npz), start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        path, start = self.index[i]
        cond, drums, style_id = _load_npz(path)
        end = start + WINDOW_STEPS
        window = apply_stats(cond[start:end], self.stats)
        return {
            "conditioning": torch.from_numpy(np.ascontiguousarray(window)),
            "drums": torch.from_numpy(drums[start:end].copy()),
            "style": torch.tensor(style_id, dtype=torch.long),
        }


def make_loaders(
    batch_size: int = 32,
    sources: list[str] | None = None,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build train/val/test ``DataLoader``s sharing the same split config."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = DrumDataset(split=split, sources=sources)
        # Only drop a ragged final batch when there is more than one batch;
        # otherwise tiny splits (e.g. the 5-song fine-tune set) yield nothing.
        drop_last = split == "train" and len(ds) > batch_size
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=drop_last,
        )
    return loaders

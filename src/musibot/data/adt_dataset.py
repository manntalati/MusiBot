"""Dataset over the prepared ADT ``(mel, grid)`` clips (data/processed/adt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import PROCESSED_DIR
from .dataset import split_of

ADT_DIR = PROCESSED_DIR / "adt"


class ADTDataset(Dataset):
    """Yields fixed-length windows of ``mel [W, n_mels]`` + ``drums [W, 9, 3]``."""

    def __init__(self, split: str = "train", win: int = 64, val_frac: float = 0.1,
                 test_frac: float = 0.1, root: Path = ADT_DIR):
        self.win = win
        self.index: list[tuple[str, int]] = []
        for npz in sorted(Path(root).rglob("*.npz")):
            if split_of(npz.stem, val_frac, test_frac) != split:
                continue
            try:
                T = len(np.load(npz)["mel"])
            except (KeyError, ValueError, OSError):
                continue
            if T < win:
                continue
            hop = win // 2 if split == "train" else win
            for start in range(0, T - win + 1, hop):
                self.index.append((str(npz), start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        path, start = self.index[i]
        z = np.load(path)
        end = start + self.win
        return {
            "mel": torch.from_numpy(z["mel"][start:end].copy()),
            "drums": torch.from_numpy(z["drums"][start:end].copy()),
        }


def make_adt_loaders(batch_size: int = 64, win: int = 64, num_workers: int = 0) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        ds = ADTDataset(split=split, win=win)
        drop_last = split == "train" and len(ds) > batch_size
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, drop_last=drop_last)
    return loaders


__all__ = ["ADTDataset", "make_adt_loaders", "ADT_DIR"]

"""Corpus-level standardization of the magnitude conditioning features.

``features.py`` emits ``rms_energy`` / ``onset_density`` / ``spectral_contrast``
in raw librosa units, whose scale varies clip-to-clip. We z-score those dims
against statistics fitted **once over the training split** and reused verbatim
at inference, so the model sees the same distribution in both regimes (the old
per-clip min-max did not — it rescaled every clip against its own range).

Stats live in a tiny ``.npz`` (``FEATURE_STATS_FILE``). When the file is absent
(e.g. unit tests on synthetic clips) ``apply_stats`` is a no-op, so the pipeline
still runs — it just isn't standardized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import FEATURE_STATS_FILE, NORM_FEATURE_INDICES, PROCESSED_DIR


def fit_stats(
    sources: list[str] | None = None,
    processed_dir: Path = PROCESSED_DIR,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict[str, np.ndarray]:
    """Compute per-dim mean/std of the magnitude features over the train split.

    Only the ``NORM_FEATURE_INDICES`` columns are summarized. Uses a streaming
    (sum / sum-of-squares) accumulation so it never holds the whole corpus in
    memory.
    """
    from .dataset import split_of  # lazy: avoids a normalize<->dataset cycle

    idx = list(NORM_FEATURE_INDICES)
    roots = [processed_dir / s for s in sources] if sources else [processed_dir]

    count = 0
    total = np.zeros(len(idx), dtype=np.float64)
    total_sq = np.zeros(len(idx), dtype=np.float64)
    for root in roots:
        for npz in sorted(Path(root).rglob("*.npz")):
            if split_of(npz.stem, val_frac, test_frac) != "train":
                continue
            cond = np.load(npz)["conditioning"][:, idx].astype(np.float64)
            count += cond.shape[0]
            total += cond.sum(axis=0)
            total_sq += (cond**2).sum(axis=0)

    if count == 0:
        mean = np.zeros(len(idx), dtype=np.float32)
        std = np.ones(len(idx), dtype=np.float32)
    else:
        mean = (total / count).astype(np.float32)
        var = np.maximum(total_sq / count - (total / count) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std[std < 1e-6] = 1.0  # guard constant/degenerate dims

    return {"indices": np.array(idx, dtype=np.int64), "mean": mean, "std": std}


def save_stats(stats: dict[str, np.ndarray], path: Path = FEATURE_STATS_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stats)
    return path


def load_stats(path: Path = FEATURE_STATS_FILE) -> dict[str, np.ndarray] | None:
    if not Path(path).exists():
        return None
    data = np.load(path)
    return {"indices": data["indices"], "mean": data["mean"], "std": data["std"]}


def apply_stats(features: np.ndarray, stats: dict[str, np.ndarray] | None) -> np.ndarray:
    """Return a standardized copy of ``[T, FEATURE_DIM]`` features.

    A ``None`` ``stats`` (no fitted file) leaves the values unchanged. Always
    returns a fresh array, so callers may hand it straight to ``from_numpy``
    without aliasing a cached clip.
    """
    if stats is None:
        return features.copy()
    out = features.copy()
    idx = stats["indices"]
    out[:, idx] = (out[:, idx] - stats["mean"]) / stats["std"]
    return out


__all__ = ["fit_stats", "save_stats", "load_stats", "apply_stats"]

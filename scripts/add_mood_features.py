#!/usr/bin/env python
"""Upgrade already-prepared clips in place with the 3 mood scalars (18 -> 21 dims).

Adding mood conditioning bumped ``FEATURE_DIM`` from 18 to 21. Rather than
re-preparing the whole corpus (hours of LMD rendering), we compute the mood
scalars directly from each clip's already-stored base features + tempo — the
same ``data.mood.compute_mood`` used at inference, with the same corpus stats —
and append them as three constant columns.

    .venv/bin/python scripts/add_mood_features.py            # migrate all clips
    .venv/bin/python scripts/add_mood_features.py --dry-run  # report only
    .venv/bin/python scripts/add_mood_features.py --limit 50 # first 50 (testing)

Idempotent: clips already at 21 dims are skipped, so re-running is safe. Writes
each clip atomically (temp file + os.replace) so an interrupt can't corrupt data.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from musibot.config import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_STATS_FILE,
    N_BASE_FEATURES,
    PROCESSED_DIR,
)
from musibot.data.mood import compute_mood  # noqa: E402
from musibot.data.normalize import load_stats  # noqa: E402

CLIP_KEYS = ("conditioning", "drums", "tempo", "source", "style_id")


def migrate_clip(path: Path, stats: dict | None, dry_run: bool) -> str:
    """Return one of: 'done', 'skip' (already 21), 'bad' (unexpected width)."""
    with np.load(path, allow_pickle=False) as z:
        cond = z["conditioning"]
        if cond.shape[1] == FEATURE_DIM:
            return "skip"
        if cond.shape[1] != N_BASE_FEATURES:
            return "bad"
        data = {k: z[k] for k in z.files}

    tempo = float(data["tempo"])
    mood = compute_mood(cond[:, :N_BASE_FEATURES], tempo, stats)  # (3,) in 0..1
    new_cond = np.concatenate(
        [cond, np.tile(mood, (cond.shape[0], 1))], axis=1
    ).astype(np.float32)
    data["conditioning"] = new_cond

    if dry_run:
        return "done"

    # Atomic replace: write to a sibling temp file, then os.replace over the
    # original. The suffix ends in .npz so numpy won't append another extension.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser(description="Append mood scalars to prepared clips")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--limit", type=int, default=None, help="process at most N clips")
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    stats = load_stats()
    if stats is None:
        print(f"WARNING: no corpus stats at {FEATURE_STATS_FILE} — mood loudness/"
              "onset terms will be uncalibrated. Fit stats first for best results.")
    else:
        print(f"using corpus stats from {FEATURE_STATS_FILE}")

    clips = sorted(
        p for p in args.processed_dir.rglob("*.npz")
        if p.name != FEATURE_STATS_FILE.name
    )
    if args.limit:
        clips = clips[: args.limit]
    print(f"{'DRY-RUN: ' if args.dry_run else ''}migrating {len(clips)} clips "
          f"({N_BASE_FEATURES} -> {FEATURE_DIM} dims)\n")

    counts = {"done": 0, "skip": 0, "bad": 0}
    sample_moods = []
    for i, path in enumerate(clips):
        try:
            result = migrate_clip(path, stats, args.dry_run)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {path}: {e}")
            result = "bad"
        counts[result] += 1
        if result == "done" and len(sample_moods) < 6:
            with np.load(path, allow_pickle=False) as z:
                sample_moods.append((path.parent.name, z["conditioning"][0, -3:]))
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(clips)}  {counts}")

    print(f"\nDone: {counts}")
    if sample_moods:
        print("sample mood [energy, valence, dance] per tier:")
        for tier, m in sample_moods:
            print(f"  {tier:8} {np.round(m, 3)}")


if __name__ == "__main__":
    main()

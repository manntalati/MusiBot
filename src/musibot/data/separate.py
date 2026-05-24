"""Stem separation with Demucs (replaces the deprecated Spleeter).

Splits a song into drums / bass / vocals / other. The *accompaniment* — the
drumless mix that the model is conditioned on — is the sum of the non-drum
stems.

Uses ``demucs.pretrained.get_model`` + ``demucs.apply.apply_model`` rather than
``demucs.api.Separator`` because the latter only exists in newer demucs
versions; the lower-level API is stable across 4.x.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

SEPARATION_SR = 44100  # Demucs' native sample rate

_MODEL = None
_DEVICE = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_model(name: str = "htdemucs"):
    """Lazily load and cache a Demucs model (network download on first call)."""
    global _MODEL, _DEVICE
    if _MODEL is None:
        from demucs.pretrained import get_model

        _DEVICE = _device()
        _MODEL = get_model(name).to(_DEVICE).eval()
    return _MODEL, _DEVICE


def _load_stereo(path: Path, sr: int) -> torch.Tensor:
    """Load audio as a stereo ``[2, T]`` float32 tensor at ``sr``."""
    audio, _ = librosa.load(str(path), sr=sr, mono=False)
    if audio.ndim == 1:                            # mono -> duplicate to stereo
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return torch.from_numpy(audio.astype(np.float32))


def separate_song(path: str | Path, model: str = "htdemucs") -> dict[str, np.ndarray]:
    """Return per-stem mono audio plus a summed drumless ``accompaniment``.

    Each value is a float32 mono array at Demucs' native 44.1 kHz.
    """
    from demucs.apply import apply_model

    net, device = _get_model(model)
    wav = _load_stereo(Path(path), SEPARATION_SR).to(device)

    # Demucs expects zero-mean, unit-std input; un-normalize the outputs.
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) or 1.0
    norm = ((wav - mean) / std).unsqueeze(0)        # [1, 2, T]

    with torch.no_grad():
        sources = apply_model(net, norm, device=device, split=True, overlap=0.25)
    sources = (sources * std + mean)[0]             # [n_sources, 2, T]

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(net.sources):          # ['drums','bass','other','vocals']
        out[name] = sources[i].mean(0).cpu().numpy().astype(np.float32)
    out["accompaniment"] = sum(out[k] for k in out if k != "drums").astype(np.float32)
    return out


def separate_to_dir(
    path: str | Path, out_dir: str | Path, model: str = "htdemucs"
) -> dict[str, Path]:
    """Separate ``path`` and write ``drums.wav`` + ``accompaniment.wav``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = separate_song(path, model=model)

    written: dict[str, Path] = {}
    for name in ("drums", "accompaniment"):
        dest = out_dir / f"{name}.wav"
        sf.write(dest, stems[name], SEPARATION_SR)
        written[name] = dest
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demucs stem separation")
    parser.add_argument("song", help="path to an audio file")
    parser.add_argument("--out", default="data/processed/stems", help="output directory")
    args = parser.parse_args()

    paths = separate_to_dir(args.song, Path(args.out) / Path(args.song).stem)
    for name, dest in paths.items():
        print(f"{name}: {dest}")

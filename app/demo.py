"""Gradio demo: upload a drumless song, get it back with AI-generated drums.

    python app/demo.py --checkpoint runs/transformer_best.pt --config configs/model_transformer.yaml

Requires the package installed (``pip install -e ".[app]"``) and a soundfont
via --soundfont or the ``MUSIBOT_SOUNDFONT`` environment variable.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr

from musibot.generate import generate_drums
from musibot.render import render


def build_app(checkpoint: str, config: str, soundfont: str | None):
    def run(song_path: str, drum_gain: float):
        if not song_path:
            raise gr.Error("Upload a drumless song first.")
        tmp = Path(tempfile.mkdtemp())
        midi = generate_drums(song_path, checkpoint, config, tmp / "drums.mid")
        final = render(song_path, midi, tmp / "final.wav", soundfont, drum_gain)
        return str(final), str(midi)

    with gr.Blocks(title="MusiBot — AI Drum Track Generation") as app:
        gr.Markdown("# MusiBot\nUpload a song with no drums; get it back with an AI drum track.")
        with gr.Row():
            song = gr.Audio(label="Drumless song", type="filepath")
            gain = gr.Slider(0.0, 1.5, value=0.8, label="Drum level")
        go = gr.Button("Generate drums", variant="primary")
        out_audio = gr.Audio(label="Song with generated drums")
        out_midi = gr.File(label="Drum MIDI")
        go.click(run, inputs=[song, gain], outputs=[out_audio, out_midi])
    return app


def main():
    parser = argparse.ArgumentParser(description="MusiBot Gradio demo")
    parser.add_argument("--checkpoint", default="runs/transformer_best.pt")
    parser.add_argument("--config", default="configs/model_transformer.yaml")
    parser.add_argument("--soundfont", default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app(args.checkpoint, args.config, args.soundfont)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()

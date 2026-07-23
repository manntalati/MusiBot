"""MusiBot — upload a song, get AI-generated drums you can actually hear.

    .venv/bin/python app.py

Opens a local web UI. Drop in an mp3/wav; MusiBot separates out the original
drums (Demucs), generates a new drum track conditioned on the drumless mix, and
plays back the final result with our drums in place of the original ones.

First run downloads the Demucs weights and can take a couple of minutes per
song on CPU/MPS; subsequent runs reuse the cached model.
"""

from __future__ import annotations

import gradio as gr

from musibot.pipeline import process_song

FEEDBACK_CHOICES = ["auto", "0.5", "0.35", "0.25"]


def _run(song_path, feedback, drum_gain, progress=gr.Progress()):
    if not song_path:
        raise gr.Error("Upload a song first.")
    progress(0.1, desc="Separating drums from the mix (Demucs)…")
    fb = feedback if feedback == "auto" else float(feedback)
    result = process_song(song_path, feedback=fb, drum_gain=drum_gain)
    progress(1.0, desc="Done")
    return (
        result["mix"],
        result["drums"],
        result["accompaniment"],
        result["midi_path"],
        result["report"],
    )


with gr.Blocks(title="MusiBot") as demo:
    gr.Markdown(
        "# 🥁 MusiBot\n"
        "Upload a song. MusiBot strips its original drums, generates a fresh "
        "drum track conditioned on the drumless mix, and plays back the result — "
        "**our** drums, not the original ones."
    )
    with gr.Row():
        with gr.Column(scale=1):
            song = gr.Audio(label="Upload a song", type="filepath", sources=["upload"])
            feedback = gr.Dropdown(
                FEEDBACK_CHOICES,
                value="auto",
                label="Feedback threshold",
                info="'auto' sweeps 0.5→0.25 and picks the least-sparse take "
                "that doesn't collapse. Lower = busier drums.",
            )
            drum_gain = gr.Slider(
                0.2, 2.0, value=1.0, step=0.1,
                label="Drum level",
                info="Loudness of the generated drums relative to the song.",
            )
            go = gr.Button("Generate drums", variant="primary")
        with gr.Column(scale=1):
            mix = gr.Audio(label="Final mix (song + generated drums)", type="numpy")
            drums = gr.Audio(label="Generated drums only", type="numpy")
            accomp = gr.Audio(label="Drumless accompaniment (input to the model)", type="numpy")
            midi = gr.File(label="Generated drum MIDI")
            report = gr.Textbox(label="Diagnostics", lines=8)

    go.click(
        _run,
        inputs=[song, feedback, drum_gain],
        outputs=[mix, drums, accomp, midi, report],
    )


if __name__ == "__main__":
    demo.launch()
"""MusiBot — an AI drum machine. Upload a song; get a groove, tuned by mood.

    .venv/bin/python app.py

MusiBot strips a song's original drums (Demucs), reads its profile (tempo /
energy / valence / danceability / key), optionally enriches it with Spotify
metadata, then generates a fresh drum track conditioned on the drumless mix —
steerable with the mood faders — and plays back the result.

Spotify is optional; set SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET to enable it
(see musibot/spotify.py). Without credentials everything else works.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) so credentials aren't in the shell
    history or the repo. Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()  # before importing spotify so credentials are visible at build time

from musibot import spotify  # noqa: E402
from musibot.pipeline import analyze_song, process_song  # noqa: E402

FEEDBACK_CHOICES = ["auto", "0.5", "0.35", "0.25"]

# Force the dark "console" theme regardless of the viewer's system setting.
_FORCE_DARK = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.href);
  }
}
"""

CSS = """
:root {
  --bg:#161211; --console:#211b18; --panel:#2a221e; --inset:#100c0b;
  --line:#3a2f29; --screw:#46392f; --ink:#f3ece4; --muted:#a6968b; --faint:#6f6058;
  --red:#e6392a; --red-hot:#ff5a3c; --red-deep:#a3160c; --red-glow:rgba(230,57,42,.45);
  --lcd-bg:#1b0c09; --lcd:#ff5a44;
  --mono:ui-monospace,"SF Mono","SFMono-Regular",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.gradio-container { max-width:1160px !important; margin:0 auto !important;
  background:
    radial-gradient(120% 80% at 50% -8%, color-mix(in srgb,var(--red) 8%,var(--bg)), var(--bg) 60%) !important;
  font-family:var(--sans) !important; }
.gradio-container *{ font-family:var(--sans); }
footer{ display:none !important; }

/* hero */
.mb-hero{ position:relative; border:1px solid var(--line); border-radius:20px;
  padding:22px 26px; margin-bottom:8px; overflow:hidden;
  background:linear-gradient(180deg, color-mix(in srgb,var(--console) 92%,#fff 8%), var(--console)); }
.mb-plate{ display:flex; align-items:center; justify-content:space-between; gap:16px; }
.mb-brand{ display:flex; align-items:center; gap:13px; }
.mb-power{ width:12px; height:12px; border-radius:50%; background:var(--red);
  box-shadow:0 0 12px 2px var(--red-glow); animation:mb-blink 2.4s ease-in-out infinite; }
@keyframes mb-blink{ 0%,100%{opacity:1} 50%{opacity:.35} }
.mb-word{ font-family:var(--mono); font-weight:700; font-size:23px; letter-spacing:.34em;
  text-transform:uppercase; color:var(--ink); }
.mb-word b{ color:var(--red); }
.mb-model{ font-family:var(--mono); font-size:11px; letter-spacing:.18em; color:var(--faint);
  border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
.mb-tag{ font-family:var(--mono); font-size:11px; letter-spacing:.1em; color:var(--muted); text-transform:uppercase; }

/* section labels */
.mb-h{ font-family:var(--mono); font-size:10.5px; letter-spacing:.22em; text-transform:uppercase;
  color:var(--faint); margin:14px 2px 6px; }

/* generic panel + notes */
.mb-profile,.mb-spotify,.mb-seq,.mb-empty,.mb-note{ border:1px solid var(--line); border-radius:14px; }
.mb-empty{ color:var(--muted); font-family:var(--mono); font-size:12px; text-align:center;
  padding:20px; border-style:dashed; letter-spacing:.06em; }
.mb-note{ color:var(--muted); font-size:12.5px; padding:11px 13px; background:var(--panel); line-height:1.5; }
.mb-note a{ color:var(--red-hot); text-decoration:none; font-weight:600; }
.mb-note.ok{ color:var(--ink); border-color:color-mix(in srgb,var(--red) 40%,var(--line)); }

/* profile card */
.mb-profile{ padding:16px 18px; background:linear-gradient(180deg,var(--panel),#141010); }
.mb-pf-head{ display:flex; gap:13px; margin-bottom:14px; flex-wrap:wrap; }
.mb-lcd{ background:var(--lcd-bg); border:1px solid #000; border-radius:9px; padding:9px 13px; min-width:120px;
  box-shadow:inset 0 2px 8px rgba(0,0,0,.6);
  background-image:linear-gradient(var(--lcd-bg),var(--lcd-bg)),
    repeating-linear-gradient(0deg,transparent 0 3px,rgba(0,0,0,.25) 3px 4px); }
.mb-lcd .n{ font-family:var(--mono); font-weight:700; font-size:30px; color:var(--lcd); line-height:1;
  font-variant-numeric:tabular-nums; text-shadow:0 0 10px color-mix(in srgb,var(--lcd) 55%,transparent); }
.mb-lcd .u{ font-family:var(--mono); font-size:9.5px; letter-spacing:.26em; color:color-mix(in srgb,var(--lcd) 70%,transparent); margin-top:5px; }
.mb-chips{ display:flex; gap:7px; flex-wrap:wrap; align-content:center; }
.mb-chip{ font-family:var(--mono); font-size:11px; color:var(--ink); padding:5px 10px; border:1px solid var(--line);
  border-radius:6px; background:var(--inset); letter-spacing:.03em; }
.mb-chip b{ color:var(--red-hot); }
.mb-meter{ margin:11px 0; }
.mb-meter .t{ display:flex; justify-content:space-between; margin-bottom:6px; font-family:var(--mono); font-size:11px; }
.mb-meter .t .l{ color:var(--ink); text-transform:uppercase; letter-spacing:.05em; }
.mb-meter .t .v{ color:var(--muted); }
.mb-track{ height:8px; border-radius:99px; background:var(--inset); border:1px solid var(--line); overflow:hidden;
  box-shadow:inset 0 1px 3px rgba(0,0,0,.4); }
.mb-fill{ height:100%; background:linear-gradient(90deg,var(--red-deep),var(--red-hot)); box-shadow:0 0 10px var(--red-glow); }

/* spotify card */
.mb-spotify{ display:flex; gap:14px; padding:14px; background:var(--panel); align-items:center; }
.mb-sp-art{ width:74px; height:74px; border-radius:8px; object-fit:cover; border:1px solid var(--line); flex:0 0 auto; }
.mb-sp-title{ font-weight:700; color:var(--ink); font-size:15px; }
.mb-sp-sub{ color:var(--muted); font-size:12.5px; margin:3px 0 8px; }
.mb-sp-foot{ font-family:var(--mono); font-size:11px; color:var(--faint); margin-top:8px; }
.mb-sp-foot b{ color:var(--red-hot); }

/* sequencer (shows the real generated pattern) */
.mb-seq{ padding:14px; background:var(--inset); overflow-x:auto; }
.mb-srow{ display:grid; grid-template-columns:52px 1fr; gap:9px; align-items:center; margin:5px 0; min-width:520px; }
.mb-rn{ font-family:var(--mono); font-size:10px; color:var(--muted); text-transform:uppercase; text-align:right; }
.mb-cells,.mb-ruler{ display:grid; grid-template-columns:repeat(16,1fr); gap:5px; }
.mb-ruler span{ font-family:var(--mono); font-size:9px; color:var(--faint); text-align:center; }
.mb-step{ height:19px; border-radius:4px; background:color-mix(in srgb,var(--inset) 55%,#000); border:1px solid var(--line); }
.mb-step.beat{ border-color:color-mix(in srgb,var(--line) 55%,var(--muted)); }
.mb-step.on{ background:linear-gradient(180deg,var(--red-hot),var(--red-deep)); border-color:var(--red-hot);
  box-shadow:0 0 8px var(--red-glow), inset 0 1px 0 rgba(255,255,255,.25); }

/* Gradio component polish */
.gradio-container .block,.gradio-container .form,.gradio-container .gr-group{ background:transparent !important; border:none !important; }
.gradio-container label span,.gradio-container .gr-check-radio{ color:var(--muted) !important; }
button.primary{ background:linear-gradient(180deg,var(--red-hot),var(--red-deep)) !important; border:none !important;
  color:#fff !important; font-family:var(--mono) !important; font-weight:700 !important; letter-spacing:.12em !important;
  text-transform:uppercase !important; box-shadow:0 10px 26px -8px var(--red-glow) !important; }
button.primary:hover{ transform:translateY(-1px); }
button.secondary{ background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--ink) !important;
  font-family:var(--mono) !important; letter-spacing:.06em !important; }
.gradio-container input[type=range]{ accent-color:var(--red) !important; }
.gradio-container input[type=range]::-webkit-slider-thumb{ background:var(--red) !important; }
.gradio-container input[type=text],.gradio-container textarea{ background:var(--inset) !important; color:var(--ink) !important;
  border:1px solid var(--line) !important; font-family:var(--mono) !important; }
"""

HERO = """
<div class="mb-hero"><div class="mb-plate">
  <div class="mb-brand"><span class="mb-power"></span>
    <span class="mb-word">musi<b>bot</b></span><span class="mb-model">MB·909</span></div>
  <span class="mb-tag">AI drum machine — song in, groove out</span>
</div></div>
"""

EMPTY_PROFILE = '<div class="mb-empty">Upload a song to read its profile ◉</div>'
EMPTY_SEQ = '<div class="mb-empty">Generate to see the pattern ◉</div>'


def _band(v: float, lo: str, mid: str, hi: str) -> str:
    return lo if v < 0.4 else (mid if v < 0.66 else hi)


def _meter(label: str, v: float, desc: str) -> str:
    pct = max(0, min(100, round(v * 100)))
    return (f'<div class="mb-meter"><div class="t"><span class="l">{label}</span>'
            f'<span class="v">{desc} · {pct}%</span></div>'
            f'<div class="mb-track"><div class="mb-fill" style="width:{pct}%"></div></div></div>')


def render_profile(p: dict) -> str:
    instr = round(p["instrumentalness"] * 100)
    chips = (f'<span class="mb-chip">KEY <b>{p["key"]}</b></span>'
             f'<span class="mb-chip">{"INSTRUMENTAL" if instr >= 55 else "VOCAL"} <b>{instr}%</b></span>')
    meters = (_meter("Energy", p["energy"], _band(p["energy"], "Laid-back", "Steady", "High-energy"))
              + _meter("Valence", p["valence"], _band(p["valence"], "Dark / moody", "Neutral", "Bright / upbeat"))
              + _meter("Danceability", p["danceability"], _band(p["danceability"], "Loose", "Grooving", "Danceable")))
    return (f'<div class="mb-profile"><div class="mb-pf-head">'
            f'<div class="mb-lcd"><div class="n">{round(p["tempo"])}</div><div class="u">BPM</div></div>'
            f'<div class="mb-chips">{chips}</div></div>{meters}</div>')


def render_sequencer(pattern) -> str:
    if not pattern:
        return EMPTY_SEQ
    ruler = "".join(f'<span>{(i // 4) + 1 if i % 4 == 0 else "·"}</span>' for i in range(16))
    rows = f'<div class="mb-srow"><div class="mb-rn"></div><div class="mb-ruler">{ruler}</div></div>'
    for name, steps in pattern:
        cells = "".join(
            f'<div class="mb-step{" on" if v else ""}{" beat" if i % 4 == 0 else ""}"></div>'
            for i, v in enumerate(steps))
        rows += f'<div class="mb-srow"><div class="mb-rn">{name}</div><div class="mb-cells">{cells}</div></div>'
    return f'<div class="mb-seq">{rows}</div>'


def render_spotify(m: dict) -> str:
    art = f'<img class="mb-sp-art" src="{m["image"]}" alt="album art">' if m.get("image") else ""
    genres = "".join(f'<span class="mb-chip">{g}</span>' for g in m.get("genres", [])) or \
        '<span class="mb-chip">no genre tags</span>'
    year = f' · {m["year"]}' if m.get("year") else ""
    pop = m.get("popularity")
    pop_s = f'Popularity {pop}/100 · ' if pop is not None else ""
    return (f'<div class="mb-spotify">{art}<div><div class="mb-sp-title">{m["name"]}</div>'
            f'<div class="mb-sp-sub">{m["artist"]} · {m["album"]}{year}</div>'
            f'<div class="mb-chips">{genres}</div>'
            f'<div class="mb-sp-foot">{pop_s}style → <b>{m["style_name"]}</b></div></div></div>')


def spotify_status() -> str:
    if not spotify.is_configured():
        return ('<div class="mb-note">Spotify off. Set <b>SPOTIFY_CLIENT_ID</b> & '
                '<b>SPOTIFY_CLIENT_SECRET</b> (in a .env file) to add album art (see musibot/spotify.py).</div>')
    if spotify.is_connected():
        return '<div class="mb-note ok">◉ Spotify connected — a match appears when you upload a song.</div>'
    return ('<div class="mb-note">Spotify credentials set but the token request failed — '
            "check the id/secret.</div>")


def analyze(song_path, progress=gr.Progress()):
    if not song_path:
        return EMPTY_PROFILE, 0.5, 0.5, 0.5, "", None, ""
    progress(0.15, desc="Separating stems & reading the song's mood…")
    p = analyze_song(song_path)
    stem = Path(song_path).stem.replace("_", " ")
    sp_html, style = "", None
    if spotify.is_connected():
        try:
            m = spotify.search_track(stem)
            if m:
                sp_html, style = render_spotify(m), m["style_name"]
        except Exception:
            sp_html = ""
    progress(1.0, desc="Done")
    return render_profile(p), p["energy"], p["valence"], p["danceability"], sp_html, style, stem


def sp_match(query):
    if not spotify.is_connected():
        raise gr.Error("Connect Spotify first (set credentials, then Finish connecting).")
    m = spotify.search_track(query or "")
    if not m:
        return '<div class="mb-empty">No Spotify match for that title.</div>', None
    return render_spotify(m), m["style_name"]


def generate(song_path, energy, valence, dance, feedback, drum_gain, style, progress=gr.Progress()):
    if not song_path:
        raise gr.Error("Upload a song first.")
    progress(0.1, desc="Generating drums…")
    fb = feedback if feedback == "auto" else float(feedback)
    r = process_song(song_path, mood=(energy, valence, dance), feedback=fb,
                     drum_gain=drum_gain, style=style)
    progress(1.0, desc="Done")
    return r["mix"], r["drums"], r["accompaniment"], r["midi_path"], r["report"], render_sequencer(r["pattern"])


with gr.Blocks(title="MusiBot") as demo:
    style_state = gr.State(None)
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.HTML('<div class="mb-h">Load a song</div>')
            song = gr.Audio(label="Song", type="filepath", sources=["upload"])
            with gr.Accordion("Spotify enrichment (optional)", open=False):
                sp_status = gr.HTML(spotify_status())
                sp_card = gr.HTML("")
                with gr.Row():
                    sp_query = gr.Textbox(label="Title to match", scale=3)
                    sp_match_btn = gr.Button("Match", scale=1)
            gr.HTML('<div class="mb-h">Song profile</div>')
            profile = gr.HTML(EMPTY_PROFILE)
            gr.HTML('<div class="mb-h">Groove controls — steer the mood</div>')
            energy = gr.Slider(0, 1, value=0.5, step=0.05, label="Energy",
                               info="Higher → tighter backbeat, more controlled groove")
            valence = gr.Slider(0, 1, value=0.5, step=0.05, label="Valence (mood)",
                                info="Bright/major ↔ dark/minor character")
            dance = gr.Slider(0, 1, value=0.5, step=0.05, label="Danceability",
                              info="Rhythmic drive of the groove")
            with gr.Accordion("Advanced", open=False):
                feedback = gr.Dropdown(FEEDBACK_CHOICES, value="auto", label="Feedback threshold",
                                       info="'auto' picks the least-sparse take that doesn't collapse")
                drum_gain = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Drum level")
            go = gr.Button("◉ Generate drums", variant="primary", size="lg")
        with gr.Column(scale=6):
            gr.HTML('<div class="mb-h">Pattern — the generated groove</div>')
            sequencer = gr.HTML(EMPTY_SEQ)
            gr.HTML('<div class="mb-h">Final mix — your song with our drums</div>')
            mix = gr.Audio(label="Final mix", type="numpy")
            drums = gr.Audio(label="Generated drums only", type="numpy")
            with gr.Accordion("Stems & details", open=False):
                accomp = gr.Audio(label="Drumless accompaniment (model input)", type="numpy")
                midi = gr.File(label="Generated drum MIDI")
                report = gr.Textbox(label="Diagnostics", lines=7)

    song.upload(analyze, inputs=song,
                outputs=[profile, energy, valence, dance, sp_card, style_state, sp_query])
    song.clear(lambda: (EMPTY_PROFILE, 0.5, 0.5, 0.5, "", None, ""),
               outputs=[profile, energy, valence, dance, sp_card, style_state, sp_query])
    sp_match_btn.click(sp_match, inputs=sp_query, outputs=[sp_card, style_state])
    go.click(generate,
             inputs=[song, energy, valence, dance, feedback, drum_gain, style_state],
             outputs=[mix, drums, accomp, midi, report, sequencer])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue=gr.themes.colors.red),
        css=CSS,
        js=_FORCE_DARK,
    )

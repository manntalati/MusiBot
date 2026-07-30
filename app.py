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

import json
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

import pandas as pd  # noqa: E402

from musibot import spotify  # noqa: E402
from musibot.pipeline import (  # noqa: E402
    analyze_song,
    infill_break,
    masked_available,
    process_song,
    render_pattern,
    suggested_controls,
)

# Editable groove: rows = backbone kit, cols = 16 sixteenth-note steps.
GROOVE_ROWS = ["Kick", "Snare", "Hat", "Tom", "Ride"]
DF_COLS = ["Drum"] + [str(i + 1) for i in range(16)]
EMPTY_GROOVE_DF = pd.DataFrame(
    [[n] + [False] * 16 for n in GROOVE_ROWS], columns=DF_COLS
)


def pattern_to_df(pattern) -> pd.DataFrame:
    if not pattern:
        return EMPTY_GROOVE_DF.copy()
    return pd.DataFrame([[n] + [bool(x) for x in steps] for n, steps in pattern], columns=DF_COLS)


def df_to_pattern(df) -> list:
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=DF_COLS)
    out = []
    for _, row in df.iterrows():
        out.append((str(row.iloc[0]), [bool(row.iloc[i + 1]) for i in range(16)]))
    return out

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

/* mood map (cross-song valence x energy scatter) */
.mb-map{ padding:16px; background:var(--inset); border:1px solid var(--line); border-radius:14px; overflow-x:auto; }
.mb-map svg{ display:block; margin:0 auto; max-width:100%; height:auto; }
.mp-grid{ stroke:var(--line); stroke-width:1; }
.mp-axis{ stroke:color-mix(in srgb,var(--line) 60%,var(--muted)); stroke-width:1.2; }
.mp-quad{ font-family:var(--mono); font-size:9px; fill:var(--faint); letter-spacing:.14em; }
.mp-axt{ font-family:var(--mono); font-size:10px; fill:var(--muted); letter-spacing:.16em; }
.mp-dot{ fill:var(--red); stroke:var(--red-hot); stroke-width:1; }
.mp-lbl{ font-family:var(--mono); font-size:10px; fill:var(--ink); }

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
EMPTY_REF = '<div class="mb-empty">Upload a song to read its own drums ◉</div>'


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


def render_reference(fp: dict) -> str:
    """Show the song's OWN drums (transcribed) — the style we imitate in Phase 1."""
    if not fp or not fp.get("has_drums"):
        return '<div class="mb-empty">No clear drums found in this track to imitate ◉</div>'
    kit = ", ".join(fp.get("kit", [])) or "—"
    chips = (f'<span class="mb-chip">DENSITY <b>{fp["density"]:.2f}</b></span>'
             f'<span class="mb-chip">BACKBONE <b>{round(fp["snare_beat24"] * 100)}%</b></span>'
             f'<span class="mb-chip">KIT <b>{kit}</b></span>')
    return (f'<div class="mb-chips" style="margin-bottom:10px">{chips}</div>'
            f'{render_sequencer(fp["pattern"])}')


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


MOOD_LOG = Path("data/mood_log.json")
EMPTY_MAP = '<div class="mb-empty">Analyze a few songs to map their moods ◉</div>'


def load_log() -> dict:
    try:
        return json.loads(MOOD_LOG.read_text())
    except Exception:
        return {}


def log_profile(name: str, p: dict) -> dict:
    """Persist a song's mood so the cross-song map accumulates across sessions."""
    log = load_log()
    log[name] = {"energy": p["energy"], "valence": p["valence"],
                 "danceability": p["danceability"], "tempo": p["tempo"], "key": p["key"]}
    MOOD_LOG.parent.mkdir(parents=True, exist_ok=True)
    MOOD_LOG.write_text(json.dumps(log, indent=2))
    return log


def render_mood_map(log: dict) -> str:
    """Valence (y) x Energy (x) scatter of every analyzed song — Russell's mood
    circumplex, so quadrants read as moods and point size = danceability."""
    if not log:
        return EMPTY_MAP
    W, H, M = 560, 400, 46
    pw, ph = W - 2 * M, H - 2 * M
    fx = lambda e: M + e * pw                       # noqa: E731
    fy = lambda v: M + (1 - v) * ph                 # noqa: E731

    grid = ""
    for t in (0.25, 0.5, 0.75):
        cls = "mp-axis" if t == 0.5 else "mp-grid"
        grid += f'<line class="{cls}" x1="{fx(t):.0f}" y1="{M}" x2="{fx(t):.0f}" y2="{M+ph}"/>'
        grid += f'<line class="{cls}" x1="{M}" y1="{fy(t):.0f}" x2="{M+pw}" y2="{fy(t):.0f}"/>'
    frame = f'<rect x="{M}" y="{M}" width="{pw}" height="{ph}" fill="none" class="mp-grid"/>'

    quads = (
        f'<text class="mp-quad" x="{M+pw-8}" y="{M+14}" text-anchor="end">BRIGHT · HYPED</text>'
        f'<text class="mp-quad" x="{M+8}" y="{M+14}">CALM · WARM</text>'
        f'<text class="mp-quad" x="{M+pw-8}" y="{M+ph-8}" text-anchor="end">DARK · DRIVING</text>'
        f'<text class="mp-quad" x="{M+8}" y="{M+ph-8}">MOODY · SPARSE</text>'
    )
    axes = (
        f'<text class="mp-axt" x="{M+pw/2}" y="{H-14}" text-anchor="middle">ENERGY →</text>'
        f'<text class="mp-axt" x="16" y="{M+ph/2}" text-anchor="middle" '
        f'transform="rotate(-90 16 {M+ph/2:.0f})">VALENCE ↑</text>'
    )
    dots = ""
    for name, e in log.items():
        x, y, r = fx(e["energy"]), fy(e["valence"]), 4 + e.get("danceability", 0.5) * 4
        dots += f'<circle class="mp-dot" cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" opacity="0.9"/>'
        anchor = "end" if e["energy"] > 0.8 else "start"
        dx = -(r + 5) if anchor == "end" else (r + 5)
        dots += (f'<text class="mp-lbl" x="{x+dx:.1f}" y="{y+3.5:.1f}" '
                 f'text-anchor="{anchor}">{name}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
           f'{grid}{frame}{quads}{axes}{dots}</svg>')
    return f'<div class="mb-map">{svg}</div>'


def spotify_status() -> str:
    if not spotify.is_configured():
        return ('<div class="mb-note">Spotify off. Set <b>SPOTIFY_CLIENT_ID</b> & '
                '<b>SPOTIFY_CLIENT_SECRET</b> (in a .env file) to add album art (see musibot/spotify.py).</div>')
    if spotify.is_connected():
        return '<div class="mb-note ok">◉ Spotify connected — a match appears when you upload a song.</div>'
    return ('<div class="mb-note">Spotify credentials set but the token request failed — '
            "check the id/secret.</div>")


def analyze(song_path, imitate, progress=gr.Progress()):
    if not song_path:
        return (EMPTY_PROFILE, 0.5, 0.5, 0.5, "", None, "",
                render_mood_map(load_log()), EMPTY_REF, None)
    progress(0.15, desc="Separating stems, reading mood & the song's own drums…")
    p = analyze_song(song_path)
    stem = Path(song_path).stem.replace("_", " ")
    log = log_profile(stem, p)

    fp = p["reference"]
    e, v, d = p["energy"], p["valence"], p["danceability"]
    fp_state = None
    if fp.get("has_drums"):
        se, sd, td = suggested_controls(fp)
        fp_state = {"target_density": td, "has_drums": True}
        if imitate:                        # pre-set faders to imitate the original
            e, d = se, sd

    sp_html, style = "", None
    if spotify.is_connected():
        try:
            m = spotify.search_track(stem)
            if m:
                sp_html, style = render_spotify(m), m["style_name"]
        except Exception:
            sp_html = ""
    progress(1.0, desc="Done")
    return (render_profile(p), e, v, d, sp_html, style, stem,
            render_mood_map(log), render_reference(fp), fp_state)


def sp_match(query):
    if not spotify.is_connected():
        raise gr.Error("Connect Spotify first (set credentials, then Finish connecting).")
    m = spotify.search_track(query or "")
    if not m:
        return '<div class="mb-empty">No Spotify match for that title.</div>', None
    return render_spotify(m), m["style_name"]


def generate(song_path, energy, valence, dance, feedback, drum_gain, style,
             imitate, fp_state, progress=gr.Progress()):
    if not song_path:
        raise gr.Error("Upload a song first.")
    progress(0.1, desc="Generating drums…")
    fb = feedback if feedback == "auto" else float(feedback)
    target_density = None
    if imitate and fp_state and fp_state.get("has_drums"):
        target_density = fp_state["target_density"]
        fb = "auto"  # the density match needs the full feedback sweep to choose from
    r = process_song(song_path, mood=(energy, valence, dance), feedback=fb,
                     drum_gain=drum_gain, style=style, target_density=target_density)
    progress(1.0, desc="Done")
    return (r["mix"], r["drums"], r["accompaniment"], r["midi_path"], r["report"],
            render_sequencer(r["pattern"]), pattern_to_df(r["pattern"]))


def render_groove_fn(song_path, df, drum_gain, progress=gr.Progress()):
    if not song_path:
        raise gr.Error("Upload a song first.")
    progress(0.2, desc="Rendering your groove…")
    pattern = df_to_pattern(df)
    r = render_pattern(song_path, pattern, drum_gain=drum_gain)
    progress(1.0, desc="Done")
    return r["mix"], r["drums"]


def infill_fn(song_path, start, end, style, drum_gain, density, progress=gr.Progress()):
    if not song_path:
        raise gr.Error("Upload a song first.")
    if not masked_available():
        raise gr.Error("The infill model isn't trained yet (runs/masked_best.pt missing).")
    if start is None or end is None or float(end) <= float(start):
        raise gr.Error("Set an End time later than the Start time.")
    progress(0.2, desc="Reconstructing the break with the infill model…")
    # Density slider (0..1) -> hit threshold (higher density = lower threshold).
    # Calibrated for the ADT-context regime, where the sweet spot is ~0.6.
    threshold = max(0.35, min(0.72, 0.70 - float(density) * 0.32))
    r = infill_break(song_path, float(start), float(end), style=style,
                     drum_gain=drum_gain, threshold=threshold)
    progress(1.0, desc="Done")
    return r["mix"], render_sequencer(r["pattern"]), r["report"]


with gr.Blocks(title="MusiBot") as demo:
    style_state = gr.State(None)
    fp_state = gr.State(None)          # the song's own-drum fingerprint (Phase-1 imitation)
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
            imitate = gr.Checkbox(value=True, label="Imitate the song's own drums",
                                  info="Read the original drums and match their density + feel")
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
            gr.HTML('<div class="mb-h">Reference groove — the song\'s own drums</div>')
            ref_seq = gr.HTML(EMPTY_REF)
            gr.HTML('<div class="mb-h">Pattern — the generated groove</div>')
            sequencer = gr.HTML(EMPTY_SEQ)
            gr.HTML('<div class="mb-h">Edit the groove — tick steps to change it, then re-render</div>')
            gr.HTML('<div class="mb-note">Tweak the main 16-step pattern (it repeats every bar, '
                    'locked to the beat). Re-render to hear your version in place of the model\'s.</div>')
            groove_df = gr.Dataframe(
                value=EMPTY_GROOVE_DF, headers=DF_COLS, datatype=["str"] + ["bool"] * 16,
                row_count=5, column_count=17, interactive=True,
                label="Groove (rows = kit, columns = 16 steps)",
            )
            render_groove_btn = gr.Button("◉ Render my groove", variant="primary")
            groove_mix = gr.Audio(label="Your groove — final mix", type="numpy")
            groove_drums = gr.Audio(label="Your groove — drums only", type="numpy")
            gr.HTML('<div class="mb-h">Final mix — your song with our drums</div>')
            mix = gr.Audio(label="Final mix", type="numpy")
            drums = gr.Audio(label="Generated drums only", type="numpy")
            with gr.Accordion("Stems & details", open=False):
                accomp = gr.Audio(label="Drumless accompaniment (model input)", type="numpy")
                midi = gr.File(label="Generated drum MIDI")
                report = gr.Textbox(label="Diagnostics", lines=7)

    gr.HTML('<div class="mb-h">Mood map — where your songs sit (valence × energy)</div>')
    mood_map = gr.HTML(render_mood_map(load_log()))

    gr.HTML('<div class="mb-h">Fill a break — masked drum infill (Phase 2)</div>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML('<div class="mb-note">Reconstruct the drums in a time range with the '
                    'bidirectional infill model — it fills the gap from <b>both sides</b> + '
                    'the audio, in the song\'s own style.</div>')
            with gr.Row():
                infill_start = gr.Number(value=0.0, label="Start (sec)", minimum=0)
                infill_end = gr.Number(value=4.0, label="End (sec)", minimum=0)
            infill_density = gr.Slider(0, 1, value=0.45, step=0.05, label="Fill density",
                                       info="Higher → busier fill")
            infill_btn = gr.Button("◉ Fill the break", variant="primary", size="lg")
            infill_report = gr.Textbox(label="Infill info", lines=2)
        with gr.Column(scale=3):
            infill_mix = gr.Audio(label="Song with the break filled", type="numpy")
            infill_seq = gr.HTML(EMPTY_SEQ)

    song.upload(analyze, inputs=[song, imitate],
                outputs=[profile, energy, valence, dance, sp_card, style_state, sp_query,
                         mood_map, ref_seq, fp_state])
    song.clear(lambda: (EMPTY_PROFILE, 0.5, 0.5, 0.5, "", None, "", EMPTY_REF, None),
               outputs=[profile, energy, valence, dance, sp_card, style_state, sp_query,
                        ref_seq, fp_state])
    sp_match_btn.click(sp_match, inputs=sp_query, outputs=[sp_card, style_state])
    go.click(generate,
             inputs=[song, energy, valence, dance, feedback, drum_gain, style_state, imitate, fp_state],
             outputs=[mix, drums, accomp, midi, report, sequencer, groove_df])
    render_groove_btn.click(render_groove_fn,
                            inputs=[song, groove_df, drum_gain],
                            outputs=[groove_mix, groove_drums])
    infill_btn.click(infill_fn,
                     inputs=[song, infill_start, infill_end, style_state, drum_gain, infill_density],
                     outputs=[infill_mix, infill_seq, infill_report])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue=gr.themes.colors.red),
        css=CSS,
        js=_FORCE_DARK,
    )

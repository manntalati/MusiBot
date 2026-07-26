"""Optional Spotify metadata enrichment (album art, artist, album).

Uses the app-only **Client Credentials** flow — no user login, no browser
redirect. Set ``SPOTIFY_CLIENT_ID`` / ``SPOTIFY_CLIENT_SECRET`` (e.g. in a
gitignored ``.env`` at the repo root) to enable it; with no credentials the app
runs fully without it.

Note on the Feb-2026 API changes: under Client Credentials, Spotify now returns
``genres`` and ``popularity`` as ``null``, so those are surfaced only when
present. Search + album art + track/artist/album metadata still work.
"""

from __future__ import annotations

import base64
import os
import time

import requests

from .config import STYLE_VOCAB, UNKNOWN_STYLE_ID, canonical_style

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API = "https://api.spotify.com/v1"

# In-memory app-token cache (Client Credentials tokens are cheap to re-request).
_tok: dict = {"value": None, "exp": 0.0}


def _cfg() -> tuple[str | None, str | None]:
    return os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")


def is_configured() -> bool:
    cid, sec = _cfg()
    return bool(cid and sec)


def _token() -> str | None:
    if not is_configured():
        return None
    if _tok["value"] and time.time() < _tok["exp"] - 60:
        return _tok["value"]
    cid, sec = _cfg()
    basic = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    try:
        r = requests.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {basic}"},
            timeout=15,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    j = r.json()
    _tok["value"] = j["access_token"]
    _tok["exp"] = time.time() + j.get("expires_in", 3600)
    return _tok["value"]


def is_connected() -> bool:
    """True when credentials are set and a token can be obtained."""
    return _token() is not None


def genres_to_style(genres: list[str]) -> tuple[str, int]:
    """Best (style_name, style_id) from Spotify genre tags -> STYLE_VOCAB.

    Spotify genres are free-form ("nu metal", "alternative rock"); match by
    substring against the vocab so "nu metal" -> metal, etc. (Genres are often
    null under Client Credentials, in which case this returns ("unknown", 0).)
    """
    for g in genres:
        gk = g.lower().replace(" ", "").replace("-", "")
        sid = canonical_style(g.split()[-1] if g.split() else g)
        if sid != UNKNOWN_STYLE_ID:
            return STYLE_VOCAB[sid], sid
        for style in STYLE_VOCAB[1:]:
            if style in gk:
                return style, STYLE_VOCAB.index(style)
    return "unknown", UNKNOWN_STYLE_ID


def search_track(query: str) -> dict | None:
    """Best-match track metadata for ``query`` (title, or "title artist"), or None."""
    at = _token()
    if not at or not query.strip():
        return None
    headers = {"Authorization": f"Bearer {at}"}
    try:
        r = requests.get(
            f"{_API}/search",
            params={"q": query, "type": "track", "limit": 1},
            headers=headers,
            timeout=15,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    items = r.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    t = items[0]
    artist = t["artists"][0]
    genres: list[str] = []
    try:
        ar = requests.get(f"{_API}/artists/{artist['id']}", headers=headers, timeout=15)
        if ar.status_code == 200:
            genres = (ar.json().get("genres") or [])[:4]
    except requests.RequestException:
        pass
    style_name, style_id = genres_to_style(genres)
    images = t["album"].get("images", [])
    return {
        "name": t["name"],
        "artist": ", ".join(a["name"] for a in t["artists"]),
        "album": t["album"]["name"],
        "year": (t["album"].get("release_date") or "")[:4],
        "image": images[0]["url"] if images else None,
        "popularity": t.get("popularity"),
        "genres": genres,
        "url": t["external_urls"].get("spotify"),
        "style_name": style_name,
        "style_id": style_id,
    }


__all__ = ["is_configured", "is_connected", "search_track", "genres_to_style"]

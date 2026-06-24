"""
Best-effort confirmed-lineup fetcher (Sofascore unofficial endpoints).

Sofascore is behind Cloudflare and frequently returns 403 to server-side
requests, so this is a *best-effort* layer: every function returns None on any
failure and never raises. The reliable path is the manual drop-in
(src/lineups.load_lineup); this scraper is the automatic bonus when it works.

Endpoints (unofficial, may change):
  scheduled events : /api/v1/sport/football/scheduled-events/{YYYY-MM-DD}
  lineups          : /api/v1/event/{id}/lineups   (field "confirmed": bool)

Team names differ from ours (e.g. "USA" vs "United States"); NAME_MAP maps the
common cases. Returned XIs are lists of player-name strings (starters only),
ready for src.lineups.match_override.
"""

from __future__ import annotations

import json
import urllib.request

API = "https://api.sofascore.com/api/v1"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

# Sofascore team name → our dataset/squad name.
NAME_MAP = {
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


def canonical_team(name: str) -> str:
    return NAME_MAP.get(name, name)


def _get_json(url: str, timeout: float = 12.0) -> dict | None:
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def scheduled_events(date: str) -> list[dict]:
    """
    World Cup events scheduled on `date` (YYYY-MM-DD):
    [{event_id, home, away, kickoff_ts, status}]. Empty on any failure.
    """
    data = _get_json(f"{API}/sport/football/scheduled-events/{date}")
    if not data:
        return []
    out = []
    for e in data.get("events", []):
        tourn = (e.get("tournament", {}) or {}).get("name", "")
        if "World Cup" not in tourn:
            continue
        out.append({
            "event_id": e.get("id"),
            "home": canonical_team((e.get("homeTeam", {}) or {}).get("name", "")),
            "away": canonical_team((e.get("awayTeam", {}) or {}).get("name", "")),
            "kickoff_ts": e.get("startTimestamp"),
            "status": (e.get("status", {}) or {}).get("type", ""),
        })
    return out


def _starters(side: dict) -> list[str]:
    names = []
    for p in side.get("players", []):
        if p.get("substitute"):
            continue
        nm = (p.get("player", {}) or {}).get("name")
        if nm:
            names.append(nm)
    return names


def fetch_lineup(event_id: int) -> dict | None:
    """
    Confirmed starting XIs for an event:
        {"confirmed": bool, "home": [names], "away": [names]}
    Returns None if unavailable, not confirmed, or incomplete.
    """
    data = _get_json(f"{API}/event/{event_id}/lineups")
    if not data or not data.get("confirmed"):
        return None
    home = _starters(data.get("home", {}) or {})
    away = _starters(data.get("away", {}) or {})
    if len(home) < 11 or len(away) < 11:
        return None
    return {"confirmed": True, "home": home[:11], "away": away[:11],
            "source": "sofascore"}


def fetch_confirmed_for_match(date: str, home: str, away: str) -> dict | None:
    """Find the event for (date, home, away) and return its confirmed XI, or None."""
    for ev in scheduled_events(date):
        if ev["home"] == home and ev["away"] == away:
            if ev["event_id"] is None:
                return None
            lu = fetch_lineup(ev["event_id"])
            if lu:
                lu["kickoff_ts"] = ev["kickoff_ts"]
            return lu
    return None

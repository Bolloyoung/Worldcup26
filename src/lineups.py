"""
Confirmed starting-XI valuation.

Roughly an hour before kickoff the official teamsheets are released. Knowing
*which eleven* actually start — first-choice vs a rotated/rested side, a key
player out — is sharper match-specific signal than a season-long projected
squad. This module turns a confirmed XI into a per-match squad-strength index
in the same normalised space as the projected per-team index, so the
probability engine can use it as a `squad_override` for that single match.

Player metadata (position, age, league, nationality) comes from a database
built from data/squads/squads.json; players not in it fall back to a
team-typical default so an XI full of unknowns still values sensibly. The
signal is strongest for the teams whose squads are seeded — which are the
contenders that matter most for the forecast.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from .config import DATA_DIR
from .player_valuation import rate_first_eleven
from .squads import field_log_centre, load_squads, value_to_index

LINEUPS_DIR = DATA_DIR / "lineups"

# Minimum identifiable starters needed to trust an XI-based valuation. Below
# this we fall back to the team's projected index (or no signal), because the
# valuation would rest on too few known players.
MIN_KNOWN = 6


def _norm(name: str) -> str:
    """Accent/case-insensitive key for fuzzy name matching."""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    return n.lower().strip()


def player_database(squads: dict[str, list[dict]] | None = None) -> dict[str, dict]:
    """name-key → player metadata, from the projected squads."""
    if squads is None:
        squads = load_squads()
    db: dict[str, dict] = {}
    for players in squads.values():
        for p in players:
            db[_norm(p["name"])] = p
            # also index by surname only, to catch "Saka" vs "Bukayo Saka"
            parts = _norm(p["name"]).split()
            if len(parts) > 1:
                db.setdefault(parts[-1], p)
    return db


def _match_player(name: str, db: dict[str, dict]) -> dict | None:
    if not name:
        return None
    return db.get(_norm(name)) or db.get(_norm(name).split()[-1])


def enrich_xi(xi: list, db: dict[str, dict]) -> tuple[list[dict], int]:
    """
    Resolve the identifiable starters in a confirmed XI.

    `xi` items may be plain names (str) or dicts with name/position.
    Returns (known_player_dicts, n_unknown). Unknowns are not valued directly
    — they are imputed at the XI's own known-player average (see xi_value).
    """
    known: list[dict] = []
    unknown = 0
    for item in xi:
        name = item if isinstance(item, str) else item.get("name", "")
        pos = None if isinstance(item, str) else item.get("position")
        meta = _match_player(name, db)
        if meta:
            player = dict(meta)
            if pos:
                player["position"] = pos     # trust the teamsheet's position
            known.append(player)
        else:
            unknown += 1
    return known, unknown


def xi_value(team: str, xi: list, db=None) -> tuple[float | None, int, int]:
    """
    Estimated total value (EUR) of a confirmed XI: 11 × the average value of
    its identifiable starters (unknowns imputed at that average, so they don't
    distort). Returns (value_or_None, n_known, n_unknown); value is None when
    too few starters are identifiable to trust.
    """
    if db is None:
        db = player_database()
    known, unknown = enrich_xi(xi, db)
    n_known = len(known)
    if n_known < MIN_KNOWN:
        return None, n_known, unknown
    total_known = rate_first_eleven(known)["squad_total_value_eur"]
    return 11.0 * (total_known / n_known), n_known, unknown


def match_override(
    home: str, home_xi: list,
    away: str, away_xi: list,
    squads: dict[str, list[dict]] | None = None,
    projected: dict[str, float] | None = None,
) -> dict | None:
    """
    Per-match squad override from two confirmed XIs.

    Each team's index comes from its XI when enough starters are identifiable,
    otherwise it falls back to the team's projected index (or None → no signal
    for that team). Returns a dict with the (home_index, away_index) tuple
    under "override" plus diagnostics, or None if the field anchor is missing.
    """
    if squads is None:
        squads = load_squads()
    centre = field_log_centre(squads)
    if centre is None:
        return None
    if projected is None:
        from .squads import build_squad_index
        projected = build_squad_index(squads)
    db = player_database(squads)

    def team_index(team: str, xi: list):
        value, n_known, unknown = xi_value(team, xi, db)
        if value is None:                       # thin coverage → fall back
            return projected.get(team), value, n_known, unknown, "projected"
        return value_to_index(value, centre), value, n_known, unknown, "confirmed_xi"

    h_idx, hv, h_known, h_unknown, h_src = team_index(home, home_xi)
    a_idx, av, a_known, a_unknown, a_src = team_index(away, away_xi)
    return {
        "override": (h_idx, a_idx),
        "home_value_eur": round(hv) if hv else None,
        "away_value_eur": round(av) if av else None,
        "home_index": round(h_idx, 3) if h_idx else None,
        "away_index": round(a_idx, 3) if a_idx else None,
        "home_known": h_known, "home_unknown": h_unknown, "home_source": h_src,
        "away_known": a_known, "away_unknown": a_unknown, "away_source": a_src,
    }


# ── Manual / cached lineup files ──────────────────────────────────────────

def lineup_path(date: str, home: str, away: str) -> Path:
    safe = lambda s: s.replace(" ", "_").replace("/", "-")
    return LINEUPS_DIR / f"{date}_{safe(home)}_{safe(away)}.json"


def load_lineup(date: str, home: str, away: str) -> dict | None:
    """
    Load a confirmed lineup file if present. Format:
        {"confirmed": true, "home": [<11 names or dicts>],
         "away": [<11 names or dicts>], "source": "manual"|"sofascore"}
    """
    p = lineup_path(date, home, away)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if not data.get("home") or not data.get("away"):
        return None
    return data


def save_lineup(date: str, home: str, away: str, data: dict) -> Path:
    LINEUPS_DIR.mkdir(parents=True, exist_ok=True)
    p = lineup_path(date, home, away)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return p

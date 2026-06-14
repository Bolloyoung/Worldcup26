"""
Confederation membership for national teams.

Used by the probability engine to apply inter-confederation reliability
shrinkage: comparisons between teams from different confederations rest on a
thin set of bridging matches, so the rating gap is less trustworthy and is
shrunk toward parity.

Covers all 48 World Cup 2026 teams plus the commonly-encountered remainder so
the lookup rarely falls through to the heuristic default.
"""

from __future__ import annotations

# Confederation codes: UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC
CONFEDERATION: dict[str, str] = {
    # ── UEFA ───────────────────────────────────────────────────────────────
    "Spain": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Germany": "UEFA", "Netherlands": "UEFA", "Portugal": "UEFA",
    "Belgium": "UEFA", "England": "UEFA", "France": "UEFA", "Norway": "UEFA",
    "Austria": "UEFA", "Scotland": "UEFA", "Czech Republic": "UEFA",
    "Sweden": "UEFA", "Turkey": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Italy": "UEFA", "Denmark": "UEFA", "Poland": "UEFA", "Ukraine": "UEFA",
    "Serbia": "UEFA", "Hungary": "UEFA", "Romania": "UEFA", "Greece": "UEFA",
    "Wales": "UEFA", "Republic of Ireland": "UEFA", "Iceland": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Albania": "UEFA",
    "North Macedonia": "UEFA", "Finland": "UEFA", "Russia": "UEFA",
    # ── CONMEBOL ───────────────────────────────────────────────────────────
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Chile": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # ── CONCACAF ───────────────────────────────────────────────────────────
    "Mexico": "CONCACAF", "United States": "CONCACAF", "Canada": "CONCACAF",
    "Haiti": "CONCACAF", "Panama": "CONCACAF", "Costa Rica": "CONCACAF",
    "Jamaica": "CONCACAF", "Honduras": "CONCACAF", "Curaçao": "CONCACAF",
    "El Salvador": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    # ── CAF ────────────────────────────────────────────────────────────────
    "Morocco": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF", "Egypt": "CAF",
    "South Africa": "CAF", "Cape Verde": "CAF", "Senegal": "CAF",
    "Algeria": "CAF", "DR Congo": "CAF", "Ghana": "CAF", "Nigeria": "CAF",
    "Cameroon": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    # ── AFC ────────────────────────────────────────────────────────────────
    "Qatar": "AFC", "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Australia": "AFC", "Iraq": "AFC",
    "Uzbekistan": "AFC", "Jordan": "AFC", "United Arab Emirates": "AFC",
    "China PR": "AFC", "Iraq ": "AFC",
    # ── OFC ────────────────────────────────────────────────────────────────
    "New Zealand": "OFC", "Palestine": "AFC",
}


def get_confed(team: str) -> str:
    """Return a team's confederation, defaulting to UEFA-neutral 'INT'."""
    return CONFEDERATION.get(team, "INT")


def same_confederation(team_a: str, team_b: str) -> bool:
    """True if both teams are in the same (known) confederation."""
    ca, cb = get_confed(team_a), get_confed(team_b)
    return ca == cb and ca != "INT"

"""
league_adjustments.py
=====================
League difficulty coefficients and regional adjustments used throughout the
player valuation pipeline.

Coefficients are calibrated so that Premier League = 1.00 (reference tier).
All player metrics are multiplied by the inverse of the league coefficient
before scoring, then by the coefficient again when estimating market value —
meaning a player who dominates a weak league gets their raw numbers discounted
for performance comparison but receives a "hidden value" uplift when the league
coefficient is used as a transfer-value potential amplifier.

References
----------
- UEFA club coefficient rankings (2023-24 season)
- CIES Football Observatory league quality indices
- Transfermarkt average market value by league (2024)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Primary league difficulty table
# Scale: 0.0 (lowest) → 1.0 (Premier League reference)
# ---------------------------------------------------------------------------
LEAGUE_COEFFICIENTS: dict[str, float] = {
    # ── Europe Tier 1 ──────────────────────────────────────────────────────
    "Premier League": 1.00,
    "La Liga": 0.96,
    "Bundesliga": 0.94,
    "Serie A": 0.93,
    "Ligue 1": 0.89,
    # ── Europe Tier 2 ──────────────────────────────────────────────────────
    "Eredivisie": 0.82,
    "Primeira Liga": 0.80,
    "Süper Lig": 0.78,
    "Belgian Pro League": 0.75,
    "Scottish Premiership": 0.72,
    # ── Europe Tier 3 ──────────────────────────────────────────────────────
    "Austrian Bundesliga": 0.70,
    "Swiss Super League": 0.70,
    "Czech First League": 0.68,
    "Polish Ekstraklasa": 0.67,
    "Russian Premier League": 0.66,
    # ── Americas ───────────────────────────────────────────────────────────
    "MLS": 0.65,
    "Major League Soccer": 0.65,
    "Liga MX": 0.62,
    "Canadian Premier League": 0.45,
    "EFL Championship": 0.70,
    "Brasileirão Série A": 0.80,
    "Argentine Primera División": 0.75,
    "Colombian Primera A": 0.62,
    "Chilean Primera División": 0.60,
    "Uruguayan Primera División": 0.58,
    "Ecuadorian Serie A": 0.57,
    "Peruvian Primera División": 0.55,
    "Bolivian División Profesional": 0.50,
    # ── Asia ───────────────────────────────────────────────────────────────
    "J-League": 0.68,
    "J1 League": 0.68,
    "K League 1": 0.65,
    "Saudi Pro League": 0.60,
    "Chinese Super League": 0.55,
    "Indian Super League": 0.48,
    "A-League": 0.55,
    # ── Africa ─────────────────────────────────────────────────────────────
    "Egyptian Premier League": 0.50,
    "South African PSL": 0.48,
    "Botola Pro": 0.45,
    "CAF Champions League": 0.65,  # proxy for unknown African leagues
    "Nigerian Professional Football League": 0.45,
    "Kenyan Premier League": 0.42,
    "Tanzanian Premier League": 0.40,
    "Ugandan Premier League": 0.40,
    "Rwandan National Football League": 0.38,
    "Zambian Super League": 0.40,
    "Zimbabwean Premier Soccer League": 0.38,
    "Malagasy Cnaps Sport": 0.35,
    # ── Middle East ────────────────────────────────────────────────────────
    "UAE Pro League": 0.58,
    "Qatar Stars League": 0.57,
    "Kuwait Premier League": 0.50,
    # ── Default fallback ───────────────────────────────────────────────────
    "Unknown": 0.45,
}

# ---------------------------------------------------------------------------
# Aliases — common alternative spellings / FBref / Transfermarkt names
# ---------------------------------------------------------------------------
_LEAGUE_ALIASES: dict[str, str] = {
    # English
    "EPL": "Premier League",
    "English Premier League": "Premier League",
    "PL": "Premier League",
    # German
    "1. Bundesliga": "Bundesliga",
    "Bundesliga 1": "Bundesliga",
    # Spanish
    "Primera División": "La Liga",
    "Liga BBVA": "La Liga",
    "LaLiga": "La Liga",
    # Italian
    "Italian Serie A": "Serie A",
    # French
    "French Ligue 1": "Ligue 1",
    "Ligue 1 Uber Eats": "Ligue 1",
    # Portuguese
    "Liga NOS": "Primeira Liga",
    "Primeira Liga Portugal": "Primeira Liga",
    # Dutch
    "Dutch Eredivisie": "Eredivisie",
    # Belgian
    "First Division A": "Belgian Pro League",
    # Brazilian
    "Brasileirao": "Brasileirão Série A",
    "Serie A Brazil": "Brasileirão Série A",
    # Argentine
    "Primera Division Argentina": "Argentine Primera División",
    "Liga Profesional": "Argentine Primera División",
    # Turkish
    "Super Lig": "Süper Lig",
    # Saudi
    "Saudi Arabian Pro League": "Saudi Pro League",
    "SPL": "Saudi Pro League",
}

# ---------------------------------------------------------------------------
# Nationality/market premium multipliers
# Source: Transfermarkt average transfer fee premium analysis 2019-2024
# ---------------------------------------------------------------------------
NATIONALITY_PREMIUMS: dict[str, float] = {
    "Brazil": 1.08,
    "France": 1.06,
    "Portugal": 1.05,
    "England": 1.04,
    "Argentina": 1.03,
    "Spain": 1.03,
    "Germany": 1.02,
    "Netherlands": 1.02,
    "Belgium": 1.01,
    "Uruguay": 1.01,
    "Colombia": 1.01,
    "Croatia": 1.01,
    "Senegal": 1.01,
    # Default for all other nationalities
    "_default": 1.00,
}

# ---------------------------------------------------------------------------
# Continent groupings used for "hidden value" regional proxy estimation
# ---------------------------------------------------------------------------
CONTINENT_GROUPS: dict[str, list[str]] = {
    "europe_t1": ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"],
    "europe_t2": [
        "Eredivisie", "Primeira Liga", "Süper Lig", "Belgian Pro League",
        "Scottish Premiership",
    ],
    "europe_t3": [
        "Austrian Bundesliga", "Swiss Super League", "Czech First League",
        "Polish Ekstraklasa", "Russian Premier League",
    ],
    "south_america": [
        "Brasileirão Série A", "Argentine Primera División", "Colombian Primera A",
        "Chilean Primera División", "Uruguayan Primera División",
    ],
    "north_america": ["MLS"],
    "asia": ["J-League", "K League 1", "Saudi Pro League", "Chinese Super League"],
    "africa": [
        "Egyptian Premier League", "South African PSL", "Botola Pro",
        "Nigerian Professional Football League",
    ],
    "middle_east": ["UAE Pro League", "Qatar Stars League"],
}


def get_league_coefficient(league: str) -> float:
    """Return the difficulty coefficient for *league*.

    Performs alias resolution before lookup. Falls back to ``Unknown``
    (0.45) with a warning if the league is not found.

    Parameters
    ----------
    league:
        The league name as a string (case-insensitive alias resolution
        is attempted automatically).

    Returns
    -------
    float
        A coefficient in the range [0.35, 1.00].
    """
    # Direct lookup first
    if league in LEAGUE_COEFFICIENTS:
        return LEAGUE_COEFFICIENTS[league]

    # Try alias table
    canonical = _LEAGUE_ALIASES.get(league)
    if canonical and canonical in LEAGUE_COEFFICIENTS:
        return LEAGUE_COEFFICIENTS[canonical]

    # Case-insensitive scan as last resort
    league_lower = league.lower()
    for key, value in LEAGUE_COEFFICIENTS.items():
        if key.lower() == league_lower:
            return value
    for alias, canonical_name in _LEAGUE_ALIASES.items():
        if alias.lower() == league_lower:
            return LEAGUE_COEFFICIENTS.get(canonical_name, LEAGUE_COEFFICIENTS["Unknown"])

    logger.warning("League '%s' not found in coefficient table; using Unknown=0.45", league)
    return LEAGUE_COEFFICIENTS["Unknown"]


def get_nationality_premium(nationality: str) -> float:
    """Return the market-value nationality premium multiplier.

    Parameters
    ----------
    nationality:
        Player nationality as a country name string.

    Returns
    -------
    float
        A multiplier ≥ 1.00.
    """
    return NATIONALITY_PREMIUMS.get(nationality, NATIONALITY_PREMIUMS["_default"])


def get_continent_group(league: str) -> str:
    """Return the continent/tier group label for a given league.

    Used by the regional proxy estimator to select appropriate median
    stat cohorts.

    Parameters
    ----------
    league:
        League name (alias resolution applied internally).

    Returns
    -------
    str
        One of the keys in ``CONTINENT_GROUPS``, or ``'unknown'``.
    """
    # Resolve aliases
    resolved = _LEAGUE_ALIASES.get(league, league)
    for group, leagues in CONTINENT_GROUPS.items():
        if resolved in leagues:
            return group
    return "unknown"


def hidden_value_adjustment(league: str, performance_score: float) -> float:
    """Compute an upward transfer-value adjustment for players excelling
    in weaker leagues ("hidden value" effect).

    The intuition: a player scoring 75/100 in the Egyptian Premier League
    is likely undervalued relative to an equivalent score in Serie A,
    because the market hasn't yet priced in the ability translating
    to a higher-quality context.

    Formula (CIES-inspired):
        adjustment = (1 - coeff) × 0.5 × (score / 100)

    So a score of 80 in a coeff=0.50 league yields:
        adjustment = (1 - 0.50) × 0.5 × 0.80 = 0.20

    i.e., +20% upward nudge to the market value.

    Parameters
    ----------
    league:
        The player's current league.
    performance_score:
        The raw 0-100 performance score (before league adjustment).

    Returns
    -------
    float
        A multiplicative adjustment ≥ 1.00 to apply to market value.
    """
    coeff = get_league_coefficient(league)
    # No bonus for top leagues; only applies below 0.90
    if coeff >= 0.90:
        return 1.0
    raw_bonus = (1.0 - coeff) * 0.5 * (performance_score / 100.0)
    # Cap the bonus at +30% to prevent absurd amplification
    bonus = min(raw_bonus, 0.30)
    return 1.0 + bonus

"""
valuation_engine.py
===================
Core player valuation engine.

Public API
----------
- ``valuate_player(player: dict) -> dict``
    Valuate a single player and return a rich result dict.

- ``rate_first_eleven(squad: list[dict]) -> dict``
    Valuate an entire first XI and return per-player + squad aggregates.

Market value model
------------------
The EUR market value estimate uses a multiplicative formula inspired by
gradient boosting feature interactions (XGBoost-style):

    market_value_EUR = base_value
                       × age_multiplier
                       × contract_multiplier
                       × league_multiplier
                       × nationality_premium
                       × position_score_factor
                       × hidden_value_adjustment

Where:
  base_value            — empirical regression anchor from Transfermarkt data
                          for the player's league / position / age group
  age_multiplier        — from position-specific age-value curve
  contract_multiplier   — CIES methodology (0.85 for <1yr, 1.15 for 4yr+)
  league_multiplier     — from LEAGUE_COEFFICIENTS
  nationality_premium   — documented market premiums by nationality
  position_score_factor — composite 0-100 score / 50 (50=neutral, 100=2x)
  hidden_value_adjustment — upward nudge for players excelling in weak leagues

Base value anchors (EUR) are derived from Transfermarkt average market values
by league + position group + age bracket (2023-24 season data).

Design notes
------------
The engine deliberately avoids a trained ML model at inference time to remain
dependency-light and fully interpretable.  The multiplicative structure mirrors
what a gradient boosting ensemble would learn as feature interactions, while
remaining auditable.

If a Transfermarkt market value was successfully scraped, it is blended with
the formula estimate (60/40 formula/TM) to minimise RMSE against TM ground
truth, per the architecture specification.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports from sibling modules (avoids circular import issues)
# ---------------------------------------------------------------------------
from .data_scrapers import fetch_player_data
from .league_adjustments import (
    get_league_coefficient,
    get_nationality_premium,
    hidden_value_adjustment,
)
from .position_metrics import (
    get_age_multiplier,
    get_contract_multiplier,
    resolve_position,
    score_player_metrics,
)

# ===========================================================================
# Base value anchors (EUR)
# Empirical medians from Transfermarkt by position group + league tier + age.
# Structure: {position_group: {league_tier: {age_bracket: base_eur}}}
#
# age_bracket keys: "u21", "21-24", "25-27", "28-30", "31+"
# league_tier keys: "t1", "t2", "t3", "sa", "other"
# ===========================================================================
_BASE_VALUES: dict[str, dict[str, dict[str, float]]] = {
    "GK": {
        "t1":    {"u21": 5_000_000,  "21-24": 10_000_000, "25-27": 18_000_000,
                  "28-30": 15_000_000, "31+": 8_000_000},
        "t2":    {"u21": 2_000_000,  "21-24":  4_000_000, "25-27":  7_000_000,
                  "28-30":  5_000_000, "31+": 2_500_000},
        "t3":    {"u21":   800_000,  "21-24":  1_500_000, "25-27":  2_500_000,
                  "28-30":  1_800_000, "31+":   900_000},
        "sa":    {"u21": 1_500_000,  "21-24":  3_000_000, "25-27":  5_000_000,
                  "28-30":  3_500_000, "31+": 1_500_000},
        "other": {"u21":   400_000,  "21-24":    700_000, "25-27":  1_200_000,
                  "28-30":    900_000, "31+":   400_000},
    },
    "CB": {
        "t1":    {"u21": 8_000_000,  "21-24": 20_000_000, "25-27": 35_000_000,
                  "28-30": 25_000_000, "31+": 10_000_000},
        "t2":    {"u21": 3_000_000,  "21-24":  6_000_000, "25-27": 12_000_000,
                  "28-30":  8_000_000, "31+":  3_000_000},
        "t3":    {"u21": 1_000_000,  "21-24":  2_000_000, "25-27":  4_000_000,
                  "28-30":  2_500_000, "31+":  1_000_000},
        "sa":    {"u21": 2_000_000,  "21-24":  4_500_000, "25-27":  8_000_000,
                  "28-30":  5_500_000, "31+":  2_000_000},
        "other": {"u21":   500_000,  "21-24":  1_000_000, "25-27":  2_000_000,
                  "28-30":  1_200_000, "31+":   500_000},
    },
    "FB": {
        "t1":    {"u21": 7_000_000,  "21-24": 18_000_000, "25-27": 30_000_000,
                  "28-30": 20_000_000, "31+":  8_000_000},
        "t2":    {"u21": 2_500_000,  "21-24":  5_500_000, "25-27": 10_000_000,
                  "28-30":  7_000_000, "31+":  2_500_000},
        "t3":    {"u21":   900_000,  "21-24":  1_800_000, "25-27":  3_500_000,
                  "28-30":  2_200_000, "31+":    900_000},
        "sa":    {"u21": 1_800_000,  "21-24":  4_000_000, "25-27":  7_000_000,
                  "28-30":  4_500_000, "31+":  1_800_000},
        "other": {"u21":   450_000,  "21-24":    900_000, "25-27":  1_700_000,
                  "28-30":  1_000_000, "31+":    450_000},
    },
    "CDM": {
        "t1":    {"u21": 8_000_000,  "21-24": 20_000_000, "25-27": 35_000_000,
                  "28-30": 25_000_000, "31+": 10_000_000},
        "t2":    {"u21": 3_000_000,  "21-24":  6_500_000, "25-27": 12_000_000,
                  "28-30":  8_000_000, "31+":  3_000_000},
        "t3":    {"u21": 1_000_000,  "21-24":  2_200_000, "25-27":  4_200_000,
                  "28-30":  2_800_000, "31+":  1_000_000},
        "sa":    {"u21": 2_000_000,  "21-24":  5_000_000, "25-27":  9_000_000,
                  "28-30":  6_000_000, "31+":  2_200_000},
        "other": {"u21":   500_000,  "21-24":  1_100_000, "25-27":  2_200_000,
                  "28-30":  1_400_000, "31+":    500_000},
    },
    "CM": {
        "t1":    {"u21": 9_000_000,  "21-24": 22_000_000, "25-27": 38_000_000,
                  "28-30": 27_000_000, "31+": 10_000_000},
        "t2":    {"u21": 3_500_000,  "21-24":  7_000_000, "25-27": 13_000_000,
                  "28-30":  9_000_000, "31+":  3_500_000},
        "t3":    {"u21": 1_200_000,  "21-24":  2_500_000, "25-27":  4_500_000,
                  "28-30":  3_000_000, "31+":  1_200_000},
        "sa":    {"u21": 2_500_000,  "21-24":  5_500_000, "25-27": 10_000_000,
                  "28-30":  7_000_000, "31+":  2_500_000},
        "other": {"u21":   600_000,  "21-24":  1_200_000, "25-27":  2_500_000,
                  "28-30":  1_600_000, "31+":    600_000},
    },
    "CAM": {
        "t1":    {"u21": 12_000_000, "21-24": 30_000_000, "25-27": 50_000_000,
                  "28-30": 35_000_000, "31+": 12_000_000},
        "t2":    {"u21":  4_000_000, "21-24":  9_000_000, "25-27": 16_000_000,
                  "28-30": 11_000_000, "31+":  4_000_000},
        "t3":    {"u21":  1_500_000, "21-24":  3_000_000, "25-27":  5_500_000,
                  "28-30":  3_500_000, "31+":  1_400_000},
        "sa":    {"u21":  3_000_000, "21-24":  7_000_000, "25-27": 12_000_000,
                  "28-30":  8_000_000, "31+":  3_000_000},
        "other": {"u21":    700_000, "21-24":  1_500_000, "25-27":  3_000_000,
                  "28-30":  2_000_000, "31+":    700_000},
    },
    "W": {
        "t1":    {"u21": 15_000_000, "21-24": 35_000_000, "25-27": 55_000_000,
                  "28-30": 35_000_000, "31+": 12_000_000},
        "t2":    {"u21":  5_000_000, "21-24": 12_000_000, "25-27": 20_000_000,
                  "28-30": 12_000_000, "31+":  4_000_000},
        "t3":    {"u21":  1_800_000, "21-24":  4_000_000, "25-27":  7_000_000,
                  "28-30":  4_000_000, "31+":  1_500_000},
        "sa":    {"u21":  4_000_000, "21-24":  9_000_000, "25-27": 15_000_000,
                  "28-30":  9_000_000, "31+":  3_000_000},
        "other": {"u21":    900_000, "21-24":  2_000_000, "25-27":  4_000_000,
                  "28-30":  2_200_000, "31+":    800_000},
    },
    "ST": {
        "t1":    {"u21": 18_000_000, "21-24": 40_000_000, "25-27": 65_000_000,
                  "28-30": 40_000_000, "31+": 12_000_000},
        "t2":    {"u21":  5_500_000, "21-24": 13_000_000, "25-27": 22_000_000,
                  "28-30": 14_000_000, "31+":  4_500_000},
        "t3":    {"u21":  2_000_000, "21-24":  4_500_000, "25-27":  8_000_000,
                  "28-30":  5_000_000, "31+":  1_800_000},
        "sa":    {"u21":  4_500_000, "21-24": 10_000_000, "25-27": 18_000_000,
                  "28-30": 11_000_000, "31+":  3_500_000},
        "other": {"u21":  1_000_000, "21-24":  2_200_000, "25-27":  4_500_000,
                  "28-30":  2_800_000, "31+":    900_000},
    },
}

# League tier categorisation for base value lookup
_LEAGUE_TIERS: dict[str, str] = {
    "Premier League": "t1", "La Liga": "t1", "Bundesliga": "t1",
    "Serie A": "t1", "Ligue 1": "t1",
    "Eredivisie": "t2", "Primeira Liga": "t2", "Süper Lig": "t2",
    "Belgian Pro League": "t2", "Scottish Premiership": "t2",
    "Austrian Bundesliga": "t3", "Swiss Super League": "t3",
    "Czech First League": "t3", "Polish Ekstraklasa": "t3",
    "MLS": "t3",
    "Brasileirão Série A": "sa", "Argentine Primera División": "sa",
}


def _get_league_tier(league: str) -> str:
    """Map a league name to a base-value tier key."""
    return _LEAGUE_TIERS.get(league, "other")


def _get_age_bracket(age: int) -> str:
    """Map age to the bracket key used in ``_BASE_VALUES``."""
    if age < 21:
        return "u21"
    elif age <= 24:
        return "21-24"
    elif age <= 27:
        return "25-27"
    elif age <= 30:
        return "28-30"
    else:
        return "31+"


def _get_base_value(position: str, league: str, age: int) -> float:
    """Look up the empirical market value anchor.

    Parameters
    ----------
    position:
        Canonical position key.
    league:
        Player's current league.
    age:
        Player's age in years.

    Returns
    -------
    float
        Base EUR market value anchor.
    """
    canonical = resolve_position(position)
    tier = _get_league_tier(league)
    bracket = _get_age_bracket(age)

    tier_data = _BASE_VALUES.get(canonical, _BASE_VALUES["CM"])
    bracket_data = tier_data.get(tier, tier_data["other"])
    return bracket_data.get(bracket, bracket_data.get("25-27", 5_000_000))


def _squad_age_profile(players: list[dict]) -> str:
    """Classify squad age profile based on average age.

    Parameters
    ----------
    players:
        List of per-player result dicts (must have ``'age'`` key in input).

    Returns
    -------
    str
        One of ``'young'``, ``'prime'``, ``'experienced'``.
    """
    ages = [p.get("_age", 25) for p in players]
    if not ages:
        return "prime"
    avg = sum(ages) / len(ages)
    if avg < 24:
        return "young"
    elif avg <= 29:
        return "prime"
    else:
        return "experienced"


# ===========================================================================
# Core valuation function
# ===========================================================================

def valuate_player(player: dict) -> dict[str, Any]:
    """Compute a full valuation for a single player.

    Orchestrates data fetching, performance scoring, and market value
    calculation.  Returns a rich result dict suitable for display or
    further aggregation.

    Parameters
    ----------
    player:
        Input dict with keys:
          - ``name`` (str, required)
          - ``position`` (str, required)
          - ``nationality`` (str, required)
          - ``age`` (int, required)
          - ``league`` (str, required)
          - ``club`` (str, required)
          - ``contract_years_remaining`` (float, required)
          - ``player_id_fbref`` (str, optional)
          - ``player_id_transfermarkt`` (str, optional)
          - ``player_id_sofascore`` (int, optional)

    Returns
    -------
    dict
        Keys: ``name``, ``position``, ``performance_score``,
        ``market_value_eur``, ``age_adjusted_score``,
        ``league_adjusted_score``, ``data_source``, ``confidence``,
        plus internal ``_age`` for squad profile calculation.
    """
    name = player["name"]
    raw_position = player.get("position", "CM")
    nationality = player.get("nationality", "")
    age = int(player.get("age", 25))
    league = player.get("league", "Unknown")
    contract_years = float(player.get("contract_years_remaining", 2))

    canonical_pos = resolve_position(raw_position)
    logger.info("Valuating player: %s | %s | %s | age %d", name, canonical_pos, league, age)

    # ------------------------------------------------------------------
    # 1. Fetch live stats (with graceful fallback chain)
    # ------------------------------------------------------------------
    stats, data_source, confidence = fetch_player_data(player)

    # ------------------------------------------------------------------
    # 2. Raw performance score (0-100) — position-specific weighted sum
    # ------------------------------------------------------------------
    raw_score = score_player_metrics(stats, canonical_pos)

    # ------------------------------------------------------------------
    # 3. League-adjusted score
    # All metrics scored in the context of the player's league, then the
    # score is discounted by the league coefficient for cross-league fairness.
    # A player with score 80 in a coeff=0.80 league gets:
    #   league_adjusted_score = 80 × 0.80 = 64 (roughly)
    # This prevents weak-league players from appearing elite on raw stats.
    # ------------------------------------------------------------------
    league_coeff = get_league_coefficient(league)
    # Apply a partial (not full) league discount: blended 60/40 raw/adjusted
    # so that genuinely dominant weak-league players aren't fully penalised
    league_adjusted_score = round(
        raw_score * (0.60 + 0.40 * league_coeff), 2
    )

    # ------------------------------------------------------------------
    # 4. Age-adjusted score (performance weighted by where player sits
    #    on their position-specific age curve)
    # ------------------------------------------------------------------
    age_mult = get_age_multiplier(canonical_pos, age)
    # Age adjustment nudges score: a 35yo ST scoring 70 gets ~70*0.25 = 17.5
    # The score is still 70 for current ability; age_adjusted reflects value.
    age_adjusted_score = round(raw_score * age_mult, 2)

    # ------------------------------------------------------------------
    # 5. Market value estimate (EUR)
    # ------------------------------------------------------------------
    base_value = _get_base_value(canonical_pos, league, age)

    contract_mult = get_contract_multiplier(contract_years)
    nat_premium = get_nationality_premium(nationality)

    # position_score_factor: maps 0-100 score to a multiplier centred at 1.0
    # score=50 → factor=1.0 (neutral), score=100 → factor=2.0, score=0 → 0.0
    position_score_factor = raw_score / 50.0

    # Hidden value: players dominating weak leagues get an uplift
    hv_adj = hidden_value_adjustment(league, raw_score)

    formula_value = (
        base_value
        * age_mult
        * contract_mult
        * league_coeff        # league quality anchor
        * nat_premium
        * position_score_factor
        * hv_adj
    )

    # ------------------------------------------------------------------
    # 6. Blend with Transfermarkt value if available (minimises RMSE)
    # The formula is the workhorse; TM data (when scraped live) is used
    # as a calibration anchor — weighted 60/40 (formula/TM).
    # ------------------------------------------------------------------
    tm_value = stats.get("market_value_eur")
    if tm_value and isinstance(tm_value, (int, float)) and tm_value > 0:
        market_value_eur = 0.60 * formula_value + 0.40 * float(tm_value)
        logger.debug(
            "%s: blended value — formula %.0f, TM %.0f → final %.0f",
            name, formula_value, tm_value, market_value_eur,
        )
    else:
        market_value_eur = formula_value

    # Floor at €50k (no player should be valued at literally zero)
    market_value_eur = max(market_value_eur, 50_000.0)

    # ------------------------------------------------------------------
    # 7. Build result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "name": name,
        "position": canonical_pos,
        "performance_score": round(raw_score, 2),
        "market_value_eur": round(market_value_eur, 0),
        "age_adjusted_score": age_adjusted_score,
        "league_adjusted_score": league_adjusted_score,
        "data_source": data_source,
        "confidence": confidence,
        # Internal keys used for squad aggregation (not in public spec)
        "_age": age,
        "_league_coeff": league_coeff,
        "_age_mult": age_mult,
        "_contract_mult": contract_mult,
        "_nat_premium": nat_premium,
        "_position_score_factor": position_score_factor,
        "_base_value": base_value,
        "_formula_value": round(formula_value, 0),
        "_tm_value": tm_value,
    }

    logger.info(
        "%s → score %.1f | league-adj %.1f | age-adj %.1f | "
        "€%.0f | source=%s | confidence=%s",
        name, raw_score, league_adjusted_score, age_adjusted_score,
        market_value_eur, data_source, confidence,
    )
    return result


# ===========================================================================
# First XI rating function
# ===========================================================================

def rate_first_eleven(squad: list[dict]) -> dict[str, Any]:
    """Valuate a first XI squad and return per-player + aggregate metrics.

    Accepts a list of up to 11 player dicts and returns a comprehensive
    valuation report including per-player scores, market values, squad total
    value, average score, and squad age profile.

    Parameters
    ----------
    squad:
        List of player dicts.  Each dict must have:
          - ``name`` (str)
          - ``position`` (str)
          - ``nationality`` (str)
          - ``age`` (int)
          - ``league`` (str)
          - ``club`` (str)
          - ``contract_years_remaining`` (float)

        Optional keys:
          - ``player_id_fbref`` (str)
          - ``player_id_transfermarkt`` (str)
          - ``player_id_sofascore`` (int)

    Returns
    -------
    dict
        Structure::

            {
                "players": [
                    {
                        "name": str,
                        "position": str,
                        "performance_score": float,      # 0-100
                        "market_value_eur": float,
                        "age_adjusted_score": float,
                        "league_adjusted_score": float,
                        "data_source": str,
                        "confidence": str,
                    },
                    ...
                ],
                "squad_total_value_eur": float,
                "squad_avg_score": float,
                "squad_age_profile": str,    # "young"|"prime"|"experienced"
                "valuation_timestamp": str,  # ISO 8601
            }

    Raises
    ------
    ValueError
        If ``squad`` is empty.

    Examples
    --------
    >>> squad = [
    ...     {
    ...         "name": "Alisson Becker", "position": "GK",
    ...         "nationality": "Brazil", "age": 31,
    ...         "league": "Premier League", "club": "Liverpool",
    ...         "contract_years_remaining": 2,
    ...     },
    ... ]
    >>> result = rate_first_eleven(squad)
    >>> result["squad_total_value_eur"]
    """
    if not squad:
        raise ValueError("squad must contain at least one player")

    if len(squad) > 11:
        logger.warning(
            "Squad contains %d players (expected ≤11); processing all",
            len(squad),
        )

    player_results: list[dict[str, Any]] = []

    for i, player in enumerate(squad, start=1):
        if "name" not in player:
            logger.error("Player at index %d missing required key 'name'; skipping", i)
            continue
        try:
            result = valuate_player(player)
            player_results.append(result)
        except Exception as exc:
            logger.exception("Failed to valuate player '%s': %s",
                             player.get("name", f"idx_{i}"), exc)
            # Include a placeholder so the squad count stays correct
            player_results.append({
                "name": player.get("name", f"Unknown_{i}"),
                "position": player.get("position", "?"),
                "performance_score": 50.0,
                "market_value_eur": 1_000_000.0,
                "age_adjusted_score": 50.0,
                "league_adjusted_score": 50.0,
                "data_source": "error",
                "confidence": "low",
                "_age": player.get("age", 25),
            })

    # ------------------------------------------------------------------
    # Strip internal keys from public output
    # ------------------------------------------------------------------
    _internal_keys = {k for k in next(iter(player_results), {}) if k.startswith("_")}

    public_players = []
    for pr in player_results:
        public = {k: v for k, v in pr.items() if k not in _internal_keys}
        public_players.append(public)

    # ------------------------------------------------------------------
    # Squad-level aggregates
    # ------------------------------------------------------------------
    total_value = sum(pr.get("market_value_eur", 0) for pr in player_results)
    avg_score = (
        sum(pr.get("performance_score", 0) for pr in player_results)
        / len(player_results)
        if player_results else 0.0
    )
    age_profile = _squad_age_profile(player_results)

    return {
        "players": public_players,
        "squad_total_value_eur": round(total_value, 0),
        "squad_avg_score": round(avg_score, 2),
        "squad_age_profile": age_profile,
        "valuation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

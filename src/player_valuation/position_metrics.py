"""
position_metrics.py
===================
Position-specific metric weight tables and scoring helpers.

Each position group has a dict mapping stat keys → weight (weights sum to 1.0).
VAEP-inspired zone weighting is applied on top: defensive actions recorded in
the player's own penalty box or defensive third receive a 1.35× multiplier,
reflecting that winning a tackle inside your own box has higher expected-value
impact than winning one in midfield.

Stat key naming convention
--------------------------
Keys follow FBref column naming where possible so that scraped DataFrames can
be merged with the weight table directly. Transfermarkt / Sofascore keys are
mapped to the same names in data_scrapers.py.

References
----------
- Pappalardo et al. (2019) "PlayeRank" — multi-dimensional performance rating
- Decroos et al. (2019) "VAEP" — valuing actions by estimating probabilities
- StatsBomb open data positional analysis reports (2022-2024)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Position weight tables
# All weights in each block sum to exactly 1.00.
# Negative weights indicate "cost" metrics (more = worse).
# ---------------------------------------------------------------------------

POSITION_WEIGHTS: dict[str, dict[str, float]] = {

    # ── Goalkeeper ─────────────────────────────────────────────────────────
    "GK": {
        "psxg_minus_ga_per90": 0.30,    # Post-Shot xG - Goals Against (saves above expected)
        "save_pct": 0.20,               # Save percentage
        "sweeper_actions_per90": 0.15,  # Keeper sweeping actions (def. 6-yd box exits)
        "launch_accuracy_pct": 0.20,    # Distribution accuracy (pass/launch completion %)
        "clean_sheet_pct": 0.15,        # Clean sheet rate (adj. for team quality)
    },

    # ── Centre-Back ────────────────────────────────────────────────────────
    "CB": {
        "aerial_duel_win_pct": 0.20,         # % aerial duels won
        "ground_duel_win_pct": 0.20,         # % ground defensive duels won
        "interceptions_per90": 0.20,         # Interceptions per 90 (normalised by possession)
        "progressive_passes_per90": 0.15,    # Progressive passes per 90
        "pressure_success_rate": 0.15,       # % pressures that win the ball
        "errors_leading_to_shot_per90": -0.10,  # Errors leading to shot/goal (negative)
    },

    # ── Full-Back / Wing-Back ──────────────────────────────────────────────
    "FB": {
        "defensive_duel_win_pct": 0.20,      # % defensive duels won
        "progressive_carries_per90": 0.20,   # Progressive ball carries per 90
        "xa_per90": 0.20,                    # Expected assists per 90
        "crossing_accuracy_pct": 0.15,       # Cross completion %
        "recoveries_per90": 0.15,            # Ball recoveries per 90
        "key_passes_per90": 0.10,            # Key passes per 90
    },

    # Full-back alias used when scrapers detect "RB" / "LB"
    "RB": None,   # resolved to FB at runtime
    "LB": None,   # resolved to FB at runtime
    "WB": None,   # resolved to FB at runtime

    # ── Defensive Midfielder ───────────────────────────────────────────────
    "CDM": {
        "interceptions_per90": 0.25,         # Interceptions per 90
        "ground_duel_win_pct": 0.20,         # % ground duels won
        "pressures_per90": 0.20,             # Pressures applied per 90
        "progressive_passes_per90": 0.20,    # Progressive passes per 90
        "recoveries_per90": 0.15,            # Ball recoveries per 90
    },

    # ── Central Midfielder ─────────────────────────────────────────────────
    "CM": {
        "progressive_passes_per90": 0.20,    # Progressive passes per 90
        "xa_per90": 0.20,                    # xA per 90
        "key_passes_per90": 0.15,            # Key passes per 90
        "pass_accuracy_pct": 0.15,           # Pass accuracy %
        "pressures_per90": 0.15,             # Pressures applied per 90
        "duel_win_pct": 0.15,                # Overall duel win %
    },

    # ── Attacking Midfielder ───────────────────────────────────────────────
    "CAM": {
        "xa_per90": 0.25,                    # xA per 90
        "key_passes_per90": 0.20,            # Key passes per 90
        "xg_per90": 0.20,                    # xG per 90
        "successful_dribbles_per90": 0.20,   # Successful take-ons per 90
        "passes_into_penalty_area_per90": 0.15,  # Passes into the box per 90
    },

    # ── Winger ────────────────────────────────────────────────────────────
    "W": {
        "xg_per90": 0.20,                    # xG per 90
        "xa_per90": 0.20,                    # xA per 90
        "successful_dribbles_per90": 0.20,   # Successful dribbles/take-ons per 90
        "progressive_carries_per90": 0.15,   # Progressive carries per 90
        "takeon_success_pct": 0.15,          # Take-on (dribble attempt) success %
        "shot_creating_actions_per90": 0.10, # SCAs per 90
    },

    # Winger aliases
    "RW": None,   # resolved to W at runtime
    "LW": None,   # resolved to W at runtime

    # ── Striker ────────────────────────────────────────────────────────────
    "ST": {
        "xg_per90": 0.30,                    # xG per 90
        "goals_per90": 0.20,                 # Actual goals per 90
        "shot_conversion_rate": 0.20,        # Goals / shots on target
        "xa_per90": 0.10,                    # xA per 90 (link-up play)
        "aerial_duel_win_pct": 0.10,         # Aerial win % (target strikers)
        "pressures_per90": 0.10,             # Pressing intensity
    },

    # Striker alias
    "CF": None,   # resolved to ST at runtime
    "SS": None,   # Second striker — resolved to ST at runtime
}

# Canonical position aliases (resolved before any weight lookup)
_POSITION_ALIASES: dict[str, str] = {
    "RB": "FB",
    "LB": "FB",
    "WB": "FB",
    "RWB": "FB",
    "LWB": "FB",
    "RW": "W",
    "LW": "W",
    "AM": "CAM",
    "DM": "CDM",
    "CF": "ST",
    "SS": "ST",
    # FBref long names
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Right-Back": "FB",
    "Left-Back": "FB",
    "Defensive Midfield": "CDM",
    "Central Midfield": "CM",
    "Attacking Midfield": "CAM",
    "Right Winger": "W",
    "Left Winger": "W",
    "Centre-Forward": "ST",
}

# ---------------------------------------------------------------------------
# VAEP-inspired zone weights
# Multiply the base metric weight by these when the action is recorded in
# the specified pitch zone.  Zones follow StatsBomb pitch orientation.
# ---------------------------------------------------------------------------
ZONE_WEIGHTS: dict[str, float] = {
    "own_penalty_box": 1.35,     # Highest danger area for defensive actions
    "defensive_third": 1.20,     # Elevated risk zone
    "middle_third": 1.00,        # Reference (neutral)
    "attacking_third": 0.85,     # Defensive action here = relatively low risk
    "opponent_penalty_box": 0.75,  # Defensive action deep in opp. box is unusual
}

# ---------------------------------------------------------------------------
# Age-value curve lookup tables, by position group
# Maps age → multiplier on base valuation
# Derived from CIES Football Observatory age-value profiles (2024)
# ---------------------------------------------------------------------------
AGE_MULTIPLIERS: dict[str, dict[int, float]] = {
    "GK": {
        # GKs peak later and retain value longer
        17: 0.55, 18: 0.65, 19: 0.72, 20: 0.78,
        21: 0.82, 22: 0.86, 23: 0.90, 24: 0.94,
        25: 0.97, 26: 0.99, 27: 1.00, 28: 1.00,
        29: 1.00, 30: 0.99, 31: 0.97, 32: 0.93,
        33: 0.88, 34: 0.80, 35: 0.70, 36: 0.58,
        37: 0.45, 38: 0.35, 39: 0.25, 40: 0.18,
    },
    "CB": {
        17: 0.50, 18: 0.60, 19: 0.68, 20: 0.76,
        21: 0.82, 22: 0.87, 23: 0.92, 24: 0.96,
        25: 1.00, 26: 1.00, 27: 1.00, 28: 0.98,
        29: 0.95, 30: 0.90, 31: 0.83, 32: 0.74,
        33: 0.64, 34: 0.54, 35: 0.43, 36: 0.33,
        37: 0.25, 38: 0.18, 39: 0.13,
    },
    "FB": {
        17: 0.52, 18: 0.62, 19: 0.71, 20: 0.79,
        21: 0.85, 22: 0.90, 23: 0.95, 24: 0.98,
        25: 1.00, 26: 1.00, 27: 0.99, 28: 0.96,
        29: 0.92, 30: 0.85, 31: 0.76, 32: 0.65,
        33: 0.54, 34: 0.43, 35: 0.33, 36: 0.24,
        37: 0.17, 38: 0.12,
    },
    "CDM": {
        17: 0.50, 18: 0.60, 19: 0.69, 20: 0.77,
        21: 0.83, 22: 0.88, 23: 0.93, 24: 0.97,
        25: 1.00, 26: 1.00, 27: 1.00, 28: 0.97,
        29: 0.93, 30: 0.86, 31: 0.77, 32: 0.67,
        33: 0.56, 34: 0.46, 35: 0.36, 36: 0.27,
        37: 0.20, 38: 0.14, 39: 0.10,
    },
    "CM": {
        17: 0.52, 18: 0.62, 19: 0.71, 20: 0.79,
        21: 0.85, 22: 0.91, 23: 0.95, 24: 0.98,
        25: 1.00, 26: 1.00, 27: 1.00, 28: 0.97,
        29: 0.92, 30: 0.85, 31: 0.76, 32: 0.65,
        33: 0.54, 34: 0.43, 35: 0.33, 36: 0.24,
        37: 0.17, 38: 0.12,
    },
    "CAM": {
        17: 0.55, 18: 0.65, 19: 0.74, 20: 0.81,
        21: 0.87, 22: 0.92, 23: 0.97, 24: 0.99,
        25: 1.00, 26: 1.00, 27: 0.98, 28: 0.95,
        29: 0.89, 30: 0.81, 31: 0.71, 32: 0.60,
        33: 0.49, 34: 0.39, 35: 0.30, 36: 0.22,
        37: 0.15, 38: 0.10,
    },
    "W": {
        17: 0.58, 18: 0.68, 19: 0.77, 20: 0.84,
        21: 0.90, 22: 0.95, 23: 1.00, 24: 1.00,
        25: 1.00, 26: 0.98, 27: 0.94, 28: 0.88,
        29: 0.80, 30: 0.71, 31: 0.60, 32: 0.50,
        33: 0.40, 34: 0.31, 35: 0.23, 36: 0.16,
        37: 0.11, 38: 0.07,
    },
    "ST": {
        17: 0.55, 18: 0.65, 19: 0.74, 20: 0.82,
        21: 0.88, 22: 0.93, 23: 1.00, 24: 1.00,
        25: 1.00, 26: 0.99, 27: 0.96, 28: 0.91,
        29: 0.84, 30: 0.75, 31: 0.64, 32: 0.53,
        33: 0.43, 34: 0.33, 35: 0.25, 36: 0.18,
        37: 0.12, 38: 0.08,
    },
}

# Contract length → multiplier (CIES methodology)
CONTRACT_MULTIPLIERS: dict[int, float] = {
    0: 0.85,   # <1 year remaining (or expired)
    1: 0.93,
    2: 1.00,
    3: 1.07,
    4: 1.15,
    # 5+ years: treated as 4
}


def resolve_position(position: str) -> str:
    """Resolve a position string (including aliases) to a canonical key.

    Parameters
    ----------
    position:
        Raw position string from scraper or squad input.

    Returns
    -------
    str
        Canonical position key used in POSITION_WEIGHTS and AGE_MULTIPLIERS.
        Falls back to ``'CM'`` with a warning if unrecognised.
    """
    pos = position.strip().upper()
    # Direct hit
    if pos in POSITION_WEIGHTS and POSITION_WEIGHTS[pos] is not None:
        return pos
    # Alias table (original case or upper)
    for alias, canonical in _POSITION_ALIASES.items():
        if alias.upper() == pos:
            return canonical
    logger.warning("Position '%s' not recognised; defaulting to CM", position)
    return "CM"


def get_position_weights(position: str) -> dict[str, float]:
    """Return the metric weight dict for a given position.

    Parameters
    ----------
    position:
        Position string (aliases resolved automatically).

    Returns
    -------
    dict[str, float]
        Metric name → weight mapping.
    """
    canonical = resolve_position(position)
    weights = POSITION_WEIGHTS.get(canonical)
    if weights is None:
        # Should not happen after resolve_position, but guard anyway
        logger.error("No weights found for position '%s'; using CM weights", canonical)
        weights = POSITION_WEIGHTS["CM"]
    return weights


def get_age_multiplier(position: str, age: int) -> float:
    """Look up the age-value curve multiplier for a given position and age.

    Ages outside the table range (below 17 or above the max) are clamped
    to the nearest defined boundary.

    Parameters
    ----------
    position:
        Position string (aliases resolved automatically).
    age:
        Player age in years (integer).

    Returns
    -------
    float
        Age-value multiplier (0.0 – 1.0).
    """
    canonical = resolve_position(position)
    # Fall back to CM curve if position group not in table
    curve = AGE_MULTIPLIERS.get(canonical, AGE_MULTIPLIERS["CM"])
    ages = sorted(curve.keys())
    if age <= ages[0]:
        return curve[ages[0]]
    if age >= ages[-1]:
        return curve[ages[-1]]
    # Linear interpolation between adjacent defined ages
    for i in range(len(ages) - 1):
        if ages[i] <= age < ages[i + 1]:
            lo, hi = ages[i], ages[i + 1]
            frac = (age - lo) / (hi - lo)
            return curve[lo] + frac * (curve[hi] - curve[lo])
    return curve.get(age, 0.50)


def get_contract_multiplier(contract_years_remaining: float) -> float:
    """Return the contract-length value multiplier (CIES methodology).

    Parameters
    ----------
    contract_years_remaining:
        Floating-point years remaining on contract (e.g. 1.5).

    Returns
    -------
    float
        Multiplier in [0.85, 1.15].
    """
    years = int(contract_years_remaining)
    # 4+ years treated as 4
    years = min(years, 4)
    return CONTRACT_MULTIPLIERS.get(years, 1.00)


def apply_zone_weight(metric_value: float, zone: Optional[str]) -> float:
    """Apply VAEP-inspired zone weighting to a metric value.

    Parameters
    ----------
    metric_value:
        The raw per-90 or percentage metric value.
    zone:
        Pitch zone string (see ZONE_WEIGHTS keys), or None for no adjustment.

    Returns
    -------
    float
        Zone-adjusted metric value.
    """
    if zone is None:
        return metric_value
    multiplier = ZONE_WEIGHTS.get(zone, 1.00)
    return metric_value * multiplier


def score_player_metrics(
    stats: dict[str, float],
    position: str,
    league_percentiles: Optional[dict[str, dict[str, float]]] = None,
) -> float:
    """Compute a 0-100 composite performance score for a player.

    The score is a weighted sum of percentile ranks for each metric.
    If ``league_percentiles`` is provided (a dict mapping metric →
    {"p10": …, "p90": …}), each raw stat is converted to a 0-100 percentile
    using a linear mapping [p10 → 10, p90 → 90].  Otherwise raw stats are
    min-max scaled using hard-coded sensible bounds.

    Negative-weight metrics (e.g. errors_leading_to_shot_per90) are inverted
    before scoring so that higher = worse translates to lower score.

    Parameters
    ----------
    stats:
        Dict of metric name → value for the player.
    position:
        Player position string.
    league_percentiles:
        Optional contextual percentile bounds per metric.

    Returns
    -------
    float
        Composite score in [0, 100].
    """
    weights = get_position_weights(position)
    total_weight = 0.0
    weighted_score = 0.0

    for metric, weight in weights.items():
        value = stats.get(metric)
        if value is None:
            # Missing stat: use neutral 50th percentile (50 points)
            score = 50.0
        else:
            is_negative = weight < 0
            abs_weight = abs(weight)

            if league_percentiles and metric in league_percentiles:
                p10 = league_percentiles[metric].get("p10", 0.0)
                p90 = league_percentiles[metric].get("p90", 1.0)
                if p90 == p10:
                    score = 50.0
                else:
                    raw = (value - p10) / (p90 - p10) * 80.0 + 10.0
                    score = max(0.0, min(100.0, raw))
            else:
                # Fallback: use hard-coded sensible bounds
                score = _fallback_scale(metric, value, position)

            # Invert negative metrics: a high error rate should hurt the score
            if is_negative:
                score = 100.0 - score

            weighted_score += abs_weight * score
            total_weight += abs_weight

    if total_weight == 0:
        return 50.0

    return round(weighted_score / total_weight, 2)


# Hard-coded sensible stat bounds for fallback scaling (no league data)
_STAT_BOUNDS: dict[str, tuple[float, float]] = {
    # GK
    "psxg_minus_ga_per90": (-0.5, 0.5),
    "save_pct": (0.55, 0.85),
    "sweeper_actions_per90": (0.0, 3.0),
    "launch_accuracy_pct": (0.30, 0.75),
    "clean_sheet_pct": (0.10, 0.55),
    # Defensive
    "aerial_duel_win_pct": (0.30, 0.75),
    "ground_duel_win_pct": (0.30, 0.70),
    "defensive_duel_win_pct": (0.30, 0.70),
    "duel_win_pct": (0.30, 0.70),
    "interceptions_per90": (0.5, 4.0),
    "pressures_per90": (3.0, 18.0),
    "pressure_success_rate": (0.20, 0.45),
    "recoveries_per90": (2.0, 10.0),
    "errors_leading_to_shot_per90": (0.0, 0.3),
    # Passing / creativity
    "progressive_passes_per90": (1.0, 8.0),
    "xa_per90": (0.0, 0.4),
    "key_passes_per90": (0.2, 2.5),
    "pass_accuracy_pct": (0.60, 0.93),
    "crossing_accuracy_pct": (0.15, 0.45),
    "passes_into_penalty_area_per90": (0.5, 5.0),
    # Carrying / dribbling
    "progressive_carries_per90": (1.0, 7.0),
    "successful_dribbles_per90": (0.3, 4.0),
    "takeon_success_pct": (0.30, 0.70),
    # Attack
    "xg_per90": (0.0, 0.9),
    "goals_per90": (0.0, 0.8),
    "shot_conversion_rate": (0.05, 0.35),
    "shot_creating_actions_per90": (1.0, 6.0),
}


def _fallback_scale(metric: str, value: float, position: str) -> float:
    """Min-max scale a raw stat to 0-100 using hard-coded bounds.

    Parameters
    ----------
    metric:
        Metric key.
    value:
        Raw stat value.
    position:
        Player position (reserved for future position-specific bounds).

    Returns
    -------
    float
        Scaled score in [0, 100].
    """
    if metric not in _STAT_BOUNDS:
        # Unknown metric: return neutral
        return 50.0
    lo, hi = _STAT_BOUNDS[metric]
    if hi == lo:
        return 50.0
    scaled = (value - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, scaled))

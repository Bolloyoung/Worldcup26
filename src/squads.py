"""
Squad-strength index: turn projected national-team line-ups into a per-team
multiplier the probability engine blends into its goal rates.

This is the current-squad / injuries / form hook that team-level historical
ratings cannot see. Edit data/squads/squads.json (e.g. drop an injured star,
swap a starter) and the forecast responds.

Index construction
------------------
Each squad is valued with the ported player_valuation engine (formula path:
market value driven by league tier, position, age, nationality, contract).
Squad totals span two orders of magnitude (≈€1.2bn for England vs ≈€20m for
the minnows), so we compress with a square-root in log space and normalise to
a geometric mean of 1.0:

    index[t] = exp( 0.5 * (log V_t - mean_t log V) )

A match adjustment is then  (index_home / index_away) ** SQUAD_WEIGHT  — small,
bounded, and exactly 1.0 (no effect) whenever a squad is missing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import DATA_DIR
from .player_valuation import rate_first_eleven

SQUADS_PATH = DATA_DIR / "squads" / "squads.json"


def load_squads(path: Path | None = None) -> dict[str, list[dict]]:
    """Load {team: [player dicts]} from JSON, or {} if the file is absent."""
    p = path or SQUADS_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def squad_values(squads: dict[str, list[dict]]) -> dict[str, float]:
    """Total estimated squad market value (EUR) per team."""
    out: dict[str, float] = {}
    for team, players in squads.items():
        if not players:
            continue
        out[team] = rate_first_eleven(players)["squad_total_value_eur"]
    return out


# Compression exponent: index = exp(SQUAD_COMPRESS * (log value - centre)).
# Squad totals span ~2 orders of magnitude; 0.5 (sqrt) keeps the multiplier
# bounded. Shared by the per-team index and per-match confirmed-XI overrides
# so both live in the same strength space.
SQUAD_COMPRESS = 0.5


def field_log_centre(
    squads: dict[str, list[dict]] | None = None,
    path: Path | None = None,
) -> float | None:
    """Mean log squad value across the projected field (the index anchor)."""
    if squads is None:
        squads = load_squads(path)
    values = squad_values(squads)
    if len(values) < 2:
        return None
    return sum(math.log(max(v, 1.0)) for v in values.values()) / len(values)


def value_to_index(value: float, centre: float) -> float:
    """Map a squad value (EUR) to the normalised strength index."""
    return math.exp(SQUAD_COMPRESS * (math.log(max(value, 1.0)) - centre))


def build_squad_index(
    squads: dict[str, list[dict]] | None = None,
    path: Path | None = None,
) -> dict[str, float]:
    """
    Per-team strength multiplier centred on geometric mean 1.0.

    Returns {} when no squads are available (engine then applies no squad
    adjustment — graceful fallback).
    """
    if squads is None:
        squads = load_squads(path)
    values = squad_values(squads)
    if len(values) < 2:
        return {}
    centre = sum(math.log(max(v, 1.0)) for v in values.values()) / len(values)
    return {t: value_to_index(v, centre) for t, v in values.items()}

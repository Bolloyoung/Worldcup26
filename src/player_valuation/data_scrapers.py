"""
Lightweight, dependency-free stand-in for the UCL project's live scrapers.

The original module scraped FBref/Sofascore/Transfermarkt (≈900 lines,
requiring requests/bs4/requests_cache). For the World Cup forecaster we don't
want a fragile live-scrape of 48 squads at fit time, and the valuation engine
already degrades gracefully when no live stats are available: with empty
stats, ``score_player_metrics`` returns a neutral 50, so the squad-strength
signal comes from the *composition* of the squad (each player's league tier,
position, age, nationality and contract) — which is exactly the
current-squad information we want to inject.

If you later want live performance stats, drop the original
``data_scrapers.py`` back in here; the public ``fetch_player_data`` contract
is identical.
"""

from __future__ import annotations

from typing import Any


def fetch_player_data(player: dict) -> tuple[dict[str, Any], str, str]:
    """
    Formula-only valuation path: return no live stats.

    Returns (stats_dict, data_source_label, confidence_level) — empty stats
    drive the engine to its neutral (50) performance score, so market value is
    determined by league/position/age/nationality/contract base anchors.
    """
    return {}, "formula", "medium"

"""Betting interpretation tests, focused on the tie-handling rules."""

import numpy as np
import pytest

from src.betting import betting_card


def _pred(home, away, ph, pd_, pa):
    """Synthesise a prediction with a score matrix matching the 1X2 split."""
    mat = np.zeros((6, 6))
    tri = [(i, j) for i in range(6) for j in range(6) if i > j]
    dia = [(i, i) for i in range(6)]
    up = [(i, j) for i in range(6) for j in range(6) if i < j]
    for cells, p in [(tri, ph), (dia, pd_), (up, pa)]:
        for c in cells:
            mat[c] = p / len(cells)
    mat /= mat.sum()
    sl = sorted(
        ({"home_goals": i, "away_goals": j, "prob": float(mat[i, j])}
         for i in range(6) for j in range(6)),
        key=lambda s: -s["prob"],
    )
    return {
        "home_team": home, "away_team": away,
        "home_win": ph, "draw": pd_, "away_win": pa,
        "score_matrix": mat, "top_scorelines": sl,
    }


def test_strong_home_pick():
    c = betting_card(_pred("Spain", "Malta", 0.72, 0.20, 0.08))
    assert c["confidence"] == "Strong"
    assert c["primary_market"]["market"] == "1X2"
    assert "Spain" in c["primary_market"]["selection"]


def test_clear_leader_at_40_30_30():
    # user's example: highest prob wins, with a safer double-chance alt
    c = betting_card(_pred("A", "B", 0.40, 0.30, 0.30))
    assert c["primary_market"]["market"] == "1X2"
    assert c["primary_market"]["selection"].startswith("A")
    assert any("Double Chance" in a for a in c["alternatives"])


def test_symmetric_30_40_30_flags_and_picks_draw():
    c = betting_card(_pred("A", "B", 0.30, 0.40, 0.30))
    assert "Draw" in c["primary_market"]["selection"]
    assert c["tie_flag"] is not None
    assert "level" in c["tie_flag"]
    assert any("12" in a for a in c["alternatives"])


def test_all_equal_33_33_33_avoids():
    c = betting_card(_pred("A", "B", 0.34, 0.33, 0.33))
    assert c["confidence"] == "Avoid"
    assert c["primary_market"]["selection"] == "No bet"
    assert c["tie_flag"] is not None


def test_top_two_tie_recommends_double_chance():
    c = betting_card(_pred("A", "B", 0.40, 0.20, 0.40))
    assert c["primary_market"]["market"] == "Double Chance"
    assert "12" in c["primary_market"]["selection"]


def test_derived_markets_consistent():
    c = betting_card(_pred("A", "B", 0.45, 0.30, 0.25))
    dm = c["derived_markets"]
    assert dm["over_2_5"] + dm["under_2_5"] == pytest.approx(1.0, abs=1e-6)
    assert dm["btts_yes"] + dm["btts_no"] == pytest.approx(1.0, abs=1e-6)
    assert dm["draw_no_bet_home"] + dm["draw_no_bet_away"] == pytest.approx(1.0, abs=1e-6)
    # double chances each exceed any single outcome
    assert dm["double_chance_1X"] >= c["outcome_probs"]["home"]


def test_scoreline_banker_present():
    c = betting_card(_pred("A", "B", 0.5, 0.3, 0.2))
    assert "-" in c["scoreline_banker"]["score"]
    assert len(c["scoreline_cover"]) == 3

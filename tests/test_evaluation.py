"""Evaluation metrics + frozen-prediction-log tests."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    aggregate,
    load_log,
    match_scores,
    outcome,
    score_log,
    score_one,
    update_log,
)


def test_outcome():
    assert outcome(2, 0) == "H"
    assert outcome(1, 1) == "D"
    assert outcome(0, 3) == "A"


def test_match_scores_reward_confident_correct():
    confident_right = match_scores({"H": 0.8, "D": 0.15, "A": 0.05}, "H")[0]
    confident_wrong = match_scores({"H": 0.8, "D": 0.15, "A": 0.05}, "A")[0]
    coin = match_scores({"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}, "H")[0]
    assert confident_right < coin < confident_wrong


def test_perfect_prediction_zero_loss():
    ll, br, rps = match_scores({"H": 1.0, "D": 0.0, "A": 0.0}, "H")
    assert ll == pytest.approx(0.0, abs=1e-6)
    assert br == pytest.approx(0.0, abs=1e-6)
    assert rps == pytest.approx(0.0, abs=1e-6)


def test_aggregate_skill_positive_for_good_model():
    rows = [
        score_one("2026-06-11", "A", "B", {"H": 0.7, "D": 0.2, "A": 0.1}, 2, 0, "2-0"),
        score_one("2026-06-12", "C", "D", {"H": 0.1, "D": 0.2, "A": 0.7}, 0, 1, "0-1"),
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["favourite_accuracy"] == 1.0
    assert agg["exact_score_rate"] == 1.0
    assert agg["skill_vs_baseline_pct"] > 0


def test_aggregate_empty():
    assert aggregate([])["n"] == 0


def _fixtures():
    return pd.DataFrame([
        {"date": pd.Timestamp("2026-06-20"), "home_team": "A", "away_team": "B",
         "neutral": True, "group": "X"},
        {"date": pd.Timestamp("2026-06-21"), "home_team": "C", "away_team": "D",
         "neutral": True, "group": "X"},
    ])


class _StubEngine:
    def predict(self, home, away, neutral=True):
        return {
            "home_win": 0.5, "draw": 0.3, "away_win": 0.2,
            "expected_home_goals": 1.4, "expected_away_goals": 1.0,
            "top_scorelines": [{"home_goals": 1, "away_goals": 0, "prob": 0.12}],
        }


def test_log_freezes_played_fixtures(tmp_path):
    path = tmp_path / "log.json"
    eng = _StubEngine()

    # First run: nothing played → both fixtures logged
    log = update_log(path, _fixtures(), eng, played={})
    assert len(log) == 2
    first_a = log["2026-06-20|A|B"]["p_home"]

    # A vs B now played; its frozen entry must NOT change even though the
    # stub would now predict it again.
    log2 = update_log(path, _fixtures(), eng, played={("A", "B"): (3, 0)})
    assert log2["2026-06-20|A|B"]["p_home"] == first_a   # unchanged / frozen
    assert len(log2) == 2

    # Scoring picks up the played fixture
    card = score_log(load_log(path), {("A", "B"): (3, 0)})
    assert card["n"] == 1
    assert card["matches"][0]["actual"] == "H"

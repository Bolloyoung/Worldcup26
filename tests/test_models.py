"""Model unit tests on synthetic data (no network needed)."""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel, _tau
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine


@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    """Round-robin league with known strength ordering."""
    rng = np.random.default_rng(7)
    strengths = {"Alpha": 1.6, "Beta": 1.2, "Gamma": 1.0, "Delta": 0.7}
    teams = list(strengths)
    rows = []
    date = pd.Timestamp("2024-01-01")
    for _ in range(60):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam_h = strengths[h] / strengths[a] * 1.2
                lam_a = strengths[a] / strengths[h]
                rows.append(
                    {
                        "date": date,
                        "home_team": h,
                        "away_team": a,
                        "goals_home": rng.poisson(lam_h),
                        "goals_away": rng.poisson(lam_a),
                        "neutral": False,
                        "weight_comp": 1.0,
                    }
                )
                date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted(synthetic_matches) -> DixonColesModel:
    return DixonColesModel(xi=0.0).fit(
        synthetic_matches, reference_date=pd.Timestamp("2026-01-01")
    )


def test_tau_correction_bounds():
    assert _tau(0, 0, 1.0, 1.0, -0.1) > 1.0
    assert _tau(1, 1, 1.0, 1.0, -0.1) > 1.0
    assert _tau(5, 2, 1.0, 1.0, -0.1) == 1.0


def test_strength_ordering_recovered(fitted):
    s = fitted.team_strengths()
    assert list(s["team"]) == ["Alpha", "Beta", "Gamma", "Delta"]


def test_probabilities_sum_to_one(fitted):
    p = fitted.predict("Alpha", "Delta", neutral=True)
    assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-6)
    assert p["home_win"] > p["away_win"]


def test_home_advantage_estimated(fitted):
    assert fitted.home_advantage > 1.05


def test_neutral_venue_removes_advantage(fitted):
    p_home = fitted.predict("Beta", "Gamma", neutral=False)
    p_neutral = fitted.predict("Beta", "Gamma", neutral=True)
    assert p_home["home_win"] > p_neutral["home_win"]


def test_serialisation_roundtrip(fitted):
    clone = DixonColesModel.from_dict(fitted.to_dict())
    a = fitted.predict("Alpha", "Beta")
    b = clone.predict("Alpha", "Beta")
    assert a["home_win"] == pytest.approx(b["home_win"])


def test_elo_winner_gains():
    elo = EloRatings()
    elo.update("X", "Y", 3, 0, tournament="FIFA World Cup", neutral=True)
    assert elo.get("X") > 1500 > elo.get("Y")
    assert elo.get("X") - 1500 == pytest.approx(1500 - elo.get("Y"))


def test_engine_elo_nudge_shifts_lambdas(fitted, synthetic_matches):
    elo = EloRatings()
    for _ in range(20):
        elo.update("Gamma", "Beta", 2, 0, neutral=True)
    engine = ProbabilityEngine(fitted, elo, nudge=0.2)
    lam1, lam2 = engine.lambdas("Gamma", "Beta", neutral=True)
    dc1, dc2 = fitted.lambdas("Gamma", "Beta", neutral=True)
    assert lam1 > dc1 and lam2 < dc2


def test_pen_shootout_bounded(fitted):
    elo = EloRatings()
    elo.ratings.update({"X": 2100.0, "Y": 1300.0})
    engine = ProbabilityEngine(fitted, elo)
    p = engine.pen_shootout_p_home("X", "Y")
    assert 0.5 < p < 0.6

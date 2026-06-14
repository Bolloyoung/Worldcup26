"""Squad valuation + integration tests (formula-only path, no network)."""

import pytest

from src.player_valuation import rate_first_eleven
from src.squads import build_squad_index, load_squads, squad_values


def _squad(league, age=26):
    return [
        {"name": f"P{i}", "position": pos, "nationality": "Spain",
         "age": age, "league": league, "club": "", "contract_years_remaining": 2.5}
        for i, pos in enumerate(
            ["GK", "RB", "CB", "CB", "LB", "CDM", "CM", "CAM", "RW", "ST", "LW"]
        )
    ]


def test_valuation_runs_without_stats():
    r = rate_first_eleven(_squad("La Liga"))
    assert r["squad_total_value_eur"] > 0
    assert r["squad_avg_score"] == pytest.approx(50.0, abs=1.0)


def test_stronger_league_values_higher():
    pl = rate_first_eleven(_squad("Premier League"))["squad_total_value_eur"]
    minor = rate_first_eleven(_squad("Kenyan Premier League"))["squad_total_value_eur"]
    assert pl > minor


def test_squad_index_geometric_mean_one():
    squads = load_squads()
    idx = build_squad_index(squads)
    assert len(idx) >= 2
    import math
    geomean = math.exp(sum(math.log(v) for v in idx.values()) / len(idx))
    assert geomean == pytest.approx(1.0, abs=1e-6)


def test_index_empty_when_no_squads():
    assert build_squad_index({}) == {}
    assert build_squad_index({"OnlyOne": _squad("La Liga")}) == {}


def test_real_squads_load():
    squads = load_squads()
    assert "Spain" in squads and "Argentina" in squads
    vals = squad_values(squads)
    assert all(v > 0 for v in vals.values())

"""Confirmed-XI valuation, override, name-matching, and graceful fallback."""

import json

import pytest

from src.lineups import (
    enrich_xi,
    load_lineup,
    match_override,
    player_database,
    save_lineup,
    xi_value,
)
from src.lineup_scraper import canonical_team, scheduled_events
from src.squads import load_squads


@pytest.fixture(scope="module")
def squads():
    return load_squads()


@pytest.fixture(scope="module")
def db(squads):
    return player_database(squads)


def test_name_matching_accent_and_surname(db):
    # full name, accented, and surname-only all resolve
    assert enrich_xi(["Bukayo Saka"], db)[1] == 0          # known
    assert enrich_xi(["Saka"], db)[1] == 0                  # surname only
    assert enrich_xi(["Alvaro Morata"], db)[1] == 0         # accent-insensitive
    assert enrich_xi(["Nobody McTotallyUnknown"], db)[1] == 1


def test_xi_value_falls_back_when_thin(squads, db):
    # fewer than MIN_KNOWN identifiable → value None (untrustworthy)
    value, n_known, _ = xi_value("Spain", ["x", "y", "z"], db)
    assert value is None and n_known < 6


def test_full_known_xi_values(squads, db):
    names = [p["name"] for p in squads["Spain"]]
    value, n_known, unknown = xi_value("Spain", names, db)
    assert n_known == 11 and unknown == 0
    assert value and value > 0


def test_override_uses_confirmed_for_covered_falls_back_for_unknown(squads):
    spain = [p["name"] for p in squads["Spain"]]
    brazil = [p["name"] for p in squads["Brazil"]]
    ov = match_override("Spain", spain, "Brazil", brazil, squads=squads)
    assert ov["home_source"] == "confirmed_xi"
    assert ov["away_source"] == "confirmed_xi"
    assert ov["override"][0] and ov["override"][1]

    # Uncovered / unidentifiable away XI → away falls back (None or projected)
    ov2 = match_override(
        "Spain", spain, "Haiti", [f"unknown{i}" for i in range(11)], squads=squads
    )
    assert ov2["away_source"] == "projected"


def test_index_monotonic_in_value():
    from src.squads import value_to_index
    assert value_to_index(50e6, 19.0) < value_to_index(500e6, 19.0)


def test_manual_lineup_roundtrip(tmp_path, monkeypatch):
    import src.lineups as L
    monkeypatch.setattr(L, "LINEUPS_DIR", tmp_path)
    save_lineup("2026-06-27", "Croatia", "Ghana",
                {"confirmed": True, "home": ["A"], "away": ["B"]})
    loaded = load_lineup("2026-06-27", "Croatia", "Ghana")
    assert loaded["home"] == ["A"]
    assert load_lineup("2026-06-27", "Nobody", "Nowhere") is None


def test_scraper_name_map():
    assert canonical_team("USA") == "United States"
    assert canonical_team("Czechia") == "Czech Republic"
    assert canonical_team("France") == "France"


def test_scraper_never_raises():
    # network blocked / 403 → empty, no exception
    assert isinstance(scheduled_events("2026-06-27"), list)

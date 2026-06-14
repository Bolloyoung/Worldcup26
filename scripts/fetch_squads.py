"""
Validate data/squads/squads.json and report coverage against the 48-team field.

The squad file is hand-maintained (projected XIs with each player's league,
position, age, nationality, contract). This script:
  - validates every player has the required keys,
  - flags leagues missing from the valuation coefficient table,
  - prints the resulting squad-strength index and field coverage.

To extend coverage, add more teams/players to squads.json — any team left out
falls back gracefully to pure Dixon-Coles + Elo (no squad adjustment).

Usage:
    python scripts/fetch_squads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.player_valuation.league_adjustments import LEAGUE_COEFFICIENTS
from src.simulation.worldcup import TEAM_TO_GROUP
from src.squads import build_squad_index, load_squads

REQUIRED = {"name", "position", "nationality", "age", "league", "contract_years_remaining"}


def main() -> None:
    squads = load_squads()
    if not squads:
        raise SystemExit("No squads.json found.")

    problems = 0
    unknown_leagues: set[str] = set()
    for team, players in squads.items():
        for p in players:
            missing = REQUIRED - set(p)
            if missing:
                print(f"  ! {team} / {p.get('name','?')}: missing {missing}")
                problems += 1
            if p.get("league") not in LEAGUE_COEFFICIENTS:
                unknown_leagues.add(p.get("league", "?"))

    if unknown_leagues:
        print("Leagues not in coefficient table (will use Unknown=0.45):")
        for lg in sorted(unknown_leagues):
            print(f"  - {lg}")

    covered = set(squads) & set(TEAM_TO_GROUP)
    missing_teams = set(TEAM_TO_GROUP) - set(squads)
    print(f"\nCoverage: {len(covered)}/48 World Cup teams have squads.")
    print(f"Falling back to DC+Elo for {len(missing_teams)} teams.")

    idx = build_squad_index(squads)
    print("\nSquad-strength index (geometric mean = 1.0):")
    for t, v in sorted(idx.items(), key=lambda kv: -kv[1]):
        tag = "" if t in TEAM_TO_GROUP else "  (not a WC team)"
        print(f"  {t:<18} {v:.3f}{tag}")

    if problems:
        raise SystemExit(f"\n{problems} validation problem(s).")
    print("\nsquads.json OK.")


if __name__ == "__main__":
    main()

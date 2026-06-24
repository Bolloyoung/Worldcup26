"""
End-to-end pipeline: fetch data → fit models → simulate the tournament →
write predictions/ artifacts consumed by the dashboard.

Usage:
    python scripts/run_simulation.py [--sims 10000] [--refresh-data]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DC_RIDGE, ELO_START_DATE, N_TOURNAMENTS, PREDICTIONS_DIR, SEED
from src.data.fetch import (
    download_results,
    elo_matches,
    played_results,
    training_matches,
    verify_groups,
    wc2026_group_fixtures,
)
from src.evaluation import update_log
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine
from src.simulation.worldcup import GROUPS, TEAM_TO_GROUP, WorldCupSimulator
from src.squads import build_squad_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=N_TOURNAMENTS)
    # Data is refreshed by default so a stale cache can never silently freeze
    # the model. --refresh-data is kept as an accepted no-op (used by the CI
    # workflow); pass --cached to deliberately reuse the local copy offline.
    parser.add_argument("--refresh-data", action="store_true",
                        help="(default behaviour; kept for compatibility)")
    parser.add_argument("--cached", action="store_true",
                        help="reuse the cached results.csv instead of refetching")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.time()
    print("1/5 Downloading international results …")
    raw = download_results(force=not args.cached)
    print(f"    {len(raw):,} matches in dataset")

    # Time-decay anchor = today, so recent results always carry the most
    # weight (never frozen to a fixed tournament date).
    as_of = pd.Timestamp.now().normalize()

    fixtures = wc2026_group_fixtures(raw)
    verify_groups(fixtures, GROUPS)
    print(f"    {len(fixtures)} WC2026 group fixtures verified against draw")

    print("2/5 Fitting Dixon-Coles …")
    train = training_matches(raw)
    dc = DixonColesModel(ridge=DC_RIDGE).fit(train, reference_date=as_of)
    print(
        f"    {len(train):,} matches, {len(dc.teams)} teams, "
        f"home_adv={dc.home_advantage:.3f}, rho={dc.rho:.3f}"
    )
    missing = [t for t in TEAM_TO_GROUP if t not in dc.attack_params]
    if missing:
        raise SystemExit(f"WC teams missing from model: {missing}")

    print("3/5 Fitting Elo …")
    elo = EloRatings().fit(elo_matches(raw, ELO_START_DATE))
    squad_index = build_squad_index()
    engine = ProbabilityEngine(dc, elo, squad_index=squad_index)
    print(f"    squad index built for {len(squad_index)} teams")

    known = played_results(raw)
    print(f"4/5 Simulating {args.sims:,} tournaments "
          f"(conditioned on {len(known)} played matches) …")
    sim = WorldCupSimulator(
        engine, fixtures, n_tournaments=args.sims, seed=args.seed,
        known_results=known,
    )
    result = sim.run()
    forecast = result["forecast"]

    print("5/5 Writing artifacts …")
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    forecast.to_json(
        PREDICTIONS_DIR / "tournament_forecast.json",
        orient="records", indent=2,
    )

    match_preds = []
    for row in fixtures.itertuples(index=False):
        p = engine.predict(
            row.home_team, row.away_team, neutral=bool(row.neutral)
        )
        best = p["top_scorelines"][0]
        match_preds.append(
            {
                "date": str(row.date.date()),
                "group": TEAM_TO_GROUP[row.home_team],
                "home": row.home_team,
                "away": row.away_team,
                "city": row.city,
                "p_home": round(p["home_win"], 4),
                "p_draw": round(p["draw"], 4),
                "p_away": round(p["away_win"], 4),
                "xg_home": round(p["expected_home_goals"], 2),
                "xg_away": round(p["expected_away_goals"], 2),
                "most_likely_score":
                    f"{best['home_goals']}-{best['away_goals']}",
            }
        )
    with open(PREDICTIONS_DIR / "group_fixtures.json", "w") as f:
        json.dump(match_preds, f, indent=2)

    # Freeze pre-match predictions for not-yet-played fixtures so the model can
    # be scored honestly later (src/evaluation.py). Played fixtures keep their
    # last pre-match entry; new results never overwrite a frozen prediction.
    fixtures_with_group = fixtures.assign(
        group=fixtures["home_team"].map(TEAM_TO_GROUP)
    )
    log = update_log(
        PREDICTIONS_DIR / "prediction_log.json",
        fixtures_with_group, engine, known,
    )
    scored = sum(1 for e in log.values() if (e["home"], e["away"]) in known)
    print(f"    prediction log: {len(log)} fixtures tracked, {scored} now scorable")

    with open(PREDICTIONS_DIR / "model.json", "w") as f:
        json.dump(
            {
                "dixon_coles": dc.to_dict(),
                "elo": {t: elo.get(t) for t in TEAM_TO_GROUP},
                "meta": {
                    "fitted": as_of.date().isoformat(),
                    "n_played_matches": len(known),
                    "n_train_matches": len(train),
                    "n_tournaments": args.sims,
                    "top_finals": result["top_finals"],
                },
            },
            f, indent=2,
        )

    top = forecast.head(10)[["team", "p_champion", "p_final", "p_sf"]]
    print(f"\nDone in {time.time() - t0:.0f}s. Top 10 title favourites:\n")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

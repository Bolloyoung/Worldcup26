"""
Backfill honest pre-match predictions for World Cup matches that were played
*before* the prediction log existed (the first matchdays).

For each missing played fixture, the model is refit on data strictly before
that match's date and used to predict it — so the stored prediction is
genuinely out-of-sample, exactly as if it had been logged the morning of the
game. Entries are marked "backfilled": true for transparency. Already-logged
fixtures are never touched.

Usage:
    python scripts/backfill_log.py
    python scripts/evaluate.py        # then refresh the scorecard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DC_RIDGE, ELO_START_DATE, PREDICTIONS_DIR
from src.data.fetch import (
    download_results,
    elo_matches,
    played_results,
    training_matches,
    wc2026_group_fixtures,
)
from src.evaluation import _fixture_id, load_log
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine
from src.simulation.worldcup import TEAM_TO_GROUP
from src.squads import build_squad_index

LOG_PATH = PREDICTIONS_DIR / "prediction_log.json"


def main() -> None:
    raw = download_results(force=True)
    fixtures = wc2026_group_fixtures(raw)
    played = played_results(raw)
    log = load_log(LOG_PATH)

    logged_pairs = {(e["home"], e["away"]) for e in log.values()}
    missing = [
        row for row in fixtures.itertuples(index=False)
        if (row.home_team, row.away_team) in played
        and (row.home_team, row.away_team) not in logged_pairs
    ]
    if not missing:
        print("Nothing to backfill — every played fixture is already logged.")
        return

    print(f"Backfilling {len(missing)} early fixture(s) with out-of-sample fits…")
    train_all = training_matches(raw)
    elo_all = elo_matches(raw, ELO_START_DATE)
    squad_index = build_squad_index()

    # Group by match date so each cutoff is fit once.
    by_date: dict[pd.Timestamp, list] = {}
    for row in missing:
        by_date.setdefault(pd.Timestamp(row.date), []).append(row)

    added = 0
    for cutoff in sorted(by_date):
        tr = train_all[train_all["date"] < cutoff]
        dc = DixonColesModel(ridge=DC_RIDGE).fit(tr, reference_date=cutoff)
        elo = EloRatings().fit(elo_all[elo_all["date"] < cutoff])
        eng = ProbabilityEngine(dc, elo, squad_index=squad_index)

        for row in by_date[cutoff]:
            home, away = row.home_team, row.away_team
            if home not in dc.attack_params or away not in dc.attack_params:
                print(f"  ! skipped {home} v {away} (insufficient pre-match data)")
                continue
            p = eng.predict(home, away, neutral=bool(row.neutral))
            best = p["top_scorelines"][0]
            fid = _fixture_id(str(cutoff.date()), home, away)
            log[fid] = {
                "date": str(cutoff.date()),
                "home": home,
                "away": away,
                "group": TEAM_TO_GROUP.get(home),
                "neutral": bool(row.neutral),
                "p_home": round(p["home_win"], 4),
                "p_draw": round(p["draw"], 4),
                "p_away": round(p["away_win"], 4),
                "xg_home": round(p["expected_home_goals"], 2),
                "xg_away": round(p["expected_away_goals"], 2),
                "top_score": f"{best['home_goals']}-{best['away_goals']}",
                "logged_at": f"{cutoff.date()}T00:00:00+00:00",
                "backfilled": True,
            }
            added += 1
            print(f"  + {home} {p['home_win']:.0%}/{p['draw']:.0%}/{p['away_win']:.0%} {away}")

    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nAdded {added} backfilled prediction(s). "
          f"Log now has {len(log)} entries. Run scripts/evaluate.py to rescore.")


if __name__ == "__main__":
    main()

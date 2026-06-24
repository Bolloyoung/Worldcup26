"""
Confirmed-XI live updater — run frequently (e.g. every 15 min via GitHub
Action). For each upcoming, not-yet-played World Cup fixture it looks for a
confirmed starting XI; when one is available it re-values both teams from the
actual eleven and recomputes that match's prediction + betting card with a
per-match squad override. The "~1 hour before kickoff" timing emerges
naturally because teamsheets are only *confirmed* shortly before the match.

Lineup sources, in priority order:
  1. Manual drop-in   data/lineups/<date>_<Home>_<Away>.json  (always reliable)
  2. Sofascore scrape (best-effort; 403s are common → silently skipped)

Writes predictions/live_updates.json (consumed by the dashboard). Idempotent:
a match is only rewritten when its lineup changes.

Usage:
    python scripts/update_live.py [--date YYYY-MM-DD] [--no-scrape]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.betting import betting_card
from src.config import PREDICTIONS_DIR
from src.data.fetch import download_results, played_results, wc2026_group_fixtures
from src.lineups import load_lineup, match_override, save_lineup
from src.lineup_scraper import fetch_confirmed_for_match
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine
from src.squads import build_squad_index, load_squads

LIVE_PATH = PREDICTIONS_DIR / "live_updates.json"


def _load_engine():
    m = json.loads((PREDICTIONS_DIR / "model.json").read_text())
    dc = DixonColesModel.from_dict(m["dixon_coles"])
    elo = EloRatings()
    elo.ratings.update(m["elo"])
    return ProbabilityEngine(dc, elo, squad_index=build_squad_index())


def _get_lineup(date, home, away, allow_scrape) -> dict | None:
    manual = load_lineup(date, home, away)
    if manual and manual.get("confirmed", True):
        return manual
    if allow_scrape:
        lu = fetch_confirmed_for_match(date, home, away)
        if lu:                                   # cache scraped lineup to disk
            save_lineup(date, home, away, lu)
            return lu
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="only this date (YYYY-MM-DD); default = all upcoming")
    ap.add_argument("--no-scrape", action="store_true", help="manual lineups only")
    args = ap.parse_args()

    raw = download_results()
    fixtures = wc2026_group_fixtures(raw)
    played = played_results(raw)
    squads = load_squads()
    projected = build_squad_index(squads)
    engine = _load_engine()

    existing = json.loads(LIVE_PATH.read_text()) if LIVE_PATH.exists() else {}
    updates = dict(existing)
    changed = 0

    for row in fixtures.itertuples(index=False):
        home, away = row.home_team, row.away_team
        date = str(row.date.date())
        if (home, away) in played:                 # already kicked off
            continue
        if args.date and date != args.date:
            continue

        lineup = _get_lineup(date, home, away, allow_scrape=not args.no_scrape)
        if not lineup:
            continue

        ov = match_override(home, lineup["home"], away, lineup["away"],
                            squads=squads, projected=projected)
        if not ov:
            continue

        key = f"{date}|{home}|{away}"
        signature = [lineup.get("source"), lineup["home"], lineup["away"]]
        if updates.get(key, {}).get("_signature") == signature:
            continue                                # unchanged → skip

        p = engine.predict(home, away, neutral=bool(row.neutral),
                           squad_override=ov["override"])
        card = betting_card(p)
        updates[key] = {
            "date": date, "home": home, "away": away,
            "lineup_source": lineup.get("source", "manual"),
            "home_xi": lineup["home"], "away_xi": lineup["away"],
            "home_known": ov["home_known"], "home_unknown": ov["home_unknown"],
            "away_known": ov["away_known"], "away_unknown": ov["away_unknown"],
            "home_index": ov["home_index"], "away_index": ov["away_index"],
            "p_home": round(p["home_win"], 4),
            "p_draw": round(p["draw"], 4),
            "p_away": round(p["away_win"], 4),
            "xg_home": round(p["expected_home_goals"], 2),
            "xg_away": round(p["expected_away_goals"], 2),
            "betting_headline": card["headline"],
            "squad_source": p["squad_source"],
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "_signature": signature,
        }
        changed += 1
        print(f"  updated {home} v {away} ({lineup.get('source')}): "
              f"H/D/A {p['home_win']:.2f}/{p['draw']:.2f}/{p['away_win']:.2f} "
              f"[{ov['home_known']}/{ov['away_known']} known]")

    if changed:
        PREDICTIONS_DIR.mkdir(exist_ok=True)
        with open(LIVE_PATH, "w") as f:
            json.dump(updates, f, indent=2, ensure_ascii=False)
        print(f"Wrote {LIVE_PATH} ({changed} match(es) updated)")
    else:
        print("No confirmed lineups available to apply.")


if __name__ == "__main__":
    main()

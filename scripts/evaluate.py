"""
Score the model against actual World Cup 2026 results.

Two scorecards:

  1. Walk-forward (default) — for every completed WC2026 match, fit the model
     on data *strictly before that matchday* and predict it. This is
     leakage-free and mirrors the production daily-refit model, so it works
     on all matches played so far (including ones played before the frozen
     prediction log existed).

  2. Frozen log — scores the genuine pre-match predictions captured in
     predictions/prediction_log.json by run_simulation.py. This is the gold
     standard going forward but only covers fixtures logged before kickoff.

Writes predictions/evaluation.json (consumed by the dashboard) and prints a
scorecard. Run after each matchday:  python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ELO_START_DATE, HOSTS, PREDICTIONS_DIR
from src.data.fetch import download_results, elo_matches, played_results, training_matches
from src.evaluation import aggregate, load_log, score_log, score_one
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine
from src.squads import build_squad_index


def _played_with_dates(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= pd.Timestamp("2026-06-01"))
        & df["home_score"].notna()
        & df["away_score"].notna()
    )
    out = df[mask].copy()
    out["neutral"] = out["neutral"].astype(str).str.upper().eq("TRUE")
    return out.sort_values("date")


def walk_forward(raw: pd.DataFrame) -> dict:
    """Fit-before-each-matchday, predict, score — leakage-free."""
    games = _played_with_dates(raw)
    if games.empty:
        return {"n": 0, "message": "No WC2026 results yet."}

    squad_index = build_squad_index()
    per_match: list[dict] = []
    for day, day_games in games.groupby(games["date"].dt.date):
        cutoff = pd.Timestamp(day)
        train = training_matches(raw)
        train = train[pd.to_datetime(train["date"]) < cutoff]
        dc = DixonColesModel().fit(train, reference_date=cutoff)
        elo = EloRatings().fit(
            elo_matches(raw, ELO_START_DATE).pipe(
                lambda d: d[pd.to_datetime(d["date"]) < cutoff]
            )
        )
        eng = ProbabilityEngine(dc, elo, squad_index=squad_index, hosts=HOSTS)
        for row in day_games.itertuples(index=False):
            if row.home_team not in dc.attack_params or row.away_team not in dc.attack_params:
                continue
            p = eng.predict(row.home_team, row.away_team, neutral=bool(row.neutral))
            best = p["top_scorelines"][0]
            per_match.append(score_one(
                str(day), row.home_team, row.away_team,
                {"H": p["home_win"], "D": p["draw"], "A": p["away_win"]},
                int(row.home_score), int(row.away_score),
                top_score=f"{best['home_goals']}-{best['away_goals']}",
            ))
    return aggregate(per_match)


def _print_card(title: str, card: dict) -> None:
    print(f"\n=== {title} ===")
    if card.get("n", 0) == 0:
        print(" ", card.get("message", "no data"))
        return
    print(f"  matches scored      : {card['n']}")
    print(f"  log-loss            : {card['log_loss']}  (baseline {card['baseline_log_loss']})")
    print(f"  Brier               : {card['brier']}")
    print(f"  RPS                 : {card['rps']}")
    print(f"  favourite accuracy  : {card['favourite_accuracy']:.0%}")
    print(f"  avg prob on actual  : {card['avg_prob_on_actual']}  (baseline {card['baseline_avg_prob']})")
    print(f"  exact scoreline rate: {card['exact_score_rate']:.0%}")
    print(f"  skill vs coin-flip  : {card['skill_vs_baseline_pct']:+.1f}%")
    if card.get("reliability"):
        print("  reliability (favourite pick):")
        print(f"    {'bin':<12}{'n':>4}{'predicted':>11}{'actual':>9}")
        for b in card["reliability"]:
            print(f"    {b['bin']:<12}{b['n']:>4}{b['mean_predicted']:>11}{b['empirical']:>9}")


def main() -> None:
    raw = download_results()
    played = played_results(raw)

    wf = walk_forward(raw)
    frozen = score_log(load_log(PREDICTIONS_DIR / "prediction_log.json"), played)

    _print_card("Walk-forward (all played WC2026 matches, leakage-free)", wf)
    _print_card("Frozen pre-match log (gold standard, grows over time)", frozen)

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    with open(PREDICTIONS_DIR / "evaluation.json", "w") as f:
        json.dump({"walk_forward": wf, "frozen_log": frozen}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {PREDICTIONS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()

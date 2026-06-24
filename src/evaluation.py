"""
Prediction logging and scoring — the measurement half of the
"optimise on previous results" loop.

Why a frozen log?
-----------------
`run_simulation.py` refits the model every run and overwrites
`group_fixtures.json`. Once a match is played its result is folded into the
training fit and locked into the simulation, so re-reading the fixture's
"prediction" afterwards is contaminated (the model has seen the answer).

To score the model honestly we must capture each prediction *before* kickoff.
`update_log` writes/refreshes an entry for every not-yet-played fixture on each
run, and stops touching an entry the moment that fixture has a result — so the
stored prediction is the last fully pre-match one. `score_log` then compares
those frozen predictions to the actual outcomes.

Metrics: log-loss and ranked probability score (RPS) are the headline numbers
(both reward well-calibrated, not just correct, probabilities); Brier and
favourite-accuracy are reported alongside, plus a reliability table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

OUTCOMES = ("H", "D", "A")


def outcome(goals_home: int, goals_away: int) -> str:
    if goals_home > goals_away:
        return "H"
    if goals_home < goals_away:
        return "A"
    return "D"


def match_scores(probs: dict[str, float], actual: str) -> tuple[float, float, float]:
    """Return (log_loss, brier, rps) for one match. probs keyed H/D/A."""
    p = np.array([probs["H"], probs["D"], probs["A"]], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    y = OUTCOMES.index(actual)
    onehot = np.zeros(3)
    onehot[y] = 1.0
    log_loss = float(-np.log(p[y]))
    brier = float(np.sum((p - onehot) ** 2))
    rps = float(np.sum((np.cumsum(p) - np.cumsum(onehot)) ** 2)) / 2.0
    return log_loss, brier, rps


def _fixture_id(date: str, home: str, away: str) -> str:
    return f"{date}|{home}|{away}"


# ── Frozen prediction log ─────────────────────────────────────────────────

def load_log(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def update_log(
    path: Path,
    fixtures,                     # DataFrame: date, home_team, away_team, group, neutral
    engine,                       # ProbabilityEngine
    played: dict[tuple[str, str], tuple[int, int]],
) -> dict[str, dict]:
    """
    Refresh the frozen prediction log for all not-yet-played fixtures.

    Entries for played fixtures are left untouched (already frozen). Returns
    the updated log and writes it to `path`.
    """
    log = load_log(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for row in fixtures.itertuples(index=False):
        home, away = row.home_team, row.away_team
        if (home, away) in played:
            continue  # frozen — never overwrite a pre-match prediction
        fid = _fixture_id(str(row.date.date()), home, away)
        p = engine.predict(home, away, neutral=bool(row.neutral))
        best = p["top_scorelines"][0]
        log[fid] = {
            "date": str(row.date.date()),
            "home": home,
            "away": away,
            "group": getattr(row, "group", None),
            "neutral": bool(row.neutral),
            "p_home": round(p["home_win"], 4),
            "p_draw": round(p["draw"], 4),
            "p_away": round(p["away_win"], 4),
            "xg_home": round(p["expected_home_goals"], 2),
            "xg_away": round(p["expected_away_goals"], 2),
            "top_score": f"{best['home_goals']}-{best['away_goals']}",
            "logged_at": now,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log


# ── Scoring ───────────────────────────────────────────────────────────────

def _reliability(rows: list[dict], n_bins: int = 5) -> list[dict]:
    """
    Reliability of the favourite pick: bin by predicted favourite probability,
    compare mean predicted prob to the empirical hit rate.
    """
    fav_p = np.array([r["fav_prob"] for r in rows])
    hit = np.array([r["fav_correct"] for r in rows], dtype=float)
    edges = np.linspace(1 / 3, 1.0, n_bins + 1)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fav_p >= lo) & (fav_p < hi if hi < 1.0 else fav_p <= hi)
        if not m.any():
            continue
        table.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(m.sum()),
            "mean_predicted": round(float(fav_p[m].mean()), 3),
            "empirical": round(float(hit[m].mean()), 3),
        })
    return table


def score_one(
    date: str, home: str, away: str, probs: dict[str, float],
    goals_home: int, goals_away: int, top_score: str | None = None,
) -> dict:
    """Score a single (prediction, result) pair → a per-match row."""
    act = outcome(goals_home, goals_away)
    ll, br, rps = match_scores(probs, act)
    fav = max(probs, key=probs.get)
    result = f"{goals_home}-{goals_away}"
    return {
        "date": date,
        "match": f"{home} v {away}",
        "predicted": f"{probs['H']:.0%}/{probs['D']:.0%}/{probs['A']:.0%}",
        "result": result,
        "actual": act,
        "favourite": fav,
        "fav_prob": probs[fav],
        "fav_correct": int(fav == act),
        "prob_on_actual": probs[act],
        "log_loss": ll,
        "brier": br,
        "rps": rps,
        "top_score": top_score,
        "exact_score": int(top_score == result) if top_score else 0,
    }


def aggregate(per_match: list[dict]) -> dict[str, Any]:
    """Aggregate per-match rows into a scorecard (metrics + reliability)."""
    n = len(per_match)
    if n == 0:
        return {"n": 0, "message": "No predictions have a result yet."}

    uniform_ll = -np.log(1 / 3)
    agg = {
        "n": n,
        "log_loss": round(float(np.mean([r["log_loss"] for r in per_match])), 4),
        "brier": round(float(np.mean([r["brier"] for r in per_match])), 4),
        "rps": round(float(np.mean([r["rps"] for r in per_match])), 4),
        "favourite_accuracy": round(float(np.mean([r["fav_correct"] for r in per_match])), 4),
        "avg_prob_on_actual": round(float(np.mean([r["prob_on_actual"] for r in per_match])), 4),
        "exact_score_rate": round(float(np.mean([r["exact_score"] for r in per_match])), 4),
        "baseline_log_loss": round(float(uniform_ll), 4),
        "baseline_avg_prob": round(1 / 3, 4),
        "skill_vs_baseline_pct": round(
            (uniform_ll - float(np.mean([r["log_loss"] for r in per_match]))) / uniform_ll * 100, 1
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    per_match = sorted(per_match, key=lambda r: r["date"])
    return {**agg, "reliability": _reliability(per_match), "matches": per_match}


def score_log(
    log: dict[str, dict],
    played: dict[tuple[str, str], tuple[int, int]],
) -> dict[str, Any]:
    """Score every frozen prediction whose match has now been played."""
    per_match = [
        score_one(
            e["date"], e["home"], e["away"],
            {"H": e["p_home"], "D": e["p_draw"], "A": e["p_away"]},
            *played[(e["home"], e["away"])], top_score=e.get("top_score"),
        )
        for e in log.values()
        if (e["home"], e["away"]) in played
    ]
    return aggregate(per_match)

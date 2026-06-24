"""
Backtest the calibration/feature settings on the 2018 & 2022 World Cups.

For each tournament we fit only on matches *before* it started, predict every
match (1X2), and score the predictions with log-loss, Brier and RPS. A small
grid search over the tunable parameters reports which settings generalise —
the objective evidence base for the defaults in src/config.py.

Note: the squad-blend weight is NOT tuned here (no historical squad data); it
is a forward-looking feature with a conservative default.

Usage:
    python scripts/backtest.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import download_results
from src.models.dixon_coles import DixonColesModel
from src.models.elo import EloRatings
from src.models.engine import ProbabilityEngine

# (hosts, tournament start, tournament end). WC2026 is in progress: only its
# completed matches carry scores (the rest are dropped by _prep), so including
# it re-tunes the calibration on the live tournament as results accumulate.
TOURNAMENTS = [
    (("Russia",), "2018-06-14", "2018-07-15"),
    (("Qatar",), "2022-11-20", "2022-12-18"),
    (("United States", "Mexico", "Canada"), "2026-06-11", "2026-07-19"),
]
TRAIN_YEARS = 8


def _prep(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df.rename(columns={"home_score": "goals_home", "away_score": "goals_away"})
    df["goals_home"] = df["goals_home"].astype(int)
    df["goals_away"] = df["goals_away"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def _outcome(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)  # H, D, A


def _scores(probs: np.ndarray, y: int) -> tuple[float, float, float]:
    p = np.clip(probs, 1e-12, 1.0)
    logloss = -np.log(p[y])
    onehot = np.zeros(3)
    onehot[y] = 1.0
    brier = float(np.sum((p - onehot) ** 2))
    cp, cy = np.cumsum(p), np.cumsum(onehot)
    rps = float(np.sum((cp - cy) ** 2)) / 2.0
    return logloss, brier, rps


# Fits depend only on `ridge` (DC) and the tournament (Elo); cache them so the
# parameter grid over nudge/host/shrink/temp is just cheap prediction.
_FIT_CACHE: dict = {}


def _fits(raw: pd.DataFrame, ridge: float):
    key = round(ridge, 5)
    if key not in _FIT_CACHE:
        df = _prep(raw)
        per_tourn = []
        for hosts, start, end in TOURNAMENTS:
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            train = df[(df["date"] < start_ts) & (df["date"] >= start_ts - pd.DateOffset(years=TRAIN_YEARS))]
            games = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
            if games.empty:        # tournament not started yet → skip
                continue
            dc = DixonColesModel(ridge=ridge).fit(train.assign(weight_comp=1.0), reference_date=start_ts)
            elo = EloRatings().fit(df[df["date"] < start_ts])
            per_tourn.append((hosts, dc, elo, games))
        _FIT_CACHE[key] = per_tourn
    return _FIT_CACHE[key]


def evaluate(raw: pd.DataFrame, params: dict) -> dict:
    ll, br, rp, n = 0.0, 0.0, 0.0, 0
    for hosts, dc, elo, games in _fits(raw, params["ridge"]):
        eng = ProbabilityEngine(
            dc, elo,
            nudge=params["nudge"],
            host_bonus=params["host_bonus"],
            inter_conf_shrink=params["shrink"],
            temperature=params["temp"],
            squad_weight=0.0,
            hosts=hosts,
        )
        for row in games.itertuples(index=False):
            if row.home_team not in dc.attack_params or row.away_team not in dc.attack_params:
                continue
            p = eng.predict(row.home_team, row.away_team, neutral=bool(row.neutral))
            probs = np.array([p["home_win"], p["draw"], p["away_win"]])
            l, b, r = _scores(probs, _outcome(row.goals_home, row.goals_away))
            ll += l; br += b; rp += r; n += 1
    return {"log_loss": ll / n, "brier": br / n, "rps": rp / n, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()

    raw = download_results()

    baseline = {"ridge": 0.0, "nudge": 0.18, "host_bonus": 0.0, "shrink": 1.0, "temp": 1.0}
    base = evaluate(raw, baseline)
    print(f"Baseline (original model): log_loss={base['log_loss']:.4f} "
          f"brier={base['brier']:.4f} rps={base['rps']:.4f} (n={base['n']})\n")

    if args.quick:
        grid = dict(ridge=[0.005], nudge=[0.10], host_bonus=[70.0], shrink=[0.85], temp=[1.0, 1.1])
    else:
        grid = dict(
            ridge=[0.002, 0.005, 0.01],
            nudge=[0.06, 0.10, 0.14],
            host_bonus=[40.0, 70.0, 100.0],
            shrink=[0.80, 0.90, 1.0],
            temp=[1.0, 1.1, 1.25],
        )

    keys = list(grid)
    best, best_ll = None, float("inf")
    results = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        m = evaluate(raw, params)
        results.append((params, m))
        if m["log_loss"] < best_ll:
            best_ll, best = m["log_loss"], (params, m)

    results.sort(key=lambda x: x[1]["log_loss"])
    print("Top 5 configs by log-loss:")
    for params, m in results[:5]:
        ps = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"  ll={m['log_loss']:.4f} brier={m['brier']:.4f} rps={m['rps']:.4f} | {ps}")

    bp, bm = best
    print("\nBest config:")
    for k, v in bp.items():
        print(f"  {k} = {v}")
    print(f"  → log_loss {bm['log_loss']:.4f} vs baseline {base['log_loss']:.4f} "
          f"({(base['log_loss'] - bm['log_loss']) / base['log_loss'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()

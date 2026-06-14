"""
Dixon-Coles bivariate Poisson model, adapted for international football.

Differences from the club version this was ported from
(ucl-prediction-system):
- Vectorised negative log-likelihood with an *analytic gradient*, so fitting
  ~200 national teams (400+ parameters) takes seconds instead of hours.
- Per-match neutral flag: home advantage gamma only applies to matches that
  were not played on neutral ground.
- Per-match competition weights multiplied into the time-decay weights, so a
  World Cup match moves the parameters more than a friendly.

Reference: Dixon, M.J. & Coles, S.G. (1997). "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market." Applied
Statistics, 46(2), 265-280.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


def _tau(x: int, y: int, lam1: float, lam2: float, rho: float) -> float:
    """Dixon-Coles low-score correlation correction (scalar form)."""
    if x == 0 and y == 0:
        return max(1e-12, 1.0 - lam1 * lam2 * rho)
    if x == 0 and y == 1:
        return 1.0 + lam1 * rho
    if x == 1 and y == 0:
        return 1.0 + lam2 * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _nll_and_grad(
    params: np.ndarray,
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    x: np.ndarray,          # home goals
    y: np.ndarray,          # away goals
    home_flag: np.ndarray,  # 1.0 where home advantage applies, else 0.0
    w: np.ndarray,
    n: int,
    ridge: float = 0.0,     # L2 penalty on log attack/defense params
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood (+ ridge) and its analytic gradient."""
    log_atk = params[:n]
    log_def = params[n:2 * n]
    log_gamma = params[2 * n]
    rho = params[2 * n + 1]

    log_lam1 = log_atk[h_idx] + log_def[a_idx] + home_flag * log_gamma
    log_lam2 = log_atk[a_idx] + log_def[h_idx]
    lam1 = np.exp(log_lam1)
    lam2 = np.exp(log_lam2)

    # tau and its partial derivatives (only low scores contribute)
    tau = np.ones_like(lam1)
    dlt_dll1 = np.zeros_like(lam1)   # d log tau / d log lam1
    dlt_dll2 = np.zeros_like(lam1)   # d log tau / d log lam2
    dlt_drho = np.zeros_like(lam1)   # d log tau / d rho

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    t00 = np.maximum(1.0 - lam1[m00] * lam2[m00] * rho, 1e-10)
    tau[m00] = t00
    dlt_dll1[m00] = -lam1[m00] * lam2[m00] * rho / t00
    dlt_dll2[m00] = dlt_dll1[m00]
    dlt_drho[m00] = -lam1[m00] * lam2[m00] / t00

    t01 = np.maximum(1.0 + lam1[m01] * rho, 1e-10)
    tau[m01] = t01
    dlt_dll1[m01] = lam1[m01] * rho / t01
    dlt_drho[m01] = lam1[m01] / t01

    t10 = np.maximum(1.0 + lam2[m10] * rho, 1e-10)
    tau[m10] = t10
    dlt_dll2[m10] = lam2[m10] * rho / t10
    dlt_drho[m10] = lam2[m10] / t10

    t11 = max(1.0 - rho, 1e-10)
    tau[m11] = t11
    dlt_drho[m11] = -1.0 / t11

    ll = (
        np.log(tau)
        + x * log_lam1 - lam1 - gammaln(x + 1)
        + y * log_lam2 - lam2 - gammaln(y + 1)
    )
    nll = -float(np.sum(w * ll))

    # gradient wrt log lambdas
    g1 = w * (x - lam1 + dlt_dll1)   # d(w*ll)/d log lam1
    g2 = w * (y - lam2 + dlt_dll2)

    grad = np.zeros_like(params)
    np.add.at(grad, h_idx, g1)            # log_atk[home] via lam1
    np.add.at(grad, n + a_idx, g1)        # log_def[away] via lam1
    np.add.at(grad, a_idx, g2)            # log_atk[away] via lam2
    np.add.at(grad, n + h_idx, g2)        # log_def[home] via lam2
    grad[2 * n] = float(np.sum(g1 * home_flag))
    grad[2 * n + 1] = float(np.sum(w * dlt_drho))

    grad = -grad

    # Ridge penalty shrinks log attack/defense toward 0 (= average team).
    # Scaled by the total weight so the penalty strength is independent of
    # how many matches / how they are weighted.
    if ridge > 0.0:
        scale = ridge * float(np.sum(w))
        ad = params[:2 * n]
        nll += 0.5 * scale * float(np.dot(ad, ad))
        grad[:2 * n] += scale * ad

    return nll, grad


class DixonColesModel:
    """Dixon-Coles model with time decay and competition weights."""

    def __init__(
        self,
        xi: float = 0.0012,
        min_weight: float = 0.01,
        ridge: float = 0.001,
    ) -> None:
        self.xi = xi
        self.min_weight = min_weight
        self.ridge = ridge

        self.teams: list[str] = []
        self._team_idx: dict[str, int] = {}
        self.attack_params: dict[str, float] = {}
        self.defense_params: dict[str, float] = {}
        self.home_advantage: float = 1.25
        self.rho: float = -0.08
        self._fitted = False

    def _time_weights(self, dates: pd.Series, ref: pd.Timestamp) -> np.ndarray:
        days = (ref - pd.to_datetime(dates)).dt.days.values.astype(float)
        w = np.exp(-self.xi * days)
        return np.maximum(w, self.min_weight)

    def fit(
        self,
        matches: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
    ) -> "DixonColesModel":
        """
        Fit by maximum weighted log-likelihood.

        `matches` needs: date, home_team, away_team, goals_home, goals_away.
        Optional: neutral (bool), weight_comp (float).
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        df = matches.dropna(subset=["goals_home", "goals_away"]).copy()
        df["date"] = pd.to_datetime(df["date"])

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        w = self._time_weights(df["date"], reference_date)
        if "weight_comp" in df.columns:
            w = w * df["weight_comp"].values
        w = w / w.sum() * len(df)

        h_idx = df["home_team"].map(self._team_idx).values.astype(np.int64)
        a_idx = df["away_team"].map(self._team_idx).values.astype(np.int64)
        x = df["goals_home"].values.astype(float)
        y = df["goals_away"].values.astype(float)
        if "neutral" in df.columns:
            home_flag = (~df["neutral"].astype(bool)).values.astype(float)
        else:
            home_flag = np.ones(len(df))

        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = np.log(1.25)
        x0[2 * n + 1] = -0.08
        bounds = (
            [(-3.0, 3.0)] * (2 * n)
            + [(np.log(1.0), np.log(2.0))]
            + [(-0.9, 0.5)]
        )

        result = minimize(
            _nll_and_grad,
            x0,
            args=(h_idx, a_idx, x, y, home_flag, w, n, self.ridge),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 5000, "maxfun": 20000, "ftol": 1e-10},
        )
        if not result.success:
            warnings.warn(
                f"Dixon-Coles optimisation did not converge: {result.message}",
                RuntimeWarning,
            )

        opt = result.x
        log_atk = opt[:n].copy()
        log_def = opt[n:2 * n].copy()
        mu = log_atk.mean()
        log_atk -= mu
        log_def += mu

        self.attack_params = {
            t: float(np.exp(log_atk[i])) for t, i in self._team_idx.items()
        }
        self.defense_params = {
            t: float(np.exp(log_def[i])) for t, i in self._team_idx.items()
        }
        self.home_advantage = float(np.exp(opt[2 * n]))
        self.rho = float(opt[2 * n + 1])
        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────

    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        adv = 1.0 if neutral else self.home_advantage
        lam1 = self.attack_params[home] * self.defense_params[away] * adv
        lam2 = self.attack_params[away] * self.defense_params[home]
        return lam1, lam2

    def score_matrix_from_lambdas(
        self, lam1: float, lam2: float, max_goals: int = 10
    ) -> np.ndarray:
        gx = np.arange(max_goals + 1)
        px = poisson.pmf(gx, lam1)
        py = poisson.pmf(gx, lam2)
        mat = np.outer(px, py)
        for x in (0, 1):
            for y in (0, 1):
                mat[x, y] *= _tau(x, y, lam1, lam2, self.rho)
        mat = np.maximum(mat, 0.0)
        return mat / mat.sum()

    def predict(
        self, home: str, away: str, neutral: bool = True, max_goals: int = 10
    ) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        for t in (home, away):
            if t not in self.attack_params:
                raise ValueError(f"Team '{t}' not seen during fitting.")

        lam1, lam2 = self.lambdas(home, away, neutral)
        mat = self.score_matrix_from_lambdas(lam1, lam2, max_goals)

        scorelines = [
            {"home_goals": i, "away_goals": j, "prob": float(mat[i, j])}
            for i in range(8) for j in range(8)
        ]
        scorelines.sort(key=lambda s: s["prob"], reverse=True)

        return {
            "home_win": float(np.sum(np.tril(mat, -1))),
            "draw": float(np.trace(mat)),
            "away_win": float(np.sum(np.triu(mat, 1))),
            "expected_home_goals": lam1,
            "expected_away_goals": lam2,
            "score_matrix": mat,
            "top_scorelines": scorelines[:10],
        }

    def team_strengths(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        records = [
            {
                "team": t,
                "attack": round(self.attack_params[t], 4),
                "defense": round(self.defense_params[t], 4),
                "overall": round(
                    self.attack_params[t] / self.defense_params[t], 4
                ),
            }
            for t in self.teams
        ]
        return (
            pd.DataFrame(records)
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "xi": self.xi,
            "min_weight": self.min_weight,
            "ridge": self.ridge,
            "teams": self.teams,
            "attack_params": self.attack_params,
            "defense_params": self.defense_params,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DixonColesModel":
        m = cls(xi=d["xi"], min_weight=d["min_weight"], ridge=d.get("ridge", 0.0))
        m.teams = d["teams"]
        m._team_idx = {t: i for i, t in enumerate(m.teams)}
        m.attack_params = d["attack_params"]
        m.defense_params = d["defense_params"]
        m.home_advantage = d["home_advantage"]
        m.rho = d["rho"]
        m._fitted = True
        return m

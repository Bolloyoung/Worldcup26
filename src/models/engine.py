"""
Probability engine: Dixon-Coles goal rates with an Elo nudge, plus three
calibration/feature adjustments motivated by the matchday-1 review:

  1. Host boost      — explicit Elo bonus for USA/Mexico/Canada (the generic
                       home-advantage term under-rated the hosts; USA beat
                       Paraguay 4-1 while the model favoured Paraguay).
  2. Inter-confed    — shrink the rating gap between teams from different
     shrink            confederations, whose cross-pool comparison rests on
                       few bridging matches (Qatar AFC vs Switzerland UEFA).
  3. Squad blend     — multiply the goal rates by a current-squad strength
                       ratio, the hook for injuries/form that team-level
                       historical ratings cannot see.

A display/decision temperature then flattens the score matrix to curb
overconfidence (the Qatar-Switzerland 91% call). All of these collapse to the
original behaviour at their neutral settings (bonus 0, shrink 1, weight 0,
temperature 1), so the backtest can switch each on independently.

    lam_home *= exp(+nudge * eff_gap / 400) * squad_ratio**w
    lam_away *= exp(-nudge * eff_gap / 400) / squad_ratio**w
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import (
    CALIB_TEMPERATURE,
    ELO_NUDGE,
    HOST_ELO_BONUS,
    HOSTS,
    INTER_CONF_SHRINK,
    MAX_GOALS,
    SQUAD_WEIGHT,
)
from ..confederations import same_confederation
from .dixon_coles import DixonColesModel
from .elo import EloRatings


class ProbabilityEngine:
    def __init__(
        self,
        dc: DixonColesModel,
        elo: EloRatings,
        nudge: float = ELO_NUDGE,
        host_bonus: float = HOST_ELO_BONUS,
        inter_conf_shrink: float = INTER_CONF_SHRINK,
        temperature: float = CALIB_TEMPERATURE,
        squad_index: dict[str, float] | None = None,
        squad_weight: float = SQUAD_WEIGHT,
        hosts: tuple[str, ...] = HOSTS,
    ) -> None:
        self.dc = dc
        self.elo = elo
        self.nudge = nudge
        self.host_bonus = host_bonus
        self.inter_conf_shrink = inter_conf_shrink
        self.temperature = temperature
        # Per-team strength multipliers centred on 1.0 (>1 = stronger squad).
        self.squad_index = squad_index or {}
        self.squad_weight = squad_weight
        # Tournament hosts (override for backtesting past World Cups).
        self.hosts = tuple(hosts)

    # ── Effective rating gap ──────────────────────────────────────────────

    def _eff_elo_gap(self, home: str, away: str, neutral: bool, host_adv_factor: float) -> float:
        gap = self.elo.get(home) - self.elo.get(away)
        # Host boost: hosts get an Elo bump in their own World Cup.
        gap += self.host_bonus * (home in self.hosts) * host_adv_factor
        gap -= self.host_bonus * (away in self.hosts) * host_adv_factor
        if not neutral:
            gap += self.elo.home_adv_elo * host_adv_factor
        # Cross-confederation comparisons are less reliable → shrink the gap.
        if not same_confederation(home, away):
            gap *= self.inter_conf_shrink
        return gap

    def _squad_ratio(
        self, home: str, away: str,
        override: tuple[float, float] | None = None,
    ) -> float:
        # A per-match override (e.g. from a confirmed starting XI) supersedes
        # the projected per-team squad index for this match.
        if override is not None:
            h, a = override
        else:
            h = self.squad_index.get(home)
            a = self.squad_index.get(away)
        if not h or not a:          # missing squad → graceful fallback
            return 1.0
        return h / a

    def lambdas(
        self, home: str, away: str, neutral: bool = True,
        host_adv_factor: float = 1.0,
        squad_override: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Blended expected goals (DC × Elo × host × confed × squad)."""
        lam1, lam2 = self.dc.lambdas(home, away, neutral=True)
        if not neutral:
            lam1 *= self.dc.home_advantage ** host_adv_factor

        shift = np.exp(self.nudge * self._eff_elo_gap(home, away, neutral, host_adv_factor) / 400.0)
        lam1 *= shift
        lam2 /= shift

        if self.squad_weight > 0.0:
            sr = self._squad_ratio(home, away, squad_override) ** self.squad_weight
            lam1 *= sr
            lam2 /= sr
        return lam1, lam2

    # ── Score matrix (with temperature) ───────────────────────────────────

    def score_matrix_from_lambdas(
        self, lam1: float, lam2: float, max_goals: int = MAX_GOALS
    ) -> np.ndarray:
        """Dixon-Coles score matrix, temperature-flattened for calibration."""
        mat = self.dc.score_matrix_from_lambdas(lam1, lam2, max_goals)
        if self.temperature != 1.0:
            mat = np.power(mat, 1.0 / self.temperature)
            mat /= mat.sum()
        return mat

    def score_matrix(
        self, home: str, away: str, neutral: bool = True,
        host_adv_factor: float = 1.0, max_goals: int = MAX_GOALS,
    ) -> np.ndarray:
        lam1, lam2 = self.lambdas(home, away, neutral, host_adv_factor)
        return self.score_matrix_from_lambdas(lam1, lam2, max_goals)

    def predict(
        self, home: str, away: str, neutral: bool = True,
        host_adv_factor: float = 1.0, max_goals: int = MAX_GOALS,
        squad_override: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        lam1, lam2 = self.lambdas(home, away, neutral, host_adv_factor, squad_override)
        mat = self.score_matrix_from_lambdas(lam1, lam2, max_goals)

        scorelines = [
            {"home_goals": i, "away_goals": j, "prob": float(mat[i, j])}
            for i in range(8) for j in range(8)
        ]
        scorelines.sort(key=lambda s: s["prob"], reverse=True)

        return {
            "home_team": home,
            "away_team": away,
            "home_win": float(np.sum(np.tril(mat, -1))),
            "draw": float(np.trace(mat)),
            "away_win": float(np.sum(np.triu(mat, 1))),
            "expected_home_goals": lam1,
            "expected_away_goals": lam2,
            "score_matrix": mat,
            "top_scorelines": scorelines[:10],
            "elo_home": round(self.elo.get(home), 1),
            "elo_away": round(self.elo.get(away), 1),
            "squad_index_home": round(
                (squad_override[0] if squad_override else self.squad_index.get(home, 1.0)), 3
            ),
            "squad_index_away": round(
                (squad_override[1] if squad_override else self.squad_index.get(away, 1.0)), 3
            ),
            "squad_source": "confirmed_xi" if squad_override else "projected",
        }

    def pen_shootout_p_home(self, home: str, away: str) -> float:
        """
        Penalty shootouts are close to a coin flip; give the stronger side a
        gentle edge bounded to ~[0.42, 0.58].
        """
        gap = self.elo.get(home) - self.elo.get(away)
        return 0.5 + 0.08 * float(np.tanh(gap / 400.0))

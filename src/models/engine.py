"""
Probability engine: Dixon-Coles goal rates with an Elo nudge.

The DC model sets the score distribution; the Elo gap then shifts the two
goal rates in opposite directions. This blends the two rating systems at the
lambda level so every downstream consumer (match predictions, Monte Carlo,
the dashboard) sees one consistent distribution:

    lam_home *= exp(+nudge * elo_gap / 400)
    lam_away *= exp(-nudge * elo_gap / 400)

where elo_gap includes the Elo home advantage when the venue is not neutral.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ELO_NUDGE, MAX_GOALS
from .dixon_coles import DixonColesModel
from .elo import EloRatings


class ProbabilityEngine:
    def __init__(
        self,
        dc: DixonColesModel,
        elo: EloRatings,
        nudge: float = ELO_NUDGE,
    ) -> None:
        self.dc = dc
        self.elo = elo
        self.nudge = nudge

    def lambdas(
        self, home: str, away: str, neutral: bool = True,
        host_adv_factor: float = 1.0,
    ) -> tuple[float, float]:
        """
        Blended expected goals.

        host_adv_factor scales the home advantage (1.0 = full group-stage
        advantage, <1 for knockout rounds where the venue is less certain).
        """
        lam1, lam2 = self.dc.lambdas(home, away, neutral=True)
        if not neutral:
            adv = self.dc.home_advantage ** host_adv_factor
            lam1 *= adv

        elo_gap = self.elo.get(home) - self.elo.get(away)
        if not neutral:
            elo_gap += self.elo.home_adv_elo * host_adv_factor
        shift = np.exp(self.nudge * elo_gap / 400.0)
        return lam1 * shift, lam2 / shift

    def predict(
        self, home: str, away: str, neutral: bool = True,
        host_adv_factor: float = 1.0, max_goals: int = MAX_GOALS,
    ) -> dict[str, Any]:
        lam1, lam2 = self.lambdas(home, away, neutral, host_adv_factor)
        mat = self.dc.score_matrix_from_lambdas(lam1, lam2, max_goals)

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
        }

    def pen_shootout_p_home(self, home: str, away: str) -> float:
        """
        Penalty shootouts are close to a coin flip; give the stronger side a
        gentle edge bounded to ~[0.42, 0.58].
        """
        gap = self.elo.get(home) - self.elo.get(away)
        return 0.5 + 0.08 * float(np.tanh(gap / 400.0))

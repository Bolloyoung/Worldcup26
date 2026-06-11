"""
International football Elo ratings (eloratings.net-style).

Ported from the UCL system's hybrid Elo; the xG component is dropped (no
reliable xG for most internationals) and K-factors are keyed on the
competition name instead of the club-competition stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import ELO_HOME_ADV, ELO_K, ELO_K_DEFAULT


@dataclass
class EloRatings:
    default_rating: float = 1500.0
    home_adv_elo: float = ELO_HOME_ADV

    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.setdefault(team, self.default_rating)

    def _expected(self, r_home: float, r_away: float, neutral: bool) -> float:
        adv = 0.0 if neutral else self.home_adv_elo
        return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - adv) / 400.0))

    @staticmethod
    def _goal_diff_mult(g_diff: int) -> float:
        return min(np.log(abs(g_diff) + 1) + 1.0, 2.5)

    def update(
        self,
        home: str,
        away: str,
        goals_home: int,
        goals_away: int,
        tournament: str = "Friendly",
        neutral: bool = False,
    ) -> None:
        r_h, r_a = self.get(home), self.get(away)
        exp_h = self._expected(r_h, r_a, neutral)

        if goals_home > goals_away:
            act_h = 1.0
        elif goals_home == goals_away:
            act_h = 0.5
        else:
            act_h = 0.0

        k = ELO_K.get(tournament, ELO_K_DEFAULT)
        gd_mult = self._goal_diff_mult(goals_home - goals_away)
        delta = k * gd_mult * (act_h - exp_h)

        self.ratings[home] = r_h + delta
        self.ratings[away] = r_a - delta

    def fit(self, matches: pd.DataFrame) -> "EloRatings":
        """Process matches chronologically (columns as in fetch.elo_matches)."""
        df = matches.sort_values("date")
        for row in df.itertuples(index=False):
            self.update(
                home=row.home_team,
                away=row.away_team,
                goals_home=int(row.goals_home),
                goals_away=int(row.goals_away),
                tournament=str(row.tournament),
                neutral=bool(row.neutral),
            )
        return self

    def predict(self, team1: str, team2: str, neutral: bool = True) -> dict:
        r1, r2 = self.get(team1), self.get(team2)
        p1 = self._expected(r1, r2, neutral)
        return {
            "team1": team1, "team2": team2,
            "team1_win_prob": round(p1, 4),
            "team2_win_prob": round(1.0 - p1, 4),
            "team1_rating": round(r1, 1),
            "team2_rating": round(r2, 1),
            "rating_diff": round(r1 - r2, 1),
        }

    def leaderboard(self, teams: list[str] | None = None) -> pd.DataFrame:
        items = (
            {t: self.get(t) for t in teams} if teams is not None
            else self.ratings
        )
        rows = [{"team": t, "elo": round(r, 1)} for t, r in items.items()]
        return (
            pd.DataFrame(rows)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

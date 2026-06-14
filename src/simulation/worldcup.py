"""
Full 2026 World Cup Monte Carlo tournament simulator.

Format (FIFA, 48 teams, 39 days, 2026-06-11 → 2026-07-19):
- 12 groups of 4; top two per group + 8 best third-placed teams → round of 32
- Official knockout bracket, matches 73-104 (round of 32 → final)
- Third-placed teams are assigned to the eight constrained bracket slots via
  a backtracking perfect matching on FIFA's allowed-group sets.

Group draw (December 2025 Washington D.C. draw + March 2026 playoff winners),
verified against the group-stage fixture graph in the results dataset.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    ET_SCALE,
    HOSTS,
    KO_HOST_ADV_FACTOR,
    MAX_GOALS,
    N_TOURNAMENTS,
    SEED,
)
from ..models.engine import ProbabilityEngine

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_TO_GROUP: dict[str, str] = {
    t: g for g, ts in GROUPS.items() for t in ts
}

# Round of 32, FIFA matches 73-88.
# ("W", "A") = winner of group A, ("R", "A") = runner-up,
# ("T", "ABCDF") = best third from one of those groups.
R32: dict[int, tuple[tuple[str, str], tuple[str, str]]] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("T", "ABCDF")),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("T", "CDFGH")),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("T", "CEFHI")),
    80: (("W", "L"), ("T", "EHIJK")),
    81: (("W", "D"), ("T", "BEFIJ")),
    82: (("W", "G"), ("T", "AEHIJ")),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("T", "EFGIJ")),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("T", "DEIJL")),
    88: (("R", "D"), ("R", "G")),
}

R16: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
}
QF: dict[int, tuple[int, int]] = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF: dict[int, tuple[int, int]] = {101: (97, 98), 102: (99, 100)}
FINAL = 104

THIRD_SLOTS: list[tuple[int, str]] = [
    (m, slots[1][1]) for m, slots in sorted(R32.items()) if slots[1][0] == "T"
]

ROUNDS = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


def allocate_thirds(qualified_groups: list[str]) -> dict[int, str] | None:
    """
    Assign the 8 qualifying third-place groups to the 8 constrained slots.
    Returns {match_number: group_letter} or None if no perfect matching
    exists (cannot happen with FIFA's table, but guarded anyway).
    """
    slots = sorted(THIRD_SLOTS, key=lambda s: len(set(s[1]) & set(qualified_groups)))
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        match_no, allowed = slots[i]
        for g in qualified_groups:
            if g in used or g not in allowed:
                continue
            assignment[match_no] = g
            used.add(g)
            if backtrack(i + 1):
                return True
            del assignment[match_no]
            used.discard(g)
        return False

    return assignment if backtrack(0) else None


class WorldCupSimulator:
    """
    Monte Carlo simulation of the whole tournament.

    Score tables for every (home, away, advantage) pairing are cached, so
    10,000 tournament runs stay fast.
    """

    def __init__(
        self,
        engine: ProbabilityEngine,
        fixtures: pd.DataFrame,
        n_tournaments: int = N_TOURNAMENTS,
        seed: int = SEED,
        max_goals: int = MAX_GOALS,
        known_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    ) -> None:
        self.engine = engine
        self.fixtures = fixtures
        self.n = n_tournaments
        self.max_goals = max_goals
        # Completed group fixtures, locked to their actual scores so the
        # forecast conditions on matches that have already been played.
        self.known_results = known_results or {}
        self._rng = np.random.default_rng(seed)
        self._table_cache: dict[tuple, np.ndarray] = {}

    # ── Score sampling ────────────────────────────────────────────────────

    def _prob_table(
        self, home: str, away: str, neutral: bool, host_adv_factor: float,
        et: bool = False,
    ) -> np.ndarray:
        key = (home, away, neutral, host_adv_factor, et)
        if key not in self._table_cache:
            lam1, lam2 = self.engine.lambdas(home, away, neutral, host_adv_factor)
            if et:
                lam1, lam2 = lam1 * ET_SCALE, lam2 * ET_SCALE
            # Use the untempered Dixon-Coles matrix here: the engine lambdas
            # already carry the host/confederation/squad adjustments, but the
            # display temperature is a single-match calibration and must not
            # compound across the 7 simulated knockout rounds.
            mat = self.engine.dc.score_matrix_from_lambdas(
                lam1, lam2, self.max_goals
            )
            self._table_cache[key] = mat.ravel()
        return self._table_cache[key]

    def _sample_scores(
        self, home: str, away: str, neutral: bool,
        host_adv_factor: float, size: int, et: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        flat = self._prob_table(home, away, neutral, host_adv_factor, et)
        idx = self._rng.choice(len(flat), size=size, p=flat)
        return np.divmod(idx, self.max_goals + 1)

    def _ko_winner(self, home: str, away: str) -> str:
        """Simulate one knockout match (90' + ET + pens). Returns winner."""
        neutral = home not in HOSTS
        f = KO_HOST_ADV_FACTOR if not neutral else 1.0
        h, a = self._sample_scores(home, away, neutral, f, 1)
        h, a = int(h[0]), int(a[0])
        if h != a:
            return home if h > a else away
        he, ae = self._sample_scores(home, away, neutral, f, 1, et=True)
        if he[0] != ae[0]:
            return home if he[0] > ae[0] else away
        p = self.engine.pen_shootout_p_home(home, away)
        return home if self._rng.random() < p else away

    # ── Group stage ───────────────────────────────────────────────────────

    def _simulate_groups(
        self, scores: dict[int, tuple[np.ndarray, np.ndarray]], sim: int
    ) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
        """
        Returns ({group: [1st, 2nd, 3rd, 4th]}, ranked list of
        (group, team) thirds best-first).
        """
        pts: dict[str, float] = defaultdict(float)
        gd: dict[str, int] = defaultdict(int)
        gf: dict[str, int] = defaultdict(int)

        for i, row in enumerate(self.fixtures.itertuples(index=False)):
            h, a = scores[i][0][sim], scores[i][1][sim]
            home, away = row.home_team, row.away_team
            gf[home] += h
            gf[away] += a
            gd[home] += h - a
            gd[away] += a - h
            if h > a:
                pts[home] += 3
            elif h < a:
                pts[away] += 3
            else:
                pts[home] += 1
                pts[away] += 1

        standings: dict[str, list[str]] = {}
        thirds: list[tuple[float, int, int, float, str, str]] = []
        for g, teams in GROUPS.items():
            order = sorted(
                teams,
                key=lambda t: (pts[t], gd[t], gf[t], self._rng.random()),
                reverse=True,
            )
            standings[g] = order
            t3 = order[2]
            thirds.append(
                (pts[t3], gd[t3], gf[t3], self._rng.random(), g, t3)
            )

        thirds.sort(reverse=True)
        ranked_thirds = [(g, t) for _, _, _, _, g, t in thirds]
        return standings, ranked_thirds

    # ── Full tournament ───────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        n = self.n
        # Pre-sample all group fixtures for every tournament at once. Fixtures
        # with a known result are locked to their actual score (a constant
        # array) so completed matches are never re-simulated.
        group_scores: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, row in enumerate(self.fixtures.itertuples(index=False)):
            res = self.known_results.get((row.home_team, row.away_team))
            if res is not None:
                hg, ag = res
                group_scores[i] = (
                    np.full(n, hg, dtype=np.int64),
                    np.full(n, ag, dtype=np.int64),
                )
            else:
                group_scores[i] = self._sample_scores(
                    row.home_team, row.away_team, bool(row.neutral), 1.0, n
                )

        reach = {r: defaultdict(int) for r in ROUNDS[1:]}
        group_pos = defaultdict(lambda: np.zeros(4, dtype=np.int64))
        final_pairs: dict[tuple[str, str], int] = defaultdict(int)

        for sim in range(n):
            standings, thirds = self._simulate_groups(group_scores, sim)

            for g, order in standings.items():
                for pos, t in enumerate(order):
                    group_pos[t][pos] += 1

            third_groups = [g for g, _ in thirds[:8]]
            third_team = {g: t for g, t in thirds[:8]}
            alloc = allocate_thirds(third_groups)
            if alloc is None:  # fall back: relax constraints, rank order
                alloc = {
                    m: g for (m, _), g in zip(THIRD_SLOTS, third_groups)
                }

            def resolve(slot: tuple[str, str], match_no: int) -> str:
                kind, ref = slot
                if kind == "W":
                    return standings[ref][0]
                if kind == "R":
                    return standings[ref][1]
                return third_team[alloc[match_no]]

            winners: dict[int, str] = {}
            for m, (s1, s2) in R32.items():
                t1, t2 = resolve(s1, m), resolve(s2, m)
                reach["r32"][t1] += 1
                reach["r32"][t2] += 1
                winners[m] = self._ko_winner(t1, t2)

            for rnd, pairs in (("r16", R16), ("qf", QF), ("sf", SF)):
                for m, (m1, m2) in pairs.items():
                    t1, t2 = winners[m1], winners[m2]
                    reach[rnd][t1] += 1
                    reach[rnd][t2] += 1
                    winners[m] = self._ko_winner(t1, t2)

            f1, f2 = winners[101], winners[102]
            reach["final"][f1] += 1
            reach["final"][f2] += 1
            champ = self._ko_winner(f1, f2)
            reach["champion"][champ] += 1
            final_pairs[tuple(sorted((f1, f2)))] += 1

        teams = sorted(TEAM_TO_GROUP)
        records = []
        for t in teams:
            rec = {"team": t, "group": TEAM_TO_GROUP[t]}
            for r in ROUNDS[1:]:
                rec[f"p_{r}"] = reach[r][t] / n
            pos = group_pos[t]
            for i in range(4):
                rec[f"p_group_pos{i + 1}"] = pos[i] / n
            records.append(rec)

        forecast = (
            pd.DataFrame(records)
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True)
        )

        top_finals = sorted(
            final_pairs.items(), key=lambda kv: kv[1], reverse=True
        )[:15]
        return {
            "forecast": forecast,
            "n_tournaments": n,
            "top_finals": [
                {"final": f"{a} vs {b}", "prob": c / n}
                for (a, b), c in top_finals
            ],
        }

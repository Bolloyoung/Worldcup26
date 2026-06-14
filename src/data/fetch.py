"""
Data acquisition and preparation.

Primary source: martj42/international_results — a community-maintained CSV of
every official men's international 'A' match since 1872, updated after every
matchday. It also carries the scheduled 2026 World Cup fixtures (scores NA),
which lets us read the real group-stage fixture list straight from the data.
"""

from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from ..config import (
    COMPETITION_WEIGHTS,
    DEFAULT_COMPETITION_WEIGHT,
    GROUP_STAGE_END,
    MIN_MATCHES_PER_TEAM,
    RAW_DIR,
    RESULTS_CSV_PATH,
    RESULTS_CSV_URL,
    TRAIN_START,
)


def download_results(force: bool = False) -> pd.DataFrame:
    """Download (or load cached) full international results CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_CSV_PATH.exists() and not force:
        return pd.read_csv(RESULTS_CSV_PATH)

    with urllib.request.urlopen(RESULTS_CSV_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    df.to_csv(RESULTS_CSV_PATH, index=False)
    return df


def training_matches(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Played matches since TRAIN_START between teams with enough data,
    in the schema the models expect:
    date, home_team, away_team, goals_home, goals_away, neutral, weight_comp.
    """
    if df is None:
        df = download_results()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["date"] >= pd.Timestamp(TRAIN_START)]
    out = out.dropna(subset=["home_score", "away_score"])

    out = out.rename(
        columns={"home_score": "goals_home", "away_score": "goals_away"}
    )
    out["goals_home"] = out["goals_home"].astype(int)
    out["goals_away"] = out["goals_away"].astype(int)
    out["neutral"] = out["neutral"].astype(str).str.upper().eq("TRUE") | (
        out["neutral"] == True  # noqa: E712 — column may already be bool
    )
    out["weight_comp"] = (
        out["tournament"]
        .map(COMPETITION_WEIGHTS)
        .fillna(DEFAULT_COMPETITION_WEIGHT)
    )

    counts = pd.concat(
        [out["home_team"], out["away_team"]]
    ).value_counts()
    keep = set(counts[counts >= MIN_MATCHES_PER_TEAM].index)
    out = out[out["home_team"].isin(keep) & out["away_team"].isin(keep)]

    cols = [
        "date", "home_team", "away_team", "goals_home", "goals_away",
        "neutral", "tournament", "weight_comp",
    ]
    return out[cols].sort_values("date").reset_index(drop=True)


def elo_matches(df: pd.DataFrame | None = None, start: str = "2010-01-01") -> pd.DataFrame:
    """Played matches since `start` for Elo fitting (all teams)."""
    if df is None:
        df = download_results()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["date"] >= pd.Timestamp(start)]
    out = out.dropna(subset=["home_score", "away_score"])
    out = out.rename(
        columns={"home_score": "goals_home", "away_score": "goals_away"}
    )
    out["goals_home"] = out["goals_home"].astype(int)
    out["goals_away"] = out["goals_away"].astype(int)
    out["neutral"] = out["neutral"].astype(str).str.upper().eq("TRUE") | (
        out["neutral"] == True  # noqa: E712
    )
    return out.sort_values("date").reset_index(drop=True)


def wc2026_group_fixtures(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    The 72 scheduled 2026 World Cup group-stage fixtures, with the dataset's
    own neutral flag (False for host-nation matches → home advantage applies).
    """
    if df is None:
        df = download_results()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    mask = (
        (out["tournament"] == "FIFA World Cup")
        & (out["date"] >= pd.Timestamp("2026-06-01"))
        & (out["date"] <= pd.Timestamp(GROUP_STAGE_END))
    )
    out = out[mask].copy()
    out["neutral"] = out["neutral"].astype(str).str.upper().eq("TRUE") | (
        out["neutral"] == True  # noqa: E712
    )
    cols = ["date", "home_team", "away_team", "city", "country", "neutral"]
    return out[cols].sort_values("date").reset_index(drop=True)


def played_results(df: pd.DataFrame | None = None) -> dict[tuple[str, str], tuple[int, int]]:
    """
    Group-stage fixtures that already have a score, as
    {(home_team, away_team): (goals_home, goals_away)}.

    Lets the tournament simulator condition on what's actually happened
    rather than re-simulating completed matches.
    """
    if df is None:
        df = download_results()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    mask = (
        (out["tournament"] == "FIFA World Cup")
        & (out["date"] >= pd.Timestamp("2026-06-01"))
        & (out["date"] <= pd.Timestamp(GROUP_STAGE_END))
        & out["home_score"].notna()
        & out["away_score"].notna()
    )
    out = out[mask]
    return {
        (r["home_team"], r["away_team"]): (int(r["home_score"]), int(r["away_score"]))
        for _, r in out.iterrows()
    }


def verify_groups(fixtures: pd.DataFrame, groups: dict[str, list[str]]) -> None:
    """
    Cross-check the hard-coded draw against the fixture graph: each group of
    four must be a closed clique in the group-stage fixture list.
    """
    team_to_group: dict[str, str] = {
        t: g for g, ts in groups.items() for t in ts
    }
    missing = set(np.unique(fixtures[["home_team", "away_team"]].values)) - set(
        team_to_group
    )
    if missing:
        raise ValueError(f"Fixture teams missing from draw table: {missing}")
    for _, row in fixtures.iterrows():
        gh = team_to_group[row["home_team"]]
        ga = team_to_group[row["away_team"]]
        if gh != ga:
            raise ValueError(
                f"Draw mismatch: {row['home_team']} ({gh}) vs "
                f"{row['away_team']} ({ga}) is a cross-group fixture."
            )

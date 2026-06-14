"""
Betting interpretation layer (decision support, model-only).

Turns a match prediction (the engine's `predict()` output) into a concrete,
human-readable betting card: a primary 1X2 suggestion with a confidence tier,
scoreline bankers/covers, and derived markets (Over/Under, BTTS, Double
Chance, Draw-No-Bet) — all computed from the model's own score matrix, with
no external odds.

Tie handling (the cases the user asked for):
  - ~33/33/33 (all three within TIE_EPS)        → "No edge — avoid".
  - top two outcomes within TIE_EPS              → Double Chance over the two
                                                   leaders (or Draw-No-Bet if
                                                   the draw is one of them).
  - symmetric 30/40/30 (draw leads, H≈A)         → draw primary, flagged, with
                                                   Double Chance 12 as the
                                                   goals-based alternative.

This is not financial advice; see DISCLAIMER.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Confidence thresholds
STRONG_PROB = 0.55        # max outcome prob for a "Strong" call
STRONG_GAP = 0.12         # lead over 2nd-best outcome for "Strong"
LEAN_GAP = 0.06           # lead over 2nd-best for "Lean"
TIE_EPS = 0.04            # outcomes within this are treated as level
ALL_TIE_EPS = 0.05        # spread (max-min) below this ≈ 33/33/33

DISCLAIMER = (
    "Model-derived probabilities for analysis/entertainment only. Not financial "
    "advice. No external bookmaker odds are used, so a suggestion is not a "
    "guaranteed value bet. Bet responsibly; 18+/21+."
)

_OUTCOME_NAME = {"H": "home", "D": "draw", "A": "away"}
_MARKET_CODE = {"H": "1", "D": "X", "A": "2"}


def _derived_markets(mat: np.ndarray) -> dict[str, float]:
    n = mat.shape[0]
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    total = i + j
    over25 = float(mat[total >= 3].sum())
    btts = float(mat[(i >= 1) & (j >= 1)].sum())
    h = float(np.tril(mat, -1).sum())
    d = float(np.trace(mat))
    a = float(np.triu(mat, 1).sum())
    hd = h + d if (h + d) else 1e-9
    ha = h + a if (h + a) else 1e-9
    return {
        "over_2_5": over25,
        "under_2_5": 1.0 - over25,
        "btts_yes": btts,
        "btts_no": 1.0 - btts,
        "double_chance_1X": h + d,
        "double_chance_12": h + a,
        "double_chance_X2": d + a,
        "draw_no_bet_home": h / ha,
        "draw_no_bet_away": a / ha,
    }


def _double_chance(o1: str, o2: str) -> tuple[str, str]:
    """Return (code, label) for a double chance covering two outcomes."""
    pair = {o1, o2}
    if pair == {"H", "D"}:
        return "1X", "Double Chance — Home or Draw (1X)"
    if pair == {"H", "A"}:
        return "12", "Double Chance — Home or Away, no draw (12)"
    return "X2", "Double Chance — Draw or Away (X2)"


def betting_card(prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Build a betting card from an engine `predict()` result.

    Requires keys: home_team, away_team, home_win, draw, away_win,
    score_matrix, top_scorelines.
    """
    home = prediction["home_team"]
    away = prediction["away_team"]
    probs = {"H": prediction["home_win"], "D": prediction["draw"], "A": prediction["away_win"]}
    mat = np.asarray(prediction["score_matrix"])

    order = sorted(probs, key=probs.get, reverse=True)  # outcomes best→worst
    o1, o2, o3 = order
    p1, p2, p3 = probs[o1], probs[o2], probs[o3]
    spread = p1 - p3
    gap12 = p1 - p2

    def name(o: str) -> str:
        return {"H": home, "D": "Draw", "A": away}[o]

    markets = _derived_markets(mat)
    sl = prediction["top_scorelines"]
    banker = sl[0]
    cover = sl[:3]

    alternatives: list[str] = []
    tie_flag: str | None = None

    # ── Decide primary suggestion + confidence ────────────────────────────
    if spread < ALL_TIE_EPS:
        confidence = "Avoid"
        tie_flag = "All three outcomes are near-level (~33/33/33)."
        primary = {
            "market": "1X2",
            "selection": "No bet",
            "probability": p1,
            "confidence": confidence,
        }
        headline = f"{home} vs {away}: no edge — skip the 1X2 market."
        alternatives.append(
            f"If you must have action: highest single market is "
            f"{'Over 2.5' if markets['over_2_5'] >= 0.5 else 'Under 2.5'} "
            f"({max(markets['over_2_5'], markets['under_2_5']) * 100:.0f}%)."
        )
    elif gap12 < TIE_EPS:
        # Top two outcomes are level → cover both with a double chance.
        confidence = "Lean"
        code, label = _double_chance(o1, o2)
        dc_prob = {"1X": markets["double_chance_1X"],
                   "12": markets["double_chance_12"],
                   "X2": markets["double_chance_X2"]}[code]
        tie_flag = (
            f"{name(o1)} ({p1*100:.0f}%) and {name(o2)} ({p2*100:.0f}%) are "
            f"level — single 1X2 is a coin-flip."
        )
        primary = {
            "market": "Double Chance",
            "selection": label,
            "probability": dc_prob,
            "confidence": confidence,
        }
        headline = f"{home} vs {away}: {label} ({dc_prob*100:.0f}%)."
        if "D" in (o1, o2):
            dnb = "home" if "H" in (o1, o2) else "away"
            alternatives.append(
                f"Or Draw-No-Bet on {name('H') if dnb=='home' else name('A')} "
                f"({markets[f'draw_no_bet_{dnb}']*100:.0f}%)."
            )
    else:
        # Clear leader.
        if p1 >= STRONG_PROB and gap12 >= STRONG_GAP:
            confidence = "Strong"
        elif gap12 >= LEAN_GAP:
            confidence = "Lean"
        else:
            confidence = "Slight lean"
        primary = {
            "market": "1X2",
            "selection": f"{name(o1)} ({_MARKET_CODE[o1]})",
            "probability": p1,
            "confidence": confidence,
        }
        headline = (
            f"{home} vs {away}: back {name(o1)} "
            f"({_MARKET_CODE[o1]}) — {p1*100:.0f}%, {confidence}."
        )
        # Symmetric 30/40/30: draw leads but home and away are level.
        if o1 == "D" and abs(probs["H"] - probs["A"]) < TIE_EPS:
            tie_flag = (
                f"Draw leads but {home} and {away} are level "
                f"({probs['H']*100:.0f}% each) — low-scoring game implied."
            )
            alternatives.append(
                f"Goals alternative: Double Chance 12 "
                f"({markets['double_chance_12']*100:.0f}%) if you expect a winner."
            )
        # For a non-draw leader, offer the safer double chance as a hedge.
        if o1 in ("H", "A"):
            code, label = _double_chance(o1, "D")
            dc_prob = markets[f"double_chance_{code}"]
            alternatives.append(f"Safer: {label} ({dc_prob*100:.0f}%).")

    # Scoreline suggestions always included.
    banker_str = f"{banker['home_goals']}-{banker['away_goals']}"
    cover_str = ", ".join(
        f"{c['home_goals']}-{c['away_goals']} ({c['prob']*100:.0f}%)" for c in cover
    )

    return {
        "match": f"{home} vs {away}",
        "headline": headline,
        "confidence": confidence,
        "primary_market": primary,
        "tie_flag": tie_flag,
        "scoreline_banker": {"score": banker_str, "probability": banker["prob"]},
        "scoreline_cover": [
            {"score": f"{c['home_goals']}-{c['away_goals']}", "probability": c["prob"]}
            for c in cover
        ],
        "derived_markets": {k: round(v, 4) for k, v in markets.items()},
        "alternatives": alternatives,
        "outcome_probs": {"home": probs["H"], "draw": probs["D"], "away": probs["A"]},
        "disclaimer": DISCLAIMER,
    }

"""World Cup 2026 prediction dashboard.

Reads the precomputed artifacts in predictions/ (so the deployed app needs
no model fitting at startup) and reconstructs the probability engine from
the serialised parameters for the interactive match predictor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.betting import betting_card  # noqa: E402
from src.models.dixon_coles import DixonColesModel  # noqa: E402
from src.models.elo import EloRatings  # noqa: E402
from src.models.engine import ProbabilityEngine  # noqa: E402
from src.simulation.worldcup import GROUPS, TEAM_TO_GROUP  # noqa: E402
from src.squads import build_squad_index  # noqa: E402

PRED = ROOT / "predictions"

st.set_page_config(
    page_title="World Cup 2026 Predictions", page_icon="🏆", layout="wide"
)


@st.cache_data
def load_artifacts():
    forecast = pd.read_json(PRED / "tournament_forecast.json")
    fixtures = pd.DataFrame(
        json.loads((PRED / "group_fixtures.json").read_text())
    )
    model = json.loads((PRED / "model.json").read_text())
    return forecast, fixtures, model


@st.cache_resource
def load_engine():
    model = json.loads((PRED / "model.json").read_text())
    dc = DixonColesModel.from_dict(model["dixon_coles"])
    elo = EloRatings()
    elo.ratings.update(model["elo"])
    return ProbabilityEngine(dc, elo, squad_index=build_squad_index())


forecast, fixtures, model = load_artifacts()
meta = model["meta"]

st.title("🏆 FIFA World Cup 2026 — Prediction Engine")
st.caption(
    f"Dixon-Coles + Elo ensemble · fitted {meta['fitted']} on "
    f"{meta['n_train_matches']:,} internationals · "
    f"{meta['n_tournaments']:,} Monte Carlo tournaments · "
    "48 teams, June 11 – July 19, 2026"
)

tab_title, tab_groups, tab_fixtures, tab_match, tab_accuracy, tab_about = st.tabs(
    ["🥇 Title Race", "📋 Groups", "📅 Fixtures", "⚽ Match Predictor",
     "📏 Accuracy", "ℹ️ Model"]
)

with tab_title:
    top = forecast.head(15).copy()
    fig = go.Figure()
    for col, name, color in [
        ("p_champion", "Champion", "#d4af37"),
        ("p_final", "Reach final", "#8884d8"),
        ("p_sf", "Reach semi-final", "#82ca9d"),
    ]:
        fig.add_trace(
            go.Bar(
                y=top["team"], x=top[col] * 100, name=name,
                orientation="h", marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group", height=600, xaxis_title="Probability (%)",
        yaxis=dict(autorange="reversed"), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most likely finals")
    finals = pd.DataFrame(meta["top_finals"])
    finals["prob"] = (finals["prob"] * 100).round(2).astype(str) + " %"
    st.dataframe(finals, hide_index=True, use_container_width=True)

    st.subheader("Full forecast (all 48 teams)")
    show = forecast.copy()
    pct_cols = [c for c in show.columns if c.startswith("p_")]
    show[pct_cols] = (show[pct_cols] * 100).round(2)
    st.dataframe(show, hide_index=True, use_container_width=True, height=500)

with tab_groups:
    cols = st.columns(3)
    for i, (g, teams) in enumerate(GROUPS.items()):
        with cols[i % 3]:
            st.markdown(f"**Group {g}**")
            sub = forecast[forecast["team"].isin(teams)][
                ["team", "p_group_pos1", "p_group_pos2", "p_r32"]
            ].copy()
            sub.columns = ["Team", "Win group", "2nd", "Reach R32"]
            for c in ["Win group", "2nd", "Reach R32"]:
                sub[c] = (sub[c] * 100).round(1)
            sub = sub.sort_values("Reach R32", ascending=False)
            st.dataframe(sub, hide_index=True, use_container_width=True)

with tab_fixtures:
    live_path = PRED / "live_updates.json"
    if live_path.exists():
        live = json.loads(live_path.read_text())
        if live:
            st.subheader("🔴 Confirmed-XI updates (≈1 hour before kickoff)")
            st.caption(
                "These matches have been re-valued from the **actual** starting "
                "eleven (official teamsheet), overriding the projected squad."
            )
            rows = []
            for u in live.values():
                rows.append({
                    "Date": u["date"],
                    "Match": f"{u['home']} v {u['away']}",
                    "Home %": round(u["p_home"] * 100, 1),
                    "Draw %": round(u["p_draw"] * 100, 1),
                    "Away %": round(u["p_away"] * 100, 1),
                    "XI known (H/A)": f"{u['home_known']}/{u['away_known']}",
                    "Source": u["lineup_source"],
                    "Suggestion": u["betting_headline"],
                })
            st.dataframe(pd.DataFrame(rows).sort_values("Date"),
                         hide_index=True, use_container_width=True)
            st.divider()

    day = st.selectbox("Date", sorted(fixtures["date"].unique()))
    sub = fixtures[fixtures["date"] == day].copy()
    for c in ["p_home", "p_draw", "p_away"]:
        sub[c] = (sub[c] * 100).round(1)
    sub = sub.rename(
        columns={
            "p_home": "Home %", "p_draw": "Draw %", "p_away": "Away %",
            "xg_home": "xG home", "xg_away": "xG away",
            "most_likely_score": "Top score",
        }
    )
    st.dataframe(sub, hide_index=True, use_container_width=True)

with tab_match:
    engine = load_engine()
    teams48 = sorted(TEAM_TO_GROUP)
    c1, c2, c3 = st.columns([2, 2, 1])
    home = c1.selectbox("Team 1", teams48, index=teams48.index("Spain"))
    away = c2.selectbox("Team 2", teams48, index=teams48.index("Argentina"))
    host_home = c3.checkbox(
        "Team 1 plays at home", value=home in ("United States", "Mexico", "Canada")
    )
    if home == away:
        st.warning("Pick two different teams.")
    else:
        p = engine.predict(home, away, neutral=not host_home)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{home} win", f"{p['home_win'] * 100:.1f} %")
        m2.metric("Draw", f"{p['draw'] * 100:.1f} %")
        m3.metric(f"{away} win", f"{p['away_win'] * 100:.1f} %")
        st.caption(
            f"Expected goals {p['expected_home_goals']:.2f} – "
            f"{p['expected_away_goals']:.2f} · Elo {p['elo_home']} vs "
            f"{p['elo_away']}"
        )
        mat = p["score_matrix"][:6, :6]
        fig = px.imshow(
            mat * 100,
            labels=dict(x=f"{away} goals", y=f"{home} goals", color="%"),
            text_auto=".1f", color_continuous_scale="Blues",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Most likely scorelines**")
        lines = pd.DataFrame(p["top_scorelines"][:8])
        lines["prob"] = (lines["prob"] * 100).round(2).astype(str) + " %"
        lines["score"] = (
            lines["home_goals"].astype(str) + "-" + lines["away_goals"].astype(str)
        )
        st.dataframe(
            lines[["score", "prob"]], hide_index=True, use_container_width=False
        )

        # ── Betting interpretation ────────────────────────────────────────
        st.divider()
        st.subheader("💡 Betting interpretation")
        card = betting_card(p)

        tier_color = {
            "Strong": "🟢", "Lean": "🟡", "Slight lean": "🟡",
            "Avoid": "🔴",
        }.get(card["confidence"], "🟡")
        st.markdown(f"### {tier_color} {card['headline']}")
        if card["tie_flag"]:
            st.warning(f"⚖️ {card['tie_flag']}")

        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Primary suggestion", card["primary_market"]["selection"],
            f"{card['primary_market']['probability'] * 100:.0f}%",
        )
        b2.metric("Scoreline banker", card["scoreline_banker"]["score"],
                  f"{card['scoreline_banker']['probability'] * 100:.1f}%")
        b3.metric("Confidence", card["confidence"])

        st.markdown(
            "**Cover (top-3 scorelines):** "
            + " · ".join(
                f"{s['score']} ({s['probability'] * 100:.0f}%)"
                for s in card["scoreline_cover"]
            )
        )
        if card["alternatives"]:
            for alt in card["alternatives"]:
                st.markdown(f"- {alt}")

        dm = card["derived_markets"]
        st.markdown("**Derived markets**")
        md1, md2, md3, md4 = st.columns(4)
        md1.metric("Over 2.5", f"{dm['over_2_5'] * 100:.0f}%")
        md2.metric("Under 2.5", f"{dm['under_2_5'] * 100:.0f}%")
        md3.metric("BTTS yes", f"{dm['btts_yes'] * 100:.0f}%")
        md4.metric("BTTS no", f"{dm['btts_no'] * 100:.0f}%")
        md5, md6, md7, md8 = st.columns(4)
        md5.metric("1X", f"{dm['double_chance_1X'] * 100:.0f}%")
        md6.metric("12", f"{dm['double_chance_12'] * 100:.0f}%")
        md7.metric("X2", f"{dm['double_chance_X2'] * 100:.0f}%")
        md8.metric("DNB home", f"{dm['draw_no_bet_home'] * 100:.0f}%")

        st.caption(card["disclaimer"])

with tab_accuracy:
    eval_path = PRED / "evaluation.json"
    if not eval_path.exists():
        st.info(
            "No evaluation yet. Run `python scripts/evaluate.py` after some "
            "matches have been played to score the model against real results."
        )
    else:
        ev = json.loads(eval_path.read_text())
        wf = ev.get("walk_forward", {})
        frozen = ev.get("frozen_log", {})
        st.caption(
            "How the model is doing against actual World Cup 2026 results. "
            "**Walk-forward** re-fits before each matchday (leakage-free, covers "
            "all played games); **frozen log** scores genuine pre-match "
            "predictions and grows as the tournament goes on."
        )
        card = wf if wf.get("n", 0) else frozen
        if not card.get("n", 0):
            st.info(card.get("message", "No results scored yet."))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matches scored", card["n"])
            c2.metric("Favourite accuracy", f"{card['favourite_accuracy']:.0%}")
            c3.metric("Log-loss", card["log_loss"],
                      f"{card['skill_vs_baseline_pct']:+.1f}% vs coin-flip",
                      delta_color="normal")
            c4.metric("Avg prob on actual", card["avg_prob_on_actual"],
                      f"baseline {card['baseline_avg_prob']}")

            if card.get("reliability"):
                st.markdown("**Reliability of the favourite pick** "
                            "(predicted vs actual hit rate — closer is better)")
                rel = pd.DataFrame(card["reliability"])
                st.dataframe(rel, hide_index=True, use_container_width=False)

            st.markdown("**Per-match results**")
            mtbl = pd.DataFrame(card["matches"])[
                ["date", "match", "predicted", "result", "favourite",
                 "fav_correct", "top_score"]
            ].rename(columns={"fav_correct": "fav_hit", "top_score": "model_top"})
            st.dataframe(mtbl, hide_index=True, use_container_width=True)
            st.caption(
                "Re-tune the model on these results: `python scripts/backtest.py` "
                "(now includes completed WC2026 matches), then adjust "
                "`src/config.py` and re-run `scripts/run_simulation.py`."
            )

with tab_about:
    st.markdown(
        """
### How it works

| Layer | Detail |
|---|---|
| **Data** | All official men's internationals since 2018 (~8k matches) from the community-maintained [martj42/international_results](https://github.com/martj42/international_results) dataset, refreshed after every matchday. Competition-weighted: World Cup 1.6× … friendlies 0.6×. |
| **Dixon-Coles** | Bivariate Poisson with low-score correlation (ρ), exponential time decay, **ridge regularisation** (curbs overconfidence), per-match neutral-venue handling, fitted by L-BFGS-B with analytic gradients over 231 national teams. |
| **Elo** | eloratings.net-style ratings since 2010, K-factor by competition (World Cup 60 … friendlies 20), goal-difference multiplier. |
| **Ensemble** | The Elo gap nudges the goal rates: λ × exp(±0.12·Δelo*/400) × squad_ratio^0.15, where Δelo* adds a **host boost** (🇺🇸🇲🇽🇨🇦) and an **inter-confederation shrink**; single-match output is **temperature-calibrated**. |
| **Squad** | Projected national XIs → a squad-strength index blended into the goal rates (the injuries/form hook); teams without squads fall back gracefully. |
| **Calibration** | Ridge + temperature tuned on the 2018 & 2022 World Cups (`scripts/backtest.py`): log-loss +2.0% vs the original, worst single-match call 91% → ~64%. |
| **Simulation** | 10,000 full tournaments, **conditioned on completed matches**: 72 group games, FIFA tiebreakers, best-8 third-place allocation via constraint matching, official bracket (matches 73–104), extra time at 33% goal rate, Elo-informed penalty shootouts. |

Built by adapting the UCL prediction system (Dixon-Coles + Elo + Monte Carlo)
to the 48-team international format, then recalibrated after the matchday-1 review.
        """
    )

# World Cup 2026 Prediction System 🏆

Probabilistic forecasts for the **FIFA World Cup 2026** (48 teams, USA/Mexico/Canada, June 11 – July 19, 2026 — a 39-day tournament), built by adapting the [UCL prediction system](../ucl-prediction-system)'s Dixon-Coles + Elo + Monte Carlo stack to international football and the new 12-group → round-of-32 format.

## Current forecast (10,000 simulated tournaments)

| Team | Champion | Final | Semi-final |
|------|---------:|------:|-----------:|
| Spain | 22.8% | 35.4% | 48.8% |
| Argentina | 20.1% | 30.9% | 44.3% |
| England | 10.3% | 18.6% | 32.5% |
| France | 6.6% | 12.9% | 26.5% |
| Brazil | 5.8% | 12.2% | 24.9% |
| Portugal | 4.9% | 11.0% | 21.0% |

Full 48-team forecast in [`predictions/tournament_forecast.json`](predictions/tournament_forecast.json).

## Quick start

```bash
pip install -r requirements.txt

# Fetch latest results, fit models, simulate 10,000 tournaments (~10 s)
python scripts/run_simulation.py --refresh-data

# Launch the dashboard
streamlit run dashboard/app.py

# Run the test suite
pytest tests/ -q
```

## Architecture

```
international results CSV (martj42/international_results, updated per matchday)
    ↓
training set: internationals since 2018, competition-weighted
    ↓                              ↓
Dixon-Coles bivariate Poisson   Elo ratings (since 2010)
(analytic-gradient L-BFGS-B,    (K by competition, GD multiplier,
 time decay, neutral venues)     home advantage 80 pts)
    ↓                              ↓
ProbabilityEngine: λ × exp(±0.18·Δelo/400)
    ↓
Monte Carlo: 10,000 full tournaments
(72 group games → FIFA tiebreakers → best-8 thirds constraint matching
 → official bracket M73–M104 → ET (33% rate) → Elo-informed penalties)
    ↓
predictions/*.json → Streamlit dashboard
```

### What changed vs the UCL version

| | UCL system | This system |
|---|---|---|
| Likelihood | Python loop (fine for 36 clubs) | **Vectorised + analytic gradient** — 231 national teams fit in ~2 s |
| Weighting | time decay only | time decay × **competition importance** (WC 1.6, friendly 0.6) |
| Venue | global home advantage | **per-match neutral flag**; host advantage for 🇺🇸🇲🇽🇨🇦, 50%-discounted in knockouts |
| Tournament | 16-team bracket | **48 teams, 12 groups, best-8 thirds allocation** (backtracking matching over FIFA's allowed-group slots), official M73–M104 bracket |
| xG | FBref xG in Elo updates | dropped (no reliable xG for internationals) |

## Data sources

- **[martj42/international_results](https://github.com/martj42/international_results)** — community-maintained CSV of every official men's international since 1872, refreshed after every matchday; also carries the scheduled WC2026 fixtures, from which the group draw is **cross-verified** at runtime (`verify_groups`).
- Group draw + knockout bracket: FIFA final draw (Dec 5, 2025) and the official match schedule (matches 73–104), as published by [FIFA](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026) / [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage).

The 49 MB raw CSV is cached in `data/raw/` (gitignored); the small JSON artifacts in `predictions/` are committed so the deployed app never has to fit models.

## Project structure

```
worldcup26-prediction/
├── src/
│   ├── config.py               — every tunable parameter
│   ├── data/fetch.py           — download, training set, fixture extraction, draw verification
│   ├── models/dixon_coles.py   — vectorised DC with analytic gradient
│   ├── models/elo.py           — international Elo
│   ├── models/engine.py        — DC × Elo ensemble
│   └── simulation/worldcup.py  — groups, thirds matching, bracket, Monte Carlo
├── scripts/run_simulation.py   — end-to-end pipeline
├── dashboard/app.py            — Streamlit app (5 tabs)
├── predictions/                — committed JSON artifacts
├── tests/                      — 15 tests (model recovery, bracket integrity, 495 thirds combos)
└── .github/workflows/update-predictions.yml — daily auto-refresh during the tournament
```

## Publish to GitHub

The repo is already initialised on branch `main`. With the [GitHub CLI](https://cli.github.com):

```bash
cd worldcup26-prediction
git add -A
git commit -m "World Cup 2026 prediction system"

# create the GitHub repo and push in one step
gh repo create worldcup26-prediction --public --source=. --push
```

Or without `gh` (create the empty repo on github.com first):

```bash
git remote add origin https://github.com/<your-username>/worldcup26-prediction.git
git push -u origin main
```

Day-to-day during the tournament:

```bash
python scripts/run_simulation.py --refresh-data   # pull latest results, refit
git add predictions/ && git commit -m "Update predictions $(date +%F)"
git push                                          # Streamlit Cloud redeploys automatically
```

…or just enable the included GitHub Action (`Actions` tab → *Update predictions* → enable), which does exactly this daily at 06:00 UTC for the duration of the tournament.

## Deployment — recommended: Streamlit Community Cloud

**[Streamlit Community Cloud](https://share.streamlit.io)** is the best fit: free, deploys straight from the GitHub repo, redeploys on every push (so the daily GitHub Action keeps the live site current), and the app is pure Streamlit with small committed artifacts — no server, database, or build step needed.

1. Push to GitHub (above)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repo: `<your-username>/worldcup26-prediction`, branch `main`, main file **`dashboard/app.py`**
4. Deploy — you get `https://<app-name>.streamlit.app`

Alternatives:

| Platform | When to prefer it |
|---|---|
| **Hugging Face Spaces** (Streamlit SDK) | similar free tier, better if you already live on HF |
| **Render / Railway** | custom domain + always-on (no cold sleep); start command: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0` |
| **GitHub Pages** | only if you export static HTML — you'd lose the interactive match predictor |

## Model notes & honest limitations

- **Tiebreakers**: points → goal difference → goals scored → random; FIFA's full head-to-head sub-criteria are approximated by the random tail.
- **Third-place bracket slots** use a constraint-respecting matching over FIFA's allowed-group sets per slot, verified across all 495 possible 8-of-12 combinations; FIFA's published allocation table may pick a different valid assignment in some combinations, which marginally shifts *who plays whom* but not aggregate team probabilities.
- **No squad/injury layer**: ratings are team-level. The UCL project's `player_valuation` module could be ported as a future feature.
- **Penalty shootouts**: 50/50 ± a small Elo-based edge (capped at 58/42).
- Probabilities are estimates with Monte Carlo noise of roughly ±0.5pp at 10,000 sims; run with `--sims 100000` for tighter estimates.

## References

- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Applied Statistics 46(2).
- World Football Elo Ratings: [eloratings.net](https://eloratings.net)

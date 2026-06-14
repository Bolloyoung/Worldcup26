# World Cup 2026 Prediction System 🏆

Probabilistic forecasts for the **FIFA World Cup 2026** (48 teams, USA/Mexico/Canada, June 11 – July 19, 2026 — a 39-day tournament), built by adapting the [UCL prediction system](../ucl-prediction-system)'s Dixon-Coles + Elo + Monte Carlo stack to international football and the new 12-group → round-of-32 format.

## Current forecast (10,000 simulated tournaments, conditioned on results to date)

| Team | Champion | Final | Semi-final |
|------|---------:|------:|-----------:|
| Argentina | 19.9% | 29.4% | 42.3% |
| Spain | 16.1% | 25.6% | 38.1% |
| England | 7.7% | 14.3% | 25.0% |
| Brazil | 6.5% | 12.3% | 22.7% |
| Morocco | 5.7% | 11.1% | 20.6% |
| France | 5.4% | 10.7% | 20.3% |

Full 48-team forecast in [`predictions/tournament_forecast.json`](predictions/tournament_forecast.json). The forecast re-conditions on completed matches each time the pipeline runs.

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
    ↓                  ↓                       ↓
Dixon-Coles        Elo ratings           squad valuation
(ridge-regularised, (K by competition)     (projected XIs → squad
 analytic gradient)                         strength index)
    ↓                  ↓                       ↓
ProbabilityEngine:  λ × exp(±nudge·Δelo*/400) × squad_ratio^w
   Δelo* includes host boost (🇺🇸🇲🇽🇨🇦) and inter-confederation shrink;
   single-match output is temperature-calibrated
    ↓
Monte Carlo: 10,000 full tournaments, conditioned on completed matches
(72 group games → FIFA tiebreakers → best-8 thirds constraint matching
 → official bracket M73–M104 → ET (33% rate) → Elo-informed penalties)
    ↓
predictions/*.json → Streamlit dashboard (+ betting interpretation)
```

### Matchday-1 review → recalibration

After matchday 1 the model was diagnosed as **overconfident, not broken**: it
called Qatar 1-1 Switzerland at **91% Switzerland** (expected score 0.55–3.66),
because fitted defense parameters collapsed toward zero with no regularisation,
and cross-confederation gaps (Qatar AFC vs Switzerland UEFA, few bridging
matches) were over-extrapolated. Five changes followed, each kept at a setting
**tuned on the 2018 & 2022 World Cups** (`scripts/backtest.py`):

| Change | Effect |
|---|---|
| **Ridge regularisation** of Dixon-Coles attack/defense | stops defense→0 param inflation; main overconfidence fix |
| **Lower Elo nudge** (0.18→0.12) | less aggressive cross-rating extrapolation |
| **Display temperature** (single-match/betting only) | flattens 1X2 for honest probabilities; *not* compounded through the simulation |
| **Explicit host boost** (🇺🇸🇲🇽🇨🇦) | hosts no longer under-rated (USA beat Paraguay 4-1 while the old model favoured Paraguay) |
| **Inter-confederation shrink** | softens unreliable cross-pool gaps |

Backtest (304 matches across the 2018 & 2022 World Cups): **log-loss 1.0032 vs
1.0236 baseline (+2.0%)**, with the worst single-match call dropping from 91% to
~64%. On the 8 matchday-1 games: favourites 5/8 (was 4/8), average probability
on the actual outcome 0.416 (was 0.393), max single-match probability 0.64
(was 0.91).

### What changed vs the UCL version

| | UCL system | This system |
|---|---|---|
| Likelihood | Python loop (fine for 36 clubs) | **Vectorised + analytic gradient** — 231 national teams fit in ~2 s |
| Weighting | time decay only | time decay × **competition importance** (WC 1.6, friendly 0.6) |
| Venue | global home advantage | **per-match neutral flag**; host advantage for 🇺🇸🇲🇽🇨🇦, 50%-discounted in knockouts |
| Tournament | 16-team bracket | **48 teams, 12 groups, best-8 thirds allocation** (backtracking matching over FIFA's allowed-group slots), official M73–M104 bracket |
| xG | FBref xG in Elo updates | dropped (no reliable xG for internationals) |
| Calibration | none | **ridge regularisation + display temperature**, backtested |
| Squad layer | club lineups | **projected national XIs → squad strength index** (graceful fallback) |
| Live update | static | **conditions on completed matches** (lock-in) |
| Betting | none | **interpretation layer** with tie handling (see below) |

## Betting interpretation layer

The Match Predictor tab turns each prediction into a betting card (decision
support from the model's own score matrix — no external odds):

- **Primary 1X2 pick** = highest-probability outcome, with a confidence tier
  (Strong / Lean / Slight lean) from the probability and its lead over 2nd.
- **Scorelines**: a top-1 "banker" and a top-3 "cover".
- **Derived markets**: Over/Under 2.5, BTTS, Double Chance (1X/12/X2), Draw-No-Bet.
- **Tie handling** (the awkward cases):
  - 40/30/30 → back the leader, with a safer Double Chance offered alongside.
  - 30/40/30 (draw leads, home≈away) → draw primary, flagged, with Double Chance 12 as the goals-based alternative.
  - top two level (e.g. 40/20/40) → **Double Chance** over the two leaders instead of a coin-flip 1X2.
  - ~33/33/33 → **"No edge — avoid"**.

`src/betting.py` exposes `betting_card(prediction)`. Decision support only —
not financial advice; bet responsibly.

## Data sources

- **[martj42/international_results](https://github.com/martj42/international_results)** — community-maintained CSV of every official men's international since 1872, refreshed after every matchday; also carries the scheduled WC2026 fixtures, from which the group draw is **cross-verified** at runtime (`verify_groups`).
- Group draw + knockout bracket: FIFA final draw (Dec 5, 2025) and the official match schedule (matches 73–104), as published by [FIFA](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026) / [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage).

The 49 MB raw CSV is cached in `data/raw/` (gitignored); the small JSON artifacts in `predictions/` are committed so the deployed app never has to fit models.

## Project structure

```
worldcup26-prediction/
├── src/
│   ├── config.py               — every tunable parameter (backtest-informed)
│   ├── confederations.py       — confederation map for inter-pool shrink
│   ├── betting.py              — betting card + tie handling
│   ├── squads.py               — squad strength index
│   ├── data/fetch.py           — download, training set, fixtures, played-results lock-in
│   ├── models/dixon_coles.py   — vectorised, ridge-regularised DC
│   ├── models/elo.py           — international Elo
│   ├── models/engine.py        — DC × Elo × host × confed × squad, temperature
│   ├── player_valuation/       — ported squad valuation (formula-only path)
│   └── simulation/worldcup.py  — groups, thirds matching, bracket, Monte Carlo
├── scripts/
│   ├── run_simulation.py       — end-to-end pipeline
│   ├── backtest.py             — 2018/2022 WC calibration tuning
│   └── fetch_squads.py         — validate squads.json + coverage
├── dashboard/app.py            — Streamlit app (5 tabs; betting card in Match Predictor)
├── data/squads/squads.json     — projected XIs (16 teams seeded; extend freely)
├── predictions/                — committed JSON artifacts
├── tests/                      — 28 tests (model, bracket, 495 thirds combos, betting, squads, lock-in)
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
- **Squad layer is forward-looking and not historically backtestable** — `scripts/backtest.py` tunes the ridge/host/confed/temperature settings on 2018/2022, but the squad-strength index relies on *current* projected XIs (no historical squad data), so `SQUAD_WEIGHT` is a conservative judgement call (0.15). It only adjusts matches where **both** teams have squads in `data/squads/squads.json` (16 seeded; the rest fall back to pure DC+Elo), so today it mainly nudges big-team knockout matchups. Formula-only valuation rewards league tier/age, not live form — it can over-rate a deep top-league squad.
- **Penalty shootouts**: 50/50 ± a small Elo-based edge (capped at 58/42).
- **8 games can't validate a model** — the matchday-1 numbers above are a sanity check, not proof; the 2018/2022 backtest is the real evidence base.
- Probabilities are estimates with Monte Carlo noise of roughly ±0.5pp at 10,000 sims; run with `--sims 100000` for tighter estimates.
- **Betting cards are decision support, not financial advice** — derived from model probabilities only, with no bookmaker odds, so a flagged pick is not a guaranteed value bet.

## References

- Dixon, M.J. & Coles, S.G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Applied Statistics 46(2).
- World Football Elo Ratings: [eloratings.net](https://eloratings.net)

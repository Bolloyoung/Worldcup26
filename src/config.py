"""Central configuration for the World Cup 2026 prediction system."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Primary data source: community-maintained mirror of all official
# international 'A' match results since 1872 (updated after every matchday,
# includes the scheduled WC2026 fixtures with NA scores).
RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
RESULTS_CSV_PATH = RAW_DIR / "results.csv"

# Training window: two full World Cup cycles.
TRAIN_START = "2018-01-01"

# Minimum matches inside the window for a team to get its own DC parameters.
MIN_MATCHES_PER_TEAM = 10

# Competition importance weights (multiplied with the time-decay weight).
COMPETITION_WEIGHTS: dict[str, float] = {
    "FIFA World Cup": 1.6,
    "FIFA World Cup qualification": 1.2,
    "UEFA Euro": 1.4,
    "UEFA Euro qualification": 1.0,
    "UEFA Nations League": 1.0,
    "Copa América": 1.4,
    "African Cup of Nations": 1.3,
    "African Cup of Nations qualification": 0.9,
    "AFC Asian Cup": 1.3,
    "AFC Asian Cup qualification": 0.9,
    "Gold Cup": 1.2,
    "CONCACAF Nations League": 0.9,
    "Friendly": 0.6,
}
DEFAULT_COMPETITION_WEIGHT = 0.8

# Dixon-Coles time decay. xi = 0.0012 → half-weight after ~580 days, a bit
# slower than the club-football default (0.0018) because national teams play
# far fewer matches.
DC_XI = 0.0012
DC_MIN_WEIGHT = 0.01

# L2 (ridge) penalty on the log attack/defense parameters. Shrinks every team
# toward the league-average (log-param 0), which stops minnow-beating teams
# from collapsing their defense parameter toward zero and emitting absurdly
# overconfident scorelines (the matchday-1 Qatar-Switzerland 91% failure).
# 0 = unregularised (old behaviour); tuned by scripts/backtest.py. The penalty
# is scaled by total match weight, so this is a small per-match value.
# A small ridge (0.001) is the sweet spot: it removes the worst single-match
# overconfidence (Qatar-Switzerland 91% → ~69% with temperature) while keeping
# enough elite separation for a sensible title race. Larger values flatten the
# tournament (e.g. Japan rising to a top-4 favourite). Best backtest log-loss.
DC_RIDGE = 0.001

# Elo settings (international football, eloratings.net-style K factors).
ELO_K: dict[str, float] = {
    "FIFA World Cup": 60.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 40.0,
    "UEFA Nations League": 40.0,
    "CONCACAF Nations League": 30.0,
    "Friendly": 20.0,
}
ELO_K_DEFAULT = 30.0
ELO_HOME_ADV = 80.0
ELO_START_DATE = "2010-01-01"

# Probability engine: how strongly the Elo gap nudges the DC goal rates.
# lambda_home *= exp(+ELO_NUDGE * elo_diff/400), lambda_away *= exp(-...).
# Lowered from 0.18 → 0.12: the original value over-extrapolated cross-pool
# Elo gaps and compounded the Dixon-Coles overconfidence. Backtest-tuned.
ELO_NUDGE = 0.12

# Display/decision calibration. After deriving the score matrix, outcome
# probabilities are flattened by a temperature T >= 1 (p_i ** (1/T), then
# renormalised at the score-matrix level so scorelines stay consistent).
# 1.0 = no change. The 2018/2022 backtest strongly favours T > 1 — it is the
# single biggest fix for the matchday-1 overconfidence (Qatar-Switzerland 91%).
CALIB_TEMPERATURE = 1.20

# Explicit World Cup host boost (Elo points) for USA / Mexico / Canada, added
# to their effective rating in WC matches. The backtest only weakly rewards
# this (Qatar 2022 hosts lost all three games), so it is kept modest rather
# than at the grid's lowest edge — USA/Mexico are stronger hosts than Qatar.
HOST_ELO_BONUS = 55.0

# Inter-confederation reliability shrink. When two teams are from different
# confederations the rating gap rests on few bridging matches, so we shrink it
# toward parity by this factor before computing goal rates (1.0 = no shrink).
# Directly softens cross-pool extremes like Qatar (AFC) vs Switzerland (UEFA).
INTER_CONF_SHRINK = 0.90

# Weight of the squad-strength index when blended into the goal rates.
# 0 = ignore squads entirely (pure DC+Elo); tuned by backtest. This is the
# hook for current-squad reality (injuries/form): editing data/squads.json
# moves the forecast.
SQUAD_WEIGHT = 0.15

# Hosts get a (reduced) home advantage in knockout rounds: every venue is in
# a host country, but which venue a knockout match lands in is bracket
# dependent, so we discount the fitted group-stage advantage.
HOSTS = ("United States", "Mexico", "Canada")
KO_HOST_ADV_FACTOR = 0.5

# Monte Carlo
N_TOURNAMENTS = 10_000
SEED = 42
ET_SCALE = 0.33          # 30-min ET goal rate as a share of the 90-min rate
MAX_GOALS = 10

# Tournament dates (39-day tournament: 2026-06-11 → 2026-07-19 final).
GROUP_STAGE_END = "2026-06-27"

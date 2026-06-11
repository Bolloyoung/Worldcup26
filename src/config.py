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
ELO_NUDGE = 0.18

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

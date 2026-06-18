# Goal: predict a per-match box score (total shots, shots on target, saves) for any
# head-to-head. This is a DISPLAY/ANALYSIS layer only — it never feeds the goals GLMs
# or the bracket Monte Carlo.
#
# Two estimators, blended by sample size:
#   1. Fitted Poisson GLMs (stat ~ elo_diff + neutral) mirroring the goals models. Robust
#      because it pools every team through the Elo differential, but needs data to trust.
#   2. A recency-weighted attack/defense rate estimate per team (own shots-for blended with
#      the opponent's shots-against), each shrunk toward a league prior so tiny samples for
#      smaller nations don't blow up.
# Saves are derived last: opponent SOT x (1 - opponent finishing), finishing shrunk to a prior.

import json
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'data' / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'
STATS_FILE = DATA_DIR / 'match_stats.csv'
META_FILE = MODEL_DIR / 'stats_model_meta.json'

# --- League priors (per team, per international match). Used to seed everything before
#     live shot data accumulates, and as the shrinkage target for sparse teams. ---
LEAGUE_SHOTS = 12.0      # total shots a team takes in an average game
LEAGUE_SOT = 4.3         # shots on target
LEAGUE_FINISHING = 0.30  # goals per shot on target

# --- Shrinkage / blend knobs ---
K_TEAM = 5.0          # pseudo-matches pulling a team rate toward the league prior
K_FIN = 3.0           # pseudo-SOT pulling a team's finishing rate toward the prior
K_MODEL = 40.0        # global sample at which the GLM earns half the weight vs the rate model
HALFLIFE_DAYS = 730.0 # recency half-life (~2 years) for weighting a team's past games

# Columns expected in match_stats.csv (written by update_live_data.py)
STATS_COLUMNS = [
    'date', 'home_team', 'away_team', 'neutral',
    'home_elo_pre', 'away_elo_pre',
    'home_goals', 'away_goals',
    'home_shots', 'away_shots',
    'home_sot', 'away_sot',
    'home_saves', 'away_saves',
]


def load_stats_models(model_dir=MODEL_DIR):
    """Load the four fitted Poisson GLMs + their global sample size.

    Returns (models_dict_or_None, n_samples). Missing models => (None, 0), which makes
    predict_match_stats fall back to the pure rate/prior estimate."""
    names = ['home_shots', 'away_shots', 'home_sot', 'away_sot']
    models = {}
    for name in names:
        path = Path(model_dir) / f'{name}_poisson.pkl'
        if not path.exists():
            return None, 0
        models[name] = joblib.load(path)

    n = 0
    meta_path = Path(model_dir) / 'stats_model_meta.json'
    if meta_path.exists():
        try:
            n = int(json.loads(meta_path.read_text()).get('n', 0))
        except (ValueError, json.JSONDecodeError):
            n = 0
    return models, n


def load_match_stats(path=STATS_FILE):
    """Load the per-match shot store, or None if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    if df.empty:
        return None
    return df


def build_rate_tables(df, ref_date=None):
    """Collapse the match store into per-team recency-weighted sums.

    For every team we accumulate weighted shots/SOT for and against, goals for, and the
    total recency weight (its effective sample size). Shrinkage happens later at query time
    so the same table serves any matchup."""
    if df is None or df.empty:
        return {}

    if ref_date is None:
        # Anchor recency at the most recent data (or now, whichever is later) so future-dated
        # World Cup fixtures still get a sensible, non-negative age.
        ref_date = max(pd.Timestamp.now().normalize(), df['date'].max())

    tables = {}

    def _bucket(team):
        return tables.setdefault(team, {
            'w': 0.0,
            'shots_for': 0.0, 'shots_against': 0.0,
            'sot_for': 0.0, 'sot_against': 0.0,
            'goals_for': 0.0,
        })

    for row in df.itertuples():
        age_days = max((ref_date - row.date).days, 0)
        weight = 0.5 ** (age_days / HALFLIFE_DAYS)

        # Home perspective
        h = _bucket(row.home_team)
        h['w'] += weight
        h['shots_for'] += weight * _num(row.home_shots)
        h['shots_against'] += weight * _num(row.away_shots)
        h['sot_for'] += weight * _num(row.home_sot)
        h['sot_against'] += weight * _num(row.away_sot)
        h['goals_for'] += weight * _num(row.home_goals)

        # Away perspective
        a = _bucket(row.away_team)
        a['w'] += weight
        a['shots_for'] += weight * _num(row.away_shots)
        a['shots_against'] += weight * _num(row.home_shots)
        a['sot_for'] += weight * _num(row.away_sot)
        a['sot_against'] += weight * _num(row.home_sot)
        a['goals_for'] += weight * _num(row.away_goals)

    return tables


def _num(v):
    """Coerce a possibly-missing stat to a float (NaN/None -> 0)."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _shrunk(weighted_sum, weight, prior, k):
    """Shrink a recency-weighted mean toward a prior by effective sample size."""
    return (weighted_sum + k * prior) / (weight + k)


def _team_rates(tables, team):
    """Return a team's shrunk attack/defense/finishing rates (league prior if unseen)."""
    t = tables.get(team)
    if t is None or t['w'] <= 0:
        return {
            'attack_shots': LEAGUE_SHOTS, 'defense_shots': LEAGUE_SHOTS,
            'attack_sot': LEAGUE_SOT, 'defense_sot': LEAGUE_SOT,
            'finishing': LEAGUE_FINISHING, 'eff_n': 0.0,
        }
    w = t['w']
    return {
        'attack_shots': _shrunk(t['shots_for'], w, LEAGUE_SHOTS, K_TEAM),
        'defense_shots': _shrunk(t['shots_against'], w, LEAGUE_SHOTS, K_TEAM),
        'attack_sot': _shrunk(t['sot_for'], w, LEAGUE_SOT, K_TEAM),
        'defense_sot': _shrunk(t['sot_against'], w, LEAGUE_SOT, K_TEAM),
        # Finishing shrunk in SOT units: goals per shot on target, pulled toward the prior.
        'finishing': min(max(
            (t['goals_for'] + K_FIN * LEAGUE_FINISHING) / (t['sot_for'] + K_FIN),
            0.05), 0.60),
        'eff_n': w,
    }


def _band(mu, z=1.0):
    """Rough +/- range for a count mean (one Poisson standard deviation)."""
    spread = z * math.sqrt(max(mu, 0.0))
    return max(mu - spread, 0.0), mu + spread


def predict_match_stats(team1, team2, elo_diff, neutral_val,
                        models=None, n_model=0, rate_tables=None):
    """Predict the box score for team1 vs team2.

    elo_diff / neutral_val follow the exact goals-model convention: elo_diff = elo1 - elo2,
    team1 uses the home_* GLMs with +elo_diff, team2 the away_* GLMs with -elo_diff.

    Returns a dict keyed by 'team1'/'team2', each with shots/sot/saves and (lo, hi) ranges,
    plus 'model_weight' (how much the fitted GLM contributed)."""
    rate_tables = rate_tables or {}
    r1 = _team_rates(rate_tables, team1)
    r2 = _team_rates(rate_tables, team2)

    # --- Rate estimate: blend a team's attack with the opponent's defense ---
    rate_shots_1 = 0.5 * (r1['attack_shots'] + r2['defense_shots'])
    rate_shots_2 = 0.5 * (r2['attack_shots'] + r1['defense_shots'])
    rate_sot_1 = 0.5 * (r1['attack_sot'] + r2['defense_sot'])
    rate_sot_2 = 0.5 * (r2['attack_sot'] + r1['defense_sot'])

    # --- GLM estimate: same elo_diff signal as the goals models ---
    if models is not None:
        glm_shots_1 = float(models['home_shots'].predict([1, elo_diff, neutral_val])[0])
        glm_shots_2 = float(models['away_shots'].predict([1, -elo_diff, neutral_val])[0])
        glm_sot_1 = float(models['home_sot'].predict([1, elo_diff, neutral_val])[0])
        glm_sot_2 = float(models['away_sot'].predict([1, -elo_diff, neutral_val])[0])
        w_glm = n_model / (n_model + K_MODEL)
    else:
        glm_shots_1 = glm_shots_2 = glm_sot_1 = glm_sot_2 = 0.0
        w_glm = 0.0

    # --- Blend: weight the fitted model up as the global sample grows ---
    shots_1 = w_glm * glm_shots_1 + (1 - w_glm) * rate_shots_1
    shots_2 = w_glm * glm_shots_2 + (1 - w_glm) * rate_shots_2
    sot_1 = w_glm * glm_sot_1 + (1 - w_glm) * rate_sot_1
    sot_2 = w_glm * glm_sot_2 + (1 - w_glm) * rate_sot_2

    # SOT can never exceed total shots
    sot_1 = min(sot_1, shots_1)
    sot_2 = min(sot_2, shots_2)

    # --- Saves derived last: keeper faces the opponent's on-target shots that don't score ---
    saves_1 = sot_2 * (1 - r2['finishing'])
    saves_2 = sot_1 * (1 - r1['finishing'])

    return {
        'team1': _pack(team1, shots_1, sot_1, saves_1),
        'team2': _pack(team2, shots_2, sot_2, saves_2),
        'model_weight': w_glm,
    }


def _pack(team, shots, sot, saves):
    s_lo, s_hi = _band(shots)
    t_lo, t_hi = _band(sot)
    v_lo, v_hi = _band(saves)
    return {
        'team': team,
        'shots': shots, 'shots_range': (s_lo, s_hi),
        'sot': sot, 'sot_range': (t_lo, t_hi),
        'saves': saves, 'saves_range': (v_lo, v_hi),
    }


if __name__ == '__main__':
    # Smoke test against whatever data exists (prints prior-based numbers if the store is empty).
    models, n = load_stats_models()
    df = load_match_stats()
    tables = build_rate_tables(df)
    out = predict_match_stats('Spain', 'Argentina', elo_diff=40.0, neutral_val=1,
                              models=models, n_model=n, rate_tables=tables)
    print(f"Stats GLM samples: {n} | model weight: {out['model_weight']:.2f}")
    for key in ('team1', 'team2'):
        p = out[key]
        print(f"{p['team']:>12}: shots {p['shots']:.1f}  sot {p['sot']:.1f}  saves {p['saves']:.1f}")

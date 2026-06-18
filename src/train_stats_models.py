# Goal: translate the Elo differential into expected box-score volume the same way
# train_model.py turns it into expected goals. Four Poisson GLMs are fitted —
# home/away total shots and home/away shots on target — each on (elo_diff + neutral).
#
# This is a symmetric sibling of the goals trainer. It is DISPLAY-only: it never touches
# the goals models or the bracket simulation. Shot data is sparse and recent, so this may
# train on very few rows (or none) — predict_match_stats.py blends the result against a
# recency-weighted rate model and league priors accordingly.

import json
import joblib
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'data' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATS_FILE = DATA_DIR / 'match_stats.csv'
META_FILE = MODEL_DIR / 'stats_model_meta.json'

# Need a minimum sample before a fitted GLM is meaningful; below this we leave the predictor
# on its rate/prior fallback rather than persisting a noisy model.
MIN_ROWS = 12

# Targets -> (feature column, target column), mirroring the goals home/away convention.
TARGETS = {
    'home_shots': ('elo_diff', 'home_shots'),
    'away_shots': ('neg_elo_diff', 'away_shots'),
    'home_sot': ('elo_diff', 'home_sot'),
    'away_sot': ('neg_elo_diff', 'away_sot'),
}


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])

    # Need both the Elo signal and the stat itself present on every training row.
    df = df.dropna(subset=['home_elo_pre', 'away_elo_pre', 'neutral',
                           'home_shots', 'away_shots', 'home_sot', 'away_sot'])

    df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']
    df['neg_elo_diff'] = -df['elo_diff']
    df['neutral_int'] = df['neutral'].astype(int)
    df['const'] = 1.0

    print(f"Shot data prepared. Matches with full stats: {len(df)}")
    return df


def train_stat_models(df):
    models = {}
    for name, (feature, target) in TARGETS.items():
        X = df[['const', feature, 'neutral_int']]
        y = df[target]
        models[name] = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return models


def persist_models(models, n, output_dir=MODEL_DIR):
    for name, model in models.items():
        joblib.dump(model, output_dir / f'{name}_poisson.pkl')
    META_FILE.write_text(json.dumps({'n': int(n)}))
    print(f"Stats models saved to {output_dir} (n={n}).")


def main():
    if not STATS_FILE.exists():
        print(f"No shot data at {STATS_FILE}; nothing to train. "
              "Predictor will use the rate/prior fallback.")
        return False

    df = load_and_prepare_data(STATS_FILE)
    if len(df) < MIN_ROWS:
        print(f"Only {len(df)} rows with shot stats (< {MIN_ROWS}); skipping GLM fit. "
              "Predictor will use the rate/prior fallback.")
        return False

    print("Training Poisson shot/SOT GLMs...")
    models = train_stat_models(df)
    persist_models(models, len(df))
    return True


if __name__ == '__main__':
    print("Starting Shot-Stats Model Training Pipeline...")
    main()

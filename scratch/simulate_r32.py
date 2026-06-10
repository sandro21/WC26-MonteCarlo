import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import random

BASE_DIR = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo")
MODEL_DIR = BASE_DIR / 'data' / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load the models
home_model = joblib.load(MODEL_DIR / 'home_poisson.pkl')
away_model = joblib.load(MODEL_DIR / 'away_poisson.pkl')

HOSTS = ['United States', 'Mexico', 'Canada']

# Get current Elos
def get_current_elos(df):
    latest_elos = {}
    latest_dates = {}
    for row in df.itertuples():
        home = row.home_team
        date = row.date
        if home not in latest_dates or date > latest_dates[home]:
            latest_dates[home] = date
            latest_elos[home] = row.home_elo_pre
        away = row.away_team
        if away not in latest_dates or date > latest_dates[away]:
            latest_dates[away] = date
            latest_elos[away] = row.away_elo_pre
    return latest_elos

df = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])
current_elos = get_current_elos(df)

mvi_df = pd.read_csv(DATA_DIR / 'squad_features.csv')
mvi_data = dict(zip(mvi_df['team_name'], mvi_df['market_value_index']))

# Name normalization helper
def get_mvi_name(name):
    if name == 'Turkey':
        return 'Türkiye'
    if name == 'South Korea':
        return 'Korea Republic'
    return name

def simulate_match(team1, team2, is_knockout=True):
    swapped = False
    if team2 in HOSTS and team1 not in HOSTS:
        team1, team2 = team2, team1
        swapped = True
    
    elo1_base = current_elos.get(team1, 1500.0)
    elo2_base = current_elos.get(team2, 1500.0)
    
    team1_mvi = max(float(mvi_data.get(get_mvi_name(team1), 1.0)), 0.01)
    team2_mvi = max(float(mvi_data.get(get_mvi_name(team2), 1.0)), 0.01)
    
    elo1 = elo1_base + (50 * np.log(team1_mvi))
    elo2 = elo2_base + (50 * np.log(team2_mvi))
    
    elo_diff = elo1 - elo2
    
    neutral_val = 0 if (team1 in HOSTS and team2 not in HOSTS) else 1
    
    lambda_1 = home_model.predict([1, elo_diff, neutral_val])[0]
    lambda_2 = away_model.predict([1, -elo_diff, neutral_val])[0]
    
    score1 = np.random.poisson(lambda_1)
    score2 = np.random.poisson(lambda_2)
    
    if is_knockout and score1 == score2:
        prob1 = 0.55 if elo1 > elo2 else 0.45
        if np.random.rand() < prob1:
            score1 += 1
        else:
            score2 += 1
            
    if swapped:
        return score2, score1
    return score1, score2

# Matches list: (Abbrev1, Abbrev2, Name1, Name2)
matches = [
    ('GER', 'PAR', 'Germany', 'Paraguay'),
    ('FRA', 'JPN', 'France', 'Japan'),
    ('CZE', 'CAN', 'Czechia', 'Canada'),
    ('NED', 'MAR', 'Netherlands', 'Morocco'),
    ('COL', 'CRO', 'Colombia', 'Croatia'),
    ('ESP', 'AUT', 'Spain', 'Austria'),
    ('TUR', 'ECU', 'Turkey', 'Ecuador'),
    ('BEL', 'KOR', 'Belgium', 'South Korea'),
    ('BRA', 'SWE', 'Brazil', 'Sweden'),
    ('CIV', 'NOR', 'Ivory Coast', 'Norway'),
    ('MEX', 'SCO', 'Mexico', 'Scotland'),
    ('ENG', 'COD', 'England', 'DR Congo'),
    ('ARG', 'URU', 'Argentina', 'Uruguay'),
    ('USA', 'EGY', 'United States', 'Egypt'),
    ('SUI', 'ALG', 'Switzerland', 'Algeria'),
    ('POR', 'SEN', 'Portugal', 'Senegal')
]

print("Simulating Round of 32 (10,000 runs per matchup)...")
np.random.seed(42)
random.seed(42)

for ab1, ab2, t1, t2 in matches:
    t1_wins = 0
    t2_wins = 0
    runs = 10000
    for _ in range(runs):
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        if s1 > s2:
            t1_wins += 1
        else:
            t2_wins += 1
    
    t1_pct = (t1_wins / runs) * 100
    t2_pct = (t2_wins / runs) * 100
    winner = ab1 if t1_wins > t2_wins else ab2
    
    # Get details for checking
    e1_base = current_elos.get(t1, 1500.0)
    e2_base = current_elos.get(t2, 1500.0)
    mvi1 = mvi_data.get(get_mvi_name(t1), 1.0)
    mvi2 = mvi_data.get(get_mvi_name(t2), 1.0)
    adj_e1 = e1_base + (50 * np.log(max(mvi1, 0.01)))
    adj_e2 = e2_base + (50 * np.log(max(mvi2, 0.01)))
    
    print(f"{ab1} vs {ab2} ({t1} vs {t2}):")
    print(f"  {ab1}: Base Elo={e1_base:.1f}, MVI={mvi1:.3f}, Adj Elo={adj_e1:.1f} | Win Prob={t1_pct:.1f}%")
    print(f"  {ab2}: Base Elo={e2_base:.1f}, MVI={mvi2:.3f}, Adj Elo={adj_e2:.1f} | Win Prob={t2_pct:.1f}%")
    print(f"  Expected Winner: {winner}\n")

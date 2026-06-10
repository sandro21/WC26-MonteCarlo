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

def get_probability_winner(t1, t2):
    t1_wins = 0
    runs = 10000
    for _ in range(runs):
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        if s1 > s2:
            t1_wins += 1
    p1 = (t1_wins / runs)
    p2 = 1.0 - p1
    return (t1, p1) if p1 >= 0.5 else (t2, p2)

# Abbrev helper
names_to_abbrev = {
    'Germany': 'GER', 'Paraguay': 'PAR', 'France': 'FRA', 'Japan': 'JPN',
    'Czechia': 'CZE', 'Canada': 'CAN', 'Netherlands': 'NED', 'Morocco': 'MAR',
    'Colombia': 'COL', 'Croatia': 'CRO', 'Spain': 'ESP', 'Austria': 'AUT',
    'Turkey': 'TUR', 'Ecuador': 'ECU', 'Belgium': 'BEL', 'South Korea': 'KOR',
    'Brazil': 'BRA', 'Sweden': 'SWE', 'Ivory Coast': 'CIV', 'Norway': 'NOR',
    'Mexico': 'MEX', 'Scotland': 'SCO', 'England': 'ENG', 'DR Congo': 'COD',
    'Argentina': 'ARG', 'Uruguay': 'URU', 'United States': 'USA', 'Egypt': 'EGY',
    'Switzerland': 'SUI', 'Algeria': 'ALG', 'Portugal': 'POR', 'Senegal': 'SEN'
}

# 1. R32 Matches
r32_matches = [
    ('Germany', 'Paraguay'),
    ('France', 'Japan'),
    ('Czechia', 'Canada'),
    ('Netherlands', 'Morocco'),
    ('Colombia', 'Croatia'),
    ('Spain', 'Austria'),
    ('Turkey', 'Ecuador'),
    ('Belgium', 'South Korea'),
    ('Brazil', 'Sweden'),
    ('Ivory Coast', 'Norway'),
    ('Mexico', 'Scotland'),
    ('England', 'DR Congo'),
    ('Argentina', 'Uruguay'),
    ('United States', 'Egypt'),
    ('Switzerland', 'Algeria'),
    ('Portugal', 'Senegal')
]

print("--- ROUND OF 32 EXPECTED WINNERS ---")
r32_winners = []
for t1, t2 in r32_matches:
    w, p = get_probability_winner(t1, t2)
    r32_winners.append(w)
    print(f"{names_to_abbrev[t1]} vs {names_to_abbrev[t2]} -> {names_to_abbrev[w]} ({p*100:.1f}%)")

# 2. R16 Matches
r16_pairs = [
    (r32_winners[0], r32_winners[1]), # GER vs FRA
    (r32_winners[2], r32_winners[3]), # CAN vs NED
    (r32_winners[4], r32_winners[5]), # COL vs ESP
    (r32_winners[6], r32_winners[7]), # TUR vs BEL
    (r32_winners[8], r32_winners[9]), # BRA vs NOR
    (r32_winners[10], r32_winners[11]), # MEX vs ENG
    (r32_winners[12], r32_winners[13]), # ARG vs USA
    (r32_winners[14], r32_winners[15])  # SUI vs POR
]

print("\n--- ROUND OF 16 EXPECTED WINNERS ---")
r16_winners = []
for t1, t2 in r16_pairs:
    w, p = get_probability_winner(t1, t2)
    r16_winners.append(w)
    print(f"{names_to_abbrev[t1]} vs {names_to_abbrev[t2]} -> {names_to_abbrev[w]} ({p*100:.1f}%)")

# 3. QF Matches
qf_pairs = [
    (r16_winners[0], r16_winners[1]),
    (r16_winners[2], r16_winners[3]),
    (r16_winners[4], r16_winners[5]),
    (r16_winners[6], r16_winners[7])
]

print("\n--- QUARTERFINALS EXPECTED WINNERS ---")
qf_winners = []
for t1, t2 in qf_pairs:
    w, p = get_probability_winner(t1, t2)
    qf_winners.append(w)
    print(f"{names_to_abbrev[t1]} vs {names_to_abbrev[t2]} -> {names_to_abbrev[w]} ({p*100:.1f}%)")

# 4. SF Matches
sf_pairs = [
    (qf_winners[0], qf_winners[1]),
    (qf_winners[2], qf_winners[3])
]

print("\n--- SEMIFINALS EXPECTED WINNERS ---")
sf_winners = []
for t1, t2 in sf_pairs:
    w, p = get_probability_winner(t1, t2)
    sf_winners.append(w)
    print(f"{names_to_abbrev[t1]} vs {names_to_abbrev[t2]} -> {names_to_abbrev[w]} ({p*100:.1f}%)")

# 5. Final
w, p = get_probability_winner(sf_winners[0], sf_winners[1])
print("\n--- FINAL EXPECTED WINNER ---")
print(f"{names_to_abbrev[sf_winners[0]]} vs {names_to_abbrev[sf_winners[1]]} -> {names_to_abbrev[w]} ({p*100:.1f}%)")

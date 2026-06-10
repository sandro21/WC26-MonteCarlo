import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import poisson

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
def get_names(team):
    mappings = {
        'MEX': ('Mexico', 'Mexico'),
        'RSA': ('South Africa', 'South Africa'),
        'KOR': ('South Korea', 'Korea Republic'),
        'CZE': ('Czechia', 'Czechia'),
        'CAN': ('Canada', 'Canada'),
        'BIH': ('Bosnia and Herzegovina', 'Bosnia and Herzegovina'),
        'USA': ('United States', 'United States'),
        'PAR': ('Paraguay', 'Paraguay'),
        'QAT': ('Qatar', 'Qatar'),
        'SUI': ('Switzerland', 'Switzerland'),
        'BRA': ('Brazil', 'Brazil'),
        'MAR': ('Morocco', 'Morocco'),
        'HAI': ('Haiti', 'Haiti'),
        'SCO': ('Scotland', 'Scotland'),
        'AUS': ('Australia', 'Australia'),
        'TUR': ('Turkey', 'Türkiye'),
        'GER': ('Germany', 'Germany'),
        'CUW': ('Curacao', 'Curaçao'),
        'NED': ('Netherlands', 'Netherlands'),
        'JPN': ('Japan', 'Japan'),
        'CIV': ('Ivory Coast', 'Ivory Coast'),
        'ECU': ('Ecuador', 'Ecuador'),
        'SWE': ('Sweden', 'Sweden'),
        'TUN': ('Tunisia', 'Tunisia'),
        'ESP': ('Spain', 'Spain'),
        'CPV': ('Cape Verde', 'Cape Verde'),
        'BEL': ('Belgium', 'Belgium'),
        'EGY': ('Egypt', 'Egypt'),
        'KSA': ('Saudi Arabia', 'Saudi Arabia'),
        'URU': ('Uruguay', 'Uruguay'),
        'IRN': ('IR Iran', 'Iran'), # check both spelling variations
        'NZL': ('New Zealand', 'New Zealand'),
        'FRA': ('France', 'France'),
        'SEN': ('Senegal', 'Senegal'),
        'IRQ': ('Iraq', 'Iraq'),
        'NOR': ('Norway', 'Norway'),
        'ARG': ('Argentina', 'Argentina'),
        'ALG': ('Algeria', 'Algeria'),
        'AUT': ('Austria', 'Austria'),
        'JOR': ('Jordan', 'Jordan'),
        'POR': ('Portugal', 'Portugal'),
        'COD': ('DR Congo', 'DR Congo'),
        'ENG': ('England', 'England'),
        'CRO': ('Croatia', 'Croatia'),
        'GHA': ('Ghana', 'Ghana'),
        'PAN': ('Panama', 'Panama'),
        'UZB': ('Uzbekistan', 'Uzbekistan'),
        'COL': ('Colombia', 'Colombia')
    }
    return mappings.get(team, (team, team))

# Try resolving different names for Iran
# If 'Iran' is in current_elos or 'IR Iran'
print("IR Iran in Elo:", 'IR Iran' in current_elos, "Iran in Elo:", 'Iran' in current_elos)
print("Curacao in Elo:", 'Curacao' in current_elos, "Curaçao in Elo:", 'Curaçao' in current_elos)

def predict_score(team1_ab, team2_ab):
    t1_elo_name, t1_mvi_name = get_names(team1_ab)
    t2_elo_name, t2_mvi_name = get_names(team2_ab)
    
    # Check fallback names
    if t1_elo_name == 'IR Iran' and t1_elo_name not in current_elos:
        t1_elo_name = 'Iran'
    if t2_elo_name == 'IR Iran' and t2_elo_name not in current_elos:
        t2_elo_name = 'Iran'
        
    swapped = False
    # Host adjustment matches simulate_match
    if t2_elo_name in HOSTS and t1_elo_name not in HOSTS:
        t1_elo_name, t2_elo_name = t2_elo_name, t1_elo_name
        t1_mvi_name, t2_mvi_name = t2_mvi_name, t1_mvi_name
        swapped = True
        
    elo1_base = current_elos.get(t1_elo_name, 1500.0)
    elo2_base = current_elos.get(t2_elo_name, 1500.0)
    
    team1_mvi = max(float(mvi_data.get(t1_mvi_name, 1.0)), 0.01)
    team2_mvi = max(float(mvi_data.get(t2_mvi_name, 1.0)), 0.01)
    
    elo1 = elo1_base + (50 * np.log(team1_mvi))
    elo2 = elo2_base + (50 * np.log(team2_mvi))
    
    elo_diff = elo1 - elo2
    
    neutral_val = 0 if (t1_elo_name in HOSTS and t2_elo_name not in HOSTS) else 1
    
    lambda_1 = home_model.predict([1, elo_diff, neutral_val])[0]
    lambda_2 = away_model.predict([1, -elo_diff, neutral_val])[0]
    
    # Find the most likely scoreline (joint probability of Poisson)
    best_prob = -1
    best_score = (0, 0)
    for g1 in range(8):
        for g2 in range(8):
            prob = poisson.pmf(g1, lambda_1) * poisson.pmf(g2, lambda_2)
            if prob > best_prob:
                best_prob = prob
                best_score = (g2, g1) if swapped else (g1, g2)
                
    # Also print expected value for context
    expected_score = f"{lambda_2:.2f} - {lambda_1:.2f}" if swapped else f"{lambda_1:.2f} - {lambda_2:.2f}"
    
    return best_score, expected_score

round1_matches = [
    ('MEX', 'RSA'), ('KOR', 'CZE'), ('CAN', 'BIH'), ('USA', 'PAR'), ('QAT', 'SUI'),
    ('BRA', 'MAR'), ('HAI', 'SCO'), ('AUS', 'TUR'), ('GER', 'CUW'), ('NED', 'JPN'),
    ('CIV', 'ECU'), ('SWE', 'TUN'), ('ESP', 'CPV'), ('BEL', 'EGY'), ('KSA', 'URU'),
    ('IRN', 'NZL'), ('FRA', 'SEN'), ('IRQ', 'NOR'), ('ARG', 'ALG'), ('AUT', 'JOR'),
    ('POR', 'COD'), ('ENG', 'CRO'), ('GHA', 'PAN'), ('UZB', 'COL')
]

out_path = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo/scratch/round1_predictions.txt")
with open(out_path, "w", encoding="utf-8") as out:
    for t1, t2 in round1_matches:
        best, exp = predict_score(t1, t2)
        out.write(f"{t1} vs {t2} -> {best[0]} - {best[1]} (Expected: {exp})\n")
print("Predictions written to scratch/round1_predictions.txt")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

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
# Standardize names to match what's in Elo results and GROUPS
# Let's print unique teams in elo_results and squad_features to find any discrepancies
# print("Elo teams:", sorted(list(current_elos.keys())))
# print("Squad features teams:", sorted(list(mvi_df['team_name'].unique())))

# Create MVI mapping
mvi_data = dict(zip(mvi_df['team_name'], mvi_df['market_value_index']))

# We'll map the abbreviations to full names used in the database
abbrev_map = {
    'GER': 'Germany',
    'PAR': 'Paraguay',
    'FRA': 'France',
    'JPN': 'Japan',
    'CZE': 'Czechia',
    'CAN': 'Canada',
    'NED': 'Netherlands',
    'MAR': 'Morocco',
    'COL': 'Colombia',
    'CRO': 'Croatia',
    'ESP': 'Spain',
    'AUT': 'Austria',
    'TUR': 'Turkey', # Check if Turkey or Türkiye
    'ECU': 'Ecuador',
    'BEL': 'Belgium',
    'KOR': 'South Korea', # Check if Korea Republic or South Korea
    'BRA': 'Brazil',
    'SWE': 'Sweden',
    'CIV': 'Ivory Coast',
    'NOR': 'Norway',
    'MEX': 'Mexico',
    'SCO': 'Scotland',
    'ENG': 'England',
    'COD': 'DR Congo',
    'ARG': 'Argentina',
    'URU': 'Uruguay',
    'USA': 'United States',
    'EGY': 'Egypt',
    'SUI': 'Switzerland',
    'ALG': 'Algeria',
    'POR': 'Portugal',
    'SEN': 'Senegal'
}

# Resolve names specifically for Turkey and South Korea if spelling differs
# Let's inspect Elos and MVI keys
for k in ['Turkey', 'Türkiye', 'South Korea', 'Korea Republic']:
    print(f"Elo has '{k}': {k in current_elos}, MVI has '{k}': {k in mvi_data}")

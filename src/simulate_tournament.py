import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict #like a python dictionary  but more features

# Goal: Simulate the 2026 World Cup 10,000 times using Monte Carlo simulation and trained Poisson models.

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'data' / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load the models we built from the saved folder, ran in prev step. 
home_model = joblib.load(MODEL_DIR / 'home_poisson.pkl')
away_model = joblib.load(MODEL_DIR / 'away_poisson.pkl')

HOSTS = ['United States', 'Mexico', 'Canada']

GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}

def get_current_elos(df): # function to extract elos of each team based on teh groups. 
    latest_elos = {}
    for team in df['home_team'].unique():
        last_home = df[df['home_team'] == team].tail(1) # finds the last time they played
        if not last_home.empty:
            latest_elos[team] = last_home['home_elo_post'].values[0]
            
    for team in df['away_team'].unique():
        last_away = df[df['away_team'] == team].tail(1)
        if not last_away.empty:
            if team not in latest_elos or last_away['date'].values[0] > last_home['date'].values[0]:
                latest_elos[team] = last_away['away_elo_post'].values[0]
                
    return latest_elos

df = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])
current_elos = get_current_elos(df)

# Step 2: Core Match Simulation (simulate_match)
def simulate_match(team1, team2, is_knockout=False):
    swapped = False
    if team2 in HOSTS and team1 not in HOSTS:
        team1, team2 = team2, team1
        swapped = True
        
    elo1 = current_elos.get(team1, 1500.0) #sets default o 1500, shouldnt happen tho. 
    elo2 = current_elos.get(team2, 1500.0)
    
    elo_diff = elo1 - elo2
    
    neutral_val = 0 if (team1 in HOSTS and team2 not in HOSTS) else 1
    
    lambda_1 = home_model.predict([1, elo_diff, neutral_val])[0] #using our created models to predict based on elo diff
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

# Step 3: Group Stage Logic (simulate_group_stage)

# Step 4: Knockout Stage Logic (simulate_knockout_stage)

# Step 5: The Monte Carlo Loop (main execution)

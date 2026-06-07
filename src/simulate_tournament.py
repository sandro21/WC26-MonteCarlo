import pandas as pd
import numpy as np
import joblib
import itertools
import random
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

def get_current_elos(df): # function to extract elos of each team based on the groups. 
    latest_elos = {}
    latest_dates = {}
    for row in df.itertuples():
        # Home team Elo
        home = row.home_team
        date = row.date
        if home not in latest_dates or date > latest_dates[home]:
            latest_dates[home] = date
            latest_elos[home] = row.home_elo_pre
            
        # Away team Elo
        away = row.away_team
        if away not in latest_dates or date > latest_dates[away]:
            latest_dates[away] = date
            latest_elos[away] = row.away_elo_pre
            
    return latest_elos

df = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])
current_elos = get_current_elos(df)

try:
    mvi_df = pd.read_csv(DATA_DIR / 'squad_features.csv')
    mvi_data = dict(zip(mvi_df['team_name'], mvi_df['market_value_index']))
except FileNotFoundError:
    print("Warning: squad_features.csv not found. All MVIs will default to 1.0")
    mvi_data = {}

def simulate_match(team1, team2, is_knockout=False):
    swapped = False
    if team2 in HOSTS and team1 not in HOSTS:
        team1, team2 = team2, team1
        swapped = True
    
    elo1_base = current_elos.get(team1, 1500.0) #sets default o 1500, shouldnt happen tho. 
    elo2_base = current_elos.get(team2, 1500.0)
    

    # this is the calcaultions for the elo
    #defualt is 1.0. max is 0.01
    team1_mvi = max(float(mvi_data.get(team1, 1.0)), 0.01)
    team2_mvi = max(float(mvi_data.get(team2, 1.0)), 0.01)
    
    #adjust elo based on market value
    elo1 = elo1_base + (50 * np.log(team1_mvi))
    elo2 = elo2_base + (50 * np.log(team2_mvi))
    
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
def simulate_group_stage(groups):
    advancing_teams = []
    third_place_teams = []
    
    for group_name, teams in groups.items():
        stats = {}
        for team in teams:
            stats[team] = {
                'Pts': 0,  # Points
                'GD': 0,   # Goal Difference
                'GF': 0    # Goals For
            }


        for team1, team2 in itertools.combinations(teams, 2): #python library used for simulating a round robin 
            score1, score2 = simulate_match(team1, team2, is_knockout=False) # use simulate match function to simulate each match
            
            stats[team1]['GF'] += score1
            stats[team2]['GF'] += score2
            stats[team1]['GD'] += (score1 - score2)
            stats[team2]['GD'] += (score2 - score1)
            
            if score1 > score2: #checking scores to add points respectively. 
                stats[team1]['Pts'] += 3
            elif score2 > score1:
                stats[team2]['Pts'] += 3
            else:
                stats[team1]['Pts'] += 1
                stats[team2]['Pts'] += 1
                
        sorted_teams = sorted(teams, key=lambda t: (stats[t]['Pts'], stats[t]['GD'], stats[t]['GF']), reverse=True)
        
        advancing_teams.extend(sorted_teams[:2])
        third_place_teams.append({
            'team': sorted_teams[2],
            'Pts': stats[sorted_teams[2]]['Pts'],
            'GD': stats[sorted_teams[2]]['GD'],
            'GF': stats[sorted_teams[2]]['GF']
        })
        
    third_place_teams.sort(key=lambda x: (x['Pts'], x['GD'], x['GF']), reverse=True)
    advancing_teams.extend([x['team'] for x in third_place_teams[:8]])
    
    return advancing_teams

# Step 4: Knockout Stage Logic (simulate_knockout_stage)
def simulate_knockout_stage(teams):
    random.shuffle(teams)
    
    current_round = teams
    while len(current_round) > 1:
        next_round = []
        for i in range(0, len(current_round), 2):
            team1 = current_round[i]
            team2 = current_round[i+1]
            score1, score2 = simulate_match(team1, team2, is_knockout=True)
            if score1 > score2:
                next_round.append(team1)
            else:
                next_round.append(team2)
        current_round = next_round
        
    return current_round[0] # Returns the tournament winner

# Step 5: The Monte Carlo Loop (main execution)
def run_monte_carlo(N=10000):
    wins = defaultdict(int)
    for i in range(N):
        if (i + 1) % 1000 == 0:
            print(f"Simulating tournament {i+1}/{N}...")
            
        advancing = simulate_group_stage(GROUPS)
        winner = simulate_knockout_stage(advancing)
        wins[winner] += 1
        
    # Calculate win probabilities
    results = []
    for team, count in wins.items():
        results.append((team, count / N * 100))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n=== TOP 10 TEAMS TO WIN THE 2026 WORLD CUP ===")
    for idx, (team, prob) in enumerate(results[:10]):
        print(f"{idx+1}. {team}: {prob:.2f}%")
        
    return results

if __name__ == '__main__':
    results = run_monte_carlo(10000)
    
    # Save results to data/processed/simulation_results.csv
    df_results = pd.DataFrame([
        {
            'Team': team,
            'Win_Probability': prob,
            'Elo': current_elos.get(team, 1500.0)
        } for team, prob in results
    ])
    df_results.to_csv(DATA_DIR / 'simulation_results.csv', index=False)
    print(f"Results successfully saved to {DATA_DIR / 'simulation_results.csv'}")


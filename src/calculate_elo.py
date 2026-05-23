import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'

# 2. Dynamic K-Factor Routing Function
#    - Classify tournaments into 5 Tiers using substring matching (T1=60, T2=40, T3=30, T4=20, T5=10).
# K factor is the importance weight of a specific math, allows us to change ELO based on K value. 

def calculate_k_factor(tournament_name):
    if "World Cup" in tournament_name and "Qualifier" not in tournament_name and "Qualification" not in tournament_name:
        return 60
    elif any(cup in tournament_name for cup in ["Copa América", "Euro", "African Cup of Nations", "Gold Cup", "Asian Cup"]):
        return 40
    elif "Qualifier" in tournament_name or "Qualification" in tournament_name:
        return 30
    elif "Friendly" in tournament_name:
        return 10
    else:
        return 20


# 3. Expected Score (Probability) Calculator
#    - Calculate EH and EA using the Elo formula. (expected home and expect away scores)
#    - Apply the +100 Home-Field Advantage boost to the Home team if neutral == False.
def calculate_expected_scores(home_elo, away_elo, neutral):
    if not neutral:
        home_elo = home_elo + 100
        
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home
    
    return expected_home, expected_away


# 4. Elo Rating Update Function
#    - Take old ratings, actual outcomes (1.0, 0.5, 0.0), and expected scores to return the new ratings.
def update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor):
    if outcome == 'Home Win':
        score_home = 1.0
        score_away = 0.0
    elif outcome == 'Away Win':
        score_home = 0.0
        score_away = 1.0
    else:
        score_home = 0.5
        score_away = 0.5
        
    new_home_elo = home_elo + k_factor * (score_home - expected_home)
    new_away_elo = away_elo + k_factor * (score_away - expected_away)
    
    return new_home_elo, new_away_elo

# 5. Main Elo Engine Loop
#    - Initialize all unique teams to 1500 Elo.
#    - Iterate row-by-row chronologically.
#    - Capture pre-match ratings, calculate updates, and instantly write back to the tracker.
#    - Append 'home_elo_pre' and 'away_elo_pre' columns to the dataframe.
def run_elo_engine(df):
    df_copy = df.copy()
    all_teams = pd.concat([df_copy['home_team'], df_copy['away_team']]).unique()
    elo_tracker = {team: 1500 for team in all_teams}
    
    home_elos_pre = []
    away_elos_pre = []
    
    for index, row in df_copy.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        neutral = row['neutral']
        outcome = row['match_outcome']
        tournament = row['tournament']
        
        home_elo = elo_tracker[home_team]
        away_elo = elo_tracker[away_team]
        
        home_elos_pre.append(home_elo)
        away_elos_pre.append(away_elo)
        
        expected_home, expected_away = calculate_expected_scores(home_elo, away_elo, neutral)
        k_factor = calculate_k_factor(tournament)
        
        new_home, new_away = update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor)
        
        elo_tracker[home_team] = new_home
        elo_tracker[away_team] = new_away
        
    df_copy['home_elo_pre'] = home_elos_pre
    df_copy['away_elo_pre'] = away_elos_pre
    
    return df_copy

def load_processed_data():
    input_path = DATA_DIR / 'clean_results.csv'
    print(f"Loading processed data from {input_path}...")
    return pd.read_csv(input_path, parse_dates=['date'])

if __name__ == "__main__":
    df = load_processed_data()
    print("Calculating Elo ratings...")
    elo_df = run_elo_engine(df)
    
    output_path = DATA_DIR / 'elo_results.csv'
    print(f"Saving final dataset to {output_path}...")
    elo_df.to_csv(output_path, index=False)
    print("Success! Elo calculation pipeline complete.")

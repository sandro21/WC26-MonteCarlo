#Goal: translate Elo ratings into actual goal-scoring probabilities. How does it do this? 
# It trains two predictive models based on home goals and away goals. 

import pandas as pd
import numpy as np
import statsmodels.api as sm #core stat modeling library in python. better than skikit-learn for regressions
import joblib #used for saving/loading data, in this case the models    
from scipy.stats import poisson # poisson distribution is a statistical function that calculates the probability of a specific number of events occurring in a fixed interval of time or space
from pathlib import Path 

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'data' / 'models'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True) # Creates directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)

#Step 1: Environment Setup & Data Loading
#Step 2: Feature Engineering & Pre-processing
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date']) #loading thte data from path
    df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre'] #modifying the dataframe, home team minus away team. main data for home goals.
    df['neg_elo_diff'] = -df['elo_diff'] #primary input for away gaosl model
    df['neutral_int'] = df['neutral'].astype(int) #converting neutral to 1 or 0 for model training
    df['const'] = 1.0
    
    print(f"Data prepared. Matches loaded: {len(df)}")
    return df

#Step 3: Training the Home Goals GLM
#Step 4: Training the Away Goals GLM
def train_poisson_models(df):
    X_home = df[['const', 'elo_diff', 'neutral_int']] #X represents features (inputs we give a model)
    y_home = df['home_score'] #Y represents the target(the thing we are trying to predict)
    home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    #sm.GLM creates a generalized linear model
    #we pass our featues and target and fit it to the model. we use poisson because goals scored is a count variable
    
    X_away = df[['const', 'neg_elo_diff', 'neutral_int']]
    y_away = df['away_score']
    away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()

    #what is poisson?, it is the mathematical distribution that best models count variables like 
    
    return home_model, away_model

#Step 5: Model Persistence
def persist_models(home_model, away_model, output_dir):
    home_path = output_dir / 'home_poisson.pkl' #pickle files. which is used for saving machine learning models
    away_path = output_dir / 'away_poisson.pkl'
    
    joblib.dump(home_model, home_path) #saving models to disk.
    joblib.dump(away_model, away_path)
    print(f"Models saved to {output_dir}")

#Step 6: Match Prediction Helper & Probabilistic Grid Engine
def predict_match(home_team, away_team, df, home_model, away_model, neutral=True):
    # Fetch most recent Elo for both teams
    def get_latest_elo(team_name):
        team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]
        if team_matches.empty:
            return 1500.0
        last_match = team_matches.iloc[-1]
        if last_match['home_team'] == team_name:
            return last_match['home_elo_pre']
        else:
            return last_match['away_elo_pre']

    home_elo = get_latest_elo(home_team)
    away_elo = get_latest_elo(away_team)
    
    # Calculate features
    elo_diff = home_elo - away_elo
    neutral_val = 1 if neutral else 0
    
    # Get expected goals (Lambdas) from models
    lambda_h = home_model.predict([1, elo_diff, neutral_val])[0]
    lambda_a = away_model.predict([1, -elo_diff, neutral_val])[0]
    
    # Create the 11x11 probability grid (0 to 10 goals)
    home_probs = poisson.pmf(np.arange(11), lambda_h)
    away_probs = poisson.pmf(np.arange(11), lambda_a)
    grid = np.outer(home_probs, away_probs)
    
    # Calculate outcome probabilities
    home_win_prob = np.sum(np.tril(grid, -1))
    draw_prob = np.sum(np.diag(grid))
    away_win_prob = np.sum(np.triu(grid, 1))
    
    print(f"\n--- Match Prediction: {home_team} vs {away_team} ---")
    print(f"Latest Ratings: {home_team} ({home_elo:.1f}) | {away_team} ({away_elo:.1f})")
    print(f"Predicted Goals: {home_team} = {lambda_h:.3f} | {away_team} = {lambda_a:.3f}")
    print(f"Win/Draw/Loss: {home_win_prob*100:.2f}% / {draw_prob*100:.2f}% / {away_win_prob*100:.2f}%")
    
    return home_win_prob, draw_prob, away_win_prob

#Step 7: Main Execution Logic
if __name__ == "__main__":
    print("Starting Model Training Pipeline...")
    input_csv = PROCESSED_DIR / 'elo_results.csv'
    
    # Execute Step 1 & 2
    df = load_and_prepare_data(input_csv)
    
    # Execute Step 3 & 4
    print("\nTraining Poisson GLM models...")
    home_model, away_model = train_poisson_models(df)
    
    # Execute Step 5
    print("\nExecuting Model Persistence...")
    persist_models(home_model, away_model, MODEL_DIR)
    
    # Execute Step 6 (Test Prediction)
    predict_match("Argentina", "Mexico", df, home_model, away_model, neutral=True)
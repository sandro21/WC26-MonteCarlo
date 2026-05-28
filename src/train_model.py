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

#Step 5: Model Persistence

#Step 6: Match Prediction Helper & Probabilistic Grid Engine

#Step 7: Main Execution Logic
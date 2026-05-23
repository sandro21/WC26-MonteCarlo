#Data Preparation logic for the model
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'


#Imports & Config: Load Pandas, set date columns to datetime.
def load_data():
    #these are all dataframes, which is liek Python's internal xlsx
    results = pd.read_csv(f"{DATA_DIR}/results.csv", parse_dates=['date']) 
    shootouts = pd.read_csv(f"{DATA_DIR}/shootouts.csv", parse_dates=['date'])
    former_names = pd.read_csv(f"{DATA_DIR}/former_names.csv", parse_dates=['start_date', 'end_date'])
    
    # print(results)
    # print(shootouts)
    # print(former_names)

    return results, shootouts, former_names

#Name Normalization Function: We will create a helper function that loops through former_names.csv and 
#replaces home_team and away_team in any dataframe using a conditional mask where the match date falls 
# between start_date and end_date.
def normalize_names(df, former_names):
    
    df_copy = df.copy() #make copy so we dont mess up the original df
    for index, row in former_names.iterrows(): #loop through every single row in former_names
        modern_name = row['current']
        previous_name = row['former']
        start_date = row['start_date']
        end_date = row['end_date']

        for col in ['home_team', 'away_team', 'team', 'winner']: #potential columns which may contain old names
            if col in df_copy.columns:
                #mask that maps rows of old names to new names. 
                mask = (df_copy[col] == previous_name) & (df_copy['date'] >= start_date) & (df_copy['date'] <= end_date)
                df_copy.loc[mask, col] = modern_name #does the actual making
                
    return df_copy

#Shootout Merge: We will merge results and shootouts on ['date', 'home_team', 'away_team']. We will create a new column shootout_winner.
def reconcile_shootouts(results, shootouts):
    # 1. Left join results and shootouts on date, home_team, and away_team
    merged_df = pd.merge(
        results, 
        shootouts, 
        on=['date', 'home_team', 'away_team'], 
        how='left'
    )
    
    # 2. Rename the 'winner' column to 'shootout_winner' for clarity
    merged_df = merged_df.rename(columns={'winner': 'shootout_winner'})
    
    return merged_df

#Elo Outcomes: Crucially, we will add a match_outcome column. If home_score > away_score, it's a Home Win. If it's a tie, it's a Draw. Even if there is a shootout_winner, the outcome remains Draw to respect Elo math, but our simulator can safely query shootout_winner to see who advanced!
def calculate_outcomes(df):
    df_copy = df.copy()
    df_copy['match_outcome'] = 'Draw'
    df_copy.loc[df_copy['home_score'] > df_copy['away_score'], 'match_outcome'] = 'Home Win'
    df_copy.loc[df_copy['home_score'] < df_copy['away_score'], 'match_outcome'] = 'Away Win'
    return df_copy

#Optimization: Drop city, ensure sorting by date ascending, and export to data/processed/clean_results.csv.
def optimize_data(df):
    df_optimized = df.drop(columns=['city'])  #drop city  
    df_optimized = df_optimized.sort_values(by='date', ascending=True)  #sort by date ascending
    df_optimized = df_optimized.reset_index(drop=True)  #reset index
    return df_optimized

if __name__ == "__main__":
    results, shootouts, former_names = load_data()
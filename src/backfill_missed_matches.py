"""
One-shot backfill script for the 4 missed World Cup matches from June 11-12, 2026.
The daily pipeline was crashing due to a date type bug, so these results never got recorded.

Match results (verified from FIFA/media sources):
  June 11: Mexico 2-0 South Africa, South Korea 2-1 Czech Republic
  June 12: Canada 1-1 Bosnia and Herzegovina, United States 4-1 Paraguay
"""
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
ELO_FILE = DATA_DIR / 'elo_results.csv'

sys.path.insert(0, str(BASE_DIR / 'src'))
from calculate_elo import calculate_k_factor, calculate_expected_scores, update_ratings

# Verified match results
RESULTS = [
    # (date, home_team, away_team, home_score, away_score)
    ('2026-06-11', 'Mexico',        'South Africa',            2, 0),
    ('2026-06-11', 'South Korea',   'Czech Republic',          2, 1),
    ('2026-06-12', 'Canada',        'Bosnia and Herzegovina',  1, 1),
    ('2026-06-12', 'United States', 'Paraguay',                4, 1),
]

HOSTS = ['USA', 'United States', 'Mexico', 'Canada']

def main():
    print(f"Loading Elo dataset from {ELO_FILE}...")
    df = pd.read_csv(ELO_FILE)
    df['date'] = pd.to_datetime(df['date'])

    # Build current Elo tracker from all data
    from simulate_tournament import get_current_elos
    current_elos = get_current_elos(df)

    updates_made = 0

    for date_str, home, away, h_score, a_score in RESULTS:
        target_date = pd.Timestamp(date_str)

        # Find the pre-seeded row
        mask = (
            (df['date'] == target_date) &
            (df['home_team'] == home) &
            (df['away_team'] == away)
        )
        matching = df[mask]

        if len(matching) == 0:
            print(f"  WARNING: No pre-seeded row found for {home} vs {away} on {date_str}. Skipping.")
            continue

        idx = matching.index[0]

        # Determine outcome
        if h_score > a_score:
            outcome = 'Home Win'
        elif a_score > h_score:
            outcome = 'Away Win'
        else:
            outcome = 'Draw'

        # Calculate Elo update
        home_elo = current_elos.get(home, 1500.0)
        away_elo = current_elos.get(away, 1500.0)
        neutral = 0 if (home in HOSTS or away in HOSTS) else 1
        expected_home, expected_away = calculate_expected_scores(home_elo, away_elo, neutral)
        k_factor = calculate_k_factor("FIFA World Cup")
        new_home_elo, new_away_elo = update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor)

        # Update the row in the dataframe
        df.at[idx, 'home_score'] = float(h_score)
        df.at[idx, 'away_score'] = float(a_score)
        df.at[idx, 'match_outcome'] = outcome
        df.at[idx, 'home_elo_pre'] = home_elo
        df.at[idx, 'away_elo_pre'] = away_elo
        df.at[idx, 'neutral'] = neutral
        df.at[idx, 'tournament'] = 'FIFA World Cup'

        # Update the tracker
        current_elos[home] = new_home_elo
        current_elos[away] = new_away_elo

        print(f"  OK: {home} {h_score}-{a_score} {away} ({outcome}) | Elo: {home} {home_elo:.0f}->{new_home_elo:.0f}, {away} {away_elo:.0f}->{new_away_elo:.0f}")
        updates_made += 1

    if updates_made > 0:
        df.to_csv(ELO_FILE, index=False)
        print(f"\nSaved {updates_made} match result(s) to {ELO_FILE}")
    else:
        print("\nNo updates were made.")

if __name__ == '__main__':
    main()

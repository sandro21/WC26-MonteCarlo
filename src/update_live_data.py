import os
import sys
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
ELO_FILE = DATA_DIR / 'elo_results.csv'
# Display-only box-score store (shots / SOT / saves). Separate file so the goals + bracket
# pipeline that reads elo_results.csv is never touched.
STATS_FILE = DATA_DIR / 'match_stats.csv'

API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
BASE_URL = os.getenv("BASE_URL", "https://v3.football.api-sports.io/fixtures")
STATS_URL = BASE_URL.rstrip('/') + "/statistics"

from calculate_elo import calculate_k_factor, calculate_expected_scores, update_ratings
from simulate_tournament import get_current_elos
from predict_match_stats import STATS_COLUMNS

# API-Sports statistic labels -> our short keys
STAT_LABELS = {
    'Total Shots': 'shots',
    'Shots on Goal': 'sot',
    'Goalkeeper Saves': 'saves',
}


def fetch_fixture_statistics(fixture_id):
    """Pull the per-team statistics for one fixture and return {team_name: {shots, sot, saves}}.

    Display-only: failures degrade silently to {} so the goals/Elo update is never blocked."""
    headers = {"x-apisports-key": API_SPORTS_KEY}
    try:
        response = requests.get(STATS_URL, headers=headers,
                                params={"fixture": fixture_id})
        response.raise_for_status()
        payload = response.json().get('response', [])
    except Exception as e:
        print(f"  Could not fetch statistics for fixture {fixture_id}: {e}")
        return {}

    out = {}
    for entry in payload:
        team_name = entry.get('team', {}).get('name', '')
        if not team_name:
            continue
        stats = {}
        for item in entry.get('statistics', []):
            key = STAT_LABELS.get(item.get('type'))
            if key is not None:
                stats[key] = item.get('value')  # may be None; predictor coerces to 0
        out[team_name] = stats
    return out

def fetch_live_matches(target_date):
    """Fetch fixtures for the specified date from API-Football."""
    headers = {
        "x-apisports-key": API_SPORTS_KEY
    }
    querystring = {
        "date": target_date,
        "timezone": "America/Los_Angeles"
    }
    
    print(f"Fetching matches for {target_date}...")
    try:
        response = requests.get(BASE_URL, headers=headers, params=querystring)
        response.raise_for_status()
        return response.json().get('response', [])
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return []

def process_and_update(matches, target_date):
    """Parse API response, calculate new Elo ratings, and update the dataset."""
    if not matches:
        print("No matches found for this date.")
        return False
        
    print(f"Loading existing Elo dataset from {ELO_FILE}...")
    df_elo = pd.read_csv(ELO_FILE, parse_dates=['date'])
    df_elo['date'] = pd.to_datetime(df_elo['date'])
    current_elos = get_current_elos(df_elo)
    
    new_rows = []
    new_stats_rows = []
    matches_added = 0

    for match in matches:
        league_name = match.get('league', {}).get('name', '')
        if "World Cup" not in league_name or "Women" in league_name:
            continue
            
        status = match.get('fixture', {}).get('status', {}).get('short', '')
        if status not in ['FT', 'AET', 'PEN']:
            print(f"Skipping match {home_team} vs {away_team}: Status is '{status}' (Not Finished).")
            continue
            
        home_team = match.get('teams', {}).get('home', {}).get('name', '')
        away_team = match.get('teams', {}).get('away', {}).get('name', '')
        
        if str(home_team).endswith(" W") or str(away_team).endswith(" W"):
            continue
        
        goals = match.get('goals', {})
        home_score = goals.get('home', 0)
        away_score = goals.get('away', 0)
        
        penalty = match.get('score', {}).get('penalty', {})
        pen_home = penalty.get('home')
        pen_away = penalty.get('away')
        
        shootout_winner = None
        if status == 'PEN' and pen_home is not None and pen_away is not None:
            if pen_home > pen_away:
                outcome = 'Home Win'
                shootout_winner = home_team
            else:
                outcome = 'Away Win'
                shootout_winner = away_team
        else:
            if home_score > away_score:
                outcome = 'Home Win'
            elif away_score > home_score:
                outcome = 'Away Win'
            else:
                outcome = 'Draw'
                
        home_elo = current_elos.get(home_team, 1500.0)
        away_elo = current_elos.get(away_team, 1500.0)
        
        HOSTS = ['USA', 'United States', 'Mexico', 'Canada']
        neutral = 0 if (home_team in HOSTS or away_team in HOSTS) else 1
        
        expected_home, expected_away = calculate_expected_scores(home_elo, away_elo, neutral)
        k_factor = calculate_k_factor("FIFA World Cup")
        
        new_row = {
            'date': target_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'tournament': 'FIFA World Cup',
            'country': 'North America',
            'neutral': neutral,
            'shootout_winner': shootout_winner,
            'first_shooter': None, 
            'match_outcome': outcome,
            'home_elo_pre': home_elo,
            'away_elo_pre': away_elo
        }
        new_rows.append(new_row)
        matches_added += 1

        # --- Box-score layer (display-only): capture pre-match Elos + this fixture's
        #     shots/SOT/saves so the stats models refresh alongside the goals data. ---
        fixture_id = match.get('fixture', {}).get('id')
        if fixture_id is not None:
            fixture_stats = fetch_fixture_statistics(fixture_id)
            home_stats = fixture_stats.get(home_team, {})
            away_stats = fixture_stats.get(away_team, {})
            if home_stats or away_stats:
                new_stats_rows.append({
                    'date': target_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'neutral': neutral,
                    'home_elo_pre': home_elo,
                    'away_elo_pre': away_elo,
                    'home_goals': home_score,
                    'away_goals': away_score,
                    'home_shots': home_stats.get('shots'),
                    'away_shots': away_stats.get('shots'),
                    'home_sot': home_stats.get('sot'),
                    'away_sot': away_stats.get('sot'),
                    'home_saves': home_stats.get('saves'),
                    'away_saves': away_stats.get('saves'),
                })

        new_home_elo, new_away_elo = update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor)
        current_elos[home_team] = new_home_elo
        current_elos[away_team] = new_away_elo

    if matches_added > 0:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_elo, df_new], ignore_index=True)
        df_combined.to_csv(ELO_FILE, index=False)
        print(f"Successfully appended {matches_added} new match(es) to {ELO_FILE}.")

        _append_match_stats(new_stats_rows)
        return True
    else:
        print("No finished World Cup matches found in the payload.")
        return False


def _append_match_stats(new_stats_rows):
    """Append captured box scores to match_stats.csv (creating it with headers if needed)."""
    if not new_stats_rows:
        print("No fixture statistics captured for this batch.")
        return
    df_stats_new = pd.DataFrame(new_stats_rows, columns=STATS_COLUMNS)
    if STATS_FILE.exists():
        df_existing = pd.read_csv(STATS_FILE)
        df_stats = pd.concat([df_existing, df_stats_new], ignore_index=True)
    else:
        df_stats = df_stats_new
    df_stats.to_csv(STATS_FILE, index=False)
    print(f"Appended {len(new_stats_rows)} box-score row(s) to {STATS_FILE}.")


def retrain_stats_models():
    """Refit the shot/SOT GLMs on the updated store. Failure here is non-fatal — the
    predictor falls back to the rate/prior estimate."""
    try:
        import train_stats_models
        train_stats_models.main()
    except Exception as e:
        print(f"Stats model retrain skipped: {e}")

def trigger_simulation():
    """Trigger the Monte Carlo simulation to update the web dashboard's probability cache."""
    print("Triggering the Monte Carlo Simulator to update win probabilities...")
    try:
        simulation_script = BASE_DIR / 'src' / 'simulate_tournament.py'
        subprocess.run([sys.executable, str(simulation_script)], check=True)
        print("Simulation complete. Dashboard data cache refreshed.")
    except subprocess.CalledProcessError as e:
        print(f"Error triggering simulation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Automation Pipeline for World Cup Matches")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format. Defaults to yesterday.")
    args = parser.parse_args()
    
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
    print("--- 2026 World Cup Live Automation Pipeline ---")
    matches = fetch_live_matches(target_date)
    
    updated = process_and_update(matches, target_date)

    if updated:
        retrain_stats_models()
        trigger_simulation()

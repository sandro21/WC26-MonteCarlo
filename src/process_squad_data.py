import json
import csv
from pathlib import Path

def load_squad_data(data_dir: str = "data/raw/squads"):
    """
    Loads all JSON files from the specified directory and combines them into one list.
    """
    all_teams_data = []
    # FIX: Get the project root folder so the path works from anywhere!
    base_dir = Path(__file__).parent.parent
    path = base_dir / data_dir
    for file_path in path.glob("*.json"): #loop through every file that has json in that holder. 
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if isinstance(data, dict) and "teams" in data: #checking if its a dict or not (json)
                all_teams_data.extend(data["teams"])
            elif isinstance(data, list):
                all_teams_data.extend(data)
                
    return all_teams_data

def extract_team_features(teams_data):
    """
    Processes the raw team data to extract specific market value features.
    """
    processed_teams = []
    
    for team in teams_data:
        team_name = team.get('team_name', team.get('name', 'Unknown')).strip() 
        players = team.get('players', [])        
        sorted_players = sorted(players, key=lambda x: x.get('value_eur_million', 0), reverse=True)
        total_squad_value = sum(p.get('value_eur_million', 0) for p in sorted_players)
        starting_xi_value = sum(p.get('value_eur_million', 0) for p in sorted_players[:11])
        
        if sorted_players:
            top_player = sorted_players[0]
            top_player_name = top_player.get('name', 'Unknown')
            top_player_value = top_player.get('value_eur_million', 0)
        else:
            top_player_name = 'Unknown'
            top_player_value = 0
            
        team_features = {
            'team_name': team_name,
            'total_squad_value': total_squad_value,
            'starting_xi_value': starting_xi_value,
            'top_player_name': top_player_name,
            'top_player_value': top_player_value
        }
        
        processed_teams.append(team_features)
        
    return processed_teams

def calculate_market_value_index(team_features_list):
    """
    Calculates the Market Value Index (MVI) for each team.
    """
    if not team_features_list:
        return []
        
    total_global_xi_value = sum(team['starting_xi_value'] for team in team_features_list)
    
    global_average_xi = total_global_xi_value / len(team_features_list)
    
    for team in team_features_list:
        if global_average_xi > 0:
            mvi = team['starting_xi_value'] / global_average_xi
        else:
            mvi = 0.0
            
        team['market_value_index'] = round(mvi, 3)
        
    return team_features_list

def export_to_csv(team_features_list, output_path: str = "data/processed/squad_features.csv"):
    """
    Saves the extracted features to a CSV file.
    """
    if not team_features_list:
        return
        
    base_dir = Path(__file__).parent.parent
    path = base_dir / output_path
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'team_name', 
        'total_squad_value', 
        'starting_xi_value', 
        'market_value_index', 
        'top_player_name', 
        'top_player_value'
    ]
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        writer.writerows(team_features_list)
        
    print(f"Data successfully saved to: {path.relative_to(base_dir)}")

if __name__ == "__main__":
    teams = load_squad_data()
    print(f"Loaded {len(teams)} teams.")
    
    if teams:
        features = extract_team_features(teams)
        final_features = calculate_market_value_index(features)
        
        first_team = final_features[0]
        print(f"\nFinal Features for {first_team['team_name']}:")
        print(f"  Total Squad Value: €{first_team['total_squad_value']}m")
        print(f"  Starting XI Value: €{first_team['starting_xi_value']}m")
        print(f"  Top Player: {first_team['top_player_name']} (€{first_team['top_player_value']}m)")
        print(f"  Market Value Index (MVI): {first_team['market_value_index']}\n")
        
        export_to_csv(final_features)

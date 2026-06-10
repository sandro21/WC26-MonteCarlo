import json
from pathlib import Path
import pandas as pd

raw_dir = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo/data/raw/squads")
goalkeepers = []

for file_path in raw_dir.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        teams = data.get("teams", []) if isinstance(data, dict) else data
        for team in teams:
            team_name = team.get("team_name", team.get("name", "Unknown"))
            # We want to identify the goalkeepers
            for player in team.get("players", []):
                # Check if name is a known goalkeeper or let's inspect player names
                # Usually goalkeepers are the first 2-3 players listed in the squad
                # Let's check if there is a position field, or look at the first 3 players
                pass

# Let's write a script to print the first 3 players of each team (which are usually the GKs)
# to see who the starting goalkeepers are.
gk_candidates = []
for file_path in raw_dir.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        teams = data.get("teams", []) if isinstance(data, dict) else data
        for team in teams:
            team_name = team.get("team_name", team.get("name", "Unknown"))
            players = team.get("players", [])
            # Let's take the first 3 players (usually GKs) and look at their real-world values
            for p in players[:3]:
                gk_candidates.append({
                    "team": team_name,
                    "name": p.get("name"),
                    "value": p.get("value_eur_million", 0)
                })

df_gks = pd.DataFrame(gk_candidates)
# Save to file to avoid encoding issues
out_path = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo/scratch/goalkeepers.txt")
with open(out_path, "w", encoding="utf-8") as out:
    for idx, row in df_gks.iterrows():
        out.write(f"{row['team']}: {row['name']} (€{row['value']}m)\n")
print("Goalkeeper candidates written to scratch/goalkeepers.txt")

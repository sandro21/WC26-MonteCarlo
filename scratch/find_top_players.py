import json
from pathlib import Path
import pandas as pd

raw_dir = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo/data/raw/squads")
all_players = []

for file_path in raw_dir.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        teams = data.get("teams", []) if isinstance(data, dict) else data
        for team in teams:
            team_name = team.get("team_name", team.get("name", "Unknown"))
            for player in team.get("players", []):
                all_players.append({
                    "team": team_name,
                    "name": player.get("name"),
                    "value_eur_million": player.get("value_eur_million", 0)
                })

df_players = pd.DataFrame(all_players)
top_players = df_players.sort_values(by="value_eur_million", ascending=False).head(100)

out_path = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo/scratch/top_players.txt")
with open(out_path, "w", encoding="utf-8") as out:
    for idx, row in top_players.iterrows():
        out.write(f"{row['name']} ({row['team']}): €{row['value_eur_million']}m\n")
print("Top players written to scratch/top_players.txt")

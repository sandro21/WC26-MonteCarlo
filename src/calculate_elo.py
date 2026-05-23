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
#    - Calculate EH and EA using the Elo formula.
#    - Apply the +100 Home-Field Advantage boost to the Home team if neutral == False.
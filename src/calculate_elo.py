import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'

# 2. Dynamic K-Factor Routing Function
#    - Classify tournaments into 5 Tiers using substring matching (T1=60, T2=40, T3=30, T4=20, T5=10).
# K factor is the importance weight of a specific math, allows us to change ELO based on K value. 
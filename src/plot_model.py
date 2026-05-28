import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
df = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'elo_results.csv')
df = df.dropna(subset=['home_score', 'away_score', 'home_elo_pre', 'away_elo_pre', 'neutral'])

df['elo_diff'] = df['home_elo_pre'] - df['away_elo_pre']
home_model = joblib.load(BASE_DIR / 'data' / 'models' / 'home_poisson.pkl')

# Sample some data for plotting to avoid overplotting 50k points
sample_df = df.sample(n=2000, random_state=42)

plt.figure(figsize=(10, 6))

# Plot historical dots. Add slight jitter to y to separate dots visually
jitter = np.random.normal(0, 0.15, size=len(sample_df))
plt.scatter(sample_df['elo_diff'], sample_df['home_score'] + jitter, alpha=0.3, s=15, color='#1f77b4', label='Historical Matches (Jittered)')

# Plot the curve
x_vals = np.linspace(-1000, 1000, 100)
pred_df = pd.DataFrame({
    'const': 1.0,
    'elo_diff': x_vals,
    'neutral_int': 1
})
y_pred = home_model.predict(pred_df)

plt.plot(x_vals, y_pred, color='#d62728', linewidth=4, label='Poisson GLM Curve (Expected Goals)')

plt.title('Home Goals vs Elo Difference (Poisson Regression Model)', fontsize=14, fontweight='bold')
plt.xlabel('Elo Difference (Home - Away)', fontsize=12)
plt.ylabel('Goals Scored', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save it to the conversation artifact directory so we can view it
out_path = r"C:\Users\avsad\.gemini\antigravity-ide\brain\a088387c-9414-4b88-b27e-75e3e31f566f\poisson_curve.png"
plt.savefig(out_path, dpi=300)
print(f"Plot saved to {out_path}")

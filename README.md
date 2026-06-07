<div align="center">
  <h1>🏆 2026 World Cup Monte Carlo Simulator</h1>
  <p>
    <strong>A probabilistic forecasting engine for the 2026 FIFA World Cup, powered by Elo ratings, Joint Poisson Regression Models, and 10,000 Monte Carlo simulations.</strong>
  </p>
  
  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aasteriskz-wc26-montecarlo.streamlit.app/)
  
  <h3><a href="https://aasteriskz-wc26-montecarlo.streamlit.app/">🔴 View Live Dashboard</a></h3>
</div>

---

## 📖 Overview
This project provides a robust, mathematically rigorous forecasting engine for the 2026 FIFA World Cup. By analyzing historical international match data, computing custom Elo ratings, and fitting Generalized Linear Models (GLMs), the engine estimates the exact probability of any two teams winning, losing, or drawing in a match. 

These probabilities are then injected into a **Monte Carlo Simulator** which plays out the entire 2026 World Cup 10,000 times to generate realistic championship odds for every country.

## ✨ Key Features
- **Live Automated Pipeline:** A completely autonomous GitHub Actions pipeline (`update_live_data.py`) runs every morning at 06:00 UTC, parsing real-world fixture results via the API-Sports API, recalculating Elo ratings, and refreshing the simulation odds.
- **Advanced Elo System:** Custom Elo tracking factoring in goal differential, match importance (friendlies vs. qualifiers vs. World Cup), and home-field advantage.
- **Joint Poisson Match Predictor:** Head-to-head simulations utilize two independent Poisson distributions for home/away expected goals (λ), accounting for the specific host nations (USA, Mexico, Canada) and neutral venues.
- **Interactive Sandbox:** A dynamic Streamlit frontend allowing users to pit any two nations against each other to view the real-time expected goals and win/draw/loss matrices.

## 🏗️ Architecture
The backend is split into five distinct, decoupled pipelines:
1. `prepare_data.py`: Ingests and cleans decades of historical international fixtures.
2. `calculate_elo.py`: Generates continuous Elo ratings tracking team momentum up to the present day.
3. `train_model.py`: Fits two Poisson GLMs (Home Goals & Away Goals) using the Elo differential as the primary feature.
4. `simulate_tournament.py`: The Monte Carlo engine. Runs 10k bracket permutations based on the GLM outputs.
5. `update_live_data.py`: The live ingest layer. Pulls recent matches, updates the state, and triggers a dashboard cache refresh.

## 🚀 Running Locally
If you wish to run the forecasting engine and dashboard on your local machine:

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/AAsteriskz7/WC26-MonteCarlo.git
cd WC26-MonteCarlo
pip install -r requirements.txt
```

**2. Add your API Key:**
Create a `.env` file in the root directory and add your API-Sports key:
```env
API_SPORTS_KEY="your_api_key_here"
```

**3. Run the Streamlit Dashboard:**
```bash
streamlit run app.py
```

## 🤖 Automation
This repository utilizes a **GitHub Action** located at `.github/workflows/daily_update.yml`. The action boots a runner every day, fetches the previous day's results, mathematically updates the Elo dataset, runs the 10,000 simulations, and executes a stateful write-back to the repository so the live dashboard remains continuously up-to-date without human intervention.

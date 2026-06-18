import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import scipy.stats as stats

# Page configuration
st.set_page_config(
    page_title="2026 World Cup Monte Carlo Simulator",
    page_icon="▲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Dark Mode UI Theme Styles
st.markdown("""
    <style>
    /* Dark theme styling overrides */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3, p, label {
        color: #fafafa !important;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #00ffcc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #8a8d93 !important;
    }
    hr {
        border-color: #262730 !important;
    }
    .stDataFrame {
        border: 1px solid #262730;
        border-radius: 8px;
        background-color: #161920;
    }
    </style>
""", unsafe_allow_html=True)

# Directory configurations
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'data' / 'models'

# 1. Cached Data Loading
@st.cache_data
def load_data():
    # Load processed ELO results
    df_elo = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])
    
    # Import optimized ELO lookup function from our tournament script
    from src.simulate_tournament import get_current_elos
    latest_elos = get_current_elos(df_elo)
    
    # Load simulation results
    sim_path = DATA_DIR / 'simulation_results.csv'
    if os.path.exists(sim_path):
        df_sim = pd.read_csv(sim_path)
    else:
        # Fallback dynamic simulation if results file not generated yet
        from src.simulate_tournament import run_monte_carlo
        results = run_monte_carlo(10000)
        df_sim = pd.DataFrame([
            {
                'Team': team,
                'Win_Probability': prob,
                'Elo': latest_elos.get(team, 1500.0)
            } for team, prob in results
        ])
        df_sim.to_csv(sim_path, index=False)
        
    return df_sim, latest_elos

# 2. Cached Model Loading
@st.cache_resource
def load_models():
    home_model = joblib.load(MODEL_DIR / 'home_poisson.pkl')
    away_model = joblib.load(MODEL_DIR / 'away_poisson.pkl')
    return home_model, away_model

# 2b. Cached Box-Score (shots / SOT / saves) resources — display-only layer
@st.cache_resource
def load_stats_resources():
    from src.predict_match_stats import load_stats_models, load_match_stats, build_rate_tables
    models, n_model = load_stats_models()
    rate_tables = build_rate_tables(load_match_stats())
    return models, n_model, rate_tables

# Load resources
df_sim, latest_elos = load_data()
home_model, away_model = load_models()
stats_models, stats_n, stats_rate_tables = load_stats_resources()
sorted_teams = sorted(list(latest_elos.keys()))

# --- Header Section ---
st.markdown("### ▲ 2026 WORLD CUP MONTE CARLO SIMULATOR")
st.markdown("---")

# Infrastructure Metrics
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric(label="Total Iterations", value="10,000")
col_m2.metric(label="Execution Speed", value="486/sec")
col_m3.metric(label="Total Matches Processed", value="49,287")

st.markdown("---")

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["🏆 Tournament Forecast", "⚔️ Match Predictor", "📊 Bracket Explorer"])

# --- Tab 1: Tournament Forecast (Lead Table) ---
with tab1:
    st.markdown("##### Simulated Tournament Standings")
    st.markdown("Win probabilities computed across 10,000 full Monte Carlo tournament brackets using historical Elo ratings and joint Poisson expected goal distribution models.")
    
    # Format and present sorted dataset
    df_display = df_sim.copy()
    df_display = df_display.rename(columns={
        'Team': 'Country',
        'Elo': 'Elo Rating',
        'Win_Probability': 'Win Probability (%)'
    })
    # Sort descending by win probability
    df_display = df_display.sort_values(by='Win Probability (%)', ascending=False).reset_index(drop=True)
    df_display.index += 1
    
    # Render interactive DataFrame
    st.dataframe(
        df_display,
        width='stretch',
        column_config={
            "Elo Rating": st.column_config.NumberColumn(format="%.1f"),
            "Win Probability (%)": st.column_config.NumberColumn(format="%.2f%%")
        },
        height=550
    )

# --- Tab 2: Match Predictor (Interactive Sandbox) ---
with tab2:
    st.markdown("##### Head-to-Head Sandbox Predictor")
    st.markdown("Compare any two international countries to simulate their matchup probabilities based on current ELO ratings.")
    
    # Team selections
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        team1 = st.selectbox(
            "Select Team 1", 
            sorted_teams, 
            index=sorted_teams.index("Spain") if "Spain" in sorted_teams else 0
        )
    with col_t2:
        team2 = st.selectbox(
            "Select Team 2", 
            sorted_teams, 
            index=sorted_teams.index("Argentina") if "Argentina" in sorted_teams else 1
        )
        
    neutral_venue = st.checkbox("Neutral Venue", value=True)
    
    if team1 == team2:
        st.warning("Please select two different teams for the matchup.")
    else:
        # Fetch ELOs
        elo1 = latest_elos.get(team1, 1500.0)
        elo2 = latest_elos.get(team2, 1500.0)
        
        st.markdown(f"**Elo Comparison**: {team1} ({elo1:.1f}) vs {team2} ({elo2:.1f})")
        
        # Venue and Host-Nation swap logic
        HOSTS = ['United States', 'Mexico', 'Canada']
        swapped = False
        
        if neutral_venue:
            neutral_val = 1
            elo_diff = elo1 - elo2
            # Predict expected goals (Home ELO, Away ELO, Neutral Venue)
            lambda_1 = home_model.predict([1, elo_diff, 1])[0]
            lambda_2 = away_model.predict([1, -elo_diff, 1])[0]
        else:
            # Handle host advantages
            if team2 in HOSTS and team1 not in HOSTS:
                team1_temp, team2_temp = team2, team1
                swapped = True
            else:
                team1_temp, team2_temp = team1, team2
                
            elo_diff = latest_elos.get(team1_temp, 1500.0) - latest_elos.get(team2_temp, 1500.0)
            neutral_val = 0 if (team1_temp in HOSTS and team2_temp not in HOSTS) else 1
            
            lambda_1 = home_model.predict([1, elo_diff, neutral_val])[0]
            lambda_2 = away_model.predict([1, -elo_diff, neutral_val])[0]
            
            if swapped:
                lambda_1, lambda_2 = lambda_2, lambda_1
                
        # Display Poisson Lambdas
        st.markdown("---")
        st.markdown("###### Expected Goals (Poisson λ)")
        col_g1, col_g2 = st.columns(2)
        col_g1.metric(label=f"Expected Goals for {team1}", value=f"{lambda_1:.2f}")
        col_g2.metric(label=f"Expected Goals for {team2}", value=f"{lambda_2:.2f}")
        
        # Calculate precise match outcome probabilities via joint Poisson distribution
        max_goals = 10
        prob_matrix = np.zeros((max_goals, max_goals))
        for x in range(max_goals):
            for y in range(max_goals):
                prob_matrix[x, y] = stats.poisson.pmf(x, lambda_1) * stats.poisson.pmf(y, lambda_2)
                
        win_prob = np.sum(np.tril(prob_matrix, -1)) * 100
        draw_prob = np.sum(np.diag(prob_matrix)) * 100
        loss_prob = np.sum(np.triu(prob_matrix, 1)) * 100
        
        # Normalize to exactly 100%
        total = win_prob + draw_prob + loss_prob
        if total > 0:
            win_prob = (win_prob / total) * 100
            draw_prob = (draw_prob / total) * 100
            loss_prob = (loss_prob / total) * 100
            
        # Display outcome probabilities
        st.markdown("###### Match Outcome Probabilities")
        col_w, col_d, col_l = st.columns(3)
        col_w.metric(label=f"{team1} Win", value=f"{win_prob:.1f}%")
        col_d.metric(label="Draw Probability", value=f"{draw_prob:.1f}%")
        col_l.metric(label=f"{team2} Win", value=f"{loss_prob:.1f}%")
        
        st.markdown("---")
        
        # Display horizontal outcome bar chart
        outcome_data = pd.DataFrame({
            'Outcome': [f'{team1} Win', 'Draw', f'{team2} Win'],
            'Probability (%)': [win_prob, draw_prob, loss_prob]
        })
        st.bar_chart(outcome_data, x='Outcome', y='Probability (%)', width='stretch')

        # --- Predicted Box Score (display-only; does NOT affect goals or the bracket) ---
        st.markdown("---")
        st.markdown("###### Predicted Box Score")
        from src.predict_match_stats import predict_match_stats
        # Reuse the exact goals-model Elo convention: team1 perspective, same neutral flag.
        stats_elo_diff = elo1 - elo2
        box = predict_match_stats(
            team1, team2, stats_elo_diff, neutral_val,
            models=stats_models, n_model=stats_n, rate_tables=stats_rate_tables
        )

        def _rng(lo, hi):
            return f"{lo:.0f}–{hi:.0f}"

        for label, key in [(team1, 'team1'), (team2, 'team2')]:
            p = box[key]
            st.markdown(f"**{label}**")
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Total Shots", f"{p['shots']:.1f}", help=f"~{_rng(*p['shots_range'])} range")
            bc2.metric("Shots on Target", f"{p['sot']:.1f}", help=f"~{_rng(*p['sot_range'])} range")
            bc3.metric("Saves", f"{p['saves']:.1f}", help=f"~{_rng(*p['saves_range'])} range")

        conf = box['model_weight']
        source = (f"fitted GLM weight {conf*100:.0f}% / rate-model {100-conf*100:.0f}%"
                  if conf > 0 else "recency-weighted rate model + league priors (no fitted GLM yet)")
        st.caption(f"Ranges are rough (≈1 SD). Blend: {source}. Display-only — shots do not affect advancement.")

# --- Tab 3: Bracket Explorer ---
with tab3:
    st.markdown("##### 🏆 Sample Tournament Bracket")
    st.markdown("A match-by-match log of a complete simulation run from the Group Stage to the Final.")
    
    bracket_path = DATA_DIR / 'sample_bracket.json'
    if os.path.exists(bracket_path):
        import json
        with open(bracket_path, 'r', encoding='utf-8') as f:
            bracket_data = json.load(f)
            
        rounds_order = ["Group Stage", "Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
        
        for round_name in rounds_order:
            if round_name in bracket_data:
                # Keep Semis and Final open by default for visual impact
                is_expanded = True if round_name in ["Semifinals", "Final"] else False
                
                with st.expander(round_name, expanded=is_expanded):
                    matches = bracket_data[round_name]
                    # Display matches split across 2 columns for a cleaner look
                    col1, col2 = st.columns(2)
                    for idx, match in enumerate(matches):
                        if idx % 2 == 0:
                            col1.markdown(f"🏆 {match}" if "Final" in round_name and "Semi" not in round_name else f"⚽ {match}")
                        else:
                            col2.markdown(f"🏆 {match}" if "Final" in round_name and "Semi" not in round_name else f"⚽ {match}")
    else:
        st.info("No sample bracket found. Please run the simulation engine first.")

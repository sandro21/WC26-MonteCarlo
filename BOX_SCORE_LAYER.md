# Box-Score Layer — Shots / Shots on Target / Saves

A per-match box-score prediction layer added on top of the existing 2026 World Cup
forecast pipeline. For any head-to-head it predicts **total shots**, **shots on target
(SOT)**, and **saves** for each team, with rough uncertainty ranges.

> **Display / analysis only.** This layer never feeds or alters the Elo ratings, the goals
> Poisson GLMs, or the bracket Monte Carlo. Shots do not decide who advances. The existing
> goals + bracket pipeline (`calculate_elo.py`, `train_model.py`, `prepare_data.py`,
> `simulate_tournament.py`, `elo_results.csv`) is untouched.

---

## How it works

Two estimators, blended by how much shot data exists:

1. **Fitted Poisson GLMs** — `stat ~ elo_diff + neutral`, mirroring the goals models exactly.
   Robust because every team is pooled through the Elo differential, but needs data to trust.
2. **Recency-weighted rate model** — per team, its own shots-for blended with the opponent's
   shots-against, each shrunk toward a league prior so small nations with few games don't blow up.

The blend weights the GLM up as the global sample grows:

```
w_glm   = n / (n + 40)              # n = matches with shot stats
estimate = w_glm * GLM + (1 - w_glm) * rate_model
```

**Saves are derived last**, never modelled directly:

```
saves(team) = opponent_expected_SOT * (1 - opponent_finishing)
finishing   = goals_per_SOT, shrunk toward a 0.30 league prior
```

### Elo convention (identical to the goals models)
`elo_diff = elo1 - elo2`. Team 1 uses the `home_*` GLMs with `+elo_diff`; team 2 uses the
`away_*` GLMs with `-elo_diff`; same `neutral` flag. The app passes the same `elo_diff` /
`neutral` it already computes for expected goals — a symmetric extension, not a bolt-on.
The rate fallback is identity-based (team names) and independent of Elo.

### League priors (per team, per match)
Used to seed everything before live shot data accumulates, and as the shrinkage target:

| Constant | Value | Meaning |
|---|---|---|
| `LEAGUE_SHOTS` | 12.0 | total shots in an average game |
| `LEAGUE_SOT` | 4.3 | shots on target |
| `LEAGUE_FINISHING` | 0.30 | goals per shot on target |

Shrinkage / blend knobs: `K_TEAM=5` (rate → prior), `K_FIN=3` (finishing → prior),
`K_MODEL=40` (GLM vs rate), `HALFLIFE_DAYS=730` (recency).

---

## Files

| File | Status | What it does |
|---|---|---|
| `src/predict_match_stats.py` | **new** | Core math: rate model, GLM read, sample-size blend, saves derivation. Degrades to pure priors with no data. |
| `src/train_stats_models.py` | **new** | Fits four Poisson GLMs (`home/away_shots`, `home/away_sot`) on `elo_diff + neutral`. Mirrors `train_model.py`. Skips fitting below 12 rows. |
| `src/update_live_data.py` | **modified** | Pulls API-Sports `/fixtures/statistics` per finished WC fixture → appends to `match_stats.csv` → retrains stats GLMs. Goals/Elo path unchanged; stat failures degrade silently. |
| `app.py` | **modified** | Tab 2 (Match Predictor): three metrics with ±ranges and a blend-source caption, beside expected goals. |
| `data/processed/match_stats.csv` | **generated** | Separate per-match store: captured Elos, goals, shots/SOT/saves. Created on first ingest. |
| `data/models/*_shots_poisson.pkl`, `*_sot_poisson.pkl`, `stats_model_meta.json` | **generated** | Fitted stats GLMs + sample size. |

---

## Data flow

```
API-Sports /fixtures/statistics
        │  (per finished WC fixture, in update_live_data.py)
        ▼
data/processed/match_stats.csv   ──►  train_stats_models.py  ──►  data/models/*_poisson.pkl
        │                                                                  │
        └──────────────► predict_match_stats.py ◄──────────────────────────┘
                         (rate model + GLM, blended)
                                   │
                                   ▼
                          app.py — Tab 2 box score
```

---

## Bootstrapping note

The repo's historical fixture set (Kaggle international results) is **goals-only — there is
no shot data anywhere**. The layer therefore bootstraps entirely from live API ingestion.
Until that accumulates, every matchup shows league-prior numbers (this is expected and honest).

To populate it, run the live updater for past World Cup dates (requires `API_SPORTS_KEY`):

```bash
python src/update_live_data.py --date 2026-06-11
python src/update_live_data.py --date 2026-06-12
# ... etc.
```

Going forward, the daily live-update path refreshes shot stats and refits the GLMs alongside
the existing Elo/goals update automatically.

---

## Verification

Tested in an isolated venv:
- **No data** → symmetric priors (12.0 shots / 4.3 SOT / 3.0 saves).
- **60 synthetic rows** → GLM weight `60/(60+40) = 0.60`.
- **Strong vs weak** (Spain vs Qatar) diverges correctly — the favourite takes more
  shots/SOT and makes fewer saves — and stays symmetric under a home/away swap.

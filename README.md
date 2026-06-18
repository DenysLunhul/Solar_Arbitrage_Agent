# Solar Arbitrage Agent

An autonomous Reinforcement Learning agent that optimally manages energy flows in a hybrid solar + battery + grid node to maximize profit on the Ukrainian Day-Ahead Market (DAM), guarantee energy autonomy during power outages, and minimize battery degradation.

---

## What it does

Every evening after 14:00 (when OREE publishes tomorrow's electricity prices), the agent fetches the full 24-hour price curve, weather forecast, and outage schedule — then runs a **Soft Actor-Critic (SAC)** model to produce a 96-step (15-min resolution) dispatch plan for the next day:

- **When to charge / discharge** the battery
- **When to buy / sell** electricity on the day-ahead market
- **How much SoC reserve** to hold before a predicted grid outage

The result is a complete dispatch plan with per-step energy flows, P&L, and a daily summary.

---

## Key Features

- **SAC RL agent** trained with Stable-Baselines3 on 35 000+ real timesteps of Ukrainian market data
- **Domain randomization** — trained across 50–250 kWh battery / 40–500 kWp solar configs; a single model generalizes to any hardware
- **10-component reward function** — market profit, LCOS degradation cost, unmet load penalty, SoC management, outage preparation, solar priority, price timing
- **96-step price lookahead** — agent sees the full next-day DAM price curve as part of its observation
- **Real market data** — DAM prices fetched live from [oree.com.ua](https://www.oree.com.ua)
- **Full-stack dashboard** — React frontend with live dispatch charts, SAC vs rule-based comparison, and historical results
- **Dockerized** — one `docker-compose up` brings up the full stack (FastAPI + PostgreSQL)

---

## Architecture

```
Solar PV ──┐
           ├──► Inverter ──► Load
Battery ───┘         │
                     ▼
                  Grid (DAM)
                  
SAC Agent observes: [17 normalized features + SoC + 96-step price curve + 16-step load/GTI lookahead]
SAC Agent outputs:  [battery action (-1=discharge, +1=charge), grid action (-1=buy, +1=sell)]
```

```
data_providers/          # Live data pipeline (DAM prices, weather, load, grid)
environment/             # Gymnasium RL environment + SAC training + inference
backend/                 # FastAPI REST API + PostgreSQL (JWT auth, predictions, history)
frontend/                # React + Recharts dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| RL Framework | Stable-Baselines3 (SAC) |
| RL Environment | Gymnasium |
| Deep Learning | PyTorch |
| Backend | FastAPI + SQLAlchemy + PostgreSQL |
| Frontend | React 18 + Recharts + Vite |
| Data | pandas, scikit-learn, Open-Meteo API |
| Auth | JWT (HS256) + Argon2 password hashing |
| Deployment | Docker + docker-compose |

---

## Results (150 kWh battery, 200 kWp solar, Chernivtsi UA)

Run the full-year backtest yourself:
```bash
python environment/testing/backtest_sac.py
python environment/testing/backtest_default.py
python environment/testing/compare.py
```

Results are written to `environment/testing/results/` as CSV files with per-step and per-month breakdowns.

---

## Quick Start

### Prerequisites
- Docker + docker-compose
- Python 3.13+ with venv (for standalone scripts)

### Full stack (API + DB + Frontend)

```bash
cp .env.example .env          # fill in SECRET_KEY
docker-compose up --build
```

API: `http://localhost:8000` | Frontend: `http://localhost:3000`

### Standalone live inference (no DB needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_live.py --soc 0.6    # runs after 14:00 UA time when DAM prices are published
```

### Train from scratch

```bash
source .venv/bin/activate
python environment/train.py     # ~20M steps, CUDA recommended
```

Training logs: `tensorboard --logdir environment/logs/tensorboard/`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT token |
| GET | `/predictions/` | Run SAC inference for tomorrow |
| GET | `/predictions/default` | Run rule-based strategy |
| GET | `/predictions/history` | Retrieve stored predictions |
| POST | `/config/` | Save hardware configuration |
| POST | `/strategy/` | Save rule-based strategy config |

---

## Dataset

Training data covers **35 040 timesteps** (15-min resolution, ~1 year) with 25 features:

- Ukrainian DAM prices + volumes (OREE)
- Solar irradiance (GTI), temperature, shortwave radiation (Open-Meteo)
- Synthetic load profile + grid outage schedule (based on real Ukrainian outage statistics)
- Cyclical time encodings

Location: **Chernivtsi, Ukraine** (48.29°N, 25.93°E)

---

## Environment & Reward

The RL environment (`environment/environment.py`) is a custom Gymnasium env with:

- **State space**: 163-dimensional (normalized features + SoC + price/load/GTI lookahead)
- **Action space**: continuous 2D — battery power + grid exchange
- **Reward**: 10 components normalized by battery capacity for hardware-agnostic training

Key reward design decisions:
- `r_market` — profit from selling / cost of buying at DAM prices
- `r_lcos` — penalizes battery degradation (Levelized Cost of Storage)
- `r_reserve` — enforces SoC reserve during outages (log-weighted by remaining outage duration)
- `r_solar_priority` — penalizes grid charging while solar is available
- `r_price_timing` — bonus for selling above / penalty for buying above daily price average

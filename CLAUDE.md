# Scalable AI Energy Management System (EMS) — Project Context

## Project Goal

Build an autonomous RL agent that optimally manages energy flows in a hybrid node (Solar PV + BESS + Grid) to:
1. **Maximize profit** from energy arbitrage on the Ukrainian Day-Ahead Market (DAM).
2. **Guarantee energy autonomy** during planned/predicted power outages by maintaining a sufficient State of Charge (SoC) reserve.
3. **Minimize battery degradation** by incorporating Levelized Cost of Storage (LCOS) into the reward function.
4. **Scale to any hardware** via a single universal SAC model trained with domain randomization across diverse hardware configs.

---

## Repository Layout

```
ds_project_demo/
├── docker-compose.yml
├── requirements.txt
├── .env                                   # DATABASE_URL, SECRET_KEY, ALGORITHM, ACCESS_EXPIRE_MINUTES
├── envoriment/                            # (folder name typo — kept as-is)
│   ├── environment.py                     # Gymnasium RL environment (fully implemented)
│   ├── train.py                           # Standalone SAC training script (domain randomization)
│   ├── inference.py                       # Inference module: run_inference() for FastAPI integration
│   ├── normalize.py                       # Feature normalization: drop, scale, save scalers.pkl
│   ├── default_strategy.py                # Configurable rule-based inverter dispatch (no price awareness)
│   ├── dataset_final.csv                  # Copy of training dataset (25 cols, 35 041 rows)
│   ├── dataset_normalized.csv             # Normalized training dataset (17 cols, used by train.py)
│   ├── models/
│   │   ├── sac_ems.zip                    # Final SAC model (last training run)
│   │   ├── best/best_model.zip            # Best checkpoint by eval reward ← used by API
│   │   ├── scalers.pkl                    # sklearn scalers fitted on training data
│   │   └── obs_rms.pkl                    # VecNormalize running stats (required for inference)
│   ├── testing/
│   │   ├── backtest_sac.py                # Full-dataset backtest using SAC model
│   │   ├── backtest_default.py            # Full-dataset backtest using default strategy
│   │   ├── compare.py                     # Side-by-side comparison + per-month breakdown
│   │   └── results/                       # Output CSVs (gitignored)
│   └── logs/
│       ├── tensorboard/SAC_*/             # TensorBoard event files
│       ├── monitor/                       # SB3 Monitor CSV logs
│       └── eval/                          # EvalCallback logs
├── data_providers/
│   ├── orchestrator/
│   │   ├── data_combiner.py               # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   └── combined.csv                   # Last generated live dataset (96 rows × 25 cols)
│   └── components/
│       ├── market_manager/IDM_DAM_features.py   # Fetches DAM prices from OREE (oree.com.ua)
│       ├── weather/weather.py             # Open-Meteo forecast (GTI, temp, radiation) — plain requests.Session
│       ├── grid/synthetic_grid.py         # Synthetic outage schedule generator
│       ├── load/synthetic_load.py         # Synthetic consumption profile
│       └── time/time_features.py          # Cyclical time encoding (sin/cos)
├── datasets/
│   └── dataset_v10/                       # CURRENT training dataset
│       └── dataset_final.csv              # 35 041 rows × 25 cols — column-aligned with live pipeline ✅
├── backend/
│   ├── main.py                            # FastAPI app entry point (CORS enabled)
│   ├── core/
│   │   ├── database.py                    # SQLAlchemy engine + session (PostgreSQL)
│   │   └── loader.py                      # Bulk-uploads DataFrame → History table
│   ├── models/site.py                     # ORM: User, SystemConfig, DefaultStrategy, History, AgentPredictions
│   ├── repositories/
│   │   ├── config_repo.py                 # SystemConfig CRUD + upsert
│   │   ├── prediction_repo.py             # AgentPredictions bulk insert / delete / last SoC query
│   │   ├── strategy_repo.py               # DefaultStrategy upsert / get / list
│   │   └── user_repo.py                   # User CRUD
│   ├── routers/
│   │   ├── auth.py                        # POST /auth/login, POST /auth/register
│   │   ├── config.py                      # POST /config/, GET /config/, GET /config/list
│   │   ├── predictions.py                 # GET /predictions/, GET /predictions/default
│   │   └── strategy.py                    # POST /strategy/, GET /strategy/, GET /strategy/list
│   ├── schemas/schemas.py                 # Pydantic: SiteConfig, DefaultStrategyConfig, PredictionResponse, …
│   ├── services/
│   │   ├── auth_service.py                # login / register logic
│   │   ├── config_service.py              # save / get / list configs
│   │   ├── prediction_service.py          # SAC + default strategy prediction pipelines
│   │   └── strategy_service.py            # save / get / list default strategies
│   └── security/security.py              # JWT (HS256), pwdlib Argon2 password hashing
└── temp/                                  # One-off data-cleaning utility scripts
```

---

## Timestep & Market Conventions

| Parameter | Value |
|---|---|
| Timestep | **15 minutes** (96 steps/day) |
| Market | Ukrainian **DAM** via [oree.com.ua](https://www.oree.com.ua) |
| Price unit | UAH / MWh |
| Sell price | **DAM_Price / 1000** UAH/kWh (dynamic, from OREE day-ahead market) |
| Buy price | **DAM_Price / 1000 + 3.0 UAH/kWh** (DAM + grid access tax) |
| Site location | Lat **48.2904** °N, Lon **25.9324** °E (Chernivtsi, Western Ukraine) |
| Horizon | Agent operates on **next-day** data assembled each evening (after 14:00 UA time) |

---

## RL Environment (`envoriment/environment.py`)

### Dual-Dataset Architecture

| Argument | Purpose |
|---|---|
| `df_raw` | Raw (unnormalized) dataset — used in `step()` for physics calculations |
| `df` | Normalized dataset — used in `get_observe()` to feed the neural network |

### Spaces

```python
PRICE_LOOKAHEAD = 16  # steps of future DAM_Price appended to observation (4-hour horizon)

action_space      = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
observation_space = Box(low=-inf, high=inf, shape=(34,), dtype=float32)
# 17 normalized feature cols + SoC + 16 price lookahead steps → shape (34,)
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge |
| `action[1]` | Grid: +1 = full export (sell to grid), −1 = full import (buy from grid) |

### Hardware Parameters (from `system_config` dict)

```python
system_config = {
    'battery': {
        'capacity_kwh':        float,   # kWh
        'max_charge_power':    float,   # kW  (C/2 rate: capacity / 2)
        'max_discharge_power': float,   # kW  (C/2 rate: capacity / 2)
        'efficiency':          float,   # round-trip per half-cycle
        'lcos':                float,   # UAH/kWh degradation cost
        'min_reserve':         int,     # minimum SoC % (20 → 0.20)
    },
    'solar': {
        'peak_power':  float,           # kWp
        'efficiency':  float,           # η; panel area = peak_power / (1000 × η)
    },
    'inverter': {
        'max_power':   float,           # kW
    },
    'grid': {
        'capacity':     float,          # kW; effective limit = min(inverter, grid)
    },
}
```

`max_grid_capacity = min(inverter.max_power, grid.capacity)` — whichever is smaller is the real limit.

Charge and discharge use **separate** power limits (`max_batt_charge_power_ts` and `max_batt_discharge_power_ts`).

### SoC Floors (BMS protection)

Two floors are enforced depending on grid state:

| Condition | Floor | Variable |
|---|---|---|
| Grid up (`grid_status == 1`) | `soc_soft_min` = `min_reserve / 100` | configurable per site |
| Grid down (`grid_status == 0`) | `soc_hard_min` = `0.05` | physical BMS absolute floor |

During outages the agent can discharge down to 5% (never to zero) to protect the battery.

```python
if grid_status == 1:
    max_drawable = max(0.0, (self.soc - self.soc_soft_min) * self.max_batt_capacity)
else:
    max_drawable = max(0.0, (self.soc - self.soc_hard_min) * self.max_batt_capacity)
```

### Episode Resets

Episodes are **day-aligned**: each reset picks a random full day from the dataset (`episode_start = day_idx * 96`). This prevents the agent from ever training on a mid-day slice that has no morning solar context.

SoC is randomized uniformly on each reset: `self.soc = uniform(0.0, 1.0)`.

### Reward Function (9 components, all tracked separately in `info`)

```
r_market      = |grid_export| × (DAM_price/1000)  OR  grid_import × -(DAM_price/1000 + 3.0)
r_lcos        = -(lcos × |batt_energy_cycled|)
r_unmet       = -(unmet_load × (DAM_price/1000 + 3.0) × 2)          # if unmet_load > 0
r_mismatch    = 0.0                                                   # disabled (kept in info for compat)
r_soc_soft    = -(50.0 × violation²)                                 # outside [soc_min, 0.80]
r_reserve     = -(50.0 × soc_deficit × log1p(outage_remaining_h))   # during outage
r_preparation = 5.0 × exp(-0.5 × hours_until_outage) × min(soc, target_soc)  # pre-outage
r_soc_target  = actual_chem_in × buy_price                           # when pre_charge_soc < target_soc
r_waste       = -(10.0 × lcos × wasted_kWh)                         # discharge exceeding demand + grid headroom

reward = sum(all components) / (battery_capacity_kwh / 100.0)        # normalized by capacity
```

**Reward normalization**: dividing by `capacity / 100` keeps reward magnitude consistent across the domain-randomized hardware range (50–250 kWh), preventing large-battery configs from dominating the replay buffer.

**r_soc_target rationale**: rewards storing energy at its avoidance value (`buy_price` per kWh stored), making charging economically competitive with selling. At midday buy_price ≈ 9 UAH/kWh: charging earns ~+9×kWh while r_lcos costs -1.5×kWh → net +7.5 UAH/kWh, stronger than spot selling.

**r_waste**: penalizes discharging more than demand + available grid export headroom can absorb.

### `info` dict (returned by `step()`)

`soc`, `target_soc`, `reward`, `solar_gen_ts_kwh`, `solar_surplus_kwh`, `actual_grid_kwh`, `battery_kwh` (+ = charging), `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward_market`, `reward_lcos`, `reward_unmet`, `reward_mismatch`, `reward_soc_soft`, `reward_reserve`, `reward_preparation`, `reward_soc_target`, `reward_waste`

---

## Normalization Pipeline (`envoriment/normalize.py`)

| Strategy | Columns |
|---|---|
| **Dropped** (8 cols) | `timestamp`, `Hour`, `Minute`, `Minute_sin`, `Minute_cos`, `Day`, `Day_of_week`, `Month` |
| **log1p → StandardScaler** | `DAM_Price` |
| **StandardScaler** | `Load`, `Temperature_2m`, `Shortwave_radiation`, `DAM_Vol_Buy`, `DAM_Vol_Sale` |
| **MinMaxScaler [0,1]** | `Global_tilted_irradiance_instant`, `hours_until_outage`, `outage_remaining_h`, `next_outage_duration` |
| **Passthrough** | `Hour_sin`, `Hour_cos`, `Day_of_week_sin`, `Day_of_week_cos`, `Grid`, `Day_sin`, `Day_cos` |

Result: 17-column `dataset_normalized.csv` + `scalers.pkl`.

Run standalone: `python normalize.py --input dataset_final.csv --output dataset_normalized.csv --scalers models/scalers.pkl`

---

## Standalone Training Script (`envoriment/train.py`)

Run from project root:

```bash
cd /home/denys/PycharmProjects/ds_demo/ds_project_demo && .venv/bin/python envoriment/train.py
```

### Key design decisions

**Domain randomization** via `RandomConfigWrapper`: on every `reset()` a new correlated hardware config is sampled:
```
capacity  = uniform(50, 250) kWh
solar     = capacity × uniform(0.8, 2.0) kWp
inverter  = solar × uniform(0.8, 1.1) kW
grid      = inverter × uniform(1.0, 1.5) kW   ← always ≥ inverter
charge/discharge power = capacity / 2          ← C/2 rate
```
Buy price is always computed dynamically as `DAM_Price/1000 + 3.0` — there is no `price_to_buy` config field.

**Per-month 75/25 split**: for each of the 12 months, first 75% of rows → train, last 25% → eval. All seasons represented in both sets. No seasonal bias.

**VecNormalize**: obs normalized with `clip_obs=10.0`; reward normalized on train env, raw on eval env.

**SyncNormalizeEvalCallback**: before each eval run, deep-copies `train_env.obs_rms` → `eval_env.obs_rms` so both use the same running stats.

**Eval env**: fixed `DEFAULT_SYSTEM_CONFIG` (150 kWh mid-range) for stable training progress tracking.

**Day-aligned episodes**: `_n_starts = len(df) // episode_len`; `episode_start = day_idx * episode_len`. Prevents mid-day slice training.

**DummyVecEnv chosen over SubprocVecEnv**: env step = 0.11 ms, SubprocVecEnv pipe overhead = ~13 ms → DummyVecEnv is ~7× faster.

### Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC (MlpPolicy) |
| Total timesteps | 10 000 000 |
| Buffer size | 1 000 000 |
| Batch size | 512 |
| Learning rate | 3e-4 |
| Network arch | [256, 256] |
| n_envs | 32 (DummyVecEnv) |
| n_eval_episodes | 20 |
| Gamma | 0.99 |
| Entropy coef | auto |
| Eval frequency | every 100 000 env steps |
| Seed | 42 (numpy + SAC) |
| Device | cuda |

### Outputs

| Path | Contents |
|---|---|
| `models/sac_ems.zip` | Final model (end of training) |
| `models/best/best_model.zip` | Best checkpoint by eval reward ← **API uses this** |
| `models/obs_rms.pkl` | VecNormalize running stats — **required for inference** |
| `logs/tensorboard/` | TensorBoard event files |

View training: `tensorboard --logdir envoriment/logs/tensorboard/`

---

## Inference Module (`envoriment/inference.py`)

```python
from environment.inference import load_model_and_scalers, run_inference

model, scalers, obs_rms = load_model_and_scalers(
    model_path='environment/models/best/best_model',
    scalers_path='environment/models/scalers.pkl',
    obs_rms_path='environment/models/obs_rms.pkl',
    model_cls=SAC,
)

result = run_inference(
    df_raw=df_raw,
    system_config=system_config,  # from SiteConfig.to_env_dict()
    model=model,
    scalers=scalers,
    obs_rms=obs_rms,
    initial_soc=0.6,  # real battery level from BMS
)
# result = {'dispatch_plan': [...96 dicts...], 'summary': {...}}
```

**`dispatch_plan`** per step includes: `step`, `action_battery`, `action_grid`, `soc`, `target_soc`, `solar_gen_kwh`, `solar_surplus_kwh`, `battery_kwh`, `grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward`, all 9 reward components.

**Model cache**: In FastAPI, `prediction_service.py` holds a single `_cached_model: tuple | None` — the global SAC model is loaded once from `best/best_model.zip` and reused for all configs. No per-config training or per-config model files.

**Standalone `__main__` mode** (`python envoriment/inference.py`):
- Calls `data_combiner.combine()` for live data; falls back to `combined.csv` if DAM unavailable
- `--soc` arg is optional; if omitted, queries DB for last prediction SoC (clamped to `min_reserve`); defaults to 0.5 if DB unavailable
- Args: `--model`, `--scalers`, `--obsrms`, `--config` (JSON), `--output`, `--soc`, `--tilt`, `--azimuth`

---

## Default Strategy (`envoriment/default_strategy.py`)

Rule-based inverter dispatch with no price awareness — reacts only to solar irradiance, grid status, and SoC.

### Configurable parameters (`DefaultStrategyConfig` / `DEFAULT_STRATEGY` dict)

| Parameter | Default | Description |
|---|---|---|
| `target_soc` | 0.70 | Charge battery to this SoC before exporting surplus |
| `max_soc` | 0.95 | Top-up ceiling when solar is abundant |
| `min_solar_threshold` | 10.0 W/m² | Below this GTI = no meaningful generation |
| `high_solar_threshold` | 400.0 W/m² | Above this GTI = abundant solar |
| `solar_surplus_priority` | `charge_first` | `charge_first`: fill to max_soc then export; `sell_first`: export at target_soc |
| `night_discharge` | True | Discharge battery at night to cover load |
| `night_sell` | False | Also export battery energy to grid at night |
| `allow_grid_charging` | False | Buy from grid to charge when SoC < target and no solar |
| `outage_reserve` | 0.0 | Minimum SoC to preserve during grid outages |

`generate_dispatch_plan(df_raw, df_norm, config, initial_soc, strategy) → dict` returns the same `{'dispatch_plan': [...], 'summary': {...}}` shape as `run_inference()`.

---

## Data Pipeline

### `data_combiner.py`

```python
combine(config_id, tilt=None, azimuth=None) → pd.DataFrame  # 96 rows × 25 cols
```

Returns `None` if DAM fetch fails. Always check for None before passing to inference. No SQLite cache — uses plain `requests.Session()`.

### `synthetic_grid.py` — Monthly outage averages (h/day)

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.50 | 3.00 | 1.50 | 0.50 | 0.20 | 0.10 | 0.21 | 0.25 | 0.40 | 2.07 | 3.12 | 6.68 |

---

## Backend API

### Stack
- **FastAPI** + **SQLAlchemy** + **PostgreSQL**
- **CORS** enabled (`allow_origins=["*"]`) — required for any browser frontend
- **JWT** (PyJWT, HS256, 30-min expiry) + **pwdlib** (Argon2 password hashing)

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | No | Health check |
| POST | `/auth/register` | No | Create user |
| POST | `/auth/login` | No | Get JWT token |
| POST | `/config/` | JWT | Save/update `SiteConfig` (upsert by name) |
| GET | `/config/` | JWT | Retrieve named config |
| GET | `/config/list` | JWT | List all configs for current user |
| POST | `/strategy/` | JWT | Save/update `DefaultStrategyConfig` (upsert by name) |
| GET | `/strategy/` | JWT | Retrieve named strategy |
| GET | `/strategy/list` | JWT | List all strategies for current user |
| GET | `/predictions/` | JWT | Run SAC inference + store results. Params: `config_name`, `initial_soc`. Blocked before 14:00 UA time. |
| GET | `/predictions/default` | JWT | Run default strategy + return results (not stored). Params: `config_name`, `strategy_name`, `initial_soc`. Blocked before 14:00 UA time. |

### Response Models (Pydantic)

Both prediction endpoints return `PredictionResponse`:

```python
class DispatchStep(BaseModel):
    timestamp: str
    soc: float;  target_soc: float
    solar_kwh: float;  load_kwh: float
    battery_kwh: float;  grid_kwh: float
    unmet_load_kwh: float;  money_earned_ts: float
    dam_price: float;  grid_status: int;  hours_until_outage: float

class DispatchSummary(BaseModel):
    total_money_earned: float;  bought_kwh: float;  sold_kwh: float
    solar_kwh: float;  unmet_load_kwh: float;  lcos_total_uah: float
    initial_soc: float;  final_soc: float;  steps: int

class PredictionResponse(BaseModel):
    summary: DispatchSummary
    dispatch_plan: list[DispatchStep]
```

Reward components and raw RL actions are stripped from the API response (stored in DB only).

### SiteConfig Schema → `to_env_dict()` mapping

```python
SiteConfig.to_env_dict() → {
    'battery': { capacity_kwh, min_reserve, lcos, max_charge_power, max_discharge_power, efficiency },
    'solar':   { peak_power, efficiency },
    'inverter':{ max_power },
    'grid':    { capacity },
}
```
Buy price is always `DAM_Price/1000 + 3.0` computed dynamically in `step()` — no `price_to_buy` field.

### Database Models

| Table | Key columns |
|---|---|
| `users` | id, username, email, hashed_password |
| `system_configs` | id, user_id (FK), config_name, settings (JSONB) |
| `default_strategies` | id, user_id (FK), strategy_name, settings (JSONB) |
| `history` | id, user_id (FK), timestamp, data (JSONB) |
| `predictions` | id, user_id (FK), config_id (FK), date, step, timestamp + 30 physics/reward cols |

`predictions` keyed by `config_id + date` — multiple configs per user don't collide.

### Prediction Service details

- **Global model cache**: `_cached_model: tuple | None` — SAC model loaded once from `environment/models/best/best_model.zip`, reused for all requests. No per-config training or model files.
- **Timezone**: gate and date calculations use `UTC+2` (Ukrainian time)
- **Paths**: absolute paths anchored to `Path(__file__).resolve().parent.parent.parent`
- **`initial_soc`**: if omitted, read from DB (last step SoC of previous prediction for this config), clamped to `min_reserve`
- **SAC predictions** (`GET /predictions/`): result stored to `predictions` table (96 rows per run)
- **Default strategy** (`GET /predictions/default`): result returned only, not stored in DB

---

## Known Issues & Open Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active |
| 2 | `requirements.txt` | ✅ Fixed | All deps present: `scikit-learn`, `psycopg2-binary`, `openpyxl`, `uvicorn` |
| 3 | Real SoC input | ⚠️ Manual | `initial_soc` auto-persisted via DB (API) but no live BMS/inverter integration |
| 4 | No `GET /predictions/history` | ❌ Open | SAC predictions stored in DB but no endpoint to retrieve past days |
| 5 | SAC training in progress | ⚠️ Training | Model needs full 10M-step run with current env fixes (reward normalization, day-aligned resets, soc_hard_min, 16-step lookahead). Predictions work but quality improves after retraining. |
| 6 | `data_combiner.py` | ✅ Fixed | `get_solar_parameters()` now has try/except — no longer crashes when DB is down |
| 7 | `inference.py __main__` | ✅ Fixed | Project root added to `sys.path` so `data_providers` import works when running directly |
| 8 | `inference.py __main__` | ✅ Fixed | NaN guard added — aborts with clear error if DAM columns are all NaN instead of crashing PyTorch |
| 9 | `combined.csv` | ⚠️ External | DAM_Price/Vol columns are NaN when OREE fetch fails — inference will abort cleanly but needs live DAM data to run |

---

## Python Dependencies

Current `requirements.txt` (complete — no missing deps):
```
fastapi, uvicorn, pandas, numpy, requests, openmeteo-requests, requests-cache,
retry-requests, gymnasium, python-calamine, sqlalchemy, psycopg2-binary, pyjwt,
python-dotenv, pwdlib, pydantic, scikit-learn, stable-baselines3, torch,
tensorboard, openpyxl
```

---

## AWS Deployment Architecture

### Decisions made

- **Database**: Amazon RDS (PostgreSQL 16) — not S3. S3 is object storage (no SQL queries, no transactions, no foreign keys). RDS is identical to local PostgreSQL — only `DATABASE_URL` in `.env` changes, zero code changes required.
- **Application hosting**: EC2 running Docker containers.
- **Target production path**: ECR + ECS (see below). For initial demo: `git clone` on EC2 + `docker-compose up`.
- **API Gateway**: planned in front of EC2 for CORS handling, SSL, and rate limiting. Not yet set up.
- **CORS**: `CORSMiddleware` with `allow_origins=["*"]` kept in `backend/main.py` until API Gateway is in place. Once API Gateway is configured, remove `CORSMiddleware` entirely — API Gateway handles it at the AWS level.

### Target architecture

```
Browser
    │
    ▼
API Gateway  ← handles CORS, SSL, rate limiting
    │
    ▼
EC2 (FastAPI + Docker)
    │  ├── RDS PostgreSQL (private VPC, port 5432)
    │  ├── OREE (oree.com.ua) — DAM prices
    │  └── Open-Meteo — weather forecast
```

### Production deployment path (ECR + ECS)

```
Local machine                  AWS
──────────────                 ──────────────────────────
docker build .      →push→     ECR  (image registry)
                                    │
                               ECS  (runs containers, replaces docker-compose)
                                    │
                               RDS  (PostgreSQL)
```

- **ECR** = AWS Docker image registry (like Docker Hub but private)
- **ECS** = AWS container orchestrator (replaces `docker-compose` in production)

Commands (once ECR repo is created):
```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t ems-api .
docker tag ems-api:latest <account>.dkr.ecr.<region>.amazonaws.com/ems-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/ems-api:latest
```

### RDS instance settings

| Setting | Value | Notes |
|---|---|---|
| Engine | PostgreSQL 16 | |
| Instance | `db.t3.micro` | ~$13/month; free tier eligible |
| Storage | 20 GB gp2 | More than enough for predictions table |
| Multi-AZ | No | Overkill for demo |
| Public access | No | Private VPC only |
| Security group | Allow 5432 from EC2 SG only | Never expose RDS to internet |

### What changes when going from local → AWS

| Item | Local | AWS |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:pass@localhost:5433/postgres` | `postgresql://user:pass@<rds-endpoint>:5432/dbname` |
| DB tables | Created by `Base.metadata.create_all()` on startup | Same — auto-created on first `docker-compose up` / ECS start |
| Model files | Local disk (tracked in git, 3.3 MB) | On EC2 after `git clone`; optionally move to S3 for easier updates |
| App process | `docker-compose up` | ECS task / EC2 docker-compose |

### Files added for deployment

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the FastAPI app image (CPU torch, python 3.13-slim) |
| `.dockerignore` | Excludes `.venv`, logs, datasets from build context |
| `docker-compose.yml` | Updated: app service added, MinIO removed (not used in code) |
| `.env.example` | Documents all required env vars; use `db` as DB host inside docker-compose |

### Local dev vs docker-compose DATABASE_URL

- **Running locally** (outside Docker): `DATABASE_URL=postgresql://postgres:pass@localhost:5433/postgres`
- **Running via docker-compose**: `DATABASE_URL=postgresql://postgres:pass@db:5432/postgres` — use service name `db`, internal port `5432` (not the mapped `5433`)

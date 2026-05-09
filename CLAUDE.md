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
│   ├── dataset_final.csv                  # Copy of training dataset (25 cols, 35 041 rows)
│   ├── dataset_normalized.csv             # Normalized training dataset (17 cols, used by train.py)
│   ├── models/
│   │   ├── sac_ems.zip                    # Final SAC model
│   │   ├── best/best_model.zip            # Best checkpoint by eval reward
│   │   ├── checkpoints/                   # Periodic checkpoints (every 50k steps)
│   │   ├── scalers.pkl                    # sklearn scalers fitted on training data
│   │   └── obs_rms.pkl                    # VecNormalize running stats (required for inference)
│   └── logs/
│       ├── tensorboard/SAC_*/             # TensorBoard event files
│       ├── monitor/                       # SB3 Monitor CSV logs
│       └── eval/                          # EvalCallback logs
├── data_providers/
│   ├── orchestrator/
│   │   ├── data_combiner.py               # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   ├── combined.csv                   # Last generated live dataset (96 rows × 25 cols)
│   │   └── .cache.sqlite                  # Open-Meteo API cache
│   └── components/
│       ├── market_manager/IDM_DAM_features.py   # Fetches DAM prices from OREE (oree.com.ua)
│       ├── weather/weather.py             # Open-Meteo forecast (GTI, temp, radiation)
│       ├── grid/synthetic_grid.py         # Synthetic outage schedule generator
│       ├── load/synthetic_load.py         # Synthetic consumption profile
│       └── time/time_features.py          # Cyclical time encoding (sin/cos)
├── datasets/
│   └── dataset_v10/                       # CURRENT training dataset
│       └── dataset_final.csv              # 35 041 rows × 25 cols — column-aligned with live pipeline ✅
├── backend/
│   ├── main.py                            # FastAPI app entry point (CORS enabled)
│   ├── trained_models/                    # Per-config PPO models saved locally as {config_id}.zip
│   ├── core/
│   │   ├── database.py                    # SQLAlchemy engine + session (PostgreSQL)
│   │   ├── loader.py                      # Bulk-uploads DataFrame → History table
│   │   └── trainer.py                     # PPO training via SB3; saves locally; writes AgentModels status
│   ├── models/site.py                     # ORM: User, SystemConfig, History, AgentPredictions, AgentModels
│   ├── repositories/
│   │   ├── config_repo.py                 # SystemConfig CRUD + upsert
│   │   ├── model_repo.py                  # AgentModels queries
│   │   ├── prediction_repo.py             # AgentPredictions bulk insert / delete by config_id
│   │   └── user_repo.py                   # User CRUD
│   ├── routers/
│   │   ├── auth.py                        # POST /auth/login, POST /auth/register
│   │   ├── config.py                      # POST /config/, POST /config/train, GET /config/, GET /config/list
│   │   └── predictions.py                 # GET /predictions/?config_name=&initial_soc= ✅
│   ├── schemas/schemas.py                 # Pydantic: SiteConfig, Battery, Inverter, SolarPanel, Grid, User
│   ├── services/
│   │   ├── auth_service.py                # login / register logic
│   │   ├── config_service.py              # save / get / list / trigger_training logic
│   │   └── prediction_service.py          # full prediction pipeline with model cache
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
| Electricity tariff | **5.5 UAH/kWh** fixed (Ukrainian regulated residential rate) |
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
action_space      = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
observation_space = Box(low=-inf, high=inf, shape=(n_features + 1,), dtype=float32)
# 17 normalized feature cols + SoC → shape (18,)
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge |
| `action[1]` | Grid: −1 = full export, +1 = full import |

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
        'price_to_buy': float,          # UAH/kWh fixed tariff
    },
}
```

`max_grid_capacity = min(inverter.max_power, grid.capacity)` — whichever is smaller is the real limit.

Charge and discharge use **separate** power limits (`max_batt_charge_power_ts` and `max_batt_discharge_power_ts`).

### Reward Function (7 components, all tracked separately in `info`)

```
r_market      = grid_import × (-price_to_buy)  OR  |grid_export| × DAM_price/1000
r_lcos        = -(lcos × |batt_energy_cycled|)
r_unmet       = -(unmet_load × price_to_buy × 2)        # if unmet_load > 0
r_mismatch    = -(2.0 × |grid_commanded - grid_actual|)
r_soc_soft    = -(3.0 × violation²)                     # outside [soc_min, 0.80]
r_reserve     = -(30.0 × soc_deficit × log1p(outage_remaining_h))  # during outage
r_preparation = 5.0 × exp(-0.5 × hours_until_outage) × min(soc, target_soc)  # pre-outage

reward = sum of all 7 components
```

### `info` dict (returned by `step()`)

`soc`, `target_soc`, `reward`, `solar_gen_ts_kwh`, `solar_surplus_kwh`, `actual_grid_kwh`, `battery_kwh` (+ = charging), `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward_market`, `reward_lcos`, `reward_unmet`, `reward_mismatch`, `reward_soc_soft`, `reward_reserve`, `reward_preparation`

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

Run from anywhere (auto `chdir` to `envoriment/`):

```bash
python envoriment/train.py
```

### Key design decisions

**Domain randomization** via `RandomConfigWrapper`: on every `reset()` a new correlated hardware config is sampled. Hardware is sized realistically:
```
capacity  = uniform(50, 500) kWh
solar     = capacity × uniform(0.8, 2.0) kWp
inverter  = solar × uniform(0.8, 1.1) kW
grid      = inverter × uniform(1.0, 1.5) kW   ← always ≥ inverter
charge/discharge power = capacity / 2          ← C/2 rate
price_to_buy = 5.5 UAH/kWh                    ← fixed (Ukrainian regulated tariff)
```

**Per-month 75/25 split**: for each of the 12 months, first 75% of rows → train, last 25% → eval. All seasons represented in both sets. No seasonal bias.

**VecNormalize**: obs normalized with `clip_obs=10.0`; reward normalized on train env, raw on eval env.

**Eval env**: fixed `DEFAULT_SYSTEM_CONFIG` (200 kWh mid-range) for stable training progress tracking.

### Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC (MlpPolicy) |
| Total timesteps | 1 000 000 |
| Buffer size | 500 000 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Network arch | [256, 256] |
| Gamma | 0.99 |
| Entropy coef | auto |
| Checkpoint frequency | every 50 000 steps |
| Eval frequency | every 50 000 steps, 1 episode |
| Seed | 42 (numpy + SAC) |

### Outputs

| Path | Contents |
|---|---|
| `models/sac_ems.zip` | Final model |
| `models/best/best_model.zip` | Best checkpoint by eval reward |
| `models/checkpoints/` | Periodic checkpoints |
| `models/obs_rms.pkl` | VecNormalize running stats — **required for inference** |
| `logs/tensorboard/` | TensorBoard event files |

View training: `tensorboard --logdir envoriment/logs/tensorboard/`

---

## Inference Module (`envoriment/inference.py`)

```python
from envoriment.inference import load_model_and_scalers, run_inference

model, scalers, obs_rms = load_model_and_scalers(
    model_path='envoriment/models/best/best_model',
    scalers_path='envoriment/models/scalers.pkl',
    obs_rms_path='envoriment/models/obs_rms.pkl',
)

result = run_inference(
    df_raw=df_raw,
    system_config=system_config,   # from SiteConfig.to_env_dict()
    model=model,
    scalers=scalers,
    obs_rms=obs_rms,
    initial_soc=0.6,               # real battery level from BMS
)
# result = {'dispatch_plan': [...96 dicts...], 'summary': {...}}
```

**`dispatch_plan`** per step includes: `step`, `action_battery`, `action_grid`, `soc`, `target_soc`, `solar_gen_kwh`, `solar_surplus_kwh`, `battery_kwh`, `grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward`, `reward_market`, `reward_lcos`, `reward_unmet`, `reward_mismatch`, `reward_soc_soft`, `reward_reserve`, `reward_preparation`

**Model cache**: `load_model_and_scalers()` must be called once at startup. In FastAPI, `prediction_service.py` caches models by `config_id` in `_model_cache` dict.

---

## Data Pipeline

### `data_combiner.py`

```python
combine(config_id, tilt=None, azimuth=None) → pd.DataFrame  # 96 rows × 25 cols
```

Returns `None` if DAM fetch fails. Always check for None before passing to inference.

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
| POST | `/config/` | JWT | Save/update `SiteConfig` (upsert by name — no training triggered) |
| POST | `/config/train` | JWT | Trigger PPO training for config (skips if model already ready) |
| GET | `/config/` | JWT | Retrieve named config |
| GET | `/config/list` | JWT | List all configs for current user |
| GET | `/predictions/` | JWT | Run inference + store results. Params: `config_name`, `initial_soc` (0–1, default 0.5). Blocked before 14:00 UA time. |

### SiteConfig Schema → `to_env_dict()` mapping

```python
SiteConfig.to_env_dict() → {
    'battery': { capacity_kwh, min_reserve, lcos, max_charge_power, max_discharge_power, efficiency },
    'solar':   { peak_power, efficiency },
    'inverter':{ max_power },
    'grid':    { capacity, price_to_buy },   # price_to_buy ← Grid.price_buy_from_grid
}
```

### Database Models

| Table | Key columns |
|---|---|
| `users` | id, username, email, hashed_password |
| `system_configs` | id, user_id (FK), config_name, settings (JSONB) |
| `history` | id, user_id (FK), timestamp, data (JSONB) |
| `predictions` | id, user_id (FK), **config_id (FK)**, date, step, timestamp + 30 physics/reward cols |
| `agent_models` | id, config_id (FK), status, **algorithm**, trained_at, total_timesteps, storage_path |

`predictions` keyed by `config_id + date` (not `user_id + date`) — multiple configs per user don't collide.

`agent_models.algorithm` stores `"PPO"` or `"SAC"` — used by prediction_service to load correct class.

### Prediction Service details

- **Model cache**: `_model_cache: dict[int, tuple]` keyed by `config_id` — model loaded from disk once, reused forever
- **Timezone**: gate and date calculations use `UTC+2` (Ukrainian time)
- **Paths**: absolute paths anchored to `Path(__file__).resolve().parent.parent.parent`
- **`initial_soc`**: passed as query param `GET /predictions/?initial_soc=0.6`, validated `[0.0, 1.0]`
- **All 30+ `AgentPredictions` columns populated** including reward breakdown and battery/solar flows

### Backend Trainer (`backend/core/trainer.py`)

Triggered by `POST /config/train` only if no ready model exists. Uses **PPO** (200k steps). Separate from the standalone SAC trainer — not unified.

---

## Known Issues & Open Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active |
| 2 | Backend vs standalone training | ⚠️ Diverged | `backend/core/trainer.py` uses PPO 200k steps; `envoriment/train.py` uses SAC 1M steps + VecNormalize + domain randomization. Not unified. |
| 3 | `requirements.txt` | ⚠️ Partial | Missing: `scikit-learn`, `psycopg2-binary`, `openpyxl`, `uvicorn` |
| 4 | Real SoC input | ⚠️ Manual | `initial_soc` must be passed manually — no BMS/inverter integration |
| 5 | No `GET /predictions/history` | ❌ Open | Predictions stored in DB but no endpoint to retrieve them without re-running inference |

---

## Python Dependencies

Current `requirements.txt`:
```
fastapi, pandas, numpy, requests, openmeteo-requests, requests-cache,
retry-requests, gymnasium, python-calamine, sqlalchemy, pyjwt,
python-dotenv, pwdlib, pydantic, stable-baselines3, torch, tensorboard, boto3
```

Missing (add before running):
```
scikit-learn       # normalize.py
psycopg2-binary    # PostgreSQL driver
openpyxl           # OREE Excel parsing
uvicorn            # FastAPI server
```

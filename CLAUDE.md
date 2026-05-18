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
│   ├── default_strategy.py                # Baseline inverter dispatch (no price awareness)
│   ├── models/
│   │   ├── sac_ems.zip                    # Final SAC model
│   │   ├── best/best_model.zip            # Best checkpoint by eval reward
│   │   ├── checkpoints/                   # Periodic checkpoints (every 100k steps)
│   │   ├── scalers.pkl                    # sklearn scalers fitted on training data
│   │   └── obs_rms.pkl                    # VecNormalize running stats (required for inference)
│   ├── results/
│   │   ├── dispatch_plan.csv              # Last inference output
│   │   └── last_soc.txt                   # Final SoC from last inference run (persisted across days)
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
| Sell price | **DAM_Price** UAH/kWh (dynamic, from OREE day-ahead market) |
| Buy price | **DAM_Price + 3.0 UAH/kWh** (DAM + grid access tax) |
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

### Hard SoC Floor (BMS protection)

When `grid_status == 1` (grid is up), the environment enforces a hard discharge floor at `soc_soft_min`. The agent physically cannot drain below this reserve. During outages (`grid_status == 0`), full discharge to 0 is allowed.

This is implemented in Block 2 (discharge):
```python
if grid_status == 1:
    max_drawable = max(0.0, (self.soc - self.soc_soft_min) * self.max_batt_capacity)
else:
    max_drawable = self.soc * self.max_batt_capacity
```

### Reward Function (9 components, all tracked separately in `info`)

```
r_market      = grid_import × (-(DAM_price/1000 + 3.0))  OR  |grid_export| × DAM_price/1000
r_lcos        = -(lcos × |batt_energy_cycled|)
r_unmet       = -(unmet_load × (DAM_price/1000 + 3.0) × 2)       # if unmet_load > 0
r_mismatch    = -(2.0 × |grid_commanded - grid_actual|)           # ONLY when net_demand==0 AND grid is up
r_soc_soft    = -(50.0 × violation²)                              # outside [soc_min, 0.80]
r_reserve     = -(30.0 × soc_deficit × log1p(outage_remaining_h)) # during outage
r_preparation = 5.0 × exp(-0.5 × hours_until_outage) × min(soc, target_soc)  # pre-outage
r_soc_target  = actual_chem_in × buy_price                        # when pre_charge_soc < target_soc
                                                                  # = avoided future purchase cost
r_waste       = -(2.0 × lcos × wasted_kWh)                       # discharge that exceeds demand + available grid headroom

reward = sum of all 9 components
```

**r_soc_target design rationale**: rewards storing energy at its avoidance value (`buy_price` per kWh stored), making charging toward the reserve target economically competitive with selling solar surplus. At midday buy_price ≈ 9 UAH/kWh: charging earns ~+9×kWh while r_lcos pays -1.5×kWh, net +7.5 UAH/kWh — stronger signal than solar spot selling.

**r_mismatch scope**: only fires when `net_demand_after_batt < 1e-6` — i.e. when the agent actually controls the grid outcome. Skipped when environment must force grid import to cover unmet load (the agent's grid action is irrelevant there).

**r_waste**: penalizes discharging more than demand + available grid export headroom can absorb. `wasted = batt_output - demand_covered - min(batt_surplus, grid_capacity - solar_surplus)`. Both solar surplus and battery surplus share the same grid capacity.

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

**Domain randomization** via `RandomConfigWrapper`: on every `reset()` a new correlated hardware config is sampled. Hardware is sized realistically:
```
capacity  = uniform(50, 500) kWh
solar     = capacity × uniform(0.8, 2.0) kWp
inverter  = solar × uniform(0.8, 1.1) kW
grid      = inverter × uniform(1.0, 1.5) kW   ← always ≥ inverter
charge/discharge power = capacity / 2          ← C/2 rate
```
Buy price is always computed dynamically as `DAM_Price/1000 + 3.0` — there is no `price_to_buy` config field.

**Per-month 75/25 split**: for each of the 12 months, first 75% of rows → train, last 25% → eval. All seasons represented in both sets. No seasonal bias.

**VecNormalize**: obs normalized with `clip_obs=10.0`; reward normalized on train env, raw on eval env.

**SyncNormalizeEvalCallback**: before each eval run, deep-copies `train_env.obs_rms` → `eval_env.obs_rms` so both use the same running stats. Without this, the eval Q-function evaluates against a different observation distribution than it was trained on.

**Eval env**: fixed `DEFAULT_SYSTEM_CONFIG` (200 kWh mid-range) for stable training progress tracking.

**reset() SoC randomization**: `self.soc = uniform(0.0, 1.0)` on each episode reset — ensures the agent sees all SoC levels during training, not just SoC=0.

**Callback freq scaling**: `save_freq = checkpoint_freq // n_envs` — SB3 callback `_on_step()` fires once per `n_envs` environment steps, so dividing keeps checkpoints at the intended absolute step count.

### Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC (MlpPolicy) |
| Total timesteps | 5 000 000 |
| Buffer size | 1 000 000 |
| Batch size | 512 |
| Learning rate | 3e-4 |
| Network arch | [512, 512] |
| n_envs | 32 (DummyVecEnv) |
| Gamma | 0.99 |
| Entropy coef | auto |
| Checkpoint frequency | every 100 000 env steps |
| Eval frequency | every 100 000 env steps, 5 episodes |
| Seed | 42 (numpy + SAC) |
| Device | cuda |

**DummyVecEnv chosen over SubprocVecEnv**: env step = 0.11 ms, SubprocVecEnv pipe overhead = ~13 ms → DummyVecEnv is ~7× faster on this hardware.

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
from environment.inference import load_model_and_scalers, run_inference

model, scalers, obs_rms = load_model_and_scalers(
    model_path='environment/models/best/best_model',
    scalers_path='environment/models/scalers.pkl',
    obs_rms_path='environment/models/obs_rms.pkl',
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

**`dispatch_plan`** per step includes: `step`, `action_battery`, `action_grid`, `soc`, `target_soc`, `solar_gen_kwh`, `solar_surplus_kwh`, `battery_kwh`, `grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward`, `reward_market`, `reward_lcos`, `reward_unmet`, `reward_mismatch`, `reward_soc_soft`, `reward_reserve`, `reward_preparation`, `reward_soc_target`, `reward_waste`

**Model cache**: `load_model_and_scalers()` must be called once at startup. In FastAPI, `prediction_service.py` caches models by `config_id` in `_model_cache` dict.

**Standalone `__main__` mode** (`python envoriment/inference.py`):
- Calls `data_combiner.combine()` for live data; falls back to `combined.csv` if DAM unavailable or error
- `--soc` arg is optional; if omitted reads `envoriment/results/last_soc.txt`, clamped to `min_reserve`; defaults to 0.5 if no file exists
- Saves final SoC to `last_soc.txt` after each run so the next day starts from real battery state
- Args: `--model`, `--scalers`, `--obsrms`, `--config` (JSON), `--output`, `--soc`, `--tilt`, `--azimuth`

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
    'grid':    { capacity },
}
```
`price_to_buy` was removed — buy price is always `DAM_Price/1000 + 3.0` computed dynamically in `step()`.

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
- **`initial_soc`**: passed as query param `GET /predictions/?initial_soc=0.6`, validated `[0.0, 1.0]`; if omitted, read from DB (last step SoC of previous prediction for this config), clamped to `min_reserve` so an outage-drained battery doesn't cascade into the next day
- **All 30+ `AgentPredictions` columns populated** including reward breakdown and battery/solar flows

### Backend Trainer (`backend/core/trainer.py`)

Triggered by `POST /config/train` only if no ready model exists. Uses **PPO** (200k steps). Separate from the standalone SAC trainer — not unified.

---

## Known Issues & Open Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active |
| 2 | Backend vs standalone training | ⚠️ Diverged | `backend/core/trainer.py` uses PPO 200k steps; `envoriment/train.py` uses SAC 5M steps + VecNormalize + domain randomization. Not unified. |
| 3 | `requirements.txt` | ⚠️ Partial | Missing: `scikit-learn`, `psycopg2-binary`, `openpyxl`, `uvicorn` |
| 4 | Real SoC input | ⚠️ Manual | `initial_soc` auto-persisted via `last_soc.txt` (standalone) and DB (API), but no live BMS/inverter integration |
| 5 | No `GET /predictions/history` | ❌ Open | Predictions stored in DB but no endpoint to retrieve them without re-running inference |
| 6 | Outage over-discharge | ⚠️ Training | Model over-discharges during outages (wastes energy). SoC clamp prevents next-day cascade. Should improve after full 5M-step training via `r_waste` / `r_reserve` penalties. |

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

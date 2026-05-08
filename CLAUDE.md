# Scalable AI Energy Management System (EMS) — Project Context

## Project Goal

Build an autonomous RL agent that optimally manages energy flows in a hybrid node (Solar PV + BESS + Grid) to:
1. **Maximize profit** from energy arbitrage on the Ukrainian Day-Ahead Market (DAM) and Intraday Market (IDM).
2. **Guarantee energy autonomy** during planned/predicted power outages by maintaining a sufficient State of Charge (SoC) reserve.
3. **Minimize battery degradation** by incorporating Levelized Cost of Storage (LCOS) into the reward function.
4. **Scale to any hardware** (PV capacity, BESS size, inverter power) via per-user `SystemConfig` stored in the DB — each config gets its own trained model.

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
│   ├── models/                            # Trained model artifacts
│   │   ├── sac_ems.zip                    # Final SAC model
│   │   ├── best/best_model.zip            # Best checkpoint by eval reward
│   │   ├── checkpoints/                   # Periodic checkpoints (every 50k steps)
│   │   └── scalers.pkl                    # sklearn scalers fitted on training data
│   └── logs/
│       ├── tensorboard/SAC_*/             # TensorBoard event files (runs SAC_4 … SAC_18+)
│       ├── monitor/                       # SB3 Monitor CSV logs
│       └── eval/                          # EvalCallback logs
├── data_providers/
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── data_combiner.py               # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   ├── combined.csv                   # Last generated live dataset (96 rows × 25 cols)
│   │   └── .cache.sqlite                  # Open-Meteo API cache
│   ├── agent_data_preprocessing/
│   │   └── __init__.py                    # Normalization now lives in envoriment/normalize.py
│   └── components/
│       ├── __init__.py
│       ├── market_manager/
│       │   ├── __init__.py
│       │   └── IDM_DAM_features.py        # Fetches DAM prices from OREE (oree.com.ua); IDM not implemented
│       ├── weather/
│       │   ├── __init__.py
│       │   └── weather.py                 # Open-Meteo forecast (GTI, temp, radiation)
│       ├── grid/
│       │   ├── __init__.py
│       │   └── synthetic_grid.py          # Synthetic outage schedule generator
│       ├── load/
│       │   ├── __init__.py
│       │   └── synthetic_load.py          # Synthetic consumption profile
│       └── time/
│           ├── __init__.py
│           └── time_features.py           # Cyclical time encoding (sin/cos)
├── datasets/
│   ├── datasets_v0/ … datasets_v8/        # Historical versions (kept for reference)
│   ├── datasets_v9/
│   │   ├── dataset.csv                    # 35 041 rows × 27 cols (legacy — IDM columns, outage_risk; not aligned with live pipeline)
│   │   ├── sorted.csv                     # Sorted version of dataset.csv
│   │   ├── build_v10.py                   # Build script used to produce v10
│   │   ├── checkkkk.py                    # Validation script
│   │   └── sort_dataset.py                # Sorting utility
│   └── dataset_v10/                       # CURRENT training dataset
│       ├── dataset_final.csv              # 35 041 rows × 25 cols — column-aligned with live pipeline ✅
│       ├── dataset.csv                    # Intermediate (has Day_sin/Day_cos, still has IDM cols)
│       ├── dataset_no_idm.csv             # Intermediate (IDM cols dropped)
│       ├── build_v10.py                   # Step 1: adds Day_sin/Day_cos to v9 sorted
│       ├── drop_idm.py                    # Step 2: drops IDM_* and Avg_Weighted_Price
│       └── drop_dam_vol.py                # Step 3: reorders columns, adds timestamp → dataset_final.csv
├── backend/
│   ├── __init__.py
│   ├── main.py                            # FastAPI app entry point
│   ├── trained_models/                    # Per-config PPO models saved locally as {config_id}.zip
│   ├── core/
│   │   ├── database.py                    # SQLAlchemy engine + session (PostgreSQL)
│   │   ├── loader.py                      # Bulk-uploads DataFrame → History table
│   │   └── trainer.py                     # PPO training via SB3; saves locally; writes AgentModels status
│   ├── models/
│   │   └── site.py                        # ORM: User, SystemConfig, History, AgentPredictions, AgentModels
│   ├── repositories/                      # Data-access layer (DB queries only, no business logic)
│   │   ├── __init__.py
│   │   ├── config_repo.py                 # SystemConfig CRUD
│   │   ├── model_repo.py                  # AgentModels queries
│   │   ├── prediction_repo.py             # AgentPredictions bulk insert / delete
│   │   └── user_repo.py                   # User CRUD
│   ├── routers/
│   │   ├── auth.py                        # POST /auth/login, POST /auth/register
│   │   ├── config.py                      # POST /config/ (+ background training), GET /config/, GET /config/list
│   │   └── predictions.py                 # GET /predictions/?config_name=… — fully wired ✅
│   ├── schemas/
│   │   └── schemas.py                     # Pydantic: SiteConfig (with to_env_dict()), Battery, Inverter, SolarPanel, Grid, User
│   ├── services/                          # Business logic layer (orchestrates repos + external calls)
│   │   ├── __init__.py
│   │   ├── auth_service.py                # login / register logic
│   │   ├── config_service.py              # save / get / list config logic
│   │   └── prediction_service.py          # full prediction pipeline (data fetch → inference → DB write)
│   └── security/
│       └── security.py                    # JWT (HS256), pwdlib Argon2 password hashing
└── temp/                                  # One-off data-cleaning utility scripts (not part of main pipeline)
    ├── add_column.py.py
    ├── clear_DAM_and_IDM.py
    ├── clear_indexs.py
    ├── clear_weather.py
    ├── concatenate_files.py
    ├── get_uah_prices_IDM.py
    ├── final_full_dataset_DAM.csv
    └── final_full_dataset_IDM.csv
```

---

## Timestep & Market Conventions

| Parameter | Value |
|---|---|
| Timestep | **15 minutes** (96 steps/day) |
| Market | Ukrainian **DAM** via [oree.com.ua](https://www.oree.com.ua) |
| Price unit | UAH / MWh |
| Site location | Lat **48.2904** °N, Lon **25.9324** °E (Chernivtsi, Western Ukraine) |
| Horizon | Agent operates on **next-day** data assembled each evening |

DAM data is fetched for `today + 1 day` from:
```
https://www.oree.com.ua/index.php/PXS/downloadxlsx/{DD.MM.YYYY}/DAM/2
```
Hourly rows are expanded ×4 to match 15-min timesteps.

---

## RL Environment (`envoriment/environment.py`)

### Dual-Dataset Architecture

The environment takes **two DataFrames** at construction:

| Argument | Purpose |
|---|---|
| `df_raw` | Raw (unnormalized) dataset — used in `step()` for physics calculations (prices, loads, grid status) |
| `df` | Normalized dataset — used in `get_observe()` to feed the neural network |

This separation means the agent sees normalized observations while the physics engine runs on real-scale values.

### Spaces

```python
action_space      = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
observation_space = Box(low=-inf, high=inf, shape=(n_features + 1,), dtype=float32)
  # n_features = df.shape[1] (normalized df, after dropping 8 cols via normalize.py)
  # With dataset_normalized.csv (17 feature cols) → shape=(18,)
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge (scaled by `max_batt_power / 4`) |
| `action[1]` | Grid exchange: −1 = full export, +1 = full import (note: `grid_power_ts = -action[1] × max_grid_capacity_ts`) |

### Hardware Parameters (from `system_config` dict)

All hardware parameters are loaded from the `system_config` dict passed at construction — nothing is hardcoded.

| `system_config` key | Environment attribute | Notes |
|---|---|---|
| `battery.capacity_kwh` | `max_batt_capacity` | kWh |
| `battery.max_charge_power` | `max_batt_power` | kW; discharge limited by SoC |
| `battery.efficiency` | `batt_efficiency` | Round-trip per half-cycle |
| `battery.lcos` | `lcos` | UAH/kWh degradation cost |
| `battery.min_reserve` | `soc_soft_min` | Percent → fraction (÷100) |
| `inverter.max_power` | `max_grid_capacity` | kW; also limits grid import/export |
| `inverter.price_to_buy` | `price_to_buy` | UAH/kWh fixed tariff for buying from grid |
| `solar.peak_power` | `solar_peak_power_kw` | kWp |
| `solar.efficiency` | `solar_efficiency` | η; panel area derived as `peak_power/(1000×η)` |

Fixed constants: `soc_soft_max = 0.80`, timestep = 15 min (¼ hour), standard irradiance = 1000 W/m².

Initial SoC at `reset()`: **0.0** (can be overridden externally for inference via `env.soc = initial_soc`).

### Observation Vector

`np.append(df_normalized.iloc[step], soc)` — 17 normalized feature columns + SoC scalar → shape `(18,)`.

### Reward Function

```
# 5.1 Market P&L
if grid_import > 0:  reward -= grid_import × price_to_buy          # buy at fixed tariff
else:                reward += |grid_export| × (DAM_Price / 1000)  # sell at DAM spot price

# 5.2 Battery degradation
reward -= lcos × |actual_batt_energy_abs|

# 5.3 Unmet load penalty
reward -= unmet_load × price_to_buy × 2

# 5.4 Mismatch penalty (unrealistic grid action)
reward -= 2.0 × |grid_commanded - grid_actual|

# 5.5 Soft SoC boundary penalty
if soc < soc_soft_min:  reward -= 3.0 × (soc_soft_min - soc)²
if soc > soc_soft_max:  reward -= 3.0 × (soc - soc_soft_max)²

# 5.6 Outage reserve penalty  (only when Grid==0)
reward -= 30.0 × max(0, target_soc - soc) × log1p(outage_remaining_h)

# 5.7 Pre-outage preparation bonus  (only when Grid==1 and hours_until_outage ≤ 3)
urgency   = exp(-0.5 × hours_until_outage)
soc_ready = min(soc, target_soc)
reward += 5.0 × urgency × soc_ready
```

`target_soc` is computed dynamically by `_calc_target_soc()`: estimated energy needed for the next outage (load minus solar) + 10% buffer, clamped to `[soc_soft_min, soc_soft_max]`.

### Episode
- Terminated when `curr_step >= len(df) - 1` (handles episodes shorter than 96 steps if data is sliced)
- `truncated = False` always
- `info` dict returns: `soc`, `target_soc`, `reward`, `solar_gen_ts_kwh`, `solar_surplus_kwh`, `actual_grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `mismatch`

---

## Normalization Pipeline (`envoriment/normalize.py`)

Normalizes `dataset_final.csv` → `dataset_normalized.csv` and saves `scalers.pkl` for inference-time use.

### Column treatment

| Strategy | Columns |
|---|---|
| **Dropped** (8 cols) | `timestamp`, `Hour`, `Minute`, `Minute_sin`, `Minute_cos`, `Day`, `Day_of_week`, `Month` |
| **log1p → StandardScaler** | `DAM_Price` |
| **StandardScaler** | `Load`, `Temperature_2m`, `Shortwave_radiation`, `DAM_Vol_Buy`, `DAM_Vol_Sale` |
| **MinMaxScaler [0, 1]** | `Global_tilted_irradiance_instant`, `hours_until_outage`, `outage_remaining_h`, `next_outage_duration` |
| **Passthrough** | `Hour_sin`, `Hour_cos`, `Day_of_week_sin`, `Day_of_week_cos`, `Grid` |
| **Also passthrough (not listed)** | `Day_sin`, `Day_cos` |

Result: 17-column `dataset_normalized.csv`. Scaler objects serialized to `models/scalers.pkl`.

### Inference-time normalization

`normalize_row(row: pd.Series, scalers: dict) → pd.Series` applies the same transformations to a single live row from `data_combiner.py`. Used in `inference.py`.

Run standalone: `python normalize.py --input dataset_final.csv --output dataset_normalized.csv --scalers models/scalers.pkl`

---

## Standalone Training Script (`envoriment/train.py`)

Trains a **SAC** (Soft Actor-Critic) agent via Stable-Baselines3. Run from inside `envoriment/`:

```bash
python train.py
```

### Key design decisions

**Domain randomization** via `RandomConfigWrapper(gym.Wrapper)`: on every `reset()` a new random `system_config` is sampled (battery 1–20 kWh, solar 1–10 kWp, inverter 3–15 kW, etc.). This trains a single universal model that generalizes across different hardware configs instead of per-config models.

**VecNormalize**: obs normalized with `clip_obs=10.0`; reward normalized on train env, raw on eval env.

**80/20 time-split**: first 80% rows → train, last 20% → eval (no shuffling to preserve time order).

### Configuration

| Parameter | Value |
|---|---|
| Algorithm | SAC (MlpPolicy) |
| Total timesteps | 500 000 |
| Buffer size | 100 000 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Network arch | [256, 256] |
| Gamma | 0.99 |
| Entropy coef | auto |
| Checkpoint frequency | every 50 000 steps |
| Eval frequency | every 50 000 steps, 3 episodes |

### Outputs

| Path | Contents |
|---|---|
| `models/sac_ems.zip` | Final model |
| `models/best/best_model.zip` | Best checkpoint by mean eval reward |
| `models/checkpoints/sac_ems_*_steps.zip` | Periodic checkpoints |
| `logs/tensorboard/SAC_N/` | TensorBoard event files |
| `logs/monitor/` | SB3 Monitor CSV |
| `logs/eval/` | EvalCallback results |

View training: `tensorboard --logdir logs/tensorboard/`

---

## Inference Module (`envoriment/inference.py`)

Designed to be called from the FastAPI predictions router.

```python
from inference import load_model_and_scalers, run_inference

model, scalers = load_model_and_scalers('models/sac_ems', 'models/scalers.pkl')

result = run_inference(
    df_raw        = df_raw,          # 96-row raw DataFrame from data_combiner.py
    system_config = system_config,   # dict from SystemConfig.settings
    model         = model,
    scalers       = scalers,
    initial_soc   = 0.6,             # current SoC from BMS/inverter
)
# result = {'dispatch_plan': [...96 dicts...], 'summary': {...}}
```

**`dispatch_plan`** — list of 96 dicts per 15-min step:
`step`, `action_battery`, `action_grid`, `soc`, `target_soc`, `solar_gen_kwh`, `grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `reward`

**`summary`** — aggregated day totals:
`total_reward_uah`, `bought_kwh`, `sold_kwh`, `solar_kwh`, `unmet_load_kwh`, `lcos_total_uah`, `initial_soc`, `final_soc`, `steps`

CLI usage: `python inference.py --data dataset_normalized.csv --model models/sac_ems --scalers models/scalers.pkl --soc 0.6`

---

## Data Pipeline

### `data_combiner.py` — Live Assembly

```python
combine(config_id, tilt=None, azimuth=None) → pd.DataFrame  # 96 rows × 25 cols
```

Reads solar `tilt`/`azimuth` from the latest `SystemConfig` for `config_id` if not passed directly. Calls each provider for `tomorrow` and concatenates results.

Output columns (combined.csv — 25 total):
```
timestamp,
Month, Day, Hour, Minute,
Hour_sin, Hour_cos, Minute_sin, Minute_cos,
Day_of_week, Day_of_week_sin, Day_of_week_cos, Day_sin, Day_cos,
Grid, next_outage_duration, outage_remaining_h, hours_until_outage,
Load,
Temperature_2m, Shortwave_radiation, Global_tilted_irradiance_instant,
DAM_Price, DAM_Vol_Buy, DAM_Vol_Sale
```

### `synthetic_grid.py` — Outage Schedule

Generates realistic outage blocks for `tomorrow` based on monthly average outage hours in Ukraine:

| Month | Avg h/day | Month | Avg h/day |
|---|---|---|---|
| Jan | 3.50 | Jul | 0.21 |
| Feb | 3.00 | Aug | 0.25 |
| Mar | 1.50 | Sep | 0.40 |
| Apr | 0.50 | Oct | 2.07 |
| May | 0.20 | Nov | 3.12 |
| Jun | 0.10 | Dec | 6.68 |

Output columns: `Grid` (1=on, 0=off), `next_outage_duration`, `outage_remaining_h`, `hours_until_outage`.

### `weather.py` — Solar Forecast

Open-Meteo API, 1-hour intervals expanded ×4 to 15-min. Panel orientation from `SystemConfig.solar.tilt / azimuth`.

Columns returned: `Temperature_2m`, `Shortwave_radiation`, `Global_tilted_irradiance_instant`.

### `synthetic_load.py` — Consumption Profile

- Daytime (08:00–18:00): 350 W base; Night: 60 W base
- Weekend factor: 0.4×
- Noise: ±15%, random spikes (+100–200 W, probability 2%)

### `time_features.py` — Cyclical Time Encoding

Returns 13 columns:
```
Month, Day, Hour, Minute,
Hour_sin, Hour_cos, Minute_sin, Minute_cos,
Day_of_week, Day_of_week_sin, Day_of_week_cos,
Day_sin, Day_cos
```

---

## Training Dataset (`datasets/dataset_v10/dataset_final.csv`) ✅ CURRENT

- **Size**: 35 041 rows (~365 days × 96 steps)
- **Period**: Full year, 15-min resolution
- **Column-aligned** with live `data_combiner.py` output
- **Copy** also stored at `envoriment/dataset_final.csv` for local training runs

### Columns (25 total)

| Column | Description |
|---|---|
| timestamp | ISO datetime string |
| Month, Day, Hour, Minute | Calendar features |
| Day_of_week | 0–6 |
| Hour_sin, Hour_cos | Cyclical hour (period 24) |
| Minute_sin, Minute_cos | Cyclical minute (period 60) |
| Day_of_week_sin, Day_of_week_cos | Cyclical weekday (period 7) |
| Day_sin, Day_cos | Cyclical day-of-month (period 31) |
| Grid | 1=grid on, 0=outage |
| next_outage_duration | Hours until end of next outage block |
| outage_remaining_h | Hours left in current outage |
| hours_until_outage | Hours until next outage starts |
| Load | Synthetic load (W) |
| Temperature_2m | °C |
| Shortwave_radiation | W/m² global |
| Global_tilted_irradiance_instant | W/m² on tilted panel |
| DAM_Price | UAH/MWh day-ahead price |
| DAM_Vol_Buy, DAM_Vol_Sale | MWh volumes |

### Dataset Build Chain (v9 → v10)

```
datasets_v9/sorted.csv
  → dataset_v10/build_v10.py     # adds Day_sin / Day_cos
  → dataset_v10/dataset.csv
  → dataset_v10/drop_idm.py      # removes IDM_*, Avg_Weighted_Price, outage_risk
  → dataset_v10/dataset_no_idm.csv
  → dataset_v10/drop_dam_vol.py  # reorders cols, renames, adds timestamp
  → dataset_v10/dataset_final.csv  ✅  FINAL
```

### Dataset Version History

| Version | Key change |
|---|---|
| v0 | Initial DAM/IDM + weather + kWh coefficients |
| v1–v2 | Market data cleaning, hour=0–24 format fix |
| v3 | Concatenation of multiple date ranges |
| v4 | Added synthetic Load column |
| v5 | Added Grid/outage columns; sin/cos time features; resized to 15-min |
| v6 | Added GTI (Global Tilted Irradiance), temperature correction |
| v7 | Validation checks |
| v8 | Added sin/cos for day-of-month and month |
| v9 | Fixed Hour=24→Hour=0 next-day bug; shifted GTI +1h alignment; recalculated all cyclical features (27 cols, has IDM) |
| **v10** | **Current**: dropped IDM, added `Day_sin`/`Day_cos`, `hours_until_outage`, `next_outage_duration`; column-aligned with live pipeline (25 cols) |

---

## Backend API

### Stack
- **FastAPI** + **SQLAlchemy** + **PostgreSQL**
- **MinIO** (S3-compatible object storage) for trained model `.zip` files
- **JWT** (PyJWT, HS256, 30-min expiry) + **pwdlib** (Argon2 password hashing)

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | No | Health check |
| POST | `/auth/register` | No | Create user |
| POST | `/auth/login` | No | Get JWT token |
| POST | `/config/` | JWT | Save `SiteConfig` + trigger PPO background training |
| GET | `/config/?config_name=...` | JWT | Retrieve named config |
| GET | `/config/list` | JWT | List all configs for current user |
| GET | `/predictions/` | JWT | ❌ Partial — time gate + data fetch only; inference loop not wired |

### SiteConfig Schema

```python
Battery:
  capacity_kwh: float
  min_reserve: float = 10             # Minimum SoC % during outages
  lcos: float                         # Levelized cost [UAH/kWh]
  max_charge_power: float             # [kW]
  max_discharge_power: float          # [kW]
  efficiency: float = 1.0

Inverter:
  max_power: float                    # [kW] — also used as max_grid_capacity in env
  efficiency: float
  # Note: price_to_buy used by environment comes from inverter config in train.py;
  # the Grid schema's price_buy_from_grid is defined but not yet wired to environment

SolarPanel:
  peak_power: float                   # [kWp]
  efficiency: float
  azimuth: float = 0                  # 0 = South
  tilt: float = 35                    # Degrees from horizontal

Grid:
  capacity: float                     # [kW]
  price_buy_from_grid: float          # [UAH/kWh]
```

### Database Models

| Table | Columns |
|---|---|
| `users` | id, username, email, hashed_password |
| `system_configs` | id, user_id (FK), config_name, settings (JSONB) |
| `history` | id, user_id (FK), timestamp, data (JSONB) — per-timestep telemetry |
| `agent_predictions` | 30+ columns per step: battery/grid actions, energy flows, SoC, reward breakdown, outage state |
| `agent_models` | id, config_id (FK), status (training\|ready\|failed), trained_at, total_timesteps, storage_path, mean_reward |

### Environment Variables (`.env`)

```
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/ems_database
SECRET_KEY=<hex secret>
ALGORITHM=HS256
ACCESS_EXPIRE_MINUTES=30
MINIO_ENDPOINT=<host:port>
MINIO_ROOT_USER=<user>
MINIO_ROOT_PASSWORD=<password>
```

### Backend Trainer (`backend/core/trainer.py`)

Triggered as `BackgroundTask` on `POST /config/`. Uses **PPO** (not SAC). Workflow:
1. Fetch `SystemConfig` from DB by `config_id`
2. Load `datasets/dataset_v10/dataset_final.csv`, drop timestamp
3. Create `Environment` with raw DataFrame and config settings
4. Train PPO for **200 000** timesteps
5. Save `.zip` to MinIO bucket `models` as `{config_id}.zip`
6. Update `agent_models` row: status → `ready` / `failed`, path, timestamp

Note: backend trainer uses **PPO + raw data** (no VecNormalize, no domain randomization). The standalone `envoriment/train.py` uses **SAC + normalized data + RandomConfigWrapper** and is the more capable training path.

---

## Known Issues & Incomplete Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `environment.py` | ✅ Fixed | Full `step()` with battery/grid physics, SoC tracking, reward, correct Gymnasium return signature |
| 2 | `environment.py` | ✅ Fixed | Reward function complete: market P&L, LCOS, unmet load, mismatch, soft SoC, outage reserve, pre-outage bonus |
| 3 | `environment.py` | ✅ Fixed | `observation_space` shape is dynamic `(n_features + 1,)` — no longer hardcoded |
| 4 | `IDM_DAM_features.py` | ✅ Fixed | DAM columns renamed to `DAM_Price`, `DAM_Vol_Buy`, `DAM_Vol_Sale` |
| 5 | `data_providers/` | ✅ Fixed | Column mismatch resolved — v10 dataset aligns with live combined output (25 cols each) |
| 6 | `environment.py` | ✅ Fixed | All hardware params loaded from `system_config` dict — nothing hardcoded |
| 7 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active; v10 training dataset has no IDM columns |
| 8 | `backend/core/trainer.py` | ✅ Fixed | PPO training via BackgroundTasks on POST /config/; uploads to MinIO; writes AgentModels status |
| 9 | `normalize.py` | ✅ Fixed | `envoriment/normalize.py` implemented: drops 8 cols, applies log/standard/minmax scalers, saves `scalers.pkl` |
| 10 | `backend/routers/predictions.py` | ❌ Open | Time gate (≥14:00) and data fetch implemented; inference loop not yet wired; `run_inference()` in `inference.py` is ready to plug in |
| 11 | `requirements.txt` | ⚠️ Partial | `stable-baselines3`, `torch`, `tensorboard`, `boto3` added ✅; missing: `scikit-learn` (needed by `normalize.py`), `psycopg2-binary` (PostgreSQL), `openpyxl` (OREE Excel), `uvicorn` |
| 12 | Backend vs standalone training | ⚠️ Diverged | `backend/core/trainer.py` uses PPO + raw data (200k steps); `envoriment/train.py` uses SAC + VecNormalize + domain randomization (500k steps). These are separate pipelines — not yet unified |

---

## Next Steps (Logical Order)

1. **Wire `inference.py` → `predictions.py`**: call `run_inference(df_raw, system_config, model, scalers)` inside `GET /predictions/`; load model from MinIO or local path; store results in `AgentPredictions` table. Return `404` if no trained model for that config.
2. **Fix `requirements.txt`**: add `scikit-learn`, `psycopg2-binary`, `openpyxl`, `uvicorn`.
3. **Unify training pipelines**: decide whether to replace `backend/core/trainer.py` (PPO) with the SAC + VecNormalize approach from `envoriment/train.py`, or keep them for different use cases.
4. **Re-enable IDM** in `IDM_DAM_features.py` for richer price signals (requires rebuilding dataset to v11 with IDM columns).
5. **Wire `Grid.price_buy_from_grid`** from `SiteConfig` into the environment as `price_to_buy` (currently only set via `inverter.price_to_buy` in `train.py`'s hardcoded config).

---

## Python Dependencies

Current `requirements.txt`:
```
fastapi
pandas
numpy
requests
openmeteo-requests
requests-cache
retry-requests
gymnasium
python-calamine
sqlalchemy
pyjwt
python-dotenv
pwdlib
pydantic
stable-baselines3
torch
tensorboard
boto3
```

Missing (need to add):
```
scikit-learn       # required by envoriment/normalize.py (StandardScaler, MinMaxScaler)
psycopg2-binary    # PostgreSQL driver for SQLAlchemy
openpyxl           # Excel parsing for OREE DAM data
uvicorn            # FastAPI ASGI server
```

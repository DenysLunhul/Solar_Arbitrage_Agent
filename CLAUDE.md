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
├── envoriment/
│   └── envoriment.py                      # Gymnasium RL environment (fully implemented)
├── data_providers/
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── data_combiner.py               # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   ├── combined.csv                   # Last generated live dataset (96 rows × 25 cols)
│   │   └── .cache.sqlite                  # Open-Meteo API cache
│   ├── agent_data_preprocessing/
│   │   └── __init__.py                    # preprocessing.py does NOT exist — no preprocessing in pipeline
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
│   ├── core/
│   │   ├── database.py                    # SQLAlchemy engine + session (PostgreSQL)
│   │   └── loader.py                      # Bulk-uploads DataFrame → History table
│   ├── models/
│   │   └── site.py                        # ORM: User, SystemConfig, History, AgentPredictions
│   ├── routers/
│   │   ├── auth.py                        # POST /auth/login, POST /auth/register
│   │   ├── config.py                      # POST /config/, GET /config/, GET /config/list
│   │   └── predictions.py                 # EMPTY — placeholder for inference endpoints
│   ├── schemas/
│   │   └── schemas.py                     # Pydantic: SiteConfig, Battery, Inverter, SolarPanel, Grid, User
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

## RL Environment (`envoriment/envoriment.py`)

### Spaces

```python
action_space      = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
observation_space = Box(low=-inf, high=inf, shape=(n_features + 1,), dtype=float32)
  # n_features = df.shape[1] computed at runtime from input DataFrame
  # With dataset_final.csv (24 feature cols after dropping timestamp) → shape=(25,)
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge (scaled by `max_batt_power / 4`) |
| `action[1]` | Grid exchange: +1 = full import, −1 = full export (scaled by `max_grid_capacity / 4`) |

### Hardcoded Hardware Parameters (not yet loaded from `SystemConfig`)

| Parameter | Value |
|---|---|
| `max_batt_capacity` | 2.0 kWh |
| `max_batt_power` | 1.0 kW |
| `max_grid_capacity` | 5.0 kW |
| `batt_efficiency` | 0.95 |
| `lcos` | 1.5 UAH/kWh |
| `soc_soft_min / soc_soft_max` | 0.20 / 0.80 |
| `solar_peak_power` | 3.0 kW |
| `solar_efficiency` | 0.18 |
| Initial SoC | 0.5 |

### Observation Vector

`np.append(df.iloc[step], soc)` — dataset row (24 feature columns, `timestamp` dropped) + SoC scalar → shape `(25,)`.

### Reward Function (fully implemented)

```
reward = -(actual_grid_kwh × curr_price)      # market P&L
       - lcos × |actual_batt_energy|           # LCOS degradation cost
       - 50 × unmet_load                       # unmet load penalty [kWh]
       - 2  × mismatch                         # power balance deviation
       - quadratic SoC soft penalty            # outside [soc_soft_min, soc_soft_max]
       - 30 × soc_deficit × log(outage_remaining_h)   # outage reserve (when Grid==0)
       + 5  × urgency × soc_ready              # pre-outage charging bonus (when Grid==1, time_to_outage ≤ 3h)
```

### Episode
- 96 steps per episode (one full day)
- `terminated = True` at step 96; `truncated = False`
- `reset()` returns `(observation, info)` per Gymnasium v26+ API
- `step()` returns `(observation, reward, terminated, truncated, info)` with 10+ metrics in `info`

---

## Data Pipeline

### `data_combiner.py` — Live Assembly

Reads `SystemConfig` from DB (tilt, azimuth), then calls each provider for `tomorrow`:

```python
combine(tilt, azimuth) → pd.DataFrame  # 96 rows × 25 cols
```

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
- **JWT** (PyJWT, HS256, 30-min expiry) + **pwdlib** (Argon2 password hashing)

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/` | No | Health check |
| POST | `/auth/register` | No | Create user |
| POST | `/auth/login` | No | Get JWT token |
| POST | `/config/` | JWT | Save `SiteConfig` for current user |
| GET | `/config/?config_name=...` | JWT | Retrieve named config |
| GET | `/config/list` | JWT | List all configs for current user |

### SiteConfig Schema

```python
Battery:
  battery_capacity_kwh: float
  battery_min_reserve: float          # Minimum SoC % during outages
  battery_lcos: float                 # Levelized cost [UAH/kWh]
  battery_max_charge_power: float     # [kW]
  battery_max_discharge_power: float  # [kW]
  battery_efficiency: float

Inverter:
  max_power: float                    # [kW]
  efficiency: float

SolarPanel:
  solar_peak_power: float             # [kWp]
  solar_efficiency: float
  solar_azimuth: float = 0            # 0 = South
  solar_tilt: float = 35              # Degrees from horizontal

Grid:
  grid_capacity: float                # [kW]
  price_buy_from_grid: float          # [UAH/kWh]
```

### Database Models

| Table | Columns |
|---|---|
| `users` | id, username, email, hashed_password |
| `system_configs` | id, user_id (FK), config_name, settings (JSONB) |
| `history` | id, user_id (FK), timestamp, data (JSONB) — per-timestep telemetry |
| `agent_predictions` | id, user_id (FK), timestamp, data (JSONB) — **defined, no endpoints yet** |

### Environment Variables (`.env`)

```
DATABASE_URL=postgresql://postgres:<password>@localhost:5432/ems_database
SECRET_KEY=<hex secret>
ALGORITHM=HS256
ACCESS_EXPIRE_MINUTES=30
```

---

## Known Issues & Incomplete Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `envoriment.py` | ✅ Fixed | `super().__init__()` correct; full `step()` implemented with battery/grid logic, SoC tracking, reward, proper return signature |
| 2 | `envoriment.py` | ✅ Fixed | Reward function complete: market P&L, LCOS, unmet load, mismatch, soft SoC, outage reserve, pre-outage bonus |
| 3 | `envoriment.py` | ✅ Fixed | `observation_space` shape is dynamic `(n_features + 1,)`; no longer hardcoded to `(31,)` |
| 4 | `IDM_DAM_features.py` | ✅ Fixed | DAM columns renamed to `DAM_Price`, `DAM_Vol_Buy`, `DAM_Vol_Sale` |
| 5 | `data_providers/` | ✅ Fixed | Column mismatch resolved — v10 dataset aligns with live combined output (25 cols each) |
| 6 | `envoriment.py` | ✅ Fixed | Hardware params loaded from `SiteConfig` passed at construction — no longer hardcoded |
| 7 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active; v10 training dataset has no IDM either |
| 8 | General | ✅ Fixed | Training triggered via `BackgroundTasks` on `POST /config/`; `backend/core/trainer.py` trains PPO with SB3, uploads model to MinIO as `{config_id}.zip`, writes status to `agent_models` table |
| 9 | `preprocessing.py` | ❌ Open | File does not exist; no feature normalization/dropping in the pipeline |
| 10 | `backend/routers/predictions.py` | ❌ Open | File exists but is incomplete — no inference loop yet; `AgentPredictions` and `AgentModels` ORM models defined |
| 11 | `requirements.txt` | ✅ Fixed | Added `stable-baselines3`, `torch`, `tensorboard`, `boto3` |

---

## Next Steps (Logical Order)

1. **Add RL training script**: integrate SB3 (`PPO` or `SAC`) with `envoriment.py` + `dataset_final.csv`; add checkpointing and TensorBoard logging. Model is trained **per `SystemConfig`** and saved as `models/{config_id}.zip`.
2. **Add RL dependencies** to `requirements.txt`: `stable-baselines3`, `torch`, `tensorboard`.
3. **Wire `SystemConfig` → Environment**: load battery/inverter/solar/grid params from DB at training time — each config trains its own model with its own hardware params.
4. **Implement `preprocessing.py`**: write `drop_features()` and `normalize_features()`; plug into training pipeline before feeding data to the env.
5. **Inference pipeline** (`backend/routers/predictions.py`): load `models/{config_id}.zip` → call `data_combiner.combine()` → step through 96-step schedule → store results in `AgentPredictions` table → expose via API. Return `404` if model for that config hasn't been trained yet.
6. **Re-enable IDM** in `IDM_DAM_features.py` for richer price signals (requires rebuilding dataset to v11 with IDM columns).

---

## Python Dependencies

From `requirements.txt` (confirmed present):
```
gymnasium
numpy
pandas
fastapi
uvicorn
sqlalchemy
psycopg2-binary
pydantic
PyJWT
pwdlib[argon2]
python-dotenv
openmeteo-requests
requests-cache
retry-requests
openpyxl
python-calamine
requests
```

Missing (needed for RL training — not yet in `requirements.txt`):
```
stable-baselines3
torch
tensorboard
```

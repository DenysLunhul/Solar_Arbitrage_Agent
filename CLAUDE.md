# Scalable AI Energy Management System (EMS) — Project Context

## Project Goal

Build an autonomous RL agent that optimally manages energy flows in a hybrid node (Solar PV + BESS + Grid) to:
1. **Maximize profit** from energy arbitrage on the Ukrainian Day-Ahead Market (DAM) and Intraday Market (IDM).
2. **Guarantee energy autonomy** during planned/predicted power outages by maintaining a sufficient State of Charge (SoC) reserve.
3. **Minimize battery degradation** by incorporating Levelized Cost of Storage (LCOS) into the reward function.
4. **Scale to any hardware** (PV capacity, BESS size, inverter power) via per-user `SystemConfig` stored in the DB — no retraining of the architecture required.

---

## Repository Layout

```
ds_project_demo/
├── docker-compose.yml
├── requirements.txt
├── .env                                   # DATABASE_URL, SECRET_KEY, ALGORITHM, ACCESS_EXPIRE_MINUTES
├── envoriment/
│   └── envoriment.py                      # Gymnasium RL environment (partially complete — see Known Issues)
├── data_providers/
│   ├── __init__.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── data_combiner.py               # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   ├── combined.csv                   # Last generated live dataset (96 rows × 25 cols)
│   │   └── .cache.sqlite                  # Open-Meteo API cache
│   ├── agent_data_preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py              # Feature dropping + normalization (INCOMPLETE — see Known Issues)
│   └── components/
│       ├── __init__.py
│       ├── market_manager/
│       │   ├── __init__.py
│       │   └── IDM_DAM_features.py        # Fetches DAM prices from OREE (oree.com.ua); IDM missing
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
│   │   ├── dataset.csv                    # Final training dataset: 35 041 rows × 27 cols (~1 year)
│   │   ├── sorted.csv                     # Sorted version of dataset.csv
│   │   ├── build_v10.py                   # Build script for v10 (not yet executed)
│   │   ├── checkkkk.py                    # Validation script
│   │   └── sort_dataset.py                # Sorting utility
│   └── datasets_v10/                      # Empty — v10 not built yet
├── backend/
│   ├── __init__.py
│   ├── main.py                            # FastAPI app entry point
│   ├── core/
│   │   ├── database.py                    # SQLAlchemy engine + session (PostgreSQL)
│   │   └── loader.py                      # Bulk-uploads DataFrame → History table
│   ├── models/
│   │   └── site.py                        # ORM: User, SystemConfig, History
│   ├── routers/
│   │   ├── auth.py                        # POST /auth/login, POST /auth/register
│   │   └── config.py                      # POST /config/, GET /config/, GET /config/list
│   ├── schemas/
│   │   └── schemas.py                     # Pydantic: SiteConfig, Battery, Inverter, SolarPanel, User
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
action_space      = Box(low=-1.0, high=1.0, shape=(2,))    # [battery_action, grid_action]
observation_space = Box(low=-inf, high=inf, shape=(31,))    # declared; actual shape ~26 — see Known Issues
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge (scaled by `max_batt_power / 4`) |
| `action[1]` | Grid exchange: +1 = full import, −1 = full export (scaled by `max_grid_capacity / 4`) |

### Default Hardware Parameters (hardcoded — not yet loaded from `SystemConfig`)

| Parameter | Value |
|---|---|
| `max_batt_capacity` | 2.0 kWh |
| `max_batt_power` | 1.0 kW |
| `max_grid_capacity` | 5.0 kW |
| Initial SoC | 0.5 |
| Charge efficiency | 0.95 (round-trip implied) |

### Observation Vector
`np.append(df.iloc[step], soc)` → dataset row + SoC scalar.  
Declared shape `(31,)` does not match the combined CSV output of 25 columns (→ actual `(26,)`). Must be reconciled before training.

### Reward Function (implemented, simplified)
```
reward = -(actual_grid_energy_ts * curr_price) - unmet_load_penalty - mismatch_penalty
```
- `unmet_load_penalty` = 50 × unmet energy [kWh]
- `mismatch_penalty` = penalty for power balance deviation
- **Missing**: LCOS degradation cost and outage reserve penalty (see Known Issues)

### Episode
- 96 steps per episode (one full day)
- `terminated = True` at step 96; `truncated = False`
- `reset()` returns `(observation, info)` per Gymnasium API ✅

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
DAM_Price, DAM_Vol_Sale, DAM_Vol_Buy
```

DAM columns are renamed to English inside `IDM_DAM_features.py` before being merged. `Day_sin` / `Day_cos` are new columns added in the latest version of `time_features.py` (not present in training dataset v9).

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
`Day_sin` / `Day_cos` are new (not in training dataset v9 — column mismatch with live pipeline).

### `preprocessing.py` — Feature Preprocessing (INCOMPLETE)

Located at `data_providers/agent_data_preprocessing/preprocessing.py`.

- `drop_features()` — defined but has a **syntax bug**: passes `[DROP_FEATURES]` (list-in-list) to `.drop(columns=...)`. Should be `.drop(columns=DROP_FEATURES)`.
- `normalize_features()` — stub only; not implemented.
- Not called from anywhere in the current pipeline.

---

## Training Dataset (`datasets/datasets_v9/dataset.csv`)

- **Size**: 35 041 rows (~365 days × 96 steps)
- **Period**: Full year, 15-min resolution

### Columns (27 total)

| Column | Description |
|---|---|
| Date | "M - D" string (dropped before training) |
| Month, Day, Day_of_week, Hour, Minute | Calendar features |
| DAM_Price | UAH/MWh day-ahead price |
| DAM_Vol_Sale, DAM_Vol_Buy | MWh volumes |
| IDM_Price, IDM_Last_Price, IDM_Vol_Sale, IDM_Vol_Buy | Intraday market |
| Avg_Weighted_Price | Weighted average of DAM+IDM |
| Load | Synthetic load (W) |
| global_tilted_irradiance_instant | W/m² on tilted panel |
| Temperature_2m | °C |
| Shortwave_Radiation | W/m² global |
| Hour_sin, Hour_cos | Cyclical hour (period 24) |
| Minute_sin, Minute_cos | Cyclical minute (period 60) |
| Day_of_week_sin, Day_of_week_cos | Cyclical weekday (period 7) |
| Grid | 1=grid on, 0=outage |
| outage_remaining_h | Hours left in current outage |
| outage_risk | Probability/fraction of outage in coming horizon |

**Note**: v9 does not have `Day_sin` / `Day_cos` (added to live pipeline later). Also lacks `hours_until_outage` and `next_outage_duration` from the live pipeline. The live combined CSV and the training dataset are not yet column-aligned.

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
| v9 | **Current**: fixed Hour=24→Hour=0 next-day bug; shifted GTI +1h alignment; recalculated all cyclical features |
| v10 | Planned — build script exists at `datasets_v9/build_v10.py`; not yet executed |

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
SiteConfig:
  battery:
    capacity_kwh: float          # BESS total capacity [kWh]
    min_reserve: float = 10      # Minimum SoC % during outages
    lcos: float                  # Levelized cost [UAH/kWh] — degradation cost
    max_charge_power: float      # [kW]
    max_discharge_power: float   # [kW]
    efficiency: float = 1        # Round-trip efficiency (override)
  inverter:
    max_power: float             # [kW]
    efficiency: float
    is_grid_tied: bool
  solar:
    peak_power: float            # [kWp]
    efficiency: float
    azimuth: float = 0           # 0 = South
    tilt: float = 35             # Degrees from horizontal
```

### Database Models

| Table | Columns |
|---|---|
| `users` | id, username, email, hashed_password |
| `system_configs` | id, user_id (FK), config_name, settings (JSONB) |
| `history` | id, timestamp, data (JSONB) — stores per-timestep telemetry |

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
| 1 | `envoriment.py` | ✅ Fixed | `super().__init__()` is correct |
| 2 | `envoriment.py` | ✅ Fixed | `step()` is implemented with battery/grid logic, SoC tracking, penalties, and proper return signature |
| 3 | `envoriment.py:32` | ❌ Open | `observation_space shape=(31,)` mismatches actual combined CSV output (25 cols → shape `(26,)` with SoC). Must align before training. |
| 4 | `IDM_DAM_features.py` | ✅ Fixed | DAM columns are renamed to `DAM_Price`, `DAM_Vol_Sale`, `DAM_Vol_Buy` |
| 5 | `envoriment.py` | ❌ Open | Hardware params (`max_batt_capacity`, `max_batt_power`, `max_grid_capacity`) are hardcoded; should be loaded from `SystemConfig` |
| 6 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM is active. Training dataset has IDM columns but live pipeline does not. |
| 7 | General | ❌ Open | No RL training script exists (no SB3 / RLlib setup, no training loop, no checkpointing) |
| 8 | `envoriment.py` | ❌ Open | Reward function is simplified — missing LCOS degradation cost and outage reserve penalty (`SoC < min_reserve` when `Grid == 0`) |
| 9 | `preprocessing.py` | ❌ Open | `drop_features()` has syntax bug (list-in-list); `normalize_features()` is a stub; not called anywhere |
| 10 | `data_providers/` | ❌ Open | Column mismatch between training dataset (v9, 27 cols) and live combined output (25 cols): live has `Day_sin`, `Day_cos`, `hours_until_outage`, `next_outage_duration`; v9 has `IDM_*`, `Avg_Weighted_Price`, `outage_risk` |
| 11 | `requirements.txt` | ❌ Open | Missing RL dependencies: `stable-baselines3`, `torch`, `tensorboard` |

---

## Next Steps (Logical Order)

1. **Align columns**: reconcile training dataset (v9) with live combined output — either rebuild v10 with matching columns or strip live output to v9 schema.
2. **Fix observation space**: update `observation_space shape` in `envoriment.py` to match the actual feature count after column alignment.
3. **Complete reward function**: add LCOS cost (`battery.lcos × |energy_delta|`) and outage reserve penalty (`SoC < min_reserve` when `Grid == 0`).
4. **Wire `SystemConfig` → Environment**: load battery/inverter/solar params from DB at env init instead of hardcoded values.
5. **Add RL training script**: integrate SB3 (`PPO` or `SAC`) with the environment and training dataset; add checkpointing.
6. **Fix `preprocessing.py`**: correct `drop_features()` syntax; implement `normalize_features()`; plug into training pipeline.
7. **Re-enable IDM** in `IDM_DAM_features.py` for richer price signals and to close the training/live column gap.
8. **Inference pipeline**: load trained model + call `data_combiner.combine()` → step through next-day schedule → return dispatch plan via new API endpoint.

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

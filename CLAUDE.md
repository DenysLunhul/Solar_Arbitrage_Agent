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
├── envoriment/
│   └── envoriment.py          # Gymnasium RL environment (INCOMPLETE — see Known Issues)
├── data_providers/
│   ├── orchestrator/
│   │   ├── data_combiner.py   # Main entry: assembles "tomorrow" DataFrame from all providers
│   │   └── combined.csv       # Last generated live dataset (96 rows × 23 cols)
│   ├── market_manager/
│   │   └── IDM_DAM_features.py  # Fetches DAM prices from OREE (oree.com.ua)
│   ├── weather/
│   │   └── weather.py         # Open-Meteo forecast (GTI, temp, radiation)
│   ├── grid/
│   │   └── synthetic_grid.py  # Synthetic outage schedule generator
│   ├── load/
│   │   └── synthetic_load.py  # Synthetic consumption profile
│   └── time/
│       └── time_features.py   # Cyclical time encoding (sin/cos)
├── datasets/
│   └── datasets_v9/
│       └── dataset.csv        # Final training dataset: 35 040 rows × 27 cols (~1 year)
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── core/
│   │   ├── database.py        # SQLAlchemy engine + session (PostgreSQL)
│   │   └── loader.py          # Bulk-uploads DataFrame → History table
│   ├── models/
│   │   └── site.py            # ORM: User, SystemConfig, History
│   ├── routers/
│   │   ├── auth.py            # POST /auth/login, POST /auth/register
│   │   └── config.py          # POST /config/, GET /config/
│   ├── schemas/
│   │   └── schemas.py         # Pydantic: SiteConfig, Battery, Inverter, SolarPanel, User
│   └── security/
│       └── security.py        # JWT (HS256), pwdlib password hashing
└── .env                       # DATABASE_URL, SECRET_KEY, ALGORITHM, ACCESS_EXPIRE_MINUTES
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
action_space  = Box(low=-1.0, high=1.0, shape=(2,))   # [battery_action, grid_action]
observation_space = Box(low=-inf, high=inf, shape=(31,))  # 30 dataset features + SoC
```

| Action dim | Meaning |
|---|---|
| `action[0]` | Battery: +1 = full charge, −1 = full discharge (scaled by `max_batt_power / 4`) |
| `action[1]` | Grid exchange: +1 = full import, −1 = full export (scaled by `max_grid_capacity / 4`) |

### Default Hardware Parameters (hardcoded, to be replaced by `SystemConfig`)

| Parameter | Value |
|---|---|
| `max_batt_capacity` | 2.0 kWh |
| `max_batt_power` | 1.0 kW |
| `max_grid_capacity` | 5.0 kW |
| Initial SoC | 0.5 |
| Charge efficiency | 0.95 (round-trip implied) |

### Observation Vector
`np.append(df.iloc[step], soc)` → 30 dataset columns + SoC scalar = 31 features.

### Reward Function (intended, not yet implemented)
```
reward = net_revenue - lcos_cost - outage_penalty
```
- `net_revenue` = grid export revenue − grid import cost
- `lcos_cost` = LCOS [UAH/kWh] × |actual_energy_delta| (from battery schema)
- `outage_penalty` = large negative if SoC < `min_reserve` when `Grid == 0`

---

## Data Pipeline

### `data_combiner.py` — Live Assembly

Reads `SystemConfig` from DB (tilt, azimuth), then calls each provider for `tomorrow`:

```python
combine(tilt, azimuth) → pd.DataFrame  # 96 rows
```

Output columns (combined.csv):
```
timestamp, Month, Day, Hour, Minute,
Hour_sin, Hour_cos, Minute_sin, Minute_cos,
Day_of_week, Day_of_week_sin, Day_of_week_cos,
Grid, next_outage_duration, outage_remaining_h, hours_until_outage,
Load,
Temperature_2m, Shortwave_radiation, Global_tilted_irradiance_instant,
Ціна грн/МВт.год, Обсяг продажу МВт.год, Обсяг купівлі МВт.год   ← need rename
```

**Note**: DAM columns arrive with Ukrainian names from the OREE API and are not yet renamed. The training dataset uses `DAM_Price`, `DAM_Vol_Sale`, `DAM_Vol_Buy`.

### `synthetic_grid.py` — Outage Schedule

Generates realistic outage blocks for `tomorrow` based on monthly average outage hours in Ukraine:

| Month | Avg h/day |
|---|---|
| Jan | 3.50 | Feb | 3.00 | Mar | 1.50 | Apr | 0.50 |
| May | 0.20 | Jun | 0.10 | Jul | 0.21 | Aug | 0.25 |
| Sep | 0.40 | Oct | 2.07 | Nov | 3.12 | Dec | 6.68 |

Output columns: `Grid` (1=on, 0=off), `next_outage_duration`, `outage_remaining_h`, `hours_until_outage`.

### `weather.py` — Solar Forecast

Open-Meteo API, 1-hour intervals expanded ×4 to 15-min. Panel orientation from `SystemConfig.solar.tilt / azimuth`.

Columns returned: `Temperature_2m`, `Shortwave_radiation`, `Global_tilted_irradiance_instant`.

### `synthetic_load.py` — Consumption Profile

- Daytime (08:00–18:00): 350 W base; Night: 60 W base
- Weekend factor: 0.4×
- Noise: ±15%, random spikes (+100–200 W, probability 2%)

---

## Training Dataset (`datasets/datasets_v9/dataset.csv`)

- **Size**: 35 040 rows (~365 days × 96 steps)
- **Period**: Full year, 15-min resolution

### Columns (27 total)

| Column | Description |
|---|---|
| Date | "M - D" string |
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
| v9 | **Final**: fixed Hour=24→Hour=0 next-day bug; shifted GTI +1h alignment; recalculated all cyclical features |

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

| # | Location | Issue |
|---|---|---|
| 1 | `envoriment.py:13` | `super().init()` → must be `super().__init__()` |
| 2 | `envoriment.py:55–107` | `step()` is incomplete: grid_delta block has no logic, no reward calc, no `done`/`truncated` return |
| 3 | `envoriment.py:29` | `observation_space shape=(31,)` assumes 30 features — verify against actual combined DataFrame column count after DAM rename |
| 4 | `data_combiner.py` | DAM columns have Ukrainian names in output; need `.rename()` to `DAM_Price`, `DAM_Vol_Sale`, `DAM_Vol_Buy` |
| 5 | `envoriment.py` | Hardware params (`max_batt_capacity`, `max_batt_power`, `max_grid_capacity`) are hardcoded; should be loaded from `SystemConfig` via backend |
| 6 | `IDM_DAM_features.py` | IDM fetching is commented out; only DAM is active |
| 7 | General | No RL training script exists yet (no SB3 / RLlib setup) |
| 8 | `config.py:7` | Import `from models.site import ...` should be `from backend.models.site import ...` |

---

## Next Steps (Logical Order)

1. **Fix environment bugs** (items 1–3 above).
2. **Complete `step()`**: implement grid interaction logic, reward calculation (arbitrage revenue − LCOS − outage penalty), and `terminated`/`truncated` flags.
3. **Wire `SystemConfig` → Environment**: load battery/inverter/solar params from DB at env init.
4. **Fix DAM column rename** in `data_combiner.py`.
5. **Add RL training script**: integrate SB3 (`PPO` or `SAC`) with the environment and the training dataset.
6. **Re-enable IDM** in `IDM_DAM_features.py` for richer price signals.
7. **Inference pipeline**: load trained model + call `data_combiner.combine()` → step through next-day schedule → return dispatch plan via API.

---

## Python Dependencies (inferred from code)

```
gymnasium
numpy
pandas
fastapi
uvicorn
sqlalchemy
psycopg2-binary
pydantic
python-jose[cryptography]   # or PyJWT
pwdlib[argon2]
python-dotenv
openmeteo-requests
requests-cache
retry-requests
openpyxl
python-calamine
requests
```

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
├── environment/                           # RL environment, training, inference, backtests
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
│   │   ├── predictions.py                 # GET /predictions/, /default, /history, /history/dates
│   │   └── strategy.py                    # POST /strategy/, GET /strategy/, GET /strategy/list
│   ├── schemas/schemas.py                 # Pydantic: SiteConfig, DefaultStrategyConfig, PredictionResponse, …
│   ├── services/
│   │   ├── auth_service.py                # login / register logic
│   │   ├── config_service.py              # save / get / list configs
│   │   ├── prediction_service.py          # SAC + default strategy prediction pipelines
│   │   └── strategy_service.py            # save / get / list default strategies
│   └── security/security.py              # JWT (HS256), pwdlib Argon2 password hashing
├── frontend/                              # React SPA (Vite + Recharts)
│   ├── package.json                       # react, react-dom, recharts, vite
│   ├── vite.config.js                     # Vite config (port 3000)
│   ├── index.html                         # Vite entry point
│   ├── .env.example                       # VITE_API_URL variable
│   └── src/
│       ├── main.jsx                       # React root mount
│       ├── api.js                         # All API calls + JWT token management
│       └── App.jsx                        # Full dashboard (login, charts, table, history)
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

## RL Environment (`environment/environment.py`)

### Dual-Dataset Architecture

| Argument | Purpose |
|---|---|
| `df_raw` | Raw (unnormalized) dataset — used in `step()` for physics calculations |
| `df` | Normalized dataset — used in `get_observe()` to feed the neural network |

### Spaces

```python
PRICE_LOOKAHEAD = 96  # steps of future DAM_Price appended to observation (24-hour horizon — full next-day curve)
PRICE_HISTORY   = 16  # steps of past  DAM_Price appended to observation (4-hour history)
LOAD_LOOKAHEAD  = 16  # steps of future Load appended to observation (4-hour horizon)
GTI_LOOKAHEAD   = 16  # steps of future GTI appended to observation (4-hour horizon)

action_space      = Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)
observation_space = Box(low=-inf, high=inf, shape=(163,), dtype=float32)
# 17 normalized feature cols + SoC + 96 price fwd + 16 price hist + 16 load + 16 GTI + 1 tomorrow solar → shape (163,)
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

### Reward Function (10 components, all tracked separately in `info`)

```
r_market         = |grid_export| × (DAM_price/1000)  OR  grid_import × -(DAM_price/1000 + 3.0)
r_lcos           = -(3.0 × lcos × |batt_energy_cycled|)
r_unmet          = -(unmet_load × (DAM_price/1000 + 3.0) × 5)        # if unmet_load > 0
r_mismatch       = 0.0                                                 # disabled (kept in info for compat)
r_soc_soft       = -(50.0 × violation²)                               # outside [soc_min, 0.80]
                 + -(100.0 × (target_soc - soc)²)                     # below target (grid up only)
r_reserve        = -(50.0 × soc_deficit × log1p(outage_remaining_h)) # during outage
r_preparation    = 20.0 × exp(-0.5 × hours_until_outage) × min(soc, target_soc)  # pre-outage (≤6 h)
r_soc_target     = solar_chem_stored × buy_price                      # solar-sourced charging while soc < target_soc
r_waste          = -(10.0 × lcos × wasted_kWh)                       # discharge exceeding demand + grid headroom
r_curtail        = -(curtailed_kWh × curr_price × 3.0)               # solar surplus discarded, not stored or exported (grid up)
r_solar_priority = -(grid_needed_for_batt × solar_fraction × curr_price × 4.0)  # grid charging while solar active (non-outage, load imports excluded)
r_price_timing   = ±price_dev × |grid_kwh| × 1.0                    # bonus sell-high / penalty buy-high
r_eod_soc        = max(0, soc - DEFAULT_SOC_TARGET) × capacity × buy_price × 0.15  # fired at step 95 only

reward = sum(all components) / (battery_capacity_kwh / 100.0)        # normalized by capacity
```

**Reward normalization**: dividing by `capacity / 100` keeps reward magnitude consistent across the domain-randomized hardware range (50–250 kWh), preventing large-battery configs from dominating the replay buffer.

**r_lcos coefficient = 2.5**: raised from 1.4 after backtest showed 1.33 cycles/day (target ≤ 1.0) and 166k UAH LCOS/year. Each kWh cycled through a 150 kWh battery costs `2.5 × 1.15 / 1.5 = 1.92` in normalized reward — comparable to the r_market benefit from arbitrage, making unnecessary cycling unprofitable.

**r_soc_soft below-target coefficient = 100.0**: raised from 25 to give a daily gap penalty of `-64` (normalized, 150 kWh battery, 10% below target over 96 steps) vs a single grid-charge step cost of `-90` — a margin of ~70% that SAC can reliably learn. At 25, the daily penalty was only `-16`, too weak to overcome charging cost.

**r_preparation window = 6h**: extended from 3h so the agent receives the positive readiness signal early enough to act (charging a 150 kWh battery at C/2 takes ~40 min; the wider window ensures the reward is seen well before charging is needed).

**r_soc_target rationale**: rewards solar-sourced charging at avoided-purchase value (`buy_price` per kWh stored). Grid charging is intentionally excluded — `r_market` already penalises the import cost, and including grid charging in `r_soc_target` would create a near-zero net signal that teaches free grid-charging. Fires whenever `solar_used_for_batt > 0` and `soc < target_soc` (no time gate).

**DEFAULT_SOC_TARGET = 0.30**: Economic baseline SoC used when no outage is forecasted and as a floor in `_calc_target_soc()` so small outages never collapse the target to `soc_soft_min`.

**r_waste**: penalizes discharging more than demand + available grid export headroom can absorb.

**r_curtail multiplier = 3.0×**: raised from 1.0× because the model was discarding ~10k kWh/year of solar surplus. Solar lost at the current step cannot be recovered at a better price later; the 3× multiplier ensures curtailment always costs more in reward than any `r_price_timing` gain from holding.

**r_solar_priority**: penalizes grid-sourced battery charging while solar is simultaneously active (> 0.1 kWh/step). Only fires when `grid_needed_for_batt > 0.01` — load-driven imports are excluded. Coefficient 2.0 × solar_fraction × curr_price × grid_needed_for_batt. Suppressed when outage is imminent (`hours_until_outage ≤ 3` and `soc < target_soc`). Backtest found 46.6% of all grid buying (69k kWh/year) happened during daylight, buying expensive grid power (8 UAH/kWh) while free solar was available.

### `info` dict (returned by `step()`)

`soc`, `target_soc`, `reward`, `solar_gen_ts_kwh`, `solar_surplus_kwh`, `actual_grid_kwh`, `battery_kwh` (+ = charging), `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward_market`, `reward_lcos`, `reward_unmet`, `reward_mismatch`, `reward_soc_soft`, `reward_reserve`, `reward_preparation`, `reward_soc_target`, `reward_waste`, `reward_curtail`, `reward_price_timing`, `reward_solar_priority`

---

## Normalization Pipeline (`environment/normalize.py`)

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

## Standalone Training Script (`environment/train.py`)

Run from project root:

```bash
cd /home/denys/PycharmProjects/ds_demo/ds_project_demo && .venv/bin/python environment/train.py
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

| Parameter | Value | Note |
|---|---|---|
| Algorithm | SAC (MlpPolicy) | |
| Total timesteps | 20 000 000 | extended from 15M; model_10 peaked at 5.9M, more headroom needed |
| Buffer size | 2 000 000 | doubled from 1M — covers ~10% of training experience at end |
| Batch size | 512 | |
| Learning rate | step decay | 1e-4 (0–12M) → 5e-5 (12–16M) → 2.5e-5 (16–20M); stabilises late training |
| Target entropy | -1.0 | explicit; 'auto'=−2 allowed near-deterministic collapse |
| Tau | 0.002 | lowered from 0.005 for more stable target network |
| Learning starts | 50 000 | raised from 10k — gives ~16 full episodes before first update |
| Clip reward | 100.0 | raised from 10.0 — 10.0 clipped r_unmet peaks (−587) to −10, losing signal |
| Network arch | [512, 512] | |
| n_envs | 32 (DummyVecEnv) | |
| n_eval_episodes | 50 | increased from 20 for a more stable eval signal |
| Gamma | 0.99 | |
| Entropy coef | auto | |
| Eval frequency | every 100 000 env steps | |
| Seed | 42 (numpy + SAC) | |
| Device | cuda | |

### Outputs

| Path | Contents |
|---|---|
| `models/sac_ems.zip` | Final model (end of training) |
| `models/best/best_model.zip` | Best checkpoint by eval reward ← **API uses this** |
| `models/obs_rms.pkl` | VecNormalize running stats — **required for inference** |
| `logs/tensorboard/` | TensorBoard event files |

View training: `tensorboard --logdir environment/logs/tensorboard/`

---

## Inference Module (`environment/inference.py`)

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

**`dispatch_plan`** per step includes: `step`, `action_battery`, `action_grid`, `soc`, `target_soc`, `solar_gen_kwh`, `solar_surplus_kwh`, `battery_kwh`, `grid_kwh`, `unmet_load_kwh`, `lcos_cost`, `mismatch`, `money_earned_ts`, `reward`, all 10 reward components.

**Model cache**: In FastAPI, `prediction_service.py` holds a single `_cached_model: tuple | None` — the global SAC model is loaded once from `best/best_model.zip` and reused for all configs. No per-config training or per-config model files.

**Standalone `__main__` mode** (`python environment/inference.py`):
- Calls `data_combiner.combine()` for live data; falls back to `combined.csv` if DAM unavailable
- `--soc` arg is optional; if omitted, queries DB for last prediction SoC (clamped to `min_reserve`); defaults to 0.5 if DB unavailable
- Args: `--model`, `--scalers`, `--obsrms`, `--config` (JSON), `--output`, `--soc`, `--tilt`, `--azimuth`

---

## Default Strategy (`environment/default_strategy.py`)

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
| GET | `/predictions/history` | JWT | Retrieve stored SAC predictions. Params: `config_name`, `date` (optional, defaults to latest). |
| GET | `/predictions/history/dates` | JWT | List all dates with stored predictions for a config. Param: `config_name`. |

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
| `predictions` | id, user_id (FK), config_id (FK), date, step, timestamp + 35 physics/reward cols (6 energy-flow cols always NULL — see issue #10) |

`predictions` keyed by `config_id + date` — multiple configs per user don't collide.

### Prediction Service details

- **Global model cache**: `_cached_model: tuple | None` — SAC model loaded once from `environment/models/best/best_model.zip`, reused for all requests. No per-config training or model files.
- **Timezone**: gate and date calculations use `UTC+2` (Ukrainian time)
- **Paths**: absolute paths anchored to `Path(__file__).resolve().parent.parent.parent`
- **`initial_soc`**: if omitted, read from DB (last step SoC of previous prediction for this config), clamped to `min_reserve`
- **SAC predictions** (`GET /predictions/`): result stored to `predictions` table (96 rows per run)
- **Default strategy** (`GET /predictions/default`): result returned only, not stored in DB
- **History retrieval** (`GET /predictions/history`): reads stored rows; accepts optional `date` param, defaults to latest stored date
- **History dates** (`GET /predictions/history/dates`): returns list of ISO date strings with stored predictions for a config

---

## Frontend (`frontend/`)

### Stack
- **Vite** + **React 18** + **Recharts** — single-page app, no router needed
- All CSS is inlined as a JS string in `App.jsx` (no separate stylesheet) — Sora + JetBrains Mono fonts

### Local dev
```bash
cd frontend
cp .env.example .env        # set VITE_API_URL=http://localhost:8000
npm install
npm run dev                 # → http://localhost:3000
```

### Build for S3 deployment
```bash
npm run build               # outputs to frontend/dist/
aws s3 sync dist/ s3://<bucket-name>/ --delete
# Invalidate CloudFront cache after upload
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"
```

### `src/api.js`
All fetch calls with JWT handling. Token stored in `localStorage` under key `ems_token`. On 401, clears token and reloads page to show login screen. Reads `VITE_API_URL` env var at build time.

### `src/App.jsx` — screens & components

| Component | Description |
|---|---|
| `Login` | Login + Register tabs. POST `/auth/login` (form-urlencoded) → stores token |
| `Sidebar` | Mode tabs (SAC / Default / Історія), config/strategy dropdowns (from API), initial SoC input, run button, CSV upload, logout |
| `Kpis` | 4 KPI cards from `summary` object (earned, sold/bought, solar, unmet) |
| `Charts` | SoC area chart (full width), Solar area chart, Grid bar chart (green=sell / red=buy) |
| `Table` | Paginated dispatch table (20 rows/page) — timestamp, SoC bar, battery direction, grid badge, solar, load, DAM price, P&L |

### Sidebar modes
| Mode | What it runs | Stored to DB? |
|---|---|---|
| SAC | `GET /predictions/` | Yes |
| Default | `GET /predictions/default` | No |
| Історія | `GET /predictions/history` | Read-only |

Config and strategy dropdowns auto-populate from `/config/list` and `/strategy/list` on login; refresh button re-fetches. If list is empty, falls back to a text input for manual entry. History dates auto-load from `/predictions/history/dates` when switching to history mode.

### CSV upload (offline/demo mode)
Accepts backtest output CSV (`environment/testing/results/sac_dispatch.csv`). Handles both `solar_kwh` (API) and `solar_gen_kwh` (CSV) column names. KPIs are computed locally from the CSV rows.

---

## Known Issues & Open Work

| # | Location | Status | Issue |
|---|---|---|---|
| 1 | `IDM_DAM_features.py` | ❌ Open | IDM fetching not implemented; only DAM active |
| 2 | `requirements.txt` | ✅ Fixed | All deps present: `scikit-learn`, `psycopg2-binary`, `openpyxl`, `uvicorn` |
| 3 | Real SoC input | ⚠️ Manual | `initial_soc` auto-persisted via DB (API) but no live BMS/inverter integration |
| 4 | `GET /predictions/history` | ✅ Fixed | Endpoints added: `/predictions/history` (by date) + `/predictions/history/dates` (list) |
| 5 | SAC retraining required | ⚠️ Pending | model_10 retrain in progress (see issues #28–#30). Run: `cd environment && ../.venv/bin/python train.py` |
| 6 | `data_combiner.py` | ✅ Fixed | `get_solar_parameters()` now has try/except — no longer crashes when DB is down |
| 7 | `inference.py __main__` | ✅ Fixed | Project root added to `sys.path` so `data_providers` import works when running directly |
| 8 | `inference.py __main__` | ✅ Fixed | NaN guard added — aborts with clear error if DAM columns are all NaN instead of crashing PyTorch |
| 9 | `combined.csv` | ⚠️ External | DAM_Price/Vol columns are NaN when OREE fetch fails — inference will abort cleanly but needs live DAM data to run |
| 10 | `AgentPredictions` energy flow cols | ⚠️ Schema only | 6 columns (`solar_to_load_kwh`, `solar_to_battery_kwh`, `solar_to_grid_kwh`, `battery_to_load_kwh`, `grid_to_load_kwh`, `grid_to_battery_kwh`) in DB model but never populated by `_build_rows()` in `prediction_service.py` — always NULL |
| 11 | `inference.py` | ✅ Fixed | `reward_curtail` now captured from `info` dict and included in dispatch plan; `_build_rows()` stores it |
| 12 | `backend/models/site.py` | ✅ Fixed | `reward_curtail` column added to `AgentPredictions` — **existing DBs need `ALTER TABLE predictions ADD COLUMN reward_curtail FLOAT;`** |
| 13 | `environment/environment.py` | ✅ Fixed | Discharge efficiency: DC request converted to chemical kWh (`chem_needed = energy_to_draw / η`) before clamping to `max_drawable`; SoC drain and DC output now physically consistent. Requires retraining. |
| 14 | `data_combiner.py` | ✅ Fixed | Year-boundary timestamps: rollover detected via `(month, day) < (today.month, today.day)` — uses `today.year + 1` on Dec 31 → Jan 1 runs |
| 15 | `environment/environment.py` | ✅ Fixed | Island-mode SoC over-drain: during outages battery was discharged at full requested rate (18 kWh/step) regardless of load (5-6 kWh/step), depleting SoC 3-4× too fast. Fixed by clamping `actual_draw = min(actual_draw, load_chem)` when `grid_status == 0`. Requires retraining. |
| 16 | `environment/environment.py` | ✅ Fixed | r_waste never fired during outages: `grid_headroom` was computed as if grid were connected (18.75 kWh), making `wasted = 0` always during outages. Fixed: `grid_headroom = 0.0 if grid_status == 0 else ...`. Requires retraining. |
| 17 | `environment/environment.py` | ✅ Fixed | Phantom SoC gain during outage: charging branch never checked grid_status, so battery received energy from a non-existent grid source during outages (SoC climbed with no solar/grid), and `grid_needed_for_batt` inflated `unmet_load` with phantom demand. Fixed by capping to solar-only when `grid_status == 0`. Requires retraining. |
| 18 | `environment/environment.py` | ✅ Fixed | r_unmet multiplier too weak (2×): model regressed on unmet load at 20M steps (7,745 kWh) vs 10M steps (5,030 kWh) because other rewards dominated. Raised to 5× so unmet load is always the worst possible outcome. Requires retraining. |
| 19 | `environment/environment.py` | ✅ Fixed | Price arbitrage backwards: model bought expensive (avg 5.65 UAH/kWh) and sold cheap (avg 4.50 UAH/kWh) despite having 32-step price lookahead in observation. Added `r_price_timing` component: bonus for selling above day-average price, penalty for buying above day-average price. Requires retraining. |
| 20 | `environment/environment.py` | ✅ Fixed | r_price_timing coefficient 2.0 → 1.0: model was too aggressive at holding energy, causing r_curtail = -52k UAH/year and summer months going negative. Reduced coefficient keeps the buy-cheap/sell-expensive signal without over-holding. Requires retraining. |
| 21 | `environment/environment.py` | ✅ Fixed | Price-timing buy penalty suppressed when outage imminent (hours_until_outage ≤ 3 and soc < target_soc): model delayed grid charging before outages to avoid price penalty, arriving underprepared. Now bypasses price check when pre-outage charging is urgent. Requires retraining. |
| 22 | `environment/environment.py` | ✅ Fixed | r_lcos coefficient 1.2 → 1.4: model cycled battery excessively for arbitrage (LCOS 166k vs 131k in prior run). Higher coefficient discourages unnecessary cycling while still allowing profitable arbitrage. Requires retraining. |
| 23 | `environment/environment.py` | ✅ Fixed | r_lcos coefficient raised 1.4 → 2.5: backtest shows 1.33 cycles/day (target ≤ 1.0) and 166k UAH LCOS/year. Stronger penalty required to suppress excessive cycling. Requires retraining. |
| 24 | `environment/environment.py` | ✅ Fixed | r_curtail multiplier 1.0× → 3.0×: model curtails ~10k kWh/year of solar because r_price_timing bonus for holding outweighs the curtailment loss. 3× penalty ensures discarded solar always costs more than any timing gain. Requires retraining. |
| 25 | `environment/environment.py` | ✅ Fixed | Added r_solar_priority: 46.6% of grid buying (69k kWh/year) happened during daylight when solar was active, wasting free solar and inflating grid costs. New component penalizes grid import proportional to concurrent solar fraction. Bypassed when outage is imminent. Requires retraining. |
| 26 | `environment/inference.py`, `default_strategy.py` | ✅ Fixed | Misleading `total_money_earned` metric: showed −614k UAH but the system actually saved +1.47M UAH vs grid-only (81% self-sufficiency). Added `economic_savings_uah` = cash_flow + solar_self_consumed × avg_buy_price − LCOS to both SAC and default strategy summaries. |
| 27 | Hardware sizing | ⚠️ Physical limit | December/January unmet load (Dec: 73 kWh/day, Jan: 41 kWh/day) is a fundamental hardware constraint: solar generates only 229 kWh/day in December but load is 781 kWh/day, and the 150 kWh battery cannot bridge a 552 kWh/day deficit during outages (6.68 h/day in December). Cannot be fixed with reward tuning; requires larger battery (400+ kWh) or accepting winter grid dependency. |
| 28 | `environment/train.py` | ✅ Fixed | SAC entropy collapse: both model_7 (SAC_69) and model_8 (SAC_70) had alpha crash to ~0.0006 by step 600k. Root causes: `target_entropy='auto'` (=-2, too permissive), `lr=3e-4` (overconfident critic), `learning_starts=10k` (only 312 steps/env before first update), `tau=0.005`, `clip_reward=10.0` (clipped r_unmet peaks of −587 to −10). Fixed: `target_entropy=-1.0`, `lr=1e-4`, `learning_starts=50k`, `tau=0.002`, `clip_reward=100.0`. |
| 29 | `environment/environment.py` | ✅ Fixed | r_lcos coefficient 2.5 → 4.0 overcorrected: model_9 hit 0.989 cycles/day but curtailed 28,475 kWh/year (vs ~10k for model_7) because the agent refused profitable cycles, wasting free solar. Economic savings dropped from 1.47M (model_7) to 483k UAH. Rolled back to 2.5 — the goal is economically optimal cycling, not minimizing cycle count at any cost. |
| 30 | `environment/models/best/` | ✅ Done | Model_10 trained (SAC_72). Economic savings 459k UAH, cycles/day 1.12. |
| 31 | `environment/environment.py`, `train.py` | ⚠️ Pending | **Model_11 retrain required.** Fixes applied: PRICE_LOOKAHEAD 32→96 (obs 98→163), r_lcos 2.5→3.0, r_solar_priority 2.0→4.0, r_eod_soc added (end-of-day carry bonus), tomorrow solar summary feature added. train.py: 20M steps, 2M buffer, n_eval_episodes=50, LR step decay. Run: `cd environment && ../.venv/bin/python train.py` |

---

## Python Dependencies

Current `requirements.txt` (complete — no missing deps):
```
fastapi, uvicorn, pandas, numpy, requests, openmeteo-requests, requests-cache,
retry-requests, gymnasium, python-calamine, sqlalchemy, psycopg2-binary, pyjwt,
python-dotenv, pwdlib, pydantic, scikit-learn, stable-baselines3, torch,
tensorboard, openpyxl
```


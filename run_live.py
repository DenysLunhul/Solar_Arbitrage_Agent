"""
run_live.py — fetch tomorrow's data and run SAC inference.
Usage: python run_live.py [--soc 0.5] [--tilt 35] [--azimuth 0]
"""
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from data_providers.components.time.time_features import fetch_time
from data_providers.components.grid.synthetic_grid import fetch_grid
from data_providers.components.load.synthetic_load import fetch_load
from data_providers.components.weather.weather import fetch_weather
from data_providers.components.market_manager.DAM_features import fetch_DAM
from environment.inference import load_model_and_scalers, run_inference

SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        100.0,
        'max_charge_power':    50.0,
        'max_discharge_power': 50.0,
        'efficiency':          0.95,
        'lcos':                1.5,
        'min_reserve':         15,
    },
    'solar': {
        'peak_power':  100.0,
        'efficiency':  0.20,
    },
    'inverter': {
        'max_power':    100.0,
        'efficiency':   0.95,
    },
    'grid': {
        'capacity': 100.0,
    },
}

def build_dataset(today, tilt, azimuth) -> pd.DataFrame:
    print("Fetching DAM prices...")
    dam = fetch_DAM(today)
    if dam is None:
        raise RuntimeError(
            "DAM prices not available yet (OREE publishes after 14:00 UA time). "
            "Try again later or check oree.com.ua manually."
        )

    print("Fetching weather forecast...")
    weather = fetch_weather(today, tilt, azimuth)

    print("Building time / grid / load features...")
    time   = fetch_time(today)
    grid   = fetch_grid(today)
    load   = fetch_load(today)

    df = pd.concat([time, grid, load, weather, dam], axis=1)

    current_year = datetime.now().year
    df['timestamp'] = df.apply(
        lambda r: datetime(current_year, int(r['Month']), int(r['Day']),
                           int(r['Hour']), int(r['Minute'])),
        axis=1,
    )
    cols = df.columns.tolist()
    df = df[['timestamp'] + [c for c in cols if c != 'timestamp']]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--soc',     type=float, default=0.5,  help='Initial battery SoC (0-1)')
    parser.add_argument('--tilt',    type=float, default=35.0, help='Solar panel tilt (degrees)')
    parser.add_argument('--azimuth', type=float, default=0.0,  help='Solar panel azimuth (degrees)')
    args = parser.parse_args()

    today = datetime.today()
    tomorrow_str = (today.replace(hour=0, minute=0, second=0, microsecond=0)
                    .__class__(today.year, today.month, today.day)).strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"Live inference  |  tomorrow: {tomorrow_str}  |  initial SoC: {args.soc:.0%}")
    print(f"{'='*60}\n")

    df_raw = build_dataset(today, args.tilt, args.azimuth)

    raw_csv = ROOT / "data_providers" / "orchestrator" / "combined.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"Raw data saved → {raw_csv}  ({len(df_raw)} rows)\n")

    model, scalers, obs_rms = load_model_and_scalers(
        model_path=str(ROOT / "environment" / "models" / "best" / "best_model.zip"),
        scalers_path=str(ROOT / "environment" / "models" / "scalers.pkl"),
        obs_rms_path=str(ROOT / "environment" / "models" / "obs_rms.pkl"),
    )

    result = run_inference(
        df_raw=df_raw,
        system_config=SYSTEM_CONFIG,
        model=model,
        scalers=scalers,
        obs_rms=obs_rms,
        initial_soc=args.soc,
    )

    plan = pd.DataFrame(result['dispatch_plan'])

    s = result['summary']
    print(f"\n{'='*60}")
    print("ПІДСУМКИ")
    print(f"{'='*60}")
    print(f"  Кроків:              {s['steps']}")
    print(f"  Зароблено:           {s['total_money_earned']:.2f} UAH")
    print(f"  Сонячна генерація:   {s['solar_kwh']:.2f} kWh")
    print(f"  Куплено з мережі:    {s['bought_kwh']:.3f} kWh")
    print(f"  Продано в мережу:    {s['sold_kwh']:.3f} kWh")
    print(f"  Непокрите:           {s['unmet_load_kwh']:.4f} kWh")
    print(f"  Деградація батареї:  {s['lcos_total_uah']:.3f} UAH")
    print(f"  SoC: {s['initial_soc']:.0%} → {s['final_soc']:.0%}")

    print(f"\n{'='*60}")
    print("DISPATCH PLAN (перші 12 кроків — перші 3 год)")
    print(f"{'='*60}")
    cols = ['step', 'soc', 'solar_gen_kwh', 'battery_kwh', 'grid_kwh',
            'money_earned_ts', 'unmet_load_kwh']
    print(plan[cols].head(12).to_string(index=False))

    out_csv = ROOT / "results" / "live_dispatch.csv"
    out_csv.parent.mkdir(exist_ok=True)
    plan.to_csv(out_csv, index=False)
    print(f"\nПовний dispatch plan → {out_csv}")

if __name__ == '__main__':
    main()

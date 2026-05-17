"""
inverter_dispatch.py
====================
Dispatch plan based on primitive hybrid inverter logic.
No price awareness or outage forecasting — reacts only to solar irradiance,
grid presence, and current SoC.
"""

import argparse
import numpy as np
import pandas as pd
from environment import Environment

DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        200.0,
        'max_charge_power':    150,
        'max_discharge_power': 150,
        'efficiency':          0.95,
        'lcos':                1.5,
        'min_reserve':         20,
    },
    'solar': {
        'peak_power':  150.0,
        'efficiency':  0.23,
    },
    'inverter': {
        'max_power':    100.0,
        'efficiency': 0.95,
    },
    'grid': {
        'capacity': 150.0,
    },
}


def inverter_action(row: pd.Series, soc: float) -> np.ndarray:
    """Primitive hybrid inverter logic — no prices, no outage forecast."""
    gti         = float(row['Global_tilted_irradiance_instant'])
    grid_status = int(row['Grid'])

    TARGET_SOC           = 0.70
    MAX_SOC              = 0.95
    MIN_SOLAR_THRESHOLD  = 10.0   # W/m²
    HIGH_SOLAR_THRESHOLD = 400.0  # W/m²

    if grid_status == 0:
        # No grid: discharge battery to serve load
        return np.array([-1.0, 0.0], dtype=np.float32)

    if gti > MIN_SOLAR_THRESHOLD:
        if soc < TARGET_SOC:
            # Charge battery to 70% first
            return np.array([1.0, 0.0], dtype=np.float32)
        elif gti > HIGH_SOLAR_THRESHOLD and soc < MAX_SOC:
            # Abundant sun: top up to 95% and export surplus
            return np.array([1.0, 1.0], dtype=np.float32)
        else:
            # Battery at target: export all solar surplus
            return np.array([0.0, 1.0], dtype=np.float32)
    else:
        # Night / low irradiance: discharge battery, grid covers the rest
        return np.array([-1.0, -1.0], dtype=np.float32)


def generate_dispatch_plan(df_raw: pd.DataFrame, df_norm: pd.DataFrame, config: dict, initial_soc: float, output_file: str):
    print("Running baseline inverter simulation...")

    env = Environment(df_raw=df_raw, df=df_norm, system_config=config)
    env.reset()
    env.soc = initial_soc

    dispatch_history = []

    while True:
        curr_step = env.curr_step
        row = df_raw.iloc[curr_step]

        action = inverter_action(row, env.soc)

        _, _, terminated, truncated, info = env.step(action)

        grid_kwh = info.get('actual_grid_kwh', 0)
        sell_price = float(row['DAM_Price']) / 1000
        buy_price  = sell_price + 3.0
        money_earned = 0.0
        if grid_kwh < 0:
            money_earned = abs(grid_kwh) * sell_price
        elif grid_kwh > 0:
            money_earned = -(grid_kwh * buy_price)

        record = {
            'step': curr_step,
            'action_battery': round(action[0], 4),
            'action_grid': round(action[1], 4),
            'soc': round(info.get('soc', env.soc), 4),
            'target_soc': None,
            'solar_gen_kwh': round(info.get('solar_gen_ts_kwh', 0), 4),
            'solar_surplus_kwh': round(info.get('solar_surplus_kwh', 0), 4),
            'battery_kwh': round(info.get('battery_kwh', 0), 4),
            'grid_kwh': round(grid_kwh, 4),
            'unmet_load_kwh': round(info.get('unmet_load_kwh', 0), 4),
            'lcos_cost': round(info.get('lcos_cost', 0), 4),
            'mismatch': round(info.get('mismatch', 0), 4),
            'money_earned_ts': round(money_earned, 4)
        }

        dispatch_history.append(record)

        if terminated or truncated:
            break

    df_plan = pd.DataFrame(dispatch_history)
    df_plan.to_csv(output_file, index=False)

    print(f"Done. Generated {len(df_plan)} steps.")
    print(f"Saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',     default='dataset_final.csv')
    parser.add_argument('--norm',    default='dataset_normalized.csv')
    parser.add_argument('--soc',     type=float, default=0.5)
    parser.add_argument('--days',    type=int,   default=1)
    parser.add_argument('--out',     default='results/dispatch_plan_inverter.csv')
    args = parser.parse_args()

    print("Loading data...")
    df_raw  = pd.read_csv(args.raw)
    df_norm = pd.read_csv(args.norm)

    if args.days is not None:
        df_raw  = df_raw.iloc[:args.days * 96].reset_index(drop=True)
        df_norm = df_norm.iloc[:args.days * 96].reset_index(drop=True)

    print(f"Data: {len(df_raw)} rows ({len(df_raw)//96} days)")

    generate_dispatch_plan(
        df_raw=df_raw,
        df_norm=df_norm,
        config=DEFAULT_SYSTEM_CONFIG,
        initial_soc=args.soc,
        output_file=args.out
    )

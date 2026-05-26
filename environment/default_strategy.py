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

try:
    from environment import Environment
except ImportError:
    from environment.environment import Environment

DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        250.0,
        'max_charge_power':    125.0,
        'max_discharge_power': 125.0,
        'efficiency':          0.95,
        'lcos':                1.15,
        'min_reserve':         20,
    },
    'solar': {
        'peak_power':  250.0,
        'efficiency':  0.20,
    },
    'inverter': {
        'max_power': 250.0,
    },
    'grid': {
        'capacity': 250.0,
    },
}

DEFAULT_STRATEGY = {
    'target_soc':             0.70,
    'max_soc':                0.95,
    'min_solar_threshold':    10.0,
    'high_solar_threshold':   400.0,
    'solar_surplus_priority': 'charge_first',
    'night_discharge':        True,
    'night_sell':             False,
    'allow_grid_charging':    False,
    'outage_reserve':         0.0,
}

def inverter_action(row: pd.Series, soc: float, strategy: dict | None = None) -> np.ndarray:
    """Rule-based hybrid inverter dispatch driven by a user-configurable strategy dict."""
    if strategy is None:
        strategy = DEFAULT_STRATEGY

    gti         = float(row['Global_tilted_irradiance_instant'])
    grid_status = int(row['Grid'])

    target_soc      = strategy.get('target_soc',             DEFAULT_STRATEGY['target_soc'])
    max_soc         = strategy.get('max_soc',                 DEFAULT_STRATEGY['max_soc'])
    min_solar       = strategy.get('min_solar_threshold',     DEFAULT_STRATEGY['min_solar_threshold'])
    high_solar      = strategy.get('high_solar_threshold',    DEFAULT_STRATEGY['high_solar_threshold'])
    sol_priority    = strategy.get('solar_surplus_priority',  DEFAULT_STRATEGY['solar_surplus_priority'])
    night_discharge = strategy.get('night_discharge',         DEFAULT_STRATEGY['night_discharge'])
    night_sell      = strategy.get('night_sell',              DEFAULT_STRATEGY['night_sell'])
    grid_charging   = strategy.get('allow_grid_charging',     DEFAULT_STRATEGY['allow_grid_charging'])
    outage_reserve  = strategy.get('outage_reserve',          DEFAULT_STRATEGY['outage_reserve'])

    if grid_status == 0:
        if soc > outage_reserve:
            return np.array([-1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    has_solar      = gti > min_solar
    abundant_solar = gti > high_solar

    if has_solar:
        if soc < target_soc:
            return np.array([1.0, 0.0], dtype=np.float32)

        if sol_priority == 'charge_first':
            if soc < max_soc:
                return np.array([1.0, 1.0], dtype=np.float32)
            return np.array([0.0, 1.0], dtype=np.float32)

        else:
            return np.array([0.0, 1.0], dtype=np.float32)

    else:
        if night_discharge:
            grid_act = 1.0 if night_sell else 0.0
            return np.array([-1.0, grid_act], dtype=np.float32)
        if grid_charging and soc < target_soc:
            return np.array([1.0, -1.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

def generate_dispatch_plan(
    df_raw:      pd.DataFrame,
    df_norm:     pd.DataFrame,
    config:      dict,
    initial_soc: float,
    strategy:    dict | None = None,
    output_file: str | None = None,
) -> dict:
    """
    Run the default strategy over df_raw and return a result dict compatible
    with run_inference() output: {'dispatch_plan': [...], 'summary': {...}}.
    Optionally saves a CSV if output_file is provided.
    """
    env = Environment(df_raw=df_raw, df=df_norm, system_config=config)
    env.reset()
    env.soc = initial_soc

    dispatch_plan = []

    while True:
        curr_step = env.curr_step
        row = df_raw.iloc[curr_step]

        action = inverter_action(row, env.soc, strategy)
        _, _, terminated, truncated, info = env.step(action)

        dispatch_plan.append({
            'step':              curr_step,
            'action_battery':    round(float(action[0]), 4),
            'action_grid':       round(float(action[1]), 4),
            'soc':               round(float(info['soc']), 4),
            'target_soc':        round(float(info['target_soc']), 4),
            'solar_gen_kwh':     round(float(info['solar_gen_ts_kwh']), 4),
            'solar_surplus_kwh': round(float(info['solar_surplus_kwh']), 4),
            'battery_kwh':       round(float(info['battery_kwh']), 4),
            'grid_kwh':          round(float(info['actual_grid_kwh']), 4),
            'unmet_load_kwh':    round(float(info['unmet_load_kwh']), 4),
            'lcos_cost':         round(float(info['lcos_cost']), 4),
            'curtailed_kwh':     round(float(info['curtailed_kwh']), 4),
            'money_earned_ts':   round(float(info['money_earned_ts']), 4),
        })

        if terminated or truncated:
            break

    _total_money = round(sum(s['money_earned_ts'] for s in dispatch_plan), 2)
    _bought_kwh  = round(sum(s['grid_kwh'] for s in dispatch_plan if s['grid_kwh'] > 0), 3)
    _sold_kwh    = round(sum(abs(s['grid_kwh']) for s in dispatch_plan if s['grid_kwh'] < 0), 3)
    _solar_kwh   = round(sum(s['solar_gen_kwh'] for s in dispatch_plan), 3)
    _lcos_uah    = round(sum(s['lcos_cost'] for s in dispatch_plan), 3)

    _total_bought_cost = sum(-s['money_earned_ts'] for s in dispatch_plan if s['grid_kwh'] > 0)
    _avg_buy_price     = (_total_bought_cost / _bought_kwh) if _bought_kwh > 0 else 8.0
    _curtailed_kwh     = round(sum(s['curtailed_kwh'] for s in dispatch_plan), 3)
    _solar_self        = max(0.0, _solar_kwh - _sold_kwh - _curtailed_kwh)
    _economic_savings  = round(_total_money + _solar_self * _avg_buy_price - _lcos_uah, 2)

    summary = {
        'total_money_earned':  _total_money,
        'economic_savings_uah': _economic_savings,
        'bought_kwh':          _bought_kwh,
        'sold_kwh':            _sold_kwh,
        'solar_kwh':           _solar_kwh,
        'curtailed_kwh':       _curtailed_kwh,
        'unmet_load_kwh':      round(sum(s['unmet_load_kwh'] for s in dispatch_plan), 4),
        'lcos_total_uah':      _lcos_uah,
        'initial_soc':         round(initial_soc, 3),
        'final_soc':           dispatch_plan[-1]['soc'] if dispatch_plan else initial_soc,
        'steps':               len(dispatch_plan),
    }

    if output_file:
        pd.DataFrame(dispatch_plan).to_csv(output_file, index=False)
        print(f"Saved {len(dispatch_plan)} steps → {output_file}")

    return {'dispatch_plan': dispatch_plan, 'summary': summary}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',     default='dataset_final.csv')
    parser.add_argument('--norm',    default='dataset_normalized.csv')
    parser.add_argument('--soc',     type=float, default=0.5)
    parser.add_argument('--days',    type=int,   default=1)
    parser.add_argument('--out',     default='results/dispatch_plan_default.csv')
    args = parser.parse_args()

    print("Loading data...")
    df_raw  = pd.read_csv(args.raw)
    df_norm = pd.read_csv(args.norm)

    if args.days is not None:
        df_raw  = df_raw.iloc[:args.days * 96].reset_index(drop=True)
        df_norm = df_norm.iloc[:args.days * 96].reset_index(drop=True)

    print(f"Data: {len(df_raw)} rows ({len(df_raw)//96} days)")

    result = generate_dispatch_plan(
        df_raw=df_raw,
        df_norm=df_norm,
        config=DEFAULT_SYSTEM_CONFIG,
        initial_soc=args.soc,
        output_file=args.out,
    )
    print(f"Total earned: {result['summary']['total_money_earned']} UAH")
    print(f"Final SoC:    {result['summary']['final_soc']:.3f}")

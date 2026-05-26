"""
SAC agent backtest — full training dataset (365 days × 96 steps).
SoC is chained day-to-day so battery state carries over realistically.

Outputs:
  testing/results/sac_dispatch.csv   — 35 040 step-level rows
  testing/results/sac_summary.csv    — 365 day-level rows
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR))
os.chdir(_DIR)

from inference import load_model_and_scalers, run_inference

EVAL_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        150.0,
        'max_charge_power':     75.0,
        'max_discharge_power':  75.0,
        'efficiency':          0.95,
        'lcos':                1.15,
        'min_reserve':         20,
    },
    'solar': {
        'peak_power':  200.0,
        'efficiency':  0.2,
    },
    'inverter': {
        'max_power': 180.0,
    },
    'grid': {
        'capacity': 220.0,
    },
}

RESULTS_DIR = Path(__file__).resolve().parent / 'results'

def run(
    dataset_path: str,
    model_path:   str,
    scalers_path: str,
    obs_rms_path: str,
    system_config: dict,
    initial_soc: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df_raw = pd.read_csv(dataset_path)
    n_days = len(df_raw) // 96
    print(f"Dataset: {len(df_raw)} rows → {n_days} days")

    model, scalers, obs_rms = load_model_and_scalers(model_path, scalers_path, obs_rms_path)

    all_steps, all_summary = [], []
    soc = initial_soc

    for day_idx in range(n_days):
        day_df   = df_raw.iloc[day_idx * 96 : (day_idx + 1) * 96].reset_index(drop=True)
        date_str = str(day_df['timestamp'].iloc[0])[:10]

        result = run_inference(
            df_raw=day_df,
            system_config=system_config,
            model=model,
            scalers=scalers,
            obs_rms=obs_rms,
            initial_soc=soc,
        )

        for step in result['dispatch_plan']:
            step['date'] = date_str
        all_steps.extend(result['dispatch_plan'])

        summary = result['summary']
        summary['date']        = date_str
        summary['initial_soc'] = round(soc, 4)
        all_summary.append(summary)

        soc = result['summary']['final_soc']

        if (day_idx + 1) % 30 == 0 or day_idx == n_days - 1:
            print(f"  Day {day_idx+1:3d}/{n_days}  {date_str}  "
                  f"earned={summary['total_money_earned']:>8.2f} UAH  "
                  f"final_soc={soc:.3f}")

    dispatch_df = pd.DataFrame(all_steps)
    summary_df  = pd.DataFrame(all_summary)

    col_order = ['date', 'step'] + [c for c in dispatch_df.columns if c not in ('date', 'step')]
    dispatch_df = dispatch_df[col_order]

    sum_cols = ['date', 'initial_soc', 'final_soc', 'total_money_earned',
                'economic_savings_uah', 'total_reward_uah', 'solar_kwh',
                'bought_kwh', 'sold_kwh', 'unmet_load_kwh', 'lcos_total_uah', 'steps']
    summary_df = summary_df[[c for c in sum_cols if c in summary_df.columns]]

    return dispatch_df, summary_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(_DIR / 'dataset_final.csv'))
    parser.add_argument('--model',   default=str(_DIR / 'models' / 'best' / 'best_model.zip'))
    parser.add_argument('--scalers', default=str(_DIR / 'models' / 'scalers.pkl'))
    parser.add_argument('--obsrms',  default=str(_DIR / 'models' / 'obs_rms.pkl'))
    parser.add_argument('--soc',     type=float, default=0.2)
    args = parser.parse_args()

    dispatch_df, summary_df = run(
        dataset_path=args.dataset,
        model_path=args.model,
        scalers_path=args.scalers,
        obs_rms_path=args.obsrms,
        system_config=EVAL_SYSTEM_CONFIG,
        initial_soc=args.soc,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    dispatch_df.to_csv(RESULTS_DIR / 'sac_dispatch.csv', index=False)
    summary_df.to_csv(RESULTS_DIR / 'sac_summary.csv', index=False)

    print("\n" + "=" * 60)
    print("SAC — OVERALL RESULTS (365 days)")
    print("=" * 60)
    print(f"  Grid cash flow:    {summary_df['total_money_earned'].sum():>12.2f} UAH  (sell - buy)")
    if 'economic_savings_uah' in summary_df.columns:
        print(f"  Economic savings:  {summary_df['economic_savings_uah'].sum():>12.2f} UAH  (vs grid-only baseline)")
    print(f"  Solar generated:   {summary_df['solar_kwh'].sum():>12.1f} kWh")
    print(f"  Grid bought:       {summary_df['bought_kwh'].sum():>12.1f} kWh")
    print(f"  Grid sold:         {summary_df['sold_kwh'].sum():>12.1f} kWh")
    print(f"  Unmet load:        {summary_df['unmet_load_kwh'].sum():>12.3f} kWh")
    print(f"  LCOS cost:         {summary_df['lcos_total_uah'].sum():>12.2f} UAH")
    print(f"  Avg daily earned:  {summary_df['total_money_earned'].mean():>12.2f} UAH")
    print(f"\nDispatch → {RESULTS_DIR / 'sac_dispatch.csv'}")
    print(f"Summary  → {RESULTS_DIR / 'sac_summary.csv'}")

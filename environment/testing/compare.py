"""
Compare SAC vs default strategy backtest results.
Run after both backtests have produced their CSVs.

Usage:
  python testing/compare.py
"""

import sys
import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'results'

def load(name: str) -> pd.DataFrame | None:
    path = RESULTS / f'{name}_summary.csv'
    if not path.exists():
        print(f"  Missing: {path}  (run backtest_{name}.py first)")
        return None
    return pd.read_csv(path)

def fmt(val: float, unit: str = '') -> str:
    return f"{val:>12.2f}{unit}"

def main():
    sac     = load('sac')
    default = load('default')

    if sac is None and default is None:
        sys.exit(1)

    print("\n" + "=" * 65)
    print(f"{'METRIC':<28} {'DEFAULT':>14} {'SAC':>14} {'DELTA':>14}")
    print("=" * 65)

    metrics = [
        ('Grid cash flow (UAH)',       'total_money_earned',   True),
        ('Economic savings (UAH)',     'economic_savings_uah', True),
        ('Avg daily cash flow (UAH)',  None,                   True),
        ('Solar generated (kWh)',      'solar_kwh',            None),
        ('Grid bought (kWh)',          'bought_kwh',           False),
        ('Grid sold (kWh)',            'sold_kwh',             True),
        ('Unmet load (kWh)',           'unmet_load_kwh',       False),
        ('LCOS cost (UAH)',            'lcos_total_uah',       False),
    ]

    for label, col, higher_is_better in metrics:
        d_val = s_val = None

        if col == 'total_money_earned':
            d_val = default['total_money_earned'].sum() if default is not None else None
            s_val = sac['total_money_earned'].sum()     if sac is not None else None
        elif col is None:
            d_val = default['total_money_earned'].mean() if default is not None else None
            s_val = sac['total_money_earned'].mean()     if sac is not None else None
        else:
            d_val = default[col].sum() if default is not None else None
            s_val = sac[col].sum()     if sac is not None else None

        d_str = f"{d_val:>12.2f}" if d_val is not None else f"{'N/A':>12}"
        s_str = f"{s_val:>12.2f}" if s_val is not None else f"{'N/A':>12}"

        if d_val is not None and s_val is not None:
            delta = s_val - d_val
            sign  = '+' if delta >= 0 else ''
            if higher_is_better is True:
                marker = '↑' if delta > 0 else ('↓' if delta < 0 else '=')
            elif higher_is_better is False:
                marker = '↑' if delta < 0 else ('↓' if delta > 0 else '=')
            else:
                marker = ''
            delta_str = f"{sign}{delta:>10.2f} {marker}"
        else:
            delta_str = f"{'':>13}"

        print(f"  {label:<26} {d_str} {s_str} {delta_str}")

    print("=" * 65)

    if sac is not None and default is not None:
        print("\nPER-MONTH BREAKDOWN (avg daily earned, UAH)")
        print(f"  {'Month':<8} {'Default':>10} {'SAC':>10} {'Delta':>10}")
        print("  " + "-" * 42)
        sac['month']     = pd.to_datetime(sac['date']).dt.month
        default['month'] = pd.to_datetime(default['date']).dt.month
        for m in range(1, 13):
            d = default[default['month'] == m]['total_money_earned'].mean()
            s = sac[sac['month'] == m]['total_money_earned'].mean()
            delta = s - d
            sign  = '+' if delta >= 0 else ''
            print(f"  {m:<8} {d:>10.1f} {s:>10.1f} {sign}{delta:>9.1f}")

    print()

if __name__ == '__main__':
    main()


import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


LOG_THEN_STANDARD_COLS = [
    'DAM_Price',
]
STANDARD_COLS = [
    'Temperature_2m',
    'Shortwave_radiation',
    'DAM_Vol_Buy',
    'DAM_Vol_Sale',
]
MINMAX_COLS = [
    'Global_tilted_irradiance_instant',
    'hours_until_outage',
    'outage_remaining_h',
    'next_outage_duration',
]
PASSTHROUGH_COLS = [
    'Hour_sin', 'Hour_cos',
    'Day_sin', 'Day_cos',
    'Day_of_week_sin', 'Day_of_week_cos',
    'Grid',
]
DROP_COLS = [
    'timestamp',
    'Load',      # not normalized: env observes load relative to configured peak_kw
    'Hour',
    'Minute',
    'Minute_sin',
    'Minute_cos',
    'Day',
    'Day_of_week',
    'Month',
]

# Canonical output column order — the exact order the model was trained on.
# Both normalize_dataset and normalize_row MUST emit this order: the observation
# feeds row.values positionally into the network, and live data sources
# (combined.csv) don't share the training CSV's column order.
NORMALIZED_COLS = [
    'Hour_sin', 'Hour_cos',
    'Day_of_week_sin', 'Day_of_week_cos',
    'Day_sin', 'Day_cos',
    'Grid',
    'next_outage_duration', 'outage_remaining_h', 'hours_until_outage',
    'Temperature_2m', 'Shortwave_radiation', 'Global_tilted_irradiance_instant',
    'DAM_Price', 'DAM_Vol_Buy', 'DAM_Vol_Sale',
]


def normalize_dataset(input_path: str, output_path: str, scalers_path: str):
    """Normalize the full dataset and save the normalized CSV + scalers.pkl."""
    print(f"\n[1/4] Reading dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"      Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    scalers = {}
    print("\n[3/4] Normalizing...")
    for col in LOG_THEN_STANDARD_COLS:
        if col not in df.columns:
            print(f"      WARNING: '{col}' not found, skipping")
            continue
        df[col] = np.log1p(df[col])
        scaler  = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = {'type': 'log_standard', 'scaler': scaler}
        print(f"      {col:45s} log1p → StandardScaler  mean={scaler.mean_[0]:.3f}  std={scaler.scale_[0]:.3f}")
    for col in STANDARD_COLS:
        if col not in df.columns:
            print(f"      WARNING: '{col}' not found, skipping")
            continue
        scaler  = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = {'type': 'standard', 'scaler': scaler}
        print(f"      {col:45s} StandardScaler          mean={scaler.mean_[0]:.3f}  std={scaler.scale_[0]:.3f}")
    for col in MINMAX_COLS:
        if col not in df.columns:
            print(f"      WARNING: '{col}' not found, skipping")
            continue
        scaler  = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = {'type': 'minmax', 'scaler': scaler}
        print(f"      {col:45s} MinMaxScaler            min={scaler.data_min_[0]:.3f}  max={scaler.data_max_[0]:.3f}")
    for col in PASSTHROUGH_COLS:
        if col in df.columns:
            scalers[col] = {'type': 'passthrough'}
            print(f"      {col:45s} passthrough")
    missing = [c for c in NORMALIZED_COLS if c not in df.columns]
    extra   = [c for c in df.columns if c not in NORMALIZED_COLS]
    if missing:
        print(f"      WARNING: canonical columns missing from input: {missing}")
    if extra:
        print(f"      WARNING: unexpected columns dropped from output: {extra}")
    df = df[[c for c in NORMALIZED_COLS if c in df.columns]]
    nan_count = df.isna().sum().sum()
    print(f"\n      Final shape: {df.shape[0]} × {df.shape[1]}")
    print(f"      NaN after normalisation: {nan_count}")
    if nan_count > 0:
        print(f"      WARNING: NaN present! Check input data.")
        print(df.isna().sum()[df.isna().sum() > 0])
    print(f"\n[4/4] Saving...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Dataset  → {output_path}")
    os.makedirs(os.path.dirname(scalers_path) or '.', exist_ok=True)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"      Scalers  → {scalers_path}")
    print(f"\nDone. Columns: {list(df.columns)}\n")
    return df, scalers


def normalize_row(row: pd.Series, scalers: dict) -> pd.Series:
    """Normalize a single raw row using the same scalers fitted on training data.

    Always returns columns in the canonical NORMALIZED_COLS order — live data
    sources don't share the training CSV's column order, and the observation
    consumes row.values positionally.
    """
    row = row.copy()
    cols_to_drop = [c for c in DROP_COLS if c in row.index]
    row = row.drop(index=cols_to_drop)
    for col, info in scalers.items():
        if col not in row.index:
            continue
        t = info['type']
        if t == 'log_standard':
            val      = np.log1p(float(row[col]))
            row[col] = info['scaler'].transform([[val]])[0][0]
        elif t == 'standard':
            row[col] = info['scaler'].transform([[float(row[col])]])[0][0]
        elif t == 'minmax':
            row[col] = info['scaler'].transform([[float(row[col])]])[0][0]
    missing = [c for c in NORMALIZED_COLS if c not in row.index]
    if missing:
        raise ValueError(f"normalize_row: input row is missing canonical columns {missing}")
    return row.reindex(NORMALIZED_COLS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='dataset_final.csv')
    parser.add_argument('--output',  default='dataset_normalized.csv')
    parser.add_argument('--scalers', default='models/scalers.pkl')
    args = parser.parse_args()
    normalize_dataset(args.input, args.output, args.scalers)

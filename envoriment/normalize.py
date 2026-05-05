
"""
normalize.py
============
Нормалізація датасету перед навчанням.
 
Запуск:
    python normalize.py --input  datasets/datasets_v9/dataset.csv
                        --output datasets/dataset_normalized.csv
                        --scalers models/scalers.pkl
"""
 
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
 
 
# ─────────────────────────────────────────────────────────────────────
# Конфігурація нормалізації
# ─────────────────────────────────────────────────────────────────────
 
# log1p → StandardScaler:
# Спочатку стискаємо великий хвіст через log1p,
# потім доводимо до mean=0, std=1 через StandardScaler.
# Для колонок з сильним правим скосом і великим діапазоном.
LOG_THEN_STANDARD_COLS = [
    'DAM_Price',
]
 
# StandardScaler (mean=0, std=1):
# Для колонок з приблизно нормальним розподілом.
# Підходить коли є від'ємні значення (температура).
STANDARD_COLS = [
    'Load',
    'Temperature_2m',
    'Shortwave_radiation',
    'DAM_Vol_Buy',
    'DAM_Vol_Sale',
]
 
# MinMaxScaler → [0, 1]:
# Для колонок де нуль — осмислене значення.
# Наприклад GTI=0 вночі, outage=0 коли немає відключення.
MINMAX_COLS = [
    'Global_tilted_irradiance_instant',
    'hours_until_outage',
    'outage_remaining_h',
    'next_outage_duration',
]
 
# Без змін — вже в правильному діапазоні для нейронки:
# sin/cos в [-1, 1], Grid в {0, 1}
PASSTHROUGH_COLS = [
    'Hour_sin', 'Hour_cos',
    'Day_of_week_sin', 'Day_of_week_cos',
    'Grid',
]
 
# Дропаємо — дублюють інформацію або не несуть сигналу
DROP_COLS = [
    'timestamp',
    'Hour',
    'Minute',
    'Minute_sin',
    'Minute_cos',
    'Day',
    'Day_of_week',
    'Month',
]
 
 
# ─────────────────────────────────────────────────────────────────────
def normalize_dataset(input_path: str, output_path: str, scalers_path: str):
    """
    Нормалізує весь датасет і зберігає:
        - нормалізований CSV
        - словник scalers для подальшого використання при інференсі
    """
 
    print(f"\n[1/4] Читаємо датасет: {input_path}")
    df = pd.read_csv(input_path)
    print(f"      Розмір: {df.shape[0]} рядків × {df.shape[1]} колонок")
 
    # ── Дропаємо непотрібні колонки ──────────────────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    scalers = {}
 
    print("\n[3/4] Нормалізуємо...")
 
    # ── log1p → StandardScaler ────────────────────────────────────
    for col in LOG_THEN_STANDARD_COLS:
        if col not in df.columns:
            print(f"      УВАГА: '{col}' не знайдена, пропускаємо")
            continue
 
        df[col] = np.log1p(df[col])
        scaler  = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
 
        scalers[col] = {'type': 'log_standard', 'scaler': scaler}
        print(f"      {col:45s} log1p → StandardScaler  mean={scaler.mean_[0]:.3f}  std={scaler.scale_[0]:.3f}")
 
    # ── StandardScaler ────────────────────────────────────────────
    for col in STANDARD_COLS:
        if col not in df.columns:
            print(f"      УВАГА: '{col}' не знайдена, пропускаємо")
            continue
 
        scaler  = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
 
        scalers[col] = {'type': 'standard', 'scaler': scaler}
        print(f"      {col:45s} StandardScaler          mean={scaler.mean_[0]:.3f}  std={scaler.scale_[0]:.3f}")
 
    # ── MinMaxScaler ──────────────────────────────────────────────
    for col in MINMAX_COLS:
        if col not in df.columns:
            print(f"      УВАГА: '{col}' не знайдена, пропускаємо")
            continue
 
        scaler  = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[[col]])
 
        scalers[col] = {'type': 'minmax', 'scaler': scaler}
        print(f"      {col:45s} MinMaxScaler            min={scaler.data_min_[0]:.3f}  max={scaler.data_max_[0]:.3f}")
 
    # ── Passthrough ───────────────────────────────────────────────
    for col in PASSTHROUGH_COLS:
        if col in df.columns:
            scalers[col] = {'type': 'passthrough'}
            print(f"      {col:45s} без змін")
 
    # ── Фінальна перевірка ────────────────────────────────────────
    nan_count = df.isna().sum().sum()
    print(f"\n      Фінальний розмір: {df.shape[0]} × {df.shape[1]}")
    print(f"      NaN після нормалізації: {nan_count}")
    if nan_count > 0:
        print(f"      УВАГА: є NaN! Перевір вхідні дані.")
        print(df.isna().sum()[df.isna().sum() > 0])
 
    # ── Зберігаємо ───────────────────────────────────────────────
    print(f"\n[4/4] Зберігаємо...")
 
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Датасет  → {output_path}")
 
    os.makedirs(os.path.dirname(scalers_path) or '.', exist_ok=True)
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"      Scalers  → {scalers_path}")
    print(f"\nГотово. Колонки: {list(df.columns)}\n")
 
    return df, scalers
 
 
# ─────────────────────────────────────────────────────────────────────
def normalize_row(row: pd.Series, scalers: dict) -> pd.Series:
    """
    Нормалізує один сирий рядок тими самими scaler-ами що були
    навчені на тренувальних даних.
 
    Використовується в inference.py і dispatch.py для живих даних
    від data_combiner.py.
 
    Параметри:
        row     — сирий (ненормалізований) рядок pd.Series
        scalers — словник завантажений з scalers.pkl
 
    Повертає:
        нормалізований pd.Series
    """
    row = row.copy()
 
    # Дропаємо ті самі колонки що при навчанні
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
 
        # passthrough — нічого не робимо
 
    return row
 
 
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='dataset_final.csv')
    parser.add_argument('--output',  default='dataset_normalized.csv')
    parser.add_argument('--scalers', default='models/scalers.pkl')
    args = parser.parse_args()
 
    normalize_dataset(args.input, args.output, args.scalers)
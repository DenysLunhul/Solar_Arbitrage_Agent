"""
inverter_dispatch.py
====================
Генерація плану роботи на основі базової логіки гібридного інвертора.
Інвертор не знає про ціни чи майбутні відключення. Він реагує лише на 
наявність сонця, стан мережі та рівень заряду батареї (SoC).
"""

import argparse
import numpy as np
import pandas as pd
from environment import Environment

# ═════════════════════════════════════════════════════════════════════
# DEFAULT конфіг
# ═════════════════════════════════════════════════════════════════════

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
        'efficiency': 0.95,  # Залишається для підрахунку витрат у CSV, інвертор цього не бачить
    },
    'grid':{
        'capacity': 150.0,
        'price_to_buy': 10.0
    },
}

# ═════════════════════════════════════════════════════════════════════
# ЛОГІКА ІНВЕРТОРА
# ═════════════════════════════════════════════════════════════════════

def inverter_action(row: pd.Series, soc: float) -> np.ndarray:
    """
    Імітує примітивну логіку гібридного інвертора.
    Ніяких цін, ніяких прогнозів відключень.
    """
    gti         = float(row['Global_tilted_irradiance_instant'])
    grid_status = int(row['Grid'])

    # Цільові показники інвертора
    TARGET_SOC           = 0.70  # Звичайний цільовий заряд
    MAX_SOC              = 0.95  # Дозволений максимум, якщо сонця дуже багато
    MIN_SOLAR_THRESHOLD  = 10.0  # Мінімальне сонце (Вт/м²)
    HIGH_SOLAR_THRESHOLD = 400.0 # Дуже багато сонця (яскравий полудень)

    # ── ПРАВИЛО 1: Немає мережі ──────────────────────────────────────
    if grid_status == 0:
        # Віддаємо все з батареї на навантаження. Продавати/купувати неможливо.
        return np.array([-1.0, 0.0], dtype=np.float32)

    # ── ПРАВИЛО 2: Робота при наявній мережі ─────────────────────────
    if gti > MIN_SOLAR_THRESHOLD:
        # СОНЦЕ Є
        if soc < TARGET_SOC:
            # Пріоритет: зарядити батарею до 70% (і покрити навантаження)
            # 1.0 = заряджати батарею, 0.0 = не намагатись експортувати
            return np.array([1.0, 0.0], dtype=np.float32)
            
        elif gti > HIGH_SOLAR_THRESHOLD and soc < MAX_SOC:
            # Сонця прям дуже багато, а батарея вже 70%
            # Дозволяємо дозарядити батарею до 95% і паралельно продаємо в мережу
            return np.array([1.0, 1.0], dtype=np.float32)
            
        else:
            # Батарея досягла цілі (70% при звичайному сонці або 95% при сильному)
            # Батарею не чіпаємо (0.0), весь надлишок сонця відправляємо в мережу (1.0)
            return np.array([0.0, 1.0], dtype=np.float32)
            
    else:
        # СОНЦЯ НЕМАЄ (ніч / сутінки)
        # Розряджаємо батарею, щоб перекрити навантаження підприємства (-1.0)
        # Якщо батареї не вистачить, інвертор добере з мережі (-1.0)
        return np.array([-1.0, -1.0], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦІЯ ПЛАНУ
# ═════════════════════════════════════════════════════════════════════

def generate_dispatch_plan(df_raw: pd.DataFrame, df_norm: pd.DataFrame, config: dict, initial_soc: float, output_file: str):
    print("Запускаємо симуляцію базової логіки інвертора...")
    
    env = Environment(df_raw=df_raw, df=df_norm, system_config=config)
    env.reset()
    env.soc = initial_soc
    
    dispatch_history = []

    while True:
        curr_step = env.curr_step
        row = df_raw.iloc[curr_step]
        
        # Отримуємо дію від "інвертора" (передаємо тільки рядок і поточний SoC)
        action = inverter_action(row, env.soc)
        
        # Робимо крок у середовищі
        _, _, terminated, truncated, info = env.step(action)
        
        # Фінансовий підрахунок (інвертор цього не знає, але для CSV нам це корисно бачити)
        grid_kwh = info.get('actual_grid_kwh', 0)
        money_earned = 0.0
        if grid_kwh < 0:
            price = float(row['DAM_Price']) / 1000
            money_earned = abs(grid_kwh) * price
        elif grid_kwh > 0:
            money_earned = - (grid_kwh * config['grid']['price_to_buy'])

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
    
    print(f"Готово! Згенеровано {len(df_plan)} кроків.")
    print(f"План збережено у файл: {output_file}")


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',     default='dataset_final.csv',         help='Сирий датасет')
    parser.add_argument('--norm',    default='dataset_normalized.csv',    help='Нормалізований датасет')
    parser.add_argument('--soc',     type=float, default=0.5,             help='Початковий SoC')
    parser.add_argument('--days',    type=int,   default=1,            help='Скільки днів (None = весь датасет)')
    parser.add_argument('--out',     default='results/dispatch_plan_inverter.csv', help='Куди зберегти CSV')
    args = parser.parse_args()

    print("Завантажуємо дані...")
    df_raw  = pd.read_csv(args.raw)
    df_norm = pd.read_csv(args.norm)

    if args.days is not None:
        df_raw  = df_raw.iloc[:args.days * 96].reset_index(drop=True)
        df_norm = df_norm.iloc[:args.days * 96].reset_index(drop=True)

    print(f"Даних: {len(df_raw)} рядків ({len(df_raw)//96} днів)")

    generate_dispatch_plan(
        df_raw=df_raw,
        df_norm=df_norm,
        config=DEFAULT_SYSTEM_CONFIG,
        initial_soc=args.soc,
        output_file=args.out
    )
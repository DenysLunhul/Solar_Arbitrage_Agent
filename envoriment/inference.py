
"""
inference.py
============
Запуск навченої моделі для отримання dispatch plan.
 
Використання з беку (FastAPI):
    from inference import run_inference
    result = run_inference(df_raw, system_config, initial_soc=0.6)
 
Використання з командного рядка:
    python inference.py --data  datasets/dataset_normalized.csv
                        --model models/sac_ems
                        --scalers models/scalers.pkl
                        --soc 0.6
                        --days 1
"""
 
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
 
from environment import Environment
from normalize import normalize_row
 
 
# ─────────────────────────────────────────────────────────────────────
# Завантаження моделі і scalers
#
# Ці об'єкти повинні завантажуватись ОДИН РАЗ при старті сервера,
# а не при кожному запиті. Тому виносимо в окрему функцію
# і зберігаємо в глобальних змінних на рівні модуля.
# ─────────────────────────────────────────────────────────────────────
 
def load_model_and_scalers(model_path: str, scalers_path: str):
    """
    Завантажує модель з .zip і scalers з .pkl.
    Викликати один раз при старті — результат зберегти глобально.
    """
    print(f"Завантажуємо модель:  {model_path}")
    model = SAC.load(model_path)
 
    print(f"Завантажуємо scalers: {scalers_path}")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
 
    print("Готово.\n")
    return model, scalers
 
 
# ─────────────────────────────────────────────────────────────────────
# Головна функція інференсу
#
# Саме її викликає FastAPI роутер dispatch.py.
# Приймає сирі дані + конфіг клієнта, повертає dispatch plan і summary.
# ─────────────────────────────────────────────────────────────────────
 
def run_inference(
    df_raw:        pd.DataFrame,
    system_config: dict,
    model:         SAC,
    scalers:       dict,
    initial_soc:   float = 0.5,
) -> dict:
    """
    Виконує інференс моделі, використовуючи сирі дані для фізики 
    та нормалізовані дані для нейромережі.[cite: 2, 3]
    """

    # ── 1. Створюємо нормалізований датасет для моделі ─────────────
    # Проганяємо кожен рядок сирих даних через скалери
    df_norm = pd.DataFrame([
        normalize_row(df_raw.iloc[i], scalers)
        for i in range(len(df_raw))
    ])

    # ── 2. Створюємо середовище з ДВОМА датасетами ────────────────
    # Передаємо df_raw для розрахунків у step()
    # Передаємо df_norm (як df) для спостережень у get_observe()[cite: 2]
    env = Environment(df_raw=df_raw, df=df_norm, system_config=system_config)

    # ── 3. Ініціалізація стану та SoC ─────────────────────────────
    obs, _ = env.reset()
    # Встановлюємо реальний SoC, отриманий від інвертора/BMS[cite: 3]
    env.soc = float(np.clip(initial_soc, 0.0, 1.0))
    # Оновлюємо початкове спостереження з урахуванням нового SoC[cite: 2, 3]
    obs = env.get_observe()   

    # ── 4. Генерація плану (Step-by-step) ────────────────────────
    dispatch_plan = []

    while True:
        # Отримуємо дію від моделі (deterministic=True для стабільності)[cite: 3]
        action, _state = model.predict(obs, deterministic=True)

        # Робимо крок у середовищі[cite: 2, 3]
        obs, reward, terminated, truncated, info = env.step(action)

        # Додаємо результати кроку в план[cite: 3]
        dispatch_plan.append({
            'step':           env.curr_step - 1,
            'action_battery': round(float(action[0]), 4),
            'action_grid':    round(float(action[1]), 4),
            'soc':            round(float(info['soc']), 4),
            'target_soc':     round(float(info['target_soc']), 4),
            'solar_gen_kwh':  round(float(info['solar_gen_ts_kwh']), 4),
            'grid_kwh':       round(float(info['actual_grid_kwh']), 4),
            'unmet_load_kwh': round(float(info['unmet_load_kwh']), 4),
            'lcos_cost':      round(float(info['lcos_cost']), 4),
            'money_earned_ts': round(float(info['money_earned_ts'])), 
            'reward':         round(float(reward), 4),
        })

        if terminated or truncated:
            break

    # ── 5. Формування підсумків ──────────────────────────────────
    # Розрахунки ведуться на основі реальних фізичних величин із df_raw[cite: 2, 3]
    summary = {
        'total_money_earned': round(sum(x['money_earned_ts'] for x in dispatch_plan)),
        'total_reward_uah': round(sum(x['reward'] for x in dispatch_plan), 2),
        'bought_kwh':       round(sum(x['grid_kwh'] for x in dispatch_plan if x['grid_kwh'] > 0), 3),
        'sold_kwh':         round(sum(abs(x['grid_kwh']) for x in dispatch_plan if x['grid_kwh'] < 0), 3),
        'solar_kwh':        round(sum(x['solar_gen_kwh'] for x in dispatch_plan), 3),
        'unmet_load_kwh':   round(sum(x['unmet_load_kwh'] for x in dispatch_plan), 4),
        'lcos_total_uah':   round(sum(x['lcos_cost'] for x in dispatch_plan), 3),
        'initial_soc':      round(initial_soc, 3),
        'final_soc':        dispatch_plan[-1]['soc'] if dispatch_plan else initial_soc,
        'steps':            len(dispatch_plan),
    }

    return {
        'dispatch_plan': dispatch_plan,
        'summary':       summary,
    }
 
 
# ─────────────────────────────────────────────────────────────────────
# CLI — запуск з командного рядка для тестування
# ─────────────────────────────────────────────────────────────────────
 
DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        2.0,
        'max_charge_power':    1.0,
        'max_discharge_power': 1.0,
        'efficiency':          0.95,
        'lcos':                1.5,
        'min_reserve':         20,
    },
    'solar': {
        'peak_power':  3.0,
        'efficiency':  0.18,
    },
    'inverter': {
        'max_power': 5.0,
        'price_to_buy': 4.32
    }
}
 
 
if __name__ == '__main__':
    import json
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='dataset_normalized.csv')
    parser.add_argument('--model',   default='models/sac_ems')
    parser.add_argument('--scalers', default='models/scalers.pkl')
    parser.add_argument('--config',  default=None, help='JSON файл з system_config')
    parser.add_argument('--output',  default='results/dispatch_plan.csv')
    parser.add_argument('--soc',     type=float, default=0.5)
    parser.add_argument('--days',    type=int,   default=1)
    args = parser.parse_args()
 
    # Завантажуємо конфіг
    if args.config:
        with open(args.config) as f:
            system_config = json.load(f)
        print(f"Конфіг: {args.config}")
    else:
        system_config = DEFAULT_SYSTEM_CONFIG
        print("Конфіг: DEFAULT_SYSTEM_CONFIG")
 
    # Завантажуємо модель і scalers
    model, scalers = load_model_and_scalers(args.model, args.scalers)
 
    # Завантажуємо датасет
    # При запуску з CLI дані вже нормалізовані (dataset_normalized.csv)
    # тому normalize_row всередині run_inference спрацює як passthrough
    # для вже нормалізованих колонок
    df = pd.read_csv(args.data)
    df = df.iloc[:args.days * 96].reset_index(drop=True)
    print(f"Даних: {len(df)} рядків ({args.days} днів)\n")
 
    # Запускаємо інференс
    result = run_inference(
        df_raw=df,
        system_config=system_config,
        model=model,
        scalers=scalers,
        initial_soc=args.soc,
    )
 
    # Виводимо summary
    print("\n" + "="*50)
    print("ПІДСУМКИ")
    print("="*50)
    for k, v in result['summary'].items():
        print(f"  {k:25s} {v}")
 
    # Зберігаємо dispatch plan
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    pd.DataFrame(result['dispatch_plan']).to_csv(args.output, index=False)
    print(f"\nDispatch plan → {args.output}")
    print(pd.DataFrame(result['dispatch_plan']).head(10).to_string())
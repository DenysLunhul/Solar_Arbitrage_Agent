
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
    Параметри:
        df_raw        — сирий (ненормалізований) DataFrame.
                        96 рядків для одного дня або більше.
                        Колонки мають відповідати датасету.
        system_config — параметри заліза клієнта з БД:
                        battery, solar, inverter
        model         — завантажена SAC модель (SAC.load)
        scalers       — завантажені scalers (pickle.load)
        initial_soc   — поточний заряд батареї з BMS (0.0–1.0)
 
    Повертає словник:
        {
            "dispatch_plan": [ ...список з N словників... ],
            "summary":       { ...підсумки... }
        }
 
    Кожен елемент dispatch_plan — один таймстеп (15 хвилин):
        step            — індекс (0..N-1)
        action_battery  — дія батареї від агента (-1..+1)
        action_grid     — дія мережі від агента (-1..+1)
        soc             — SoC після цього кроку
        target_soc      — динамічна ціль резерву
        solar_gen_kwh   — генерація сонця
        grid_kwh        — обмін з мережею (+ купівля, - продаж)
        unmet_load_kwh  — непокрите навантаження (= 0 в нормі)
        lcos_cost       — вартість деградації батареї за крок
        reward          — нагорода за крок
    """
 
    # ── 1. Нормалізуємо сирі дані ─────────────────────────────────
    # normalize_row застосовує ті самі scaler-и що були при навчанні
    df_norm = pd.DataFrame([
        normalize_row(df_raw.iloc[i], scalers)
        for i in range(len(df_raw))
    ])
 
    # ── 2. Інстанціюємо Environment з конфігом цього клієнта ──────
    # Environment — новий об'єкт при кожному запиті
    # model — той самий глобальний об'єкт, не змінюється
    env = Environment(df_norm, system_config=system_config)
 
    # ── 3. Виставляємо початковий SoC ────────────────────────────
    # reset() скидає soc на 0.5, тому виставляємо реальний після
    obs, _ = env.reset()
    env.soc = float(np.clip(initial_soc, 0.0, 1.0))
    obs = env.get_observe()   # перераховуємо observation з новим soc
 
    # ── 4. Проганяємо модель крок за кроком ──────────────────────
    dispatch_plan = []
 
    while True:
        # model.predict — це і є "предікшин"
        # obs:   numpy array shape=(N_features + 1,) — всі фічі + SoC
        # action: numpy array shape=(2,) — [battery_action, grid_action]
        # deterministic=True — обов'язково при інференсі (без exploration)
        action, _state = model.predict(obs, deterministic=True)
 
        obs, reward, terminated, truncated, info = env.step(action)
 
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
            'reward':         round(float(reward), 4),
        })
 
        if terminated or truncated:
            break
 
    # ── 5. Підсумки ───────────────────────────────────────────────
    total_reward = sum(x['reward']         for x in dispatch_plan)
    bought_kwh   = sum(x['grid_kwh']       for x in dispatch_plan if x['grid_kwh'] > 0)
    sold_kwh     = sum(abs(x['grid_kwh'])  for x in dispatch_plan if x['grid_kwh'] < 0)
    solar_kwh    = sum(x['solar_gen_kwh']  for x in dispatch_plan)
    unmet_kwh    = sum(x['unmet_load_kwh'] for x in dispatch_plan)
    lcos_total   = sum(x['lcos_cost']      for x in dispatch_plan)
 
    summary = {
        'total_reward_uah': round(total_reward, 2),
        'bought_kwh':       round(bought_kwh, 3),
        'sold_kwh':         round(sold_kwh, 3),
        'solar_kwh':        round(solar_kwh, 3),
        'unmet_load_kwh':   round(unmet_kwh, 4),
        'lcos_total_uah':   round(lcos_total, 3),
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
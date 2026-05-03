
"""
train.py
========
Навчання SAC-агента через Stable-Baselines3.
 
Запуск:
    python train.py
 
Що буде створено:
    models/scalers.pkl           — якщо ще не існує, запустить normalize.py
    models/sac_ems.zip           — фінальна модель
    models/best/best_model.zip   — найкраща модель за eval reward
    models/checkpoints/          — проміжні збереження
    logs/tensorboard/            — графіки для tensorboard
"""
 
import os
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym
 
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
 
from environment import Environment
 
 
# ═════════════════════════════════════════════════════════════════════
# КОНФІГ НАВЧАННЯ
# ═════════════════════════════════════════════════════════════════════
 
CONFIG = {
    # Шляхи
    'dataset_path':    'dataset_normalized.csv',
    'scalers_path':    'models/scalers.pkl',
    'model_save_path': 'models/sac_ems',
    'checkpoint_dir':  'models/checkpoints/',
    'tensorboard_dir': 'logs/tensorboard/',
    'monitor_dir':     'logs/monitor/',
 
    # Навчання
    'total_timesteps': 500_000,
    'checkpoint_freq': 50_000,
    'log_interval':    1_000,
 
    # SAC гіперпараметри
    'sac_params': {
        'buffer_size':    100_000,
        'learning_starts': 1_000,
        'batch_size':      256,
        'learning_rate':   3e-4,
        'gamma':           0.99,
        'tau':             0.005,
        'ent_coef':        'auto',
        'policy_kwargs': {
            'net_arch': [256, 256],
        },
        'verbose': 1,
        'seed':    42,
        'target_entropy': 'auto',
        'use_sde':        False,
    }
}
 
# ─────────────────────────────────────────────────────────────────────
# DEFAULT конфіг системи для навчання
#
# Якщо хочеш одну модель для всіх клієнтів — використовуй
# RandomConfigWrapper (нижче) щоб навчати на різних конфігах.
# Якщо хочеш модель для конкретного заліза — виставь тут реальні
# параметри і вимкни RandomConfigWrapper в make_envs().
# ─────────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        5.0,
        'max_charge_power':    2.5,
        'max_discharge_power': 2.5,
        'efficiency':          0.95,
        'lcos':                1.5,
        'min_reserve':         20,    # відсотки (20 = 20%)
    },
    'solar': {
        'peak_power':  5.0,
        'efficiency':  0.2,
    },
    'inverter': {
        'max_power': 5.0,
        'price_to_buy': 4.32
    }
}
 
 
# ═════════════════════════════════════════════════════════════════════
# RandomConfigWrapper
#
# При кожному reset() генерує новий рандомний конфіг системи.
# Завдяки цьому модель вчиться бути універсальною — вона бачить
# різні батареї і панелі і вчиться приймати правильні рішення
# незалежно від розміру заліза конкретного клієнта.
#
# Це називається domain randomization.
# ═════════════════════════════════════════════════════════════════════
 
class RandomConfigWrapper(gym.Wrapper):
 
    def __init__(self, df: pd.DataFrame):
        config = self._sample_config()
        env    = Environment(df, system_config=config)
        super().__init__(env)
        self.df = df
 
    def _sample_config(self) -> dict:
        capacity = float(np.random.uniform(1.0, 20.0))
        return {
            'battery': {
                'capacity_kwh':        capacity,
                'max_charge_power':    float(np.random.uniform(0.5, min(capacity, 10.0))),
                'max_discharge_power': float(np.random.uniform(0.5, min(capacity, 10.0))),
                'efficiency':          float(np.random.uniform(0.90, 0.98)),
                'lcos':                float(np.random.uniform(0.5, 3.0)),
                'min_reserve':         int(np.random.randint(10, 30)),
            },
            'solar': {
                'peak_power':  float(np.random.uniform(1.0, 10.0)),
                'efficiency':  float(np.random.uniform(0.15, 0.22)),
            },
            'inverter': {
                'max_power': float(np.random.uniform(3.0, 15.0)),
                'price_to_buy' : 4.32
            },
        }
 
    def reset(self, **kwargs):
        # Новий конфіг і новий env при кожному епізоді
        new_config = self._sample_config()
        self.env   = Environment(self.df, system_config=new_config)
        return self.env.reset(**kwargs)
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 1: Дані
# ═════════════════════════════════════════════════════════════════════
 
def load_data():
    print("\n" + "="*60)
    print("КРОК 1: Завантаження датасету")
    print("="*60)
 
    # Якщо нормалізований датасет ще не існує — запускаємо normalize.py
    if not os.path.exists(CONFIG['dataset_path']):
        print("Нормалізований датасет не знайдено. Запускаємо normalize.py...")
        from normalize import normalize_dataset
        normalize_dataset(
            input_path='datasets/datasets_v9/dataset.csv',
            output_path=CONFIG['dataset_path'],
            scalers_path=CONFIG['scalers_path'],
        )
 
    df = pd.read_csv(CONFIG['dataset_path'])
    print(f"Датасет: {df.shape[0]} рядків × {df.shape[1]} колонок")
 
    # Розбиваємо 80/20 БЕЗ перемішування — часовий ряд!
    # eval завжди на майбутніх даних яких агент не бачив при навчанні
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx].reset_index(drop=True)
    df_eval   = df.iloc[split_idx:].reset_index(drop=True)
 
    print(f"Train: {len(df_train)} рядків (~{len(df_train)//96} днів)")
    print(f"Eval:  {len(df_eval)} рядків  (~{len(df_eval)//96} днів)")
 
    return df_train, df_eval
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 2: Середовища
# ═════════════════════════════════════════════════════════════════════
 
def make_envs(df_train, df_eval):
    os.makedirs(CONFIG['monitor_dir'], exist_ok=True)

    # Обгортаємо в DummyVecEnv — потрібно для VecNormalize
    train_env = DummyVecEnv([
        lambda: Monitor(
            RandomConfigWrapper(df_train),
            filename=os.path.join(CONFIG['monitor_dir'], 'train')
        )
    ])

    eval_env = DummyVecEnv([
        lambda: Monitor(
            Environment(df_eval, system_config=DEFAULT_SYSTEM_CONFIG),
            filename=os.path.join(CONFIG['monitor_dir'], 'eval')
        )
    ])

    # VecNormalize автоматично нормалізує reward і observation
    # norm_obs=True  — нормалізує observation онлайн (running mean/std)
    # norm_reward=True — нормалізує reward онлайн
    # clip_obs=10.0  — обрізає аномальні значення observation
    # clip_reward=10.0 — обрізає аномальні значення reward
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # при eval reward не нормалізуємо щоб бачити реальні цифри
        clip_obs=10.0,
    )

    return train_env, eval_env
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 3: Модель
# ═════════════════════════════════════════════════════════════════════
 
def make_model(train_env):
    print("\n" + "="*60)
    print("КРОК 3: Ініціалізація SAC")
    print("="*60)
 
    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['tensorboard_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
 
    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        tensorboard_log=CONFIG['tensorboard_dir'],
        **CONFIG['sac_params']
    )
 
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Параметрів в policy: {total_params:,}")
    print(f"Архітектура: {CONFIG['sac_params']['policy_kwargs']['net_arch']}")
 
    return model
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 4: Callbacks
# ═════════════════════════════════════════════════════════════════════
 
def make_callbacks(eval_env):
 
    checkpoint_cb = CheckpointCallback(
        save_freq=CONFIG['checkpoint_freq'],
        save_path=CONFIG['checkpoint_dir'],
        name_prefix='sac_ems',
        verbose=1,
    )
 
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join('models', 'best'),
        log_path=os.path.join('logs', 'eval'),
        eval_freq=CONFIG['checkpoint_freq'],
        n_eval_episodes=3,
        deterministic=True,
        verbose=1,
    )
 
    return CallbackList([checkpoint_cb, eval_cb])
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 5: Навчання
# ═════════════════════════════════════════════════════════════════════
 
def train(model, callbacks):
    print("\n" + "="*60)
    print("КРОК 4: Навчання")
    print(f"Кроків: {CONFIG['total_timesteps']:,}")
    print("Tensorboard: tensorboard --logdir logs/tensorboard/")
    print("="*60 + "\n")
 
    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=callbacks,
        log_interval=CONFIG['log_interval'],
        progress_bar=False,
        reset_num_timesteps=True,
    )
 
    return model
 
 
# ═════════════════════════════════════════════════════════════════════
# КРОК 6: Збереження і швидкий тест
# ═════════════════════════════════════════════════════════════════════
 
def save_and_test(model, eval_env):
    print("\n" + "="*60)
    print("КРОК 5: Збереження і тест")
    print("="*60)
 
    model.save(CONFIG['model_save_path'])
    print(f"Модель → {CONFIG['model_save_path']}.zip")
 
    # Один тестовий епізод
    print("\nТестуємо один епізод...")
    obs, _ = eval_env.reset()
    total_reward = 0.0
    total_unmet  = 0.0
    steps        = 0
 
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        total_unmet  += info.get('unmet_load_kwh', 0.0)
        steps        += 1
        if terminated or truncated:
            break
 
    print(f"Кроків:                  {steps}")
    print(f"Сумарний reward:         {total_reward:.2f} UAH")
    print(f"Непокрите навантаження:  {total_unmet:.4f} кВт·год")
    print(f"Фінальний SoC:           {info['soc']:.3f}")
 
 
# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
 
if __name__ == '__main__':
    df_train, df_eval     = load_data()
    train_env, eval_env   = make_envs(df_train, df_eval)
    model                 = make_model(train_env)
    callbacks             = make_callbacks(eval_env)
    model                 = train(model, callbacks)
    save_and_test(model, train_env, eval_env)
 
    print("\n" + "="*60)
    print("Готово!")
    print(f"Модель:      {CONFIG['model_save_path']}.zip")
    print(f"Tensorboard: tensorboard --logdir {CONFIG['tensorboard_dir']}")
    print("="*60)
 
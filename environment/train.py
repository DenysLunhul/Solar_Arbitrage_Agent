import copy
import os
import pickle
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from environment import Environment


CONFIG = {
    'dataset_path':    'dataset_normalized.csv',
    'dataset_raw_path': 'dataset_final.csv',
    'scalers_path':    'models/scalers.pkl',
    'model_save_path': 'models/sac_ems',
    'tensorboard_dir': 'logs/tensorboard/',
    'monitor_dir':     'logs/monitor/',

    'total_timesteps': 10_000_000,
    'eval_freq':       100_000,
    'log_interval':    100_000,  # large value — suppress SB3 default episode logging
    'n_envs':          32,       # DummyVecEnv: no IPC overhead, env step ~0.11 ms each

    'sac_params': {
        'device':          'cuda',
        'buffer_size':     1_000_000,
        'learning_starts': 10_000,   # ~6 full episodes across 16 envs before first update
        'batch_size':      512,
        'learning_rate':   3e-4,
        'gamma':           0.99,
        'tau':             0.005,
        'ent_coef':        'auto',
        'policy_kwargs': {
            'net_arch': [256, 256],
        },
        'verbose': 0,
        'seed':    42,
        'target_entropy': 'auto',
        'use_sde':        False,
    }
}

DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        150.0,
        'max_charge_power':     75.0,  # C/2
        'max_discharge_power':  75.0,  # C/2
        'efficiency':          0.95,
        'lcos':                1.5,
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


# Domain randomization: each reset() samples a new hardware config so the model
# learns to operate correctly across diverse battery/solar/inverter/grid sizes.
class RandomConfigWrapper(gym.Wrapper):
    def __init__(self, df: pd.DataFrame, df_raw: pd.DataFrame, episode_len: int = 96):
        self.df = df
        self.df_raw = df_raw
        self.episode_len = episode_len
        env = Environment(df_raw=df_raw, df=df, system_config=self._sample_config(), episode_len=episode_len)
        super().__init__(env)

    def _sample_config(self) -> dict:
        capacity      = float(np.random.uniform(50, 250))
        solar_peak    = float(capacity * np.random.uniform(0.8, 2.0))
        inverter_max  = float(solar_peak * np.random.uniform(0.8, 1.1))
        grid_capacity = float(inverter_max * np.random.uniform(1.0, 1.5))
        return {
            'battery': {
                'capacity_kwh':        capacity,
                'max_charge_power':    capacity / 2,
                'max_discharge_power': capacity / 2,
                'efficiency':          float(np.random.uniform(0.90, 0.98)),
                'lcos':                float(np.random.uniform(0.5, 3.0)),
                'min_reserve':         int(np.random.randint(10, 31)),
            },
            'solar': {
                'peak_power':  solar_peak,
                'efficiency':  float(np.random.uniform(0.17, 0.23)),
            },
            'inverter': {
                'max_power': inverter_max,
            },
            'grid': {
                'capacity': grid_capacity,
            },
        }

    def reset(self, **kwargs):
        self.env = Environment(df_raw=self.df_raw, df=self.df, system_config=self._sample_config(), episode_len=self.episode_len)
        return self.env.reset(**kwargs)


def load_data():
    df     = pd.read_csv(CONFIG['dataset_path'])
    df_raw = pd.read_csv(CONFIG['dataset_raw_path'])

    train_idx, eval_idx = [], []
    for month in range(1, 13):
        idx = df_raw.index[df_raw['Month'] == month].tolist()
        split = int(len(idx) * 0.75)   # ~3 weeks train, ~1 week eval per month
        train_idx.extend(idx[:split])
        eval_idx.extend(idx[split:])

    df_train     = df.iloc[train_idx].reset_index(drop=True)
    df_train_raw = df_raw.iloc[train_idx].reset_index(drop=True)
    df_eval      = df.iloc[eval_idx].reset_index(drop=True)
    df_eval_raw  = df_raw.iloc[eval_idx].reset_index(drop=True)

    print(f"Train: {len(df_train)} rows across all 12 months")
    print(f"Eval:  {len(df_eval)} rows across all 12 months")

    return df_train, df_train_raw, df_eval, df_eval_raw


def make_envs(df_train, df_train_raw, df_eval, df_eval_raw):
    os.makedirs(CONFIG['monitor_dir'], exist_ok=True)

    n = CONFIG['n_envs']
    def make_train_env(i):
        def _init():
            return Monitor(
                RandomConfigWrapper(df_train, df_train_raw),
                filename=os.path.join(CONFIG['monitor_dir'], f'train_{i}')
            )
        return _init

    # DummyVecEnv (single process, no IPC): faster than SubprocVecEnv when env step < pipe overhead.
    # Benchmarked: env step = 0.11 ms, SubprocVecEnv pipe = ~13 ms → DummyVecEnv wins 7×.
    train_env = DummyVecEnv([make_train_env(i) for i in range(n)])

    eval_env = DummyVecEnv([
        lambda: Monitor(
            Environment(df_raw=df_eval_raw, df=df_eval, system_config=DEFAULT_SYSTEM_CONFIG, episode_len=96),
            filename=os.path.join(CONFIG['monitor_dir'], 'eval')
        )
    ])

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    return train_env, eval_env


def make_model(train_env):
    print("\n" + "="*60)
    print("Initializing SAC")
    print("="*60)

    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['tensorboard_dir'], exist_ok=True)

    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        tensorboard_log=CONFIG['tensorboard_dir'],
        **CONFIG['sac_params']
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy parameters: {total_params:,}")
    print(f"Architecture: {CONFIG['sac_params']['policy_kwargs']['net_arch']}")

    return model


class StatsCallback(BaseCallback):
    """Prints one summary line to stdout every `log_every` env steps."""

    def __init__(self, log_every: int = 100_000):
        super().__init__(verbose=0)
        self.log_every  = log_every
        self._last_log  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_every:
            buf = self.model.ep_info_buffer
            if buf:
                mean_rew = np.mean([e['r'] for e in buf])
                mean_len = np.mean([e['l'] for e in buf])
                print(
                    f"step {self.num_timesteps:>9,} | "
                    f"ep_rew_mean {mean_rew:>10.2f} | "
                    f"ep_len_mean {mean_len:>5.0f}"
                )
            else:
                print(f"step {self.num_timesteps:>9,} | collecting...")
            self._last_log = self.num_timesteps
        return True


class SyncNormalizeEvalCallback(EvalCallback):
    """EvalCallback that copies obs_rms from train_env → eval_env before each evaluation.

    Without this, train and eval VecNormalize instances diverge over time, making
    the Q-function evaluate against a different observation distribution than it was
    trained on. The copy is shallow-cloned so train_env continues updating its own stats.
    """
    def __init__(self, train_env: VecNormalize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_env = train_env

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = copy.deepcopy(self._train_env.obs_rms)
        return super()._on_step()


def make_callbacks(train_env, eval_env):
    # SB3 _on_step() fires once per n_envs steps — divide to keep freq at intended absolute step count.
    freq = max(CONFIG['eval_freq'] // CONFIG['n_envs'], 1)

    eval_cb = SyncNormalizeEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        best_model_save_path=os.path.join('models', 'best'),
        log_path=os.path.join('logs', 'eval'),
        eval_freq=freq,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    stats_cb = StatsCallback(log_every=100_000)

    return CallbackList([eval_cb, stats_cb])


def train(model, callbacks):
    print("\n" + "="*60)
    print("Training")
    print(f"Steps: {CONFIG['total_timesteps']:,}")
    print("Tensorboard: tensorboard --logdir logs/tensorboard/")
    print("="*60 + "\n")

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=callbacks,
        log_interval=CONFIG['log_interval'],
        progress_bar=True,
        reset_num_timesteps=True,
    )

    return model


def save_and_test(model, eval_env):
    print("\n" + "="*60)
    print("Saving and testing")
    print("="*60)

    model.save(CONFIG['model_save_path'])
    print(f"Model → {CONFIG['model_save_path']}.zip")

    print("\nRunning one test episode...")
    obs = eval_env.reset()
    total_money_earned = 0.0
    total_reward = 0.0
    total_unmet  = 0.0
    steps        = 0
    last_info    = {}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = eval_env.step(action)
        info = infos[0]
        total_reward       += float(reward[0])
        total_money_earned += float(info.get('money_earned_ts', 0.0))
        total_unmet        += float(info.get('unmet_load_kwh', 0.0))
        steps += 1
        last_info = info
        if dones[0]:
            break

    print(f"Steps:        {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Unmet load:   {total_unmet:.4f} kWh")
    print(f"Final SoC:    {last_info.get('soc', 0.0):.3f}")
    print(f"Earned:       {total_money_earned:.2f} UAH")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    np.random.seed(42)

    d_train, d_train_raw, d_eval, d_eval_raw = load_data()

    train_env, eval_env = make_envs(d_train, d_train_raw, d_eval, d_eval_raw)

    model = make_model(train_env)
    callbacks = make_callbacks(train_env, eval_env)
    model = train(model, callbacks)

    obs_rms_path = 'models/obs_rms.pkl'
    with open(obs_rms_path, 'wb') as f:
        pickle.dump(train_env.obs_rms, f)
    print(f"obs_rms → {obs_rms_path}")

    save_and_test(model, eval_env)

    print("\n" + "="*60)
    print("Done!")
    print(f"Model:       {CONFIG['model_save_path']}.zip")
    print(f"Tensorboard: tensorboard --logdir {CONFIG['tensorboard_dir']}")
    print("="*60)

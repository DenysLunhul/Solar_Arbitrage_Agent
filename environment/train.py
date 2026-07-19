import argparse
import copy
import glob
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# project root appended (not prepended) so `environment` still resolves to
# environment.py in this directory rather than the environment/ package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from environment import Environment
from data_providers.components.load.load_profiles import PROFILES, generate_series

def lr_schedule(progress_remaining: float) -> float:
    """Step decay: 1e-4 for the first 60%, 5e-5 for 60-80%, 2.5e-5 for the final 20%.
    Stabilises the critic in late training where oscillations previously prevented improvement."""
    if progress_remaining > 0.40:
        return 1e-4
    elif progress_remaining > 0.20:
        return 5e-5
    else:
        return 2.5e-5

CONFIG = {
    'dataset_path':    'dataset_normalized.csv',
    'dataset_raw_path': 'dataset_final.csv',
    'scalers_path':    'models/scalers.pkl',
    'model_save_path': 'models/sac_ems',
    'tensorboard_dir': 'logs/tensorboard/',
    'monitor_dir':     'logs/monitor/',

    'total_timesteps': 20_000_000,
    'eval_freq':       100_000,
    'log_interval':    100_000,
    'n_envs':          32,

    'sac_params': {
        'device':          'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_size':     2_000_000,
        'learning_starts': 50_000,
        'batch_size':      512,
        # 32 gradient updates per collected 32-env step = 1.0 update-to-data ratio
        # (canonical SAC). Keeps the GPU busy instead of idling at the old 1/32 ratio;
        # set back to 1 to restore the low-ratio behavior of pre-2026-07 runs.
        'gradient_steps':  32,
        'learning_rate':   lr_schedule,
        'gamma':           0.99,
        'tau':             0.002,
        'ent_coef':        'auto',
        'policy_kwargs': {
            'net_arch': [512, 512],
        },
        'verbose': 0,
        'seed':    42,
        'target_entropy': -1.0,
    }
}

DEFAULT_SYSTEM_CONFIG = {
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
    'load': {
        'peak_kw': 60.0,
        'profile': 'office',
    },
}

class RandomConfigWrapper(gym.Wrapper):

    def __init__(self, df: pd.DataFrame, df_raw: pd.DataFrame, episode_len: int = 96):
        self.df = df
        self.df_raw = df_raw
        self.episode_len = episode_len
        self._hours   = df_raw['Hour'].values
        self._minutes = df_raw['Minute'].values
        self._dow     = df_raw['Day_of_week'].values
        super().__init__(self._make_env())

    def _sample_config(self) -> dict:
        # Load-first correlated sampling: the site's consumption sets the scale,
        # hardware is sized around it so combos stay realistic.
        profile       = str(np.random.choice(PROFILES))
        peak_load     = float(np.random.uniform(10.0, 150.0))                    # kW
        capacity      = float(peak_load * np.random.uniform(1.5, 5.0))           # 1.5-5 h of peak load
        solar_peak    = float(peak_load * np.random.uniform(0.6, 2.5))
        inverter_max  = float(max(solar_peak * np.random.uniform(0.8, 1.1),
                                  peak_load  * np.random.uniform(1.1, 1.4)))     # headroom over peak+noise/spikes
        grid_capacity = float(inverter_max * np.random.uniform(1.0, 1.5))
        return {
            'battery': {
                'capacity_kwh':        capacity,
                'max_charge_power':    capacity / 2,
                'max_discharge_power': capacity / 2,
                'efficiency':          float(np.random.uniform(0.90, 0.98)),
                'lcos':                float(np.random.uniform(0.95, 1.25)),
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
            'load': {
                'peak_kw': peak_load,
                'profile': profile,
            },
        }

    def _make_env(self) -> Environment:
        config = self._sample_config()
        rng = np.random.default_rng(np.random.randint(2**31))
        load_kw = generate_series(config['load']['profile'], config['load']['peak_kw'],
                                  self._hours, self._minutes, self._dow, rng)
        return Environment(df_raw=self.df_raw, df=self.df, system_config=config,
                           episode_len=self.episode_len, load_kw=load_kw)

    def reset(self, **kwargs):
        self.env = self._make_env()
        return self.env.reset(**kwargs)

def load_data():
    df     = pd.read_csv(CONFIG['dataset_path'])
    df_raw = pd.read_csv(CONFIG['dataset_raw_path'])

    train_idx, eval_idx = [], []
    for month in range(1, 13):
        idx = df_raw.index[df_raw['Month'] == month].tolist()
        # round the split point down to a full day so day-aligned episodes never straddle the boundary
        split = (int(len(idx) * 0.75) // 96) * 96
        assert split > 0, f"Month {month} has only {len(idx)} rows — too few for a day-aligned 75/25 split"
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

    train_env = DummyVecEnv([make_train_env(i) for i in range(n)])

    # Eval load comes from the SAME generator as training and the live pipeline
    # (load_profiles), seeded once so every evaluation sees the identical series.
    # The dataset's Load column is a legacy shape (temp_scripts/change_load.py) kept
    # only as the fixed backtest benchmark — using it here would select checkpoints
    # on a distribution that matches neither training nor production.
    eval_load = generate_series(
        DEFAULT_SYSTEM_CONFIG['load']['profile'], DEFAULT_SYSTEM_CONFIG['load']['peak_kw'],
        df_eval_raw['Hour'].values, df_eval_raw['Minute'].values,
        df_eval_raw['Day_of_week'].values, np.random.default_rng(4242),
    )

    eval_env = DummyVecEnv([
        lambda: Monitor(
            Environment(df_raw=df_eval_raw, df=df_eval, system_config=DEFAULT_SYSTEM_CONFIG,
                        episode_len=96, load_kw=eval_load),
            filename=os.path.join(CONFIG['monitor_dir'], 'eval')
        )
    ])

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=100.0)
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

class RichProgressCallback(BaseCallback):
    """Live terminal dashboard: percentage bar, steps/s, ETA, reward, entropy,
    losses, buffer fill and eval scores — all in the terminal that launched
    training. Falls back to plain flushed prints when stdout is not a TTY
    (nohup / log-file runs), so those still show progress lines.
    """

    def __init__(self, total_timesteps: int, eval_cb: EvalCallback,
                 refresh_seconds: float = 1.0, plain_log_every: int = 100_000):
        super().__init__(verbose=0)
        self._total          = total_timesteps
        self._eval_cb        = eval_cb
        self._refresh        = refresh_seconds
        self._plain_every    = plain_log_every
        self._last_render    = 0.0
        self._plain_next     = 0
        self._n_evals_seen   = 0
        self._progress       = None
        self._task           = None
        self._is_tty         = sys.stdout.isatty()

    def _metrics_text(self) -> str:
        vals = self.model.logger.name_to_value
        buf  = self.model.ep_info_buffer
        parts = []
        if buf:
            parts.append(f"rew {np.mean([e['r'] for e in buf]):9.2f}")
        else:
            parts.append("collecting…")
        ent = vals.get('train/ent_coef')
        if ent is not None:
            parts.append(f"ent {ent:.4f}")
        aloss = vals.get('train/actor_loss')
        closs = vals.get('train/critic_loss')
        if aloss is not None:
            parts.append(f"actor {aloss:8.2f}")
        if closs is not None:
            parts.append(f"critic {closs:8.2f}")
        fill = self.model.replay_buffer.size() / self.model.replay_buffer.buffer_size
        parts.append(f"buf {fill:4.0%}")
        if self._eval_cb.evaluations_results:
            parts.append(f"eval {self._eval_cb.last_mean_reward:9.2f} "
                         f"(best {self._eval_cb.best_mean_reward:9.2f})")
        return " │ ".join(parts)

    def _on_training_start(self) -> None:
        if not self._is_tty:
            print(f"stdout is not a TTY — plain progress lines every "
                  f"{self._plain_every:,} steps", flush=True)
            return
        from rich.progress import (Progress, BarColumn, TaskProgressColumn,
                                   MofNCompleteColumn, TimeElapsedColumn,
                                   TimeRemainingColumn, TextColumn)
        self._progress = Progress(
            TextColumn("[bold blue]SAC"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[metrics]}"),
            refresh_per_second=4,
        )
        self._progress.start()
        self._task = self._progress.add_task(
            "train", total=self._total,
            completed=self.model.num_timesteps, metrics="warming up…",
        )

    def _on_step(self) -> bool:
        now = time.monotonic()
        if self._is_tty:
            if now - self._last_render >= self._refresh:
                self._last_render = now
                self._progress.update(self._task, completed=self.model.num_timesteps,
                                      metrics=self._metrics_text())
            n_evals = len(self._eval_cb.evaluations_results)
            if n_evals > self._n_evals_seen:
                self._n_evals_seen = n_evals
                self._progress.console.print(
                    f"[green]eval[/green] @ {self.model.num_timesteps:>11,} steps: "
                    f"mean reward {self._eval_cb.last_mean_reward:9.2f} "
                    f"(best {self._eval_cb.best_mean_reward:9.2f})"
                )
        else:
            if self.model.num_timesteps >= self._plain_next:
                self._plain_next = self.model.num_timesteps + self._plain_every
                pct = 100.0 * self.model.num_timesteps / self._total
                print(f"{pct:5.1f}% | step {self.model.num_timesteps:>11,} | "
                      f"{self._metrics_text()}", flush=True)
        return True

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        if self._progress is not None:
            self._progress.update(self._task, completed=self.model.num_timesteps,
                                  metrics=self._metrics_text())
            self._progress.stop()
            self._progress = None

class SyncNormalizeEvalCallback(EvalCallback):
    """EvalCallback that copies obs_rms from train_env → eval_env before each evaluation,
    and saves obs_rms to disk whenever a new best_model.zip is written.

    Without the sync, train and eval VecNormalize instances diverge over time.
    Without saving obs_rms at the best checkpoint, inference uses end-of-training
    normalisation stats against mid-training weights — a distribution mismatch.
    """

    def __init__(self, train_env: VecNormalize, obs_rms_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_env   = train_env
        self._obs_rms_path = obs_rms_path
        self._prev_best   = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.obs_rms = copy.deepcopy(self._train_env.obs_rms)
        result = super()._on_step()
        if self.best_mean_reward > self._prev_best:
            self._prev_best = self.best_mean_reward
            with open(self._obs_rms_path, 'wb') as f:
                pickle.dump(self._train_env.obs_rms, f)
        return result

class CheckpointAndNormalizeCallback(BaseCallback):
    """Saves model + obs_rms together every `save_freq` env steps.

    Unlike the stock CheckpointCallback, this pairs each model .zip with an
    obs_rms.pkl so --resume can restore normalisation stats exactly.
    """

    def __init__(self, train_env: VecNormalize, save_freq: int, save_path: str,
                 name_prefix: str = 'sac_ems', verbose: int = 1):
        super().__init__(verbose)
        self._train_env   = train_env
        self._save_freq   = save_freq
        self._save_path   = save_path
        self._name_prefix = name_prefix
        self._next_save   = save_freq
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save:
            stem = os.path.join(
                self._save_path,
                f'{self._name_prefix}_{self.num_timesteps}_steps',
            )
            self.model.save(stem)
            with open(stem + '_obs_rms.pkl', 'wb') as f:
                pickle.dump(self._train_env.obs_rms, f)
            if self.verbose:
                print(f'Checkpoint saved: {stem}.zip + obs_rms.pkl')
            self._next_save += self._save_freq
        return True


def _checkpoint_step(path: str) -> int:
    m = re.search(r'_(\d+)_steps\.zip$', path)
    return int(m.group(1)) if m else 0


def load_checkpoint(checkpoint_dir: str, train_env: VecNormalize, eval_env: VecNormalize):
    """Find the latest checkpoint, load weights + obs_rms. Returns model or None."""
    files = glob.glob(os.path.join(checkpoint_dir, 'sac_ems_*_steps.zip'))
    if not files:
        print('No checkpoints found — starting fresh.')
        return None

    latest = max(files, key=_checkpoint_step)
    steps  = _checkpoint_step(latest)
    print(f'Resuming from step {steps:,}: {latest}')

    try:
        model = SAC.load(latest, env=train_env, device=CONFIG['sac_params']['device'])
    except ValueError as e:
        raise SystemExit(
            f"Checkpoint {latest} is incompatible with the current observation space "
            f"{train_env.observation_space.shape} — it predates an env change. "
            f"Move or delete models/checkpoints/ and start fresh.\nOriginal error: {e}"
        )

    obs_rms_path = latest.replace('.zip', '_obs_rms.pkl')
    if os.path.exists(obs_rms_path):
        with open(obs_rms_path, 'rb') as f:
            obs_rms = pickle.load(f)
        train_env.obs_rms = obs_rms
        eval_env.obs_rms  = copy.deepcopy(obs_rms)
        print('obs_rms restored from checkpoint.')
    else:
        print('No obs_rms paired with this checkpoint — normalisation will re-warm.')

    return model


def make_callbacks(train_env, eval_env):
    freq = max(CONFIG['eval_freq'] // CONFIG['n_envs'], 1)

    eval_cb = SyncNormalizeEvalCallback(
        train_env=train_env,
        obs_rms_path='models/obs_rms.pkl',
        eval_env=eval_env,
        best_model_save_path=os.path.join('models', 'best'),
        log_path=os.path.join('logs', 'eval'),
        eval_freq=freq,
        n_eval_episodes=50,
        deterministic=True,
        verbose=0,   # eval results are surfaced by the dashboard instead
    )

    checkpoint_cb = CheckpointAndNormalizeCallback(
        train_env=train_env,
        save_freq=2_000_000,
        save_path=os.path.join('models', 'checkpoints'),
        verbose=1,
    )

    dashboard = RichProgressCallback(
        total_timesteps=CONFIG['total_timesteps'],
        eval_cb=eval_cb,
    )

    return CallbackList([eval_cb, checkpoint_cb, dashboard]), dashboard

def train(model, callbacks, reset_num_timesteps: bool = True):
    print("\n" + "="*60)
    print("Training" + (" (resumed)" if not reset_num_timesteps else ""))
    print(f"Steps: {CONFIG['total_timesteps']:,}  |  device: {model.device}")
    print("Tensorboard: tensorboard --logdir logs/tensorboard/")
    print("Ctrl-C saves a checkpoint you can continue from with --resume")
    print("="*60 + "\n", flush=True)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=callbacks,
        log_interval=CONFIG['log_interval'],
        progress_bar=False,   # RichProgressCallback owns the terminal display
        reset_num_timesteps=reset_num_timesteps,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint in models/checkpoints/')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    np.random.seed(42)

    d_train, d_train_raw, d_eval, d_eval_raw = load_data()

    train_env, eval_env = make_envs(d_train, d_train_raw, d_eval, d_eval_raw)

    if args.resume:
        model = load_checkpoint('models/checkpoints', train_env, eval_env)
        if model is None:
            model = make_model(train_env)
            args.resume = False
    else:
        model = make_model(train_env)

    callbacks, dashboard = make_callbacks(train_env, eval_env)
    try:
        model = train(model, callbacks, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        dashboard.close()
        steps = model.num_timesteps
        stem = os.path.join('models', 'checkpoints', f'sac_ems_{steps}_steps')
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        model.save(stem)
        with open(stem + '_obs_rms.pkl', 'wb') as f:
            pickle.dump(train_env.obs_rms, f)
        print(f"\nInterrupted at {steps:,} steps — checkpoint saved: {stem}.zip")
        print("Continue with: python train.py --resume")
        sys.exit(0)

    save_and_test(model, eval_env)

    print("\n" + "="*60)
    print("Done!")
    print(f"Model:       {CONFIG['model_save_path']}.zip")
    print(f"Tensorboard: tensorboard --logdir {CONFIG['tensorboard_dir']}")
    print("="*60)

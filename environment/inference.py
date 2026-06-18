
import os
import argparse
import pickle
import numpy as np
import pandas as pd
try:
    from environment import Environment
    from normalize import normalize_row
except ImportError:
    from environment.environment import Environment
    from environment.normalize import normalize_row


def load_model_and_scalers(
    model_path: str,
    scalers_path: str,
    obs_rms_path: str = None,
    model_cls=None,
):
    """Load model, scalers and obs_rms. Call once at startup and cache the result."""
    if model_cls is None:
        from stable_baselines3 import SAC
        model_cls = SAC
    print(f"Loading model:   {model_path}")
    model = model_cls.load(model_path, device='cpu')
    print(f"Loading scalers: {scalers_path}")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    obs_rms = None
    if obs_rms_path and os.path.exists(obs_rms_path):
        print(f"Loading obs_rms: {obs_rms_path}")
        with open(obs_rms_path, 'rb') as f:
            obs_rms = pickle.load(f)
    else:
        print("obs_rms not found — observation normalisation disabled")
    print("Done.\n")
    return model, scalers, obs_rms


def _check_obs_rms(obs_rms, obs_dim: int):
    """Return obs_rms if shape matches obs, else None with a warning."""
    if obs_rms is None:
        return None
    rms_dim = obs_rms.mean.shape[0]
    if rms_dim != obs_dim:
        print(
            f"WARNING: obs_rms shape ({rms_dim},) != env obs shape ({obs_dim},) — "
            f"saved from an older training run, skipping normalisation. Retrain to fix."
        )
        return None
    return obs_rms


def run_inference(
    df_raw:        pd.DataFrame,
    system_config: dict,
    model,
    scalers:       dict,
    initial_soc:   float = 0.5,
    obs_rms=None,
) -> dict:
    """Run the trained model on df_raw and return dispatch_plan + summary."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    df_norm = pd.DataFrame([
        normalize_row(df_raw.iloc[i], scalers)
        for i in range(len(df_raw))
    ])
    env = Environment(df_raw=df_raw, df=df_norm, system_config=system_config)
    obs, _ = env.reset()
    env.soc = float(np.clip(initial_soc, 0.0, 1.0))
    obs = env.get_observe()
    obs_rms = _check_obs_rms(obs_rms, obs.shape[0])
    dispatch_plan = []
    while True:
        if obs_rms is not None:
            obs_input = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8),
                -10.0, 10.0,
            ).astype(np.float32)
        else:
            obs_input = obs
        action, _ = model.predict(obs_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        dispatch_plan.append({
            'step':              env.curr_step - 1,
            'action_battery':    round(float(action[0]), 4),
            'action_grid':       round(float(action[1]), 4),
            'soc':               round(float(info['soc']), 4),
            'target_soc':        round(float(info['target_soc']), 4),
            'solar_gen_kwh':     round(float(info['solar_gen_ts_kwh']), 4),
            'solar_surplus_kwh': round(float(info['solar_surplus_kwh']), 4),
            'battery_kwh':       round(float(info['battery_kwh']), 4),
            'grid_kwh':          round(float(info['actual_grid_kwh']), 4),
            'unmet_load_kwh':    round(float(info['unmet_load_kwh']), 4),
            'lcos_cost':         round(float(info['lcos_cost']), 4),
            'mismatch':          round(float(info['mismatch']), 4),
            'money_earned_ts':   round(float(info['money_earned_ts']), 4),
            'reward':            round(float(reward), 4),
            'reward_market':     round(float(info['reward_market']), 4),
            'reward_lcos':       round(float(info['reward_lcos']), 4),
            'reward_unmet':      round(float(info['reward_unmet']), 4),
            'reward_mismatch':   round(float(info['reward_mismatch']), 4),
            'reward_soc_soft':   round(float(info['reward_soc_soft']), 4),
            'reward_reserve':    round(float(info['reward_reserve']), 4),
            'reward_preparation':round(float(info['reward_preparation']), 4),
            'reward_soc_target': round(float(info['reward_soc_target']), 4),
            'curtailed_kwh':          round(float(info['curtailed_kwh']), 4),
            'reward_waste':           round(float(info['reward_waste']), 4),
            'reward_curtail':         round(float(info['reward_curtail']), 4),
            'reward_price_timing':    round(float(info['reward_price_timing']), 4),
            'reward_solar_priority':  round(float(info['reward_solar_priority']), 4),
            'reward_eod_soc':         round(float(info['reward_eod_soc']), 4),
        })
        if terminated or truncated:
            break
    _total_money_earned = round(sum(x['money_earned_ts'] for x in dispatch_plan))
    _bought_kwh  = round(sum(x['grid_kwh'] for x in dispatch_plan if x['grid_kwh'] > 0), 3)
    _sold_kwh    = round(sum(abs(x['grid_kwh']) for x in dispatch_plan if x['grid_kwh'] < 0), 3)
    _solar_kwh   = round(sum(x['solar_gen_kwh'] for x in dispatch_plan), 3)
    _lcos_uah    = round(sum(x['lcos_cost'] for x in dispatch_plan), 3)
    _total_bought_cost = sum(-x['money_earned_ts'] for x in dispatch_plan if x['grid_kwh'] > 0)
    _avg_buy_price     = (_total_bought_cost / _bought_kwh) if _bought_kwh > 0 else 8.0
    _curtailed_kwh       = round(sum(x['curtailed_kwh'] for x in dispatch_plan), 3)
    _solar_self_consumed = max(0.0, _solar_kwh - _sold_kwh - _curtailed_kwh)
    _economic_savings    = round(
        _total_money_earned + _solar_self_consumed * _avg_buy_price - _lcos_uah, 2
    )
    summary = {
        'total_money_earned':  _total_money_earned,
        'economic_savings_uah': _economic_savings,
        'total_reward_uah':    round(sum(x['reward'] for x in dispatch_plan), 2),
        'bought_kwh':          _bought_kwh,
        'sold_kwh':            _sold_kwh,
        'solar_kwh':           _solar_kwh,
        'curtailed_kwh':       _curtailed_kwh,
        'unmet_load_kwh':      round(sum(x['unmet_load_kwh'] for x in dispatch_plan), 4),
        'lcos_total_uah':      _lcos_uah,
        'initial_soc':         round(initial_soc, 3),
        'final_soc':           dispatch_plan[-1]['soc'] if dispatch_plan else initial_soc,
        'steps':               len(dispatch_plan),
    }
    return {'dispatch_plan': dispatch_plan, 'summary': summary}


DEFAULT_SYSTEM_CONFIG = {
    'battery': {
        'capacity_kwh':        250.0,
        'max_charge_power':    125.0,
        'max_discharge_power': 125.0,
        'efficiency':          0.95,
        'lcos':                1.15,
        'min_reserve':         20,
    },
    'solar': {
        'peak_power':  250.0,
        'efficiency':  0.20,
    },
    'inverter': {
        'max_power': 250.0,
    },
    'grid': {
        'capacity': 250.0,
    },
}


def _fetch_soc_from_db(min_reserve: float) -> tuple[float, str]:
    """Try to get the last SoC from the database. Returns (soc, source_label)."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backend.core.database import SessionLocal
        from backend.repositories.prediction_repo import get_latest_soc
        db = SessionLocal()
        try:
            persisted = get_latest_soc(db)
        finally:
            db.close()
        if persisted is not None:
            soc = max(float(persisted), min_reserve)
            label = f"from DB (clamped to {soc:.3f})" if soc > persisted else "from DB"
            return soc, label
    except Exception as e:
        print(f"DB unavailable ({e}) — using default SoC")
    return min_reserve, "default (DB unavailable or empty)"


if __name__ == '__main__':
    import json
    import sys
    from pathlib import Path
    _ROOT    = Path(__file__).resolve().parent.parent
    _ENV_DIR = _ROOT / 'environment'
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   default=str(_ENV_DIR / 'models' / 'best' / 'best_model.zip'))
    parser.add_argument('--scalers', default=str(_ENV_DIR / 'models' / 'scalers.pkl'))
    parser.add_argument('--obsrms',  default=str(_ENV_DIR / 'models' / 'obs_rms.pkl'))
    parser.add_argument('--config',  default=None, help='JSON file with system_config')
    parser.add_argument('--output',  default=str(_ENV_DIR / 'results' / 'dispatch_plan.csv'))
    parser.add_argument('--soc',     type=float, default=None, help='Initial SoC (0-1). Overrides DB lookup.')
    parser.add_argument('--tilt',    type=float, default=35.0, help='Solar panel tilt (degrees)')
    parser.add_argument('--azimuth', type=float, default=0.0,  help='Solar panel azimuth (degrees)')
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            system_config = json.load(f)
        print(f"Config: {args.config}")
    else:
        system_config = DEFAULT_SYSTEM_CONFIG
        print("Config: DEFAULT_SYSTEM_CONFIG")
    model, scalers, obs_rms = load_model_and_scalers(args.model, args.scalers, args.obsrms)
    _COMBINED = _ROOT / 'data_providers' / 'orchestrator' / 'combined.csv'
    try:
        from data_providers.orchestrator.data_combiner import combine
        df = combine(0, tilt=args.tilt, azimuth=args.azimuth)
        if df is not None:
            print(f"Data: fetched live from orchestrator ({len(df)} rows)")
            df.to_csv(_COMBINED, index=False)
        else:
            print("DAM data unavailable — falling back to cached combined.csv")
            df = pd.read_csv(_COMBINED)
    except Exception as e:
        print(f"Orchestrator error ({e}) — falling back to cached combined.csv")
        df = pd.read_csv(_COMBINED)
    df = df.iloc[:96].reset_index(drop=True)
    print(f"Date: {str(df['timestamp'].iloc[0])[:10]}")
    nan_dam_cols = [c for c in ('DAM_Price', 'DAM_Vol_Sale', 'DAM_Vol_Buy') if df[c].isnull().all()]
    if nan_dam_cols:
        raise SystemExit(f"ERROR: DAM data missing ({', '.join(nan_dam_cols)} are all NaN) — OREE fetch failed. Inference aborted.")
    min_reserve = system_config['battery']['min_reserve'] / 100
    if args.soc is not None:
        initial_soc = args.soc
        print(f"SoC:  {initial_soc:.3f} (from --soc argument)")
    else:
        initial_soc, label = _fetch_soc_from_db(min_reserve)
        print(f"SoC:  {initial_soc:.3f} ({label})")
    print()
    result = run_inference(
        df_raw=df,
        system_config=system_config,
        model=model,
        scalers=scalers,
        initial_soc=initial_soc,
        obs_rms=obs_rms,
    )
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in result['summary'].items():
        print(f"  {k:25s} {v}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result['dispatch_plan']).to_csv(output_path, index=False)
    print(f"\nDispatch plan → {output_path}")
    print(pd.DataFrame(result['dispatch_plan']).head(10).to_string())

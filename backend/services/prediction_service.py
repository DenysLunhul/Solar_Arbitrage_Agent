from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from stable_baselines3 import PPO, SAC

from backend.models.site import AgentPredictions
from backend.repositories import config_repo, model_repo, prediction_repo
from backend.schemas.schemas import SiteConfig
from data_providers.orchestrator.data_combiner import combine
from envoriment.inference import load_model_and_scalers, run_inference

_BASE_DIR    = Path(__file__).resolve().parent.parent.parent
SCALERS_PATH = str(_BASE_DIR / "envoriment" / "models" / "scalers.pkl")
OBS_RMS_PATH = str(_BASE_DIR / "envoriment" / "models" / "obs_rms.pkl")

UA_TZ = timezone(timedelta(hours=2))

_ALGO_CLS = {"PPO": PPO, "SAC": SAC}

_model_cache: dict[int, tuple] = {}


def _load_model(agent_model) -> tuple:
    config_id = agent_model.config_id
    if config_id not in _model_cache:
        algo = agent_model.algorithm or "PPO"
        model_cls = _ALGO_CLS.get(algo, PPO)
        _model_cache[config_id] = load_model_and_scalers(
            model_path=agent_model.storage_path.replace(".zip", ""),
            scalers_path=SCALERS_PATH,
            obs_rms_path=OBS_RMS_PATH,
            model_cls=model_cls,
        )
    return _model_cache[config_id]


def get_predictions(db: Session, config_name: str, user_id: int, initial_soc: float) -> dict:
    if datetime.now(UA_TZ).hour < 14:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="DAM data for tomorrow is not yet available. Please call after 14:00.",
        )

    raw_config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")

    agent_model = model_repo.get_ready_for_config(db, raw_config.id)
    if agent_model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model for this config. POST /config/train first.",
        )

    df_raw = combine(raw_config.id)
    if df_raw is None or df_raw.isnull().values.any():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has null values")

    model, scalers, obs_rms = _load_model(agent_model)

    result = run_inference(
        df_raw=df_raw,
        system_config=SiteConfig(**raw_config.settings).to_env_dict(),
        model=model,
        scalers=scalers,
        initial_soc=initial_soc,
        obs_rms=obs_rms,
    )

    tomorrow = (datetime.now(UA_TZ) + timedelta(days=1)).date()
    prediction_repo.delete_for_date(db, raw_config.id, tomorrow)
    prediction_repo.bulk_create(
        db, _build_rows(result["dispatch_plan"], df_raw, user_id, raw_config.id, tomorrow)
    )

    return result


def _build_rows(
    dispatch_plan: list, df_raw: pd.DataFrame, user_id: int, config_id: int, target_date
) -> list[AgentPredictions]:
    rows = []
    for step_data in dispatch_plan:
        row = df_raw.iloc[step_data["step"]]
        rows.append(AgentPredictions(
            user_id=user_id,
            config_id=config_id,
            date=target_date,
            step=step_data["step"],
            timestamp=pd.to_datetime(row["timestamp"]),
            battery_action=step_data["action_battery"],
            grid_action=step_data["action_grid"],
            load_kwh=float(row["Load"]) / 1000 / 4,
            solar_kwh=step_data["solar_gen_kwh"],
            solar_surplus_kwh=step_data["solar_surplus_kwh"],
            battery_kwh=step_data["battery_kwh"],
            grid_kwh=step_data["grid_kwh"],
            unmet_load_kwh=step_data["unmet_load_kwh"],
            soc=step_data["soc"],
            target_soc=step_data["target_soc"],
            lcos_cost=step_data["lcos_cost"],
            mismatch=step_data["mismatch"],
            dam_price=float(row["DAM_Price"]) / 1000,
            grid_status=int(row["Grid"]),
            hours_until_outage=float(row["hours_until_outage"]),
            outage_remaining_h=float(row["outage_remaining_h"]),
            next_outage_duration=float(row["next_outage_duration"]),
            reward_market=step_data["reward_market"],
            reward_lcos=step_data["reward_lcos"],
            reward_unmet=step_data["reward_unmet"],
            reward_soc_soft=step_data["reward_soc_soft"],
            reward_reserve=step_data["reward_reserve"],
            reward_preparation=step_data["reward_preparation"],
            reward_total=step_data["reward"],
        ))
    return rows

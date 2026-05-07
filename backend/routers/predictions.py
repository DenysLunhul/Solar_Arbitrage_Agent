import os
import tempfile
from datetime import datetime, timedelta

import boto3
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from stable_baselines3 import PPO

from backend.core.trainer import site_config_to_env_dict, MINIO_BUCKET
from backend.models.site import SystemConfig, AgentModels, AgentPredictions
from backend.schemas.schemas import SiteConfig
from backend.security.security import get_current_user, get_db
from data_providers.orchestrator.data_combiner import combine
from envoriment.inference import load_model_and_scalers, run_inference

SCALERS_PATH = "envoriment/models/scalers.pkl"
OBS_RMS_PATH = "envoriment/models/obs_rms.pkl"

router = APIRouter(prefix="/predictions", tags=["predictions"])


def _minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )


@router.get("/")
def get_predictions(config_name: str, db=Depends(get_db), user=Depends(get_current_user)):
    if datetime.now().hour < 14:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="DAM data for tomorrow is not yet available. Please call after 14:00.",
        )

    raw_config = db.query(SystemConfig).filter(
        SystemConfig.config_name == config_name,
        SystemConfig.user_id == user.id,
    ).first()
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")

    agent_model = db.query(AgentModels).filter(
        AgentModels.config_id == raw_config.id,
        AgentModels.status == "ready",
    ).first()
    if agent_model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained model for this config. POST /config/ first and wait for training.",
        )

    df_raw = combine(raw_config.id)
    if df_raw is None or df_raw.isnull().values.any():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has null values")

    # Download PPO model from MinIO to a temp file
    client = _minio_client()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        client.download_file(MINIO_BUCKET, agent_model.storage_path, tmp_path)
        model_load_path = tmp_path[:-4]  # SB3 save/load strips .zip
        os.rename(tmp_path, model_load_path + ".zip")
        model, scalers, obs_rms = load_model_and_scalers(
            model_path=model_load_path,
            scalers_path=SCALERS_PATH,
            obs_rms_path=OBS_RMS_PATH,
            model_cls=PPO,
        )
    finally:
        for p in (tmp_path, model_load_path + ".zip"):
            if os.path.exists(p):
                os.unlink(p)

    site_config = SiteConfig(**raw_config.settings)
    system_config = site_config_to_env_dict(site_config)

    result = run_inference(
        df_raw=df_raw,
        system_config=system_config,
        model=model,
        scalers=scalers,
        initial_soc=0.5,
        obs_rms=obs_rms,
    )

    # Persist predictions for tomorrow, replacing any existing ones
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    db.query(AgentPredictions).filter(
        AgentPredictions.user_id == user.id,
        AgentPredictions.date == tomorrow,
    ).delete()

    for step_data in result['dispatch_plan']:
        row = df_raw.iloc[step_data['step']]
        db.add(AgentPredictions(
            user_id=user.id,
            date=tomorrow,
            step=step_data['step'],
            timestamp=datetime.now(),
            battery_action=step_data['action_battery'],
            grid_action=step_data['action_grid'],
            load_kwh=float(row['Load']) / 1000 / 4,
            solar_kwh=step_data['solar_gen_kwh'],
            grid_kwh=step_data['grid_kwh'],
            unmet_load_kwh=step_data['unmet_load_kwh'],
            soc=step_data['soc'],
            target_soc=step_data['target_soc'],
            lcos_cost=step_data['lcos_cost'],
            dam_price=float(row['DAM_Price']) / 1000,
            grid_status=int(row['Grid']),
            hours_until_outage=float(row['hours_until_outage']),
            outage_remaining_h=float(row['outage_remaining_h']),
            next_outage_duration=float(row['next_outage_duration']),
            reward_total=step_data['reward'],
        ))
    db.commit()

    return result

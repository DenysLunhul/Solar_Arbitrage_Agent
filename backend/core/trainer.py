import os
import tempfile
from datetime import datetime

import boto3
import pandas as pd
from stable_baselines3 import PPO

from backend.core.database import SessionLocal
from backend.models.site import AgentModels, SystemConfig
from backend.schemas.schemas import SiteConfig
from envoriment.environment import Environment
from envoriment.normalize import normalize_dataset

DATASET_PATH            = "datasets/dataset_v10/dataset_final.csv"
DATASET_NORMALIZED_PATH = "envoriment/dataset_normalized.csv"
SCALERS_PATH            = "envoriment/models/scalers.pkl"
TOTAL_TIMESTEPS         = 200_000
MINIO_BUCKET            = "models"


def _minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
    )


def _ensure_bucket(client):
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except Exception:
        client.create_bucket(Bucket=MINIO_BUCKET)


def site_config_to_env_dict(site_config: SiteConfig) -> dict:
    """Maps SiteConfig Pydantic model → system_config dict expected by Environment."""
    return {
        'battery': {
            'capacity_kwh':        site_config.battery.battery_capacity_kwh,
            'min_reserve':         site_config.battery.battery_min_reserve,
            'lcos':                site_config.battery.battery_lcos,
            'max_charge_power':    site_config.battery.battery_max_charge_power,
            'max_discharge_power': site_config.battery.battery_max_discharge_power,
            'efficiency':          site_config.battery.battery_efficiency,
        },
        'solar': {
            'peak_power': site_config.solar.solar_peak_power,
            'efficiency': site_config.solar.solar_efficiency,
        },
        'inverter': {
            'max_power':    site_config.inverter.max_power,
            'price_to_buy': site_config.grid.price_buy_from_grid,
        },
    }


def train_model_for_config(config_id: int):
    db = SessionLocal()
    record = None
    try:
        record = db.query(AgentModels).filter(AgentModels.config_id == config_id).first()
        if record is None:
            record = AgentModels(config_id=config_id, status="training", trained_at=datetime.now())
            db.add(record)
        else:
            record.status = "training"
        db.commit()

        config_row = db.query(SystemConfig).filter(SystemConfig.id == config_id).first()
        site_config = SiteConfig(**config_row.settings)
        system_config = site_config_to_env_dict(site_config)

        df_raw = pd.read_csv(DATASET_PATH)

        # Use pre-built normalized dataset; rebuild if missing
        if not os.path.exists(DATASET_NORMALIZED_PATH) or not os.path.exists(SCALERS_PATH):
            normalize_dataset(DATASET_PATH, DATASET_NORMALIZED_PATH, SCALERS_PATH)
        df_norm = pd.read_csv(DATASET_NORMALIZED_PATH)

        env = Environment(df_raw=df_raw, df=df_norm, system_config=system_config)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

        model.save(tmp_path.replace(".zip", ""))

        client = _minio_client()
        _ensure_bucket(client)
        storage_key = f"{config_id}.zip"
        client.upload_file(tmp_path, MINIO_BUCKET, storage_key)
        os.unlink(tmp_path)

        record.status = "ready"
        record.trained_at = datetime.now()
        record.total_timesteps = TOTAL_TIMESTEPS
        record.storage_path = storage_key
        db.commit()

    except Exception:
        if record is not None:
            record.status = "failed"
            db.commit()
        raise
    finally:
        db.close()

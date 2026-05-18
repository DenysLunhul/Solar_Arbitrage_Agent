import os
from datetime import datetime

import pandas as pd
from stable_baselines3 import PPO

from backend.core.database import SessionLocal
from backend.models.site import AgentModels, SystemConfig
from backend.schemas.schemas import SiteConfig
from environment.environment import Environment
from environment.normalize import normalize_dataset

DATASET_PATH            = "datasets/dataset_v10/dataset_final.csv"
DATASET_NORMALIZED_PATH = "envoriment/dataset_normalized.csv"
SCALERS_PATH            = "envoriment/models/scalers.pkl"
MODELS_DIR              = "backend/trained_models"
TOTAL_TIMESTEPS         = 200_000


def train_model_for_config(config_id: int):
    db = SessionLocal()
    record = None
    try:
        record = db.query(AgentModels).filter(AgentModels.config_id == config_id).first()
        if record is None:
            record = AgentModels(config_id=config_id, status="training", algorithm="PPO", trained_at=datetime.now())
            db.add(record)
        else:
            record.status = "training"
        db.commit()

        config_row = db.query(SystemConfig).filter(SystemConfig.id == config_id).first()
        system_config = SiteConfig(**config_row.settings).to_env_dict()

        df_raw = pd.read_csv(DATASET_PATH)

        if not os.path.exists(DATASET_NORMALIZED_PATH) or not os.path.exists(SCALERS_PATH):
            normalize_dataset(DATASET_PATH, DATASET_NORMALIZED_PATH, SCALERS_PATH)
        df_norm = pd.read_csv(DATASET_NORMALIZED_PATH)

        env = Environment(df_raw=df_raw, df=df_norm, system_config=system_config)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, str(config_id))
        model.save(model_path)

        record.status = "ready"
        record.trained_at = datetime.now()
        record.total_timesteps = TOTAL_TIMESTEPS
        record.storage_path = model_path + ".zip"
        db.commit()

    except Exception:
        if record is not None:
            record.status = "failed"
            db.commit()
        raise
    finally:
        db.close()

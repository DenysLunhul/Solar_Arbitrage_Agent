from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from backend.models.site import SystemConfig
from backend.repositories import config_repo
from backend.schemas.schemas import SiteConfig


def save_config(db: Session, config_name: str, settings: SiteConfig, user_id: int) -> SystemConfig:
    return config_repo.upsert(db, config_name, settings.model_dump(), user_id)


def get_config(db: Session, config_name: str, user_id: int) -> SystemConfig:
    config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    return config


def list_configs(db: Session, user_id: int) -> list[SystemConfig]:
    return config_repo.get_all_by_user(db, user_id)

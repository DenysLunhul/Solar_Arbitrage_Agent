from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.schemas.schemas import SiteConfig, SystemConfigResponse

from backend.security.security import get_current_user
from backend.core.database import save_to_db
from backend.models.site import SystemConfig
from backend.core.trainer import train_model_for_config

router = APIRouter(prefix="/config", tags=["config"])

@router.post("/", status_code=status.HTTP_201_CREATED)
def save_config(
    config_name: str,
    settings: SiteConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> SystemConfigResponse:
    config_in_db = SystemConfig(config_name=config_name, settings=settings.model_dump(), user_id=user.id)
    save_to_db(db, config_in_db)
    background_tasks.add_task(train_model_for_config, config_in_db.id)
    return config_in_db

@router.get("/")
def get_config(config_name: str, db: Session = Depends(get_db), user = Depends(get_current_user)) -> SystemConfigResponse:
    config = db.query(SystemConfig).filter(SystemConfig.config_name == config_name).first()
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wrong config name")
    if user.id != config.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your config")
    return config


@router.get("/list")
def get_all_configs(db: Session = Depends(get_db), user = Depends(get_current_user)) -> list[SystemConfigResponse]:
    configs = db.query(SystemConfig).filter(SystemConfig.user_id == user.id).all()
    return configs
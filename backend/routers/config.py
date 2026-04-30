from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.schemas.schemas import SiteConfig

from backend.security.security import get_current_user
from models.site import SystemConfig, User

router = APIRouter(prefix="/config", tags=["config"])

@router.post("/")
def save_config(config_name: str, settings: SiteConfig, db: Session = Depends(get_db), user = Depends(get_current_user)) -> SystemConfig:
    config_in_db = SystemConfig(config_name=config_name, settings=settings.model_dump())
    db.add(config_in_db)
    db.commit()
    db.refresh(config_in_db)
    return config_in_db

@router.get("/")
def get_config(config_name: str, db: Session = Depends(get_db), user = Depends(get_current_user)) -> SystemConfig:
    config = db.query(SystemConfig).filter(SystemConfig.config_name == config_name).first()
    if config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wrong config name")
    if user.id != config.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your congif")
    return config
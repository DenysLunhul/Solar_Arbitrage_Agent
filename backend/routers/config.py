from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.schemas.schemas import SiteConfig, SystemConfigResponse
from backend.security.security import get_current_user
from backend.services import config_service
router = APIRouter(prefix="/config", tags=["config"])
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=SystemConfigResponse)
def save_config(
    config_name: str,
    settings: SiteConfig,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return config_service.save_config(db, config_name, settings, user.id)
@router.get("/", response_model=SystemConfigResponse)
def get_config(config_name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return config_service.get_config(db, config_name, user.id)
@router.get("/list", response_model=list[SystemConfigResponse])
def list_configs(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return config_service.list_configs(db, user.id)

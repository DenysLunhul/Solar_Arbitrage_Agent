from fastapi import FastAPI, APIRouter, Depends
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.models.site import SystemConfig
from backend.schemas.schemas import SiteConfig



router = APIRouter(prefix="/config", tags=["config"])

#TODO
@router.post("/")
def save_config(item: SiteConfig, db: Session = Depends(get_db())):
    return
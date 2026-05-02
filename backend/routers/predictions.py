from fastapi import APIRouter, Depends, HTTPException, status

from backend.models.site import User, SystemConfig
from backend.security.security import get_current_user, get_db
from backend.routers.config import get_config
from backend.schemas.schemas import SiteConfig
from envoriment.envoriment import Environment

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def get_predictions(config_name: str, db = Depends(get_db), user = Depends(get_current_user)):
    raw_config = db.query(SystemConfig).filter(SystemConfig.config_name == config_name).first()
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wrong config name")
    site_config = SiteConfig(**raw_config.settings)
    #TODO
    #Fetch the df, and format for env initialization
    # env = Environment()
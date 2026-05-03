import pandas as pd
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from backend.models.site import SystemConfig
from backend.security.security import get_current_user, get_db
from backend.schemas.schemas import SiteConfig
from envoriment.envoriment import Environment
from data_providers.orchestrator.data_combiner import combine

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def get_predictions(config_name: str, db = Depends(get_db), user = Depends(get_current_user)):
    if datetime.now().hour < 14:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="DAM data for tomorrow is not yet available. Please call this endpoint after 18:00."
        )

    raw_config = db.query(SystemConfig).filter(SystemConfig.config_name == config_name).first()
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wrong config name")
    site_config = SiteConfig(**raw_config.settings)
    df: pd.DataFrame = combine(raw_config.id)
    if df is None or df.isnull().values.any():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has null values")
from datetime import date
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.schemas.schemas import PredictionResponse
from backend.security.security import get_current_user
from backend.services import prediction_service
router = APIRouter(prefix="/predictions", tags=["predictions"])
@router.get("/", response_model=PredictionResponse)
def get_predictions(
    config_name: str,
    initial_soc: float | None = Query(default=None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return prediction_service.get_predictions(db, config_name, user.id, initial_soc)
@router.get("/default", response_model=PredictionResponse)
def get_default_predictions(
    config_name: str,
    strategy_name: str,
    initial_soc: float | None = Query(default=None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return prediction_service.get_default_predictions(db, config_name, user.id, strategy_name, initial_soc)
@router.get("/history/dates", response_model=list[str])
def get_history_dates(
    config_name: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return prediction_service.get_history_dates(db, config_name, user.id)
@router.get("/history", response_model=PredictionResponse)
def get_history(
    config_name: str,
    date: date | None = Query(default=None),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return prediction_service.get_history(db, config_name, user.id, date)
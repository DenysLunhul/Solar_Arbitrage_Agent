from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.security.security import get_current_user
from backend.services import prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def get_predictions(
    config_name: str,
    initial_soc: float | None = Query(default=None, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return prediction_service.get_predictions(db, config_name, user.id, initial_soc)
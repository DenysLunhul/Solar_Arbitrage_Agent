from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.security.security import get_current_user
from backend.services import prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/")
def get_predictions(config_name: str, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return prediction_service.get_predictions(db, config_name, user.id)
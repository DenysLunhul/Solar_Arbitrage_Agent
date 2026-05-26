from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from backend.core.database import get_db
from backend.schemas.schemas import DefaultStrategyConfig, DefaultStrategyResponse
from backend.security.security import get_current_user
from backend.services import strategy_service
router = APIRouter(prefix="/strategy", tags=["strategy"])
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=DefaultStrategyResponse)
def save_strategy(
    payload: DefaultStrategyConfig,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return strategy_service.save_strategy(db, payload, user.id)
@router.get("/", response_model=DefaultStrategyResponse)
def get_strategy(
    strategy_name: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return strategy_service.get_strategy(db, strategy_name, user.id)
@router.get("/list", response_model=list[DefaultStrategyResponse])
def list_strategies(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return strategy_service.list_strategies(db, user.id)

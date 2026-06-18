from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from backend.repositories import strategy_repo
from backend.schemas.schemas import DefaultStrategyConfig


def save_strategy(db: Session, payload: DefaultStrategyConfig, user_id: int):
    return strategy_repo.upsert(db, user_id, payload.strategy_name, payload.to_strategy_dict())


def get_strategy(db: Session, strategy_name: str, user_id: int):
    row = strategy_repo.get(db, user_id, strategy_name)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return row


def list_strategies(db: Session, user_id: int):
    return strategy_repo.list_all(db, user_id)

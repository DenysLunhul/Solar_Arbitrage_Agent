from sqlalchemy.orm import Session
from backend.models.site import DefaultStrategy


def upsert(db: Session, user_id: int, strategy_name: str, settings: dict) -> DefaultStrategy:
    row = (
        db.query(DefaultStrategy)
        .filter(DefaultStrategy.user_id == user_id, DefaultStrategy.strategy_name == strategy_name)
        .first()
    )
    if row:
        row.settings = settings
    else:
        row = DefaultStrategy(user_id=user_id, strategy_name=strategy_name, settings=settings)
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get(db: Session, user_id: int, strategy_name: str) -> DefaultStrategy | None:
    return (
        db.query(DefaultStrategy)
        .filter(DefaultStrategy.user_id == user_id, DefaultStrategy.strategy_name == strategy_name)
        .first()
    )


def list_all(db: Session, user_id: int) -> list[DefaultStrategy]:
    return db.query(DefaultStrategy).filter(DefaultStrategy.user_id == user_id).all()

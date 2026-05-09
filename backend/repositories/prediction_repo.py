from datetime import date

from sqlalchemy.orm import Session

from backend.models.site import AgentPredictions


def delete_for_date(db: Session, config_id: int, target_date: date) -> None:
    db.query(AgentPredictions).filter(
        AgentPredictions.config_id == config_id,
        AgentPredictions.date == target_date,
    ).delete()


def bulk_create(db: Session, rows: list[AgentPredictions]) -> None:
    db.add_all(rows)
    db.commit()
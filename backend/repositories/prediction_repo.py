from datetime import date
from sqlalchemy.orm import Session
from backend.models.site import AgentPredictions


def get_last_soc(db: Session, config_id: int) -> float | None:
    row = (
        db.query(AgentPredictions.soc)
        .filter(AgentPredictions.config_id == config_id)
        .order_by(AgentPredictions.date.desc(), AgentPredictions.step.desc())
        .first()
    )
    return row.soc if row else None


def get_latest_soc(db: Session) -> float | None:
    row = (
        db.query(AgentPredictions.soc)
        .order_by(AgentPredictions.date.desc(), AgentPredictions.step.desc())
        .first()
    )
    return row.soc if row else None


def delete_for_date(db: Session, config_id: int, target_date: date) -> None:
    db.query(AgentPredictions).filter(
        AgentPredictions.config_id == config_id,
        AgentPredictions.date == target_date,
    ).delete()


def bulk_create(db: Session, rows: list[AgentPredictions]) -> None:
    db.add_all(rows)
    db.commit()


def get_by_config_and_date(db: Session, config_id: int, target_date: date) -> list[AgentPredictions]:
    return (
        db.query(AgentPredictions)
        .filter(AgentPredictions.config_id == config_id, AgentPredictions.date == target_date)
        .order_by(AgentPredictions.step)
        .all()
    )


def get_available_dates(db: Session, config_id: int) -> list[date]:
    rows = (
        db.query(AgentPredictions.date)
        .filter(AgentPredictions.config_id == config_id)
        .distinct()
        .order_by(AgentPredictions.date.desc())
        .all()
    )
    return [r.date for r in rows]


def get_latest_date(db: Session, config_id: int) -> date | None:
    row = (
        db.query(AgentPredictions.date)
        .filter(AgentPredictions.config_id == config_id)
        .order_by(AgentPredictions.date.desc())
        .first()
    )
    return row.date if row else None

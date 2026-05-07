from sqlalchemy.orm import Session

from backend.models.site import AgentModels


def get_ready_for_config(db: Session, config_id: int) -> AgentModels | None:
    return db.query(AgentModels).filter(
        AgentModels.config_id == config_id,
        AgentModels.status == "ready",
    ).first()

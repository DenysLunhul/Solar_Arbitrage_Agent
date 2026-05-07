from sqlalchemy.orm import Session

from backend.models.site import SystemConfig


def get_by_name_and_user(db: Session, config_name: str, user_id: int) -> SystemConfig | None:
    return db.query(SystemConfig).filter(
        SystemConfig.config_name == config_name,
        SystemConfig.user_id == user_id,
    ).first()


def get_all_by_user(db: Session, user_id: int) -> list[SystemConfig]:
    return db.query(SystemConfig).filter(SystemConfig.user_id == user_id).all()


def create(db: Session, config_name: str, settings: dict, user_id: int) -> SystemConfig:
    config = SystemConfig(config_name=config_name, settings=settings, user_id=user_id)
    db.add(config)
    db.commit()
    db.refresh(config)
    return config

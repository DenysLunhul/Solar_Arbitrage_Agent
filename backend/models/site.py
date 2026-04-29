from sqlalchemy import PrimaryKeyConstraint, Column, Integer, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB

from backend.core.database import Base



class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    data = Column(JSONB)


class SystemConfig(Base):
    __tablename__ = "system_configs"
    id = Column(Integer, primary_key=True, index=True)
    config_name = Column(String)
    settings = Column(JSONB)
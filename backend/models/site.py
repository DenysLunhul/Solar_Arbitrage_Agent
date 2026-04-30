from sqlalchemy import PrimaryKeyConstraint, Column, Integer, DateTime, String, ForeignKey
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
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    config_name = Column(String)
    settings = Column(JSONB)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
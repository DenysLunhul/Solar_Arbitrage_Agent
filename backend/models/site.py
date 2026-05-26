from sqlalchemy import Column, Integer, DateTime, String, ForeignKey, Float, Date
from sqlalchemy.dialects.postgresql import JSONB
from backend.core.database import Base


class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True)
    email           = Column(String, unique=True, index=True)
    hashed_password = Column(String)


class SystemConfig(Base):
    __tablename__ = "system_configs"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), index=True)
    config_name = Column(String)
    settings    = Column(JSONB)


class DefaultStrategy(Base):
    __tablename__ = "default_strategies"
    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), index=True)
    strategy_name = Column(String)
    settings      = Column(JSONB)


class History(Base):
    __tablename__ = "history"
    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True)
    timestamp = Column(DateTime, index=True)
    data      = Column(JSONB)


class AgentPredictions(Base):
    __tablename__ = "predictions"
    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True)
    config_id = Column(Integer, ForeignKey("system_configs.id"), index=True)
    date      = Column(Date, index=True)
    step      = Column(Integer)
    timestamp = Column(DateTime, index=True)
    battery_action = Column(Float)
    grid_action    = Column(Float)

    load_kwh          = Column(Float)
    solar_kwh         = Column(Float)
    solar_surplus_kwh = Column(Float)
    battery_kwh       = Column(Float)
    grid_kwh          = Column(Float)
    unmet_load_kwh    = Column(Float)

    solar_to_load_kwh    = Column(Float)
    solar_to_battery_kwh = Column(Float)
    solar_to_grid_kwh    = Column(Float)
    battery_to_load_kwh  = Column(Float)
    grid_to_load_kwh     = Column(Float)
    grid_to_battery_kwh  = Column(Float)

    soc        = Column(Float)
    target_soc = Column(Float)
    lcos_cost  = Column(Float)
    dam_price  = Column(Float)

    grid_status          = Column(Integer)
    hours_until_outage   = Column(Float)
    outage_remaining_h   = Column(Float)
    next_outage_duration = Column(Float)

    mismatch        = Column(Float)
    money_earned_ts = Column(Float)

    reward_market      = Column(Float)
    reward_lcos        = Column(Float)
    reward_unmet       = Column(Float)
    reward_mismatch    = Column(Float)
    reward_soc_soft    = Column(Float)
    reward_reserve     = Column(Float)
    reward_preparation = Column(Float)
    reward_soc_target  = Column(Float)
    reward_waste          = Column(Float)
    reward_curtail        = Column(Float)
    reward_price_timing   = Column(Float)
    reward_solar_priority = Column(Float)
    reward_eod_soc        = Column(Float)
    reward_total          = Column(Float)

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


class History(Base):
    __tablename__ = "history"
    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True)
    timestamp = Column(DateTime, index=True)
    data      = Column(JSONB)


class AgentPredictions(Base):
    __tablename__ = "predictions"

    # ── Identity ──────────────────────────────────────────────────
    id        = Column(Integer, primary_key=True, index=True)
    user_id   = Column(Integer, ForeignKey("users.id"), index=True)
    date      = Column(Date, index=True)
    step      = Column(Integer)       # 0–95 (15-min timestep index)
    timestamp = Column(DateTime, index=True)

    # ── Raw agent actions ─────────────────────────────────────────
    battery_action = Column(Float)    # action[0] in [-1, 1]
    grid_action    = Column(Float)    # action[1] in [-1, 1]

    # ── Energy flows (kWh per 15-min timestep) ────────────────────
    load_kwh          = Column(Float)  # enterprise consumption
    solar_kwh         = Column(Float)  # solar generation
    solar_surplus_kwh = Column(Float)  # excess solar (exported or wasted)
    battery_kwh       = Column(Float)  # actual kWh moved (+ charge, - discharge)
    grid_kwh          = Column(Float)  # net grid exchange (+ buy, - sell)
    unmet_load_kwh    = Column(Float)  # load not covered (blackout indicator)

    # ── Load coverage breakdown (for stacked charts / Sankey) ─────
    solar_to_load_kwh    = Column(Float)  # solar directly covering load
    solar_to_battery_kwh = Column(Float)  # solar going into battery
    solar_to_grid_kwh    = Column(Float)  # solar exported to grid
    battery_to_load_kwh  = Column(Float)  # battery discharge covering load
    grid_to_load_kwh     = Column(Float)  # grid import covering load
    grid_to_battery_kwh  = Column(Float)  # grid import charging battery

    # ── Battery state ─────────────────────────────────────────────
    soc        = Column(Float)         # state of charge [0, 1]
    target_soc = Column(Float)         # dynamic reserve target computed by agent
    lcos_cost  = Column(Float)         # battery degradation cost this step (UAH)

    # ── Market ────────────────────────────────────────────────────
    dam_price = Column(Float)          # UAH/kWh

    # ── Grid / outage ─────────────────────────────────────────────
    grid_status          = Column(Integer)  # 1 = on, 0 = outage
    hours_until_outage   = Column(Float)
    outage_remaining_h   = Column(Float)    # hours left in current outage
    next_outage_duration = Column(Float)    # predicted duration of next outage

    # ── Agent quality ─────────────────────────────────────────────
    mismatch = Column(Float)               # gap between requested and actual grid action

    # ── Reward breakdown ──────────────────────────────────────────
    reward_market      = Column(Float)  # P&L from grid trading
    reward_lcos        = Column(Float)  # degradation penalty
    reward_unmet       = Column(Float)  # unmet load penalty
    reward_soc_soft    = Column(Float)  # soft SoC boundary penalty
    reward_reserve     = Column(Float)  # outage reserve penalty
    reward_preparation = Column(Float)  # pre-outage preparation bonus
    reward_total       = Column(Float)  # sum of all reward components


class AgentModels(Base):
    __tablename__ = "agent_models"

    id               = Column(Integer, primary_key=True, index=True)
    config_id        = Column(Integer, ForeignKey("system_configs.id"), index=True)
    status           = Column(String)   # "training" | "ready" | "failed"
    trained_at       = Column(DateTime)
    total_timesteps  = Column(Integer)
    storage_path     = Column(String)   # MinIO key: {config_id}.zip
    mean_reward      = Column(Float, nullable=True)
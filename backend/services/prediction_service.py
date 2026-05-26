import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from stable_baselines3 import SAC
from backend.models.site import AgentPredictions
from backend.repositories import config_repo, prediction_repo, strategy_repo
from backend.schemas.schemas import SiteConfig
from data_providers.orchestrator.data_combiner import combine
from environment.inference import load_model_and_scalers, run_inference
from environment.default_strategy import generate_dispatch_plan
from environment.normalize import normalize_row

_BASE_DIR    = Path(__file__).resolve().parent.parent.parent
MODEL_PATH   = str(_BASE_DIR / "environment" / "models" / "best" / "best_model")
SCALERS_PATH = str(_BASE_DIR / "environment" / "models" / "scalers.pkl")
OBS_RMS_PATH = str(_BASE_DIR / "environment" / "models" / "obs_rms.pkl")
UA_TZ = timezone(timedelta(hours=2))
_cached_model: tuple | None = None


def _check_time_gate() -> None:
    now = datetime.now(UA_TZ)
    if now.hour < 14:
        available_at = now.replace(hour=14, minute=0, second=0, microsecond=0)
        retry_in = int((available_at - now).total_seconds() / 60)
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail={
                "message": "DAM prices for tomorrow are not yet published.",
                "available_after": "14:00 UTC+2",
                "retry_after_minutes": retry_in,
            },
        )


def _get_model() -> tuple:
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model_and_scalers(
            model_path=MODEL_PATH,
            scalers_path=SCALERS_PATH,
            obs_rms_path=OBS_RMS_PATH,
            model_cls=SAC,
        )
    return _cached_model


def get_predictions(db: Session, config_name: str, user_id: int, initial_soc: float | None) -> dict:
    _check_time_gate()
    raw_config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    if initial_soc is None:
        persisted_soc = prediction_repo.get_last_soc(db, raw_config.id)
        if persisted_soc is not None:
            system_config = SiteConfig(**raw_config.settings).to_env_dict()
            min_reserve = system_config['battery']['min_reserve'] / 100
            initial_soc = max(persisted_soc, min_reserve)
        else:
            initial_soc = 0.5
    df_raw = combine(raw_config.id)
    if df_raw is None or df_raw.isnull().values.any():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has null values")
    model, scalers, obs_rms = _get_model()
    result = run_inference(
        df_raw=df_raw,
        system_config=SiteConfig(**raw_config.settings).to_env_dict(),
        model=model,
        scalers=scalers,
        initial_soc=initial_soc,
        obs_rms=obs_rms,
    )
    tomorrow = (datetime.now(UA_TZ) + timedelta(days=1)).date()
    prediction_repo.delete_for_date(db, raw_config.id, tomorrow)
    prediction_repo.bulk_create(
        db, _build_rows(result["dispatch_plan"], df_raw, user_id, raw_config.id, tomorrow)
    )
    return _build_response(result, df_raw)


def _compute_flows(solar: float, battery: float, grid: float, load: float, unmet: float) -> dict:
    effective_load  = max(0.0, load - unmet)
    batt_charge     = max(0.0, battery)
    batt_discharge  = max(0.0, -battery)
    grid_import     = max(0.0, grid)
    grid_export     = max(0.0, -grid)
    solar_to_load    = min(solar, effective_load)
    solar_remaining  = solar - solar_to_load
    solar_to_battery = min(solar_remaining, batt_charge)
    solar_exportable = solar_remaining - solar_to_battery
    remaining_load  = max(0.0, effective_load - solar_to_load)
    battery_to_load = min(batt_discharge, remaining_load)
    batt_for_grid   = batt_discharge - battery_to_load
    solar_to_grid   = min(solar_exportable, grid_export)
    battery_to_grid = min(batt_for_grid, grid_export - solar_to_grid)
    grid_to_battery = min(max(0.0, batt_charge - solar_to_battery), grid_import)
    grid_to_load    = max(0.0, grid_import - grid_to_battery)
    return {
        "solar_to_load_kwh":    round(solar_to_load, 4),
        "solar_to_battery_kwh": round(solar_to_battery, 4),
        "solar_to_grid_kwh":    round(solar_to_grid, 4),
        "battery_to_load_kwh":  round(battery_to_load, 4),
        "battery_to_grid_kwh":  round(battery_to_grid, 4),
        "grid_to_load_kwh":     round(grid_to_load, 4),
        "grid_to_battery_kwh":  round(grid_to_battery, 4),
    }


def _build_response(result: dict, df_raw: pd.DataFrame) -> dict:
    steps = []
    for s in result["dispatch_plan"]:
        row = df_raw.iloc[s["step"]]
        load_kwh = round(float(row["Load"]) / 4, 4)
        steps.append({
            "timestamp":          str(row["timestamp"]),
            "soc":                s["soc"],
            "target_soc":         s["target_soc"],
            "solar_kwh":          s["solar_gen_kwh"],
            "load_kwh":           load_kwh,
            "battery_kwh":        s["battery_kwh"],
            "grid_kwh":           s["grid_kwh"],
            "unmet_load_kwh":     s["unmet_load_kwh"],
            "money_earned_ts":    s["money_earned_ts"],
            "dam_price":          round(float(row["DAM_Price"]) / 1000, 4),
            "grid_status":        int(row["Grid"]),
            "hours_until_outage": round(float(row["hours_until_outage"]), 2),
            **_compute_flows(s["solar_gen_kwh"], s["battery_kwh"], s["grid_kwh"], load_kwh, s["unmet_load_kwh"]),
        })
    raw = result["summary"]
    summary = {
        "total_money_earned": raw["total_money_earned"],
        "bought_kwh":         raw["bought_kwh"],
        "sold_kwh":           raw["sold_kwh"],
        "solar_kwh":          raw["solar_kwh"],
        "unmet_load_kwh":     raw["unmet_load_kwh"],
        "lcos_total_uah":     raw["lcos_total_uah"],
        "initial_soc":        raw["initial_soc"],
        "final_soc":          raw["final_soc"],
        "steps":              raw["steps"],
    }
    return {
        "summary": summary,
        "dispatch_plan": steps,
    }


def get_default_predictions(db, config_name: str, user_id: int, strategy_name: str, initial_soc: float | None) -> dict:
    _check_time_gate()
    raw_config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    strategy_row = strategy_repo.get(db, user_id, strategy_name)
    if strategy_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    if initial_soc is None:
        persisted_soc = prediction_repo.get_last_soc(db, raw_config.id)
        if persisted_soc is not None:
            system_config = SiteConfig(**raw_config.settings).to_env_dict()
            min_reserve = system_config['battery']['min_reserve'] / 100
            initial_soc = max(persisted_soc, min_reserve)
        else:
            initial_soc = 0.5
    df_raw = combine(raw_config.id)
    if df_raw is None or df_raw.isnull().values.any():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has null values")
    with open(SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    df_norm = pd.DataFrame([normalize_row(df_raw.iloc[i], scalers) for i in range(len(df_raw))])
    system_config = SiteConfig(**raw_config.settings).to_env_dict()
    result = generate_dispatch_plan(
        df_raw=df_raw,
        df_norm=df_norm,
        config=system_config,
        initial_soc=initial_soc,
        strategy=strategy_row.settings,
    )
    return _build_response(result, df_raw)


def get_history(
    db: Session, config_name: str, user_id: int, target_date: date | None
) -> dict:
    raw_config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    if target_date is None:
        target_date = prediction_repo.get_latest_date(db, raw_config.id)
        if target_date is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No predictions stored for this config")
    rows = prediction_repo.get_by_config_and_date(db, raw_config.id, target_date)
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No predictions for {target_date}")
    steps = [
        {
            "timestamp":          str(r.timestamp),
            "soc":                r.soc,
            "target_soc":         r.target_soc,
            "solar_kwh":          r.solar_kwh,
            "load_kwh":           r.load_kwh,
            "battery_kwh":        r.battery_kwh,
            "grid_kwh":           r.grid_kwh,
            "unmet_load_kwh":     r.unmet_load_kwh,
            "money_earned_ts":    r.money_earned_ts,
            "dam_price":          r.dam_price,
            "grid_status":        r.grid_status,
            "hours_until_outage": r.hours_until_outage,
            **_compute_flows(r.solar_kwh or 0.0, r.battery_kwh or 0.0, r.grid_kwh or 0.0, r.load_kwh or 0.0, r.unmet_load_kwh or 0.0),
        }
        for r in rows
    ]
    summary = {
        "total_money_earned": round(sum(r.money_earned_ts or 0 for r in rows), 2),
        "bought_kwh":         round(sum(r.grid_kwh for r in rows if (r.grid_kwh or 0) > 0), 3),
        "sold_kwh":           round(sum(abs(r.grid_kwh) for r in rows if (r.grid_kwh or 0) < 0), 3),
        "solar_kwh":          round(sum(r.solar_kwh or 0 for r in rows), 3),
        "unmet_load_kwh":     round(sum(r.unmet_load_kwh or 0 for r in rows), 4),
        "lcos_total_uah":     round(sum(r.lcos_cost or 0 for r in rows), 3),
        "initial_soc":        rows[0].soc,
        "final_soc":          rows[-1].soc,
        "steps":              len(rows),
    }
    return {"summary": summary, "dispatch_plan": steps}


def get_history_dates(db: Session, config_name: str, user_id: int) -> list[str]:
    raw_config = config_repo.get_by_name_and_user(db, config_name, user_id)
    if raw_config is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Config not found")
    return [str(d) for d in prediction_repo.get_available_dates(db, raw_config.id)]


def _build_rows(
    dispatch_plan: list, df_raw: pd.DataFrame, user_id: int, config_id: int, target_date
) -> list[AgentPredictions]:
    rows = []
    for step_data in dispatch_plan:
        row = df_raw.iloc[step_data["step"]]
        rows.append(AgentPredictions(
            user_id=user_id,
            config_id=config_id,
            date=target_date,
            step=step_data["step"],
            timestamp=pd.to_datetime(row["timestamp"]),
            battery_action=step_data["action_battery"],
            grid_action=step_data["action_grid"],
            load_kwh=float(row["Load"]) / 4,
            solar_kwh=step_data["solar_gen_kwh"],
            solar_surplus_kwh=step_data["solar_surplus_kwh"],
            battery_kwh=step_data["battery_kwh"],
            grid_kwh=step_data["grid_kwh"],
            unmet_load_kwh=step_data["unmet_load_kwh"],
            soc=step_data["soc"],
            target_soc=step_data["target_soc"],
            lcos_cost=step_data["lcos_cost"],
            mismatch=step_data["mismatch"],
            money_earned_ts=step_data["money_earned_ts"],
            dam_price=float(row["DAM_Price"]) / 1000,
            grid_status=int(row["Grid"]),
            hours_until_outage=float(row["hours_until_outage"]),
            outage_remaining_h=float(row["outage_remaining_h"]),
            next_outage_duration=float(row["next_outage_duration"]),
            reward_market=step_data["reward_market"],
            reward_lcos=step_data["reward_lcos"],
            reward_unmet=step_data["reward_unmet"],
            reward_mismatch=step_data["reward_mismatch"],
            reward_soc_soft=step_data["reward_soc_soft"],
            reward_reserve=step_data["reward_reserve"],
            reward_preparation=step_data["reward_preparation"],
            reward_soc_target=step_data["reward_soc_target"],
            reward_waste=step_data["reward_waste"],
            reward_curtail=step_data["reward_curtail"],
            reward_price_timing=step_data["reward_price_timing"],
            reward_solar_priority=step_data["reward_solar_priority"],
            reward_eod_soc=step_data["reward_eod_soc"],
            reward_total=step_data["reward"],
        ))
    return rows

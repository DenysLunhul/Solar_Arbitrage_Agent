from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class Battery(BaseModel):
    battery_capacity_kwh: float
    battery_min_reserve: float | None = Field(default=10)
    battery_lcos: float
    battery_max_charge_power: float
    battery_max_discharge_power: float
    battery_efficiency: float = Field(default=1)


class Inverter(BaseModel):
    max_power: float
    efficiency: float

class SolarPanel(BaseModel):
    solar_peak_power: float
    solar_efficiency: float
    solar_azimuth: float | None = Field(default=0)
    solar_tilt: float | None = Field(default=35)

class Grid(BaseModel):
    grid_capacity: float
    price_buy_from_grid: float


class SiteConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    battery: Battery
    inverter: Inverter
    solar: SolarPanel
    grid: Grid


class SystemConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    config_name: str
    settings: dict


class BaseUser(BaseModel):
    username: str
    email: str

class UserCreate(BaseUser):
    password: str

class UserResponse(BaseUser):
    model_config = ConfigDict(from_attributes=True)
    id: int



# class AgentDataIn(BaseModel):
#     model_config = ConfigDict(from_attributes=True)
#     timestamp: datetime
#     Month: int
#     Day_of_week: int
#     Day: int
#     Hour: int
#     Minute: int
#     Day_of_week_sin: float
#     Day_of_week_cos: float
#     Day_sin: float
#     Day_cos: float
#     Hour_sin: float
#     Hour_cos: float
#     Minute_sin: float
#     Minute_cos: float
#
#     Grid: int
#     next_outage_duration: float
#     outage_remaining_h: float
#     hours_until_outage: float
#     Load: float
#
#     Temperature_2m: float
#     Shortwave_radiation: float
#     Global_tilted_irradiance_instant: float
#
#     DAM_Price: float
#
#
#
# class AgentDataOutput(AgentDataIn):
#     model_config = ConfigDict(from_attributes=True)
#     battery_current_charge: float
#     grid_sell_energy: float
#
#     grid_to_enterprise: float
#
#     grid_to_battery: float
#     grid_from_battery: float
#
#     grid_from_solar_panels: float
#
#     reward: float
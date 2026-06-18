from typing import Literal
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


class SiteConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    battery: Battery
    inverter: Inverter
    solar: SolarPanel
    grid: Grid

    def to_env_dict(self) -> dict:
        return {
            'battery': {
                'capacity_kwh':        self.battery.battery_capacity_kwh,
                'min_reserve':         self.battery.battery_min_reserve,
                'lcos':                self.battery.battery_lcos,
                'max_charge_power':    self.battery.battery_max_charge_power,
                'max_discharge_power': self.battery.battery_max_discharge_power,
                'efficiency':          self.battery.battery_efficiency,
            },
            'solar': {
                'peak_power': self.solar.solar_peak_power,
                'efficiency': self.solar.solar_efficiency,
            },
            'inverter': {
                'max_power': self.inverter.max_power,
            },
            'grid': {
                'capacity': self.grid.grid_capacity,
            },
        }


class DefaultStrategyConfig(BaseModel):
    strategy_name: str
    target_soc: float = Field(default=0.70, ge=0.0, le=1.0, description="Charge battery to this SoC before exporting surplus")
    max_soc: float    = Field(default=0.95, ge=0.0, le=1.0, description="Top-up ceiling when solar is abundant")
    min_solar_threshold:  float = Field(default=10.0,  ge=0.0, description="Below this GTI the inverter treats it as night")
    high_solar_threshold: float = Field(default=400.0, ge=0.0, description="Above this GTI solar is considered abundant")
    solar_surplus_priority: Literal["charge_first", "sell_first"] = Field(
        default="charge_first",
        description="charge_first: top up toward max_soc before selling; sell_first: export immediately",
    )
    night_discharge: bool = Field(default=True,  description="Discharge battery at night to cover load")
    night_sell:      bool = Field(default=False, description="Also export battery energy to grid at night (requires night_discharge=true)")
    allow_grid_charging: bool = Field(default=False, description="Buy from grid to charge battery when SoC < target and no solar")
    outage_reserve: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum SoC to preserve during grid outages")

    def to_strategy_dict(self) -> dict:
        return {
            'target_soc':             self.target_soc,
            'max_soc':                self.max_soc,
            'min_solar_threshold':    self.min_solar_threshold,
            'high_solar_threshold':   self.high_solar_threshold,
            'solar_surplus_priority': self.solar_surplus_priority,
            'night_discharge':        self.night_discharge,
            'night_sell':             self.night_sell,
            'allow_grid_charging':    self.allow_grid_charging,
            'outage_reserve':         self.outage_reserve,
        }


class DefaultStrategyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:            int
    user_id:       int
    strategy_name: str
    settings:      dict


class DispatchStep(BaseModel):
    timestamp:            str
    soc:                  float
    target_soc:           float
    solar_kwh:            float
    load_kwh:             float
    battery_kwh:          float
    grid_kwh:             float
    unmet_load_kwh:       float
    money_earned_ts:      float
    dam_price:            float
    grid_status:          int
    hours_until_outage:   float
    solar_to_load_kwh:    float
    solar_to_battery_kwh: float
    solar_to_grid_kwh:    float
    battery_to_load_kwh:  float
    battery_to_grid_kwh:  float
    grid_to_load_kwh:     float
    grid_to_battery_kwh:  float


class DispatchSummary(BaseModel):
    total_money_earned:   float
    economic_savings_uah: float | None = None
    bought_kwh:           float
    sold_kwh:             float
    solar_kwh:            float
    unmet_load_kwh:       float
    lcos_total_uah:       float
    initial_soc:          float
    final_soc:            float
    steps:                int


class PredictionResponse(BaseModel):
    summary:       DispatchSummary
    dispatch_plan: list[DispatchStep]


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

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

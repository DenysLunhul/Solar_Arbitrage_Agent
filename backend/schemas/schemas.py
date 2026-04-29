from pydantic import BaseModel, Field


class Battery(BaseModel):
    capacity_kwh: float
    min_reserve: float | None = Field(default=10)
    lcos: float
    max_charge_power: float
    max_discharge_power: float
    efficiency: float = Field(default=1)


class Inverter(BaseModel):
    max_power: float
    efficiency: float
    is_grid_tied: bool

class SolarPanel(BaseModel):
    peak_power: float
    efficiency: float
    azimuth: float | None = Field(default=0)
    tilt: float | None = Field(default=35)

class SiteConfig(BaseModel):
    battery: Battery
    inverter: Inverter
    solar: SolarPanel
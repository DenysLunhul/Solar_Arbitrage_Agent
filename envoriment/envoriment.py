from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    """
    All hardware parameters for the RL environment.

    Defaults represent a small residential system.
    Use EnvConfig.from_site_config(site_config) to load from the backend.
    """

    # Battery
    max_batt_capacity: float = Field(default=2.0,  gt=0)
    max_batt_power: float    = Field(default=1.0,  gt=0)
    batt_efficiency: float   = Field(default=0.95, gt=0, le=1)
    lcos: float              = Field(default=1.5,  ge=0)
    min_soc_reserve: float   = Field(default=0.20, ge=0, le=1)

    # Inverter
    inverter_max_power: float  = Field(default=3.0,  gt=0)
    inverter_efficiency: float = Field(default=0.97, gt=0, le=1)

    # Solar
    solar_peak_power_kw: float = Field(default=3.0,  gt=0)
    solar_efficiency: float    = Field(default=0.18, gt=0, le=1)

    # Grid
    max_grid_capacity: float = Field(default=5.0, gt=0)

    @classmethod
    def from_site_config(cls, config) -> EnvConfig:
        """Build EnvConfig from a SiteConfig Pydantic model (backend schema)."""
        return cls(
            max_batt_capacity   = config.battery.capacity_kwh,
            max_batt_power      = config.battery.max_charge_power,
            batt_efficiency     = config.battery.efficiency,
            lcos                = config.battery.lcos,
            min_soc_reserve     = config.battery.min_reserve / 100,  # % → fraction
            inverter_max_power  = config.inverter.max_power,
            inverter_efficiency = config.inverter.efficiency,
            solar_peak_power_kw = config.solar.peak_power,
            solar_efficiency    = config.solar.efficiency,
        )


class Environment(gym.Env):
    """
    Середовище для RL-агента керування енергією (EMS).

    Пріоритет покриття навантаження (від найвищого до найнижчого):
        1. Сонячні панелі  — безкоштовна генерація, завжди використовується першою
        2. Батарея         — накопичений заряд, без вартості імпорту але з деградацією
        3. Мережа          — платний імпорт, штраф якщо Grid=0 (відключення)

    Дії агента (action_space: Box[-1, 1] shape=(2,)):
        action[0] → батарея:  +1 = повний заряд, -1 = повний розряд
        action[1] → мережа:   +1 = повний експорт, -1 = повний імпорт
        (знак мережі інвертований щоб +1 = "продати" = логічна "позитивна" дія)

    Спостереження (observation_space shape=(N_features + 1,)):
        всі колонки датасету + SoC (стан заряду батареї, 0.0–1.0)
    """

    def __init__(self, df: pd.DataFrame, config: EnvConfig | None = None):

        super().__init__()

        self.df  = df
        self.cfg = config or EnvConfig()

        # ── Похідні параметри (розраховуються один раз при ініціалізації) ──
        self.max_batt_power_ts    = self.cfg.max_batt_power / 4
        self.max_grid_capacity_ts = self.cfg.max_grid_capacity / 4
        self.max_export_ts        = min(self.cfg.max_grid_capacity, self.cfg.inverter_max_power) / 4
        # Площа панелей: Area = P_peak [Вт] / (GTI_stc [Вт/м²] × η)
        self.panel_area_m2        = self.cfg.solar_peak_power_kw * 1000 / (1000 * self.cfg.solar_efficiency)

        # ── Початковий стан ──────────────────────────────────────
        self.soc       = 0.5
        self.curr_step = 0

        # ── Простори дій та спостережень ────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1] + 1,), dtype=np.float32
        )

    # ──────────────────────────────────────────────────────────────
    # Допоміжні методи
    # ──────────────────────────────────────────────────────────────

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        """Генерація панелей за один таймстеп (15 хв) в кВт·год."""
        power_w = gti_w_m2 * self.cfg.solar_efficiency * self.panel_area_m2
        return max(0.0, power_w / 1000.0 / 4)

    def get_observe(self) -> np.ndarray:
        row = self.df.iloc[self.curr_step]
        return np.append(row.values, self.soc).astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc       = 0.5
        self.curr_step = 0
        return self.get_observe(), {}

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────

    def step(self, action):

        row = self.df.iloc[self.curr_step]

        # ── Зчитуємо дані поточного таймстепу ────────────────────
        curr_price         = row['DAM_Price'] / 1000
        curr_load_ts       = row['Load'] / 1000 / 4
        grid_status        = int(row['Grid'])
        gti                = row['Global_tilted_irradiance_instant']
        hours_until_outage = row['hours_until_outage']
        outage_remaining_h = row['outage_remaining_h']

        # ════════════════════════════════════════════════════════
        # БЛОК 1: СОНЯЧНА ГЕНЕРАЦІЯ
        # ════════════════════════════════════════════════════════

        solar_gen_ts       = self._calc_solar_generation_ts(gti)
        demand_after_solar = curr_load_ts - solar_gen_ts
        solar_surplus_ts   = max(0.0, -demand_after_solar)
        residual_demand_ts = max(0.0,  demand_after_solar)

        # ════════════════════════════════════════════════════════
        # БЛОК 2: БАТАРЕЯ
        # ════════════════════════════════════════════════════════

        battery_action       = float(action[0])
        battery_energy_delta = battery_action * self.cfg.max_batt_power / 4

        if battery_energy_delta > 0:
            chem_to_add           = battery_energy_delta * self.cfg.batt_efficiency
            room_in_batt          = (1.0 - self.soc) * self.cfg.max_batt_capacity
            actual_chem_in        = min(chem_to_add, room_in_batt)
            energy_drawn_for_batt = actual_chem_in / self.cfg.batt_efficiency
            solar_used_for_batt   = min(solar_surplus_ts, energy_drawn_for_batt)
            grid_needed_for_batt  = max(0.0, energy_drawn_for_batt - solar_used_for_batt)
            self.soc              = min(1.0, self.soc + actual_chem_in / self.cfg.max_batt_capacity)
            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)
            batt_contribution_ts  = -energy_drawn_for_batt
            actual_batt_energy_abs = actual_chem_in
        else:
            energy_to_draw        = abs(battery_energy_delta)
            max_drawable          = self.soc * self.cfg.max_batt_capacity
            actual_draw           = min(energy_to_draw, max_drawable)
            batt_output_ts        = actual_draw * self.cfg.batt_efficiency
            self.soc              = max(0.0, self.soc - actual_draw / self.cfg.max_batt_capacity)
            batt_contribution_ts  = batt_output_ts
            grid_needed_for_batt  = 0.0
            remaining_solar_surplus = solar_surplus_ts
            actual_batt_energy_abs = actual_draw

        # ════════════════════════════════════════════════════════
        # БЛОК 3: БАЛАНС ПІСЛЯ СОНЦЯ І БАТАРЕЇ
        # ════════════════════════════════════════════════════════

        if battery_energy_delta >= 0:
            net_demand_after_batt = residual_demand_ts + grid_needed_for_batt
        else:
            net_demand_after_batt = max(0.0, residual_demand_ts - batt_contribution_ts)
        solar_export_possible = remaining_solar_surplus

        # ════════════════════════════════════════════════════════
        # БЛОК 4: МЕРЕЖА
        # ════════════════════════════════════════════════════════

        grid_action   = float(action[1])
        grid_power_ts = -grid_action * self.max_grid_capacity_ts

        unmet_load     = 0.0
        actual_grid_ts = 0.0

        if grid_status == 1:
            if net_demand_after_batt > 0:
                actual_grid_ts = min(net_demand_after_batt, self.max_grid_capacity_ts)
                if actual_grid_ts < net_demand_after_batt:
                    unmet_load = net_demand_after_batt - actual_grid_ts
            else:
                exportable = solar_export_possible
                if grid_power_ts < 0:
                    actual_grid_ts = max(
                        -exportable,
                        -self.max_export_ts,
                        grid_power_ts
                    )
                else:
                    actual_grid_ts = 0.0
        else:
            actual_grid_ts = 0.0
            if net_demand_after_batt > 0:
                unmet_load = net_demand_after_batt

        # ════════════════════════════════════════════════════════
        # БЛОК 5: ФУНКЦІЯ ВИНАГОРОДИ
        # ════════════════════════════════════════════════════════

        reward = 0.0

        reward -= actual_grid_ts * curr_price

        lcos_cost = self.cfg.lcos * actual_batt_energy_abs
        reward -= lcos_cost

        if unmet_load > 0:
            reward -= unmet_load * 50.0

        mismatch = abs(grid_power_ts - actual_grid_ts)
        reward -= 2.0 * mismatch

        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, self.cfg.min_soc_reserve - self.soc)
            if soc_deficit > 0:
                reward -= 30.0 * soc_deficit * np.log1p(outage_remaining_h)

        if grid_status == 1 and 0 < hours_until_outage <= 3.0:
            urgency     = np.exp(-0.5 * hours_until_outage)
            soc_surplus = max(0.0, self.soc - self.cfg.min_soc_reserve)
            reward += 5.0 * urgency * soc_surplus

        # ════════════════════════════════════════════════════════
        # ЗАВЕРШЕННЯ КРОКУ
        # ════════════════════════════════════════════════════════

        self.curr_step += 1
        terminated  = self.curr_step >= len(self.df)
        truncated   = False
        observation = self.get_observe() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "soc":                 self.soc,
            "reward":              reward,
            "solar_gen_ts_kwh":    solar_gen_ts,
            "solar_surplus_kwh":   solar_export_possible,
            "residual_demand_kwh": residual_demand_ts,
            "net_demand_kwh":      net_demand_after_batt,
            "unmet_load_kwh":      unmet_load,
            "actual_grid_kwh":     actual_grid_ts,
            "lcos_cost":           lcos_cost,
            "mismatch":            mismatch,
        }

        return observation, reward, terminated, truncated, info
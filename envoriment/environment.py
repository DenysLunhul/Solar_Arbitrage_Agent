import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
 
 
class Environment(gym.Env):
 
    def __init__(self, df_raw: pd.DataFrame, df: pd.DataFrame, system_config: dict):
        super().__init__()
 
        self.df = df
        self.df_raw = df_raw
 
        batt  = system_config['battery']
        solar = system_config['solar']
        inv   = system_config['inverter']
        grid  = system_config['grid']

        # ── Батарея ───────────────────────────────────────────────
        self.max_batt_capacity         = batt['capacity_kwh']
        self.max_batt_charge_power     = batt['max_charge_power']
        self.max_batt_discharge_power  = batt['max_discharge_power']
        self.max_batt_charge_power_ts  = self.max_batt_charge_power / 4
        self.max_batt_discharge_power_ts = self.max_batt_discharge_power / 4
        self.batt_efficiency           = batt['efficiency']
        self.lcos                      = batt['lcos']

        # min_reserve приходить як відсотки (20), конвертуємо в частку (0.20)
        self.soc_soft_min         = batt['min_reserve'] / 100
        self.soc_soft_max         = 0.80

        # ── Мережа ───────────────────────────────────────────────
        self.max_grid_capacity    = min(inv['max_power'], grid['capacity'])
        self.max_grid_capacity_ts = self.max_grid_capacity / 4
        self.price_to_buy         = grid['price_to_buy']
 
        # ── Сонячні панелі ────────────────────────────────────────
        self.solar_peak_power_kw  = solar['peak_power']
        self.solar_efficiency     = solar['efficiency']
 
        # Площа панелей рахується автоматично з пікової потужності і ККД
        # Формула: P_peak[Вт] = GTI_stc[Вт/м²] × η × Area[м²]
        # GTI_stc = 1000 Вт/м² (стандартні умови)
        # Звідси: Area = P_peak / (1000 × η)
        self.panel_area_m2        = (self.solar_peak_power_kw * self.solar_efficiency)
 
        # ── Початковий стан ───────────────────────────────────────
        self.soc       = 0.0
        self.curr_step = 0
 
        # ── Простори дій та спостережень ─────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features + 1,), dtype=np.float32)

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        power_w       = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        energy_kwh_ts = (power_w / 1000.0) / 4
        return max(0.0, energy_kwh_ts)
 
    # ─────────────────────────────────────────────────────────────
    def _calc_target_soc(self, next_outage_h: float, curr_load_kw: float, gti_w_m2: float) -> float:
        """
        Динамічний цільовий SoC — скільки треба запасти щоб пережити відключення.
 
        Логіка:
            1. Рахуємо скільки споживе будинок за час відключення
            2. Віднімаємо прогнозовану генерацію сонця за той самий час
            3. Додаємо 10% буфер
            4. Переводимо в частку від ємності батареї
            5. Обмежуємо в [soc_soft_min, soc_soft_max]
        """
        if next_outage_h <= 0:
            return self.soc_soft_min
 
        # Скільки кВт·год споживе будинок за час відключення
        energy_load = curr_load_kw * next_outage_h
 
        # Груба оцінка генерації сонця під час відключення
        # (беремо поточний GTI як proxy)
        solar_per_hour = self._calc_solar_generation_ts(gti_w_m2) * 4
        energy_solar   = solar_per_hour * next_outage_h
 
        # Чистий дефіцит + 10% буфер
        net_energy_needed  = max(0.0, energy_load - energy_solar)
        energy_with_buffer = net_energy_needed * 1.10
 
        # Переводимо в частку від ємності і обмежуємо
        target_soc = energy_with_buffer / self.max_batt_capacity
        target_soc = np.clip(target_soc, self.soc_soft_min, self.soc_soft_max)
 
        return float(target_soc)
 
    # ─────────────────────────────────────────────────────────────
    def get_observe(self) -> np.ndarray:
        row = self.df.iloc[self.curr_step]
        return np.append(row.values, self.soc).astype(np.float32)
 
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc       = 0.0
        self.curr_step = 0
        return self.get_observe(), {}
 
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
 
        row = self.df_raw.iloc[self.curr_step]
 
        # ── Дані поточного таймстепу ──────────────────────────────
        curr_price         = row['DAM_Price'] / 1000
        curr_load_ts       = row['Load'] / 1000 / 4
        curr_load_kw       = row['Load'] / 1000
        grid_status        = int(row['Grid'])
        gti                = row['Global_tilted_irradiance_instant']
        hours_until_outage = row['hours_until_outage']
        outage_remaining_h = row['outage_remaining_h']
        next_outage_h      = row['next_outage_duration']
 
        # ════════════════════════════════════════════════════════
        # БЛОК 0: ДИНАМІЧНИЙ ЦІЛЬОВИЙ SoC
        # ════════════════════════════════════════════════════════
        target_soc = self._calc_target_soc(next_outage_h, curr_load_kw, gti)
 
        # ════════════════════════════════════════════════════════
        # БЛОК 1: СОНЯЧНА ГЕНЕРАЦІЯ
        # ════════════════════════════════════════════════════════
        solar_gen_ts = self._calc_solar_generation_ts(gti)
 
        demand_after_solar = curr_load_ts - solar_gen_ts
        solar_surplus_ts   = max(0.0, -demand_after_solar)
        residual_demand_ts = max(0.0,  demand_after_solar)
 
        # ════════════════════════════════════════════════════════
        # БЛОК 2: БАТАРЕЯ
        # ════════════════════════════════════════════════════════
        battery_action = float(action[0])
        if battery_action >= 0:
            battery_energy_delta = battery_action * self.max_batt_charge_power_ts
        else:
            battery_energy_delta = battery_action * self.max_batt_discharge_power_ts

        if battery_energy_delta > 0:
            # ЗАРЯДКА
            chem_to_add    = battery_energy_delta * self.batt_efficiency
            room_in_batt   = (1.0 - self.soc) * self.max_batt_capacity
            actual_chem_in = min(chem_to_add, room_in_batt)
 
            energy_drawn_for_batt   = actual_chem_in / self.batt_efficiency
            solar_used_for_batt     = min(solar_surplus_ts, energy_drawn_for_batt)
            grid_needed_for_batt    = max(0.0, energy_drawn_for_batt - solar_used_for_batt)
 
            self.soc = min(1.0, self.soc + actual_chem_in / self.max_batt_capacity)
 
            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)
            batt_contribution_ts    = -energy_drawn_for_batt
            actual_batt_energy_abs  = actual_chem_in
 
        else:
            # РОЗРЯД
            energy_to_draw = abs(battery_energy_delta)
            max_drawable   = self.soc * self.max_batt_capacity
            actual_draw    = min(energy_to_draw, max_drawable)
            batt_output_ts = actual_draw * self.batt_efficiency
 
            self.soc = max(0.0, self.soc - actual_draw / self.max_batt_capacity)
 
            batt_contribution_ts    =  batt_output_ts
            grid_needed_for_batt    =  0.0
            remaining_solar_surplus =  solar_surplus_ts
            actual_batt_energy_abs  =  actual_draw
 
        # ════════════════════════════════════════════════════════
        # БЛОК 3: БАЛАНС
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
                if grid_power_ts < 0:
                    actual_grid_ts = max( -solar_export_possible, -self.max_grid_capacity_ts, grid_power_ts)
                else:
                    actual_grid_ts = 0.0
        else:
            actual_grid_ts = 0.0
            if net_demand_after_batt > 0:
                unmet_load = net_demand_after_batt
 
        # ════════════════════════════════════════════════════════
        # БЛОК 5: REWARD (компоненти окремо для логування)
        # ════════════════════════════════════════════════════════
        r_market      = 0.0
        r_unmet       = 0.0
        r_soc_soft    = 0.0
        r_reserve     = 0.0
        r_preparation = 0.0

        # 5.1 Ринковий P&L
        if actual_grid_ts > 0:
            r_market = -actual_grid_ts * self.price_to_buy
        else:
            r_market = abs(actual_grid_ts) * curr_price

        if actual_grid_ts < 0:
            money_earned_ts = abs(actual_grid_ts * curr_price)
        else:
            money_earned_ts = actual_grid_ts * self.price_to_buy

        # 5.2 Деградація батареї
        lcos_cost = self.lcos * actual_batt_energy_abs
        r_lcos    = -lcos_cost

        # 5.3 Непокрите навантаження
        if unmet_load > 0:
            r_unmet = -unmet_load * self.price_to_buy * 2

        # 5.4 Штраф за нереалістичну дію
        mismatch   = abs(grid_power_ts - actual_grid_ts)
        r_mismatch = -2.0 * mismatch

        # 5.5 М'який штраф за SoC поза діапазоном 20%–80%
        if self.soc < self.soc_soft_min:
            r_soc_soft -= 3.0 * ((self.soc_soft_min - self.soc) ** 2)
        if self.soc > self.soc_soft_max:
            r_soc_soft -= 3.0 * ((self.soc - self.soc_soft_max) ** 2)

        # 5.6 Штраф за недостатній резерв під час відключення
        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, target_soc - self.soc)
            if soc_deficit > 0:
                r_reserve = -30.0 * soc_deficit * np.log1p(outage_remaining_h)

        # 5.7 Бонус за підготовку до відключення
        if grid_status == 1 and 0 < hours_until_outage <= 3.0:
            urgency       = np.exp(-0.5 * hours_until_outage)
            soc_ready     = min(self.soc, target_soc)
            r_preparation = 5.0 * urgency * soc_ready

        reward = r_market + r_lcos + r_unmet + r_mismatch + r_soc_soft + r_reserve + r_preparation

        # ════════════════════════════════════════════════════════
        # ЗАВЕРШЕННЯ
        # ════════════════════════════════════════════════════════
        self.curr_step += 1
        terminated  = self.curr_step >= len(self.df) - 1
        truncated   = False
        observation = self.get_observe()

        info = {
            'soc':               self.soc,
            'target_soc':        target_soc,
            'reward':            reward,
            'solar_gen_ts_kwh':  solar_gen_ts,
            'solar_surplus_kwh': solar_export_possible,
            'actual_grid_kwh':   actual_grid_ts,
            'battery_kwh':       -batt_contribution_ts,   # + = charging, - = discharging
            'unmet_load_kwh':    unmet_load,
            'lcos_cost':         lcos_cost,
            'mismatch':          mismatch,
            'money_earned_ts':   money_earned_ts,
            'reward_market':     r_market,
            'reward_lcos':       r_lcos,
            'reward_unmet':      r_unmet,
            'reward_mismatch':   r_mismatch,
            'reward_soc_soft':   r_soc_soft,
            'reward_reserve':    r_reserve,
            'reward_preparation':r_preparation,
        }

        return observation, reward, terminated, truncated, info

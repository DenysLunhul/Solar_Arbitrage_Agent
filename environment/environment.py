import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class Environment(gym.Env):

    PRICE_LOOKAHEAD = 16  # 4-hour price window fed to the network

    def __init__(self, df_raw: pd.DataFrame, df: pd.DataFrame, system_config: dict, episode_len: int = 96):
        super().__init__()

        self.df = df
        self.df_raw = df_raw
        self.episode_len = episode_len

        batt  = system_config['battery']
        solar = system_config['solar']
        inv   = system_config['inverter']
        grid  = system_config['grid']

        self.max_batt_capacity         = batt['capacity_kwh']
        self.max_batt_charge_power     = batt['max_charge_power']
        self.max_batt_discharge_power  = batt['max_discharge_power']
        self.max_batt_charge_power_ts  = self.max_batt_charge_power / 4
        self.max_batt_discharge_power_ts = self.max_batt_discharge_power / 4
        self.batt_efficiency           = batt['efficiency']
        self.lcos                      = batt['lcos']

        # min_reserve comes in as percentage (e.g. 20), convert to fraction
        self.soc_soft_min         = batt['min_reserve'] / 100
        self.soc_soft_max         = 0.80
        self.soc_hard_min         = 0.05   # physical BMS cutoff — never crossed regardless of grid status

        self.max_grid_capacity    = min(inv['max_power'], grid['capacity'])
        self.max_grid_capacity_ts = self.max_grid_capacity / 4

        self.solar_peak_power_kw  = solar['peak_power']
        self.solar_efficiency     = solar['efficiency']

        # P_peak[kW] = GTI_stc[W/m²] × η × Area[m²], GTI_stc=1000 → Area = peak_power / η
        self.panel_area_m2        = self.solar_peak_power_kw / self.solar_efficiency

        self.soc            = 0.0
        self.curr_step      = 0
        self.episode_start  = 0

        # Number of complete day-aligned episodes available
        self._n_starts = max(1, len(df) // episode_len)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        n_features = df.shape[1]
        # 17 normalized features + SoC + PRICE_LOOKAHEAD next DAM prices (4-hour horizon)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features + 1 + self.PRICE_LOOKAHEAD,), dtype=np.float32)

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        power_w       = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        energy_kwh_ts = (power_w / 1000.0) / 4
        return max(0.0, energy_kwh_ts)

    # Economic buffer SoC to maintain on days with no forecasted outage.
    # Gives r_soc_target something to reward and r_soc_below a continuous signal.
    DEFAULT_SOC_TARGET = 0.50

    def _calc_target_soc(self, next_outage_h: float, curr_load_kw: float, gti_w_m2: float) -> float:
        """Dynamic target SoC: energy needed to survive the upcoming outage, clipped to [soc_soft_min, soc_soft_max]."""
        if next_outage_h <= 0:
            return max(self.soc_soft_min, self.DEFAULT_SOC_TARGET)

        energy_load  = curr_load_kw * next_outage_h
        solar_per_hour = self._calc_solar_generation_ts(gti_w_m2) * 4
        energy_solar   = solar_per_hour * next_outage_h
        net_energy_needed  = max(0.0, energy_load - energy_solar)
        energy_with_buffer = net_energy_needed * 1.10
        target_soc = energy_with_buffer / self.max_batt_capacity
        target_soc = np.clip(target_soc, self.soc_soft_min, self.soc_soft_max)

        return float(target_soc)

    def get_observe(self) -> np.ndarray:
        idx = min(self.curr_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        base = np.append(row.values, self.soc)

        # Next PRICE_LOOKAHEAD DAM prices (4-hour horizon); zeros pad at end of episode.
        lookahead_end = idx + self.PRICE_LOOKAHEAD
        remaining     = self.df['DAM_Price'].iloc[idx:lookahead_end].values
        price_vec     = np.zeros(self.PRICE_LOOKAHEAD, dtype=np.float32)
        price_vec[:len(remaining)] = remaining

        return np.concatenate([base, price_vec]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc           = float(np.random.uniform(0.0, 1.0))
        day_idx            = int(np.random.randint(0, self._n_starts))
        self.episode_start = day_idx * self.episode_len
        self.curr_step     = self.episode_start
        return self.get_observe(), {}

    def step(self, action):

        row = self.df_raw.iloc[self.curr_step]

        curr_price         = row['DAM_Price'] / 1000   # UAH/kWh (sell price = DAM)
        buy_price          = curr_price + 3.0           # UAH/kWh (DAM + 3 UAH/kWh grid tax)
        curr_load_ts       = row['Load'] / 4            # kW → kWh per 15-min timestep
        curr_load_kw       = row['Load']
        grid_status        = int(row['Grid'])
        gti                = row['Global_tilted_irradiance_instant']
        hours_until_outage = row['hours_until_outage']
        outage_remaining_h = row['outage_remaining_h']
        next_outage_h      = row['next_outage_duration']

        target_soc = self._calc_target_soc(next_outage_h, curr_load_kw, gti)

        solar_gen_ts = self._calc_solar_generation_ts(gti)

        demand_after_solar = curr_load_ts - solar_gen_ts
        solar_surplus_ts   = max(0.0, -demand_after_solar)
        residual_demand_ts = max(0.0,  demand_after_solar)

        battery_action = float(action[0])
        if battery_action >= 0:
            battery_energy_delta = battery_action * self.max_batt_charge_power_ts
        else:
            battery_energy_delta = battery_action * self.max_batt_discharge_power_ts

        if battery_energy_delta > 0:
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
            energy_to_draw = abs(battery_energy_delta)
            # Grid up: strategic floor (min_reserve). Grid down: physical BMS floor only.
            if grid_status == 1:
                max_drawable = max(0.0, (self.soc - self.soc_soft_min) * self.max_batt_capacity)
            else:
                max_drawable = max(0.0, (self.soc - self.soc_hard_min) * self.max_batt_capacity)
            actual_draw    = min(energy_to_draw, max_drawable)
            batt_output_ts = actual_draw * self.batt_efficiency

            self.soc = max(self.soc_hard_min, self.soc - actual_draw / self.max_batt_capacity)

            batt_contribution_ts    =  batt_output_ts
            grid_needed_for_batt    =  0.0
            remaining_solar_surplus =  solar_surplus_ts
            actual_batt_energy_abs  =  actual_draw

        if battery_energy_delta >= 0:
            net_demand_after_batt = residual_demand_ts + grid_needed_for_batt
            batt_export_possible  = 0.0
        else:
            net_demand_after_batt = max(0.0, residual_demand_ts - batt_contribution_ts)
            # Battery output exceeding load demand becomes exportable surplus alongside solar.
            batt_export_possible  = max(0.0, batt_contribution_ts - residual_demand_ts)

        solar_export_possible = remaining_solar_surplus                          # pure solar (for info)
        total_export_possible = remaining_solar_surplus + batt_export_possible   # all exportable energy

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
                    actual_grid_ts = max(-total_export_possible, -self.max_grid_capacity_ts, grid_power_ts)
                else:
                    actual_grid_ts = 0.0
        else:
            actual_grid_ts = 0.0
            if net_demand_after_batt > 0:
                unmet_load = net_demand_after_batt

        r_market      = 0.0
        r_unmet       = 0.0
        r_soc_soft    = 0.0
        r_reserve     = 0.0
        r_preparation = 0.0
        r_soc_target  = 0.0

        if actual_grid_ts > 0:
            r_market = -actual_grid_ts * buy_price
        else:
            r_market = abs(actual_grid_ts) * curr_price

        if actual_grid_ts < 0:
            money_earned_ts = abs(actual_grid_ts) * curr_price   # revenue from selling
        else:
            money_earned_ts = -actual_grid_ts * buy_price         # cost of buying (negative)

        lcos_cost = self.lcos * actual_batt_energy_abs
        r_lcos    = -lcos_cost

        if unmet_load > 0:
            r_unmet = -unmet_load * buy_price * 2

        # Mismatch removed: r_market already penalises/rewards actual grid transactions;
        # a separate mismatch term dominated the gradient without teaching new behaviour.
        mismatch   = 0.0
        r_mismatch = 0.0

        # Hard boundary violations — large coefficient keeps SoC inside [soc_soft_min, soc_soft_max].
        if self.soc < self.soc_soft_min:
            r_soc_soft -= 50.0 * ((self.soc_soft_min - self.soc) ** 2)
        if self.soc > self.soc_soft_max:
            r_soc_soft -= 50.0 * ((self.soc - self.soc_soft_max) ** 2)

        # Continuous below-target penalty (grid up only): fires every step when SoC < target_soc.
        # Without this, target_soc = DEFAULT_SOC_TARGET has no gradient when the agent never charges.
        # Coefficient 5.0: at SoC=0.20, target=0.50 → -5×0.09=-0.45/step ≈ -29 over a full night,
        # comparable to the cost savings from discharging stored energy instead of buying from grid.
        if grid_status == 1 and self.soc < target_soc:
            r_soc_soft -= 5.0 * ((target_soc - self.soc) ** 2)

        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, target_soc - self.soc)
            if soc_deficit > 0:
                r_reserve = -50.0 * soc_deficit * np.log1p(outage_remaining_h)

        if grid_status == 1 and 0 < hours_until_outage <= 3.0:
            urgency       = np.exp(-0.5 * hours_until_outage)
            soc_ready     = min(self.soc, target_soc)
            r_preparation = 5.0 * urgency * soc_ready

        # Reward = energy stored × buy_price (avoided future purchase cost), making charging
        # toward reserve target economically competitive with selling solar surplus.
        # Check pre-charge SoC to handle large actions that overshoot target in one step.
        if battery_energy_delta > 0:
            pre_charge_soc = self.soc - actual_chem_in / self.max_batt_capacity
            if pre_charge_soc < target_soc:
                r_soc_target = actual_chem_in * buy_price  # economic value of stored reserve

        # Wasted-discharge: battery energy that can't reach demand OR be exported to grid.
        # Grid headroom = capacity remaining after solar takes its share.
        r_waste = 0.0
        if battery_energy_delta < 0 and batt_contribution_ts > 0:
            demand_covered  = min(batt_contribution_ts, residual_demand_ts)
            grid_headroom   = max(0.0, self.max_grid_capacity_ts - remaining_solar_surplus)
            exportable_batt = min(batt_export_possible, grid_headroom)
            wasted          = max(0.0, batt_contribution_ts - demand_covered - exportable_batt)
            r_waste         = -10.0 * self.lcos * wasted

        # Normalize by capacity so episodes with different hardware produce comparable gradient scales.
        # Without this, a 250 kWh episode dominates a 50 kWh one by 5× in the replay buffer.
        reward = (r_market + r_lcos + r_unmet + r_mismatch + r_soc_soft + r_reserve + r_preparation + r_soc_target + r_waste) / (self.max_batt_capacity / 100.0)

        self.curr_step += 1
        terminated = self.curr_step >= self.episode_start + self.episode_len
        truncated  = False
        observation = self.get_observe()

        info = {
            'soc':               self.soc,
            'target_soc':        target_soc,
            'reward':            reward,
            'solar_gen_ts_kwh':  solar_gen_ts,
            'solar_surplus_kwh': remaining_solar_surplus,
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
            'reward_soc_target': r_soc_target,
            'reward_waste':      r_waste,
        }

        return observation, reward, terminated, truncated, info

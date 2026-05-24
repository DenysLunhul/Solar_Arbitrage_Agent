import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class Environment(gym.Env):

    PRICE_LOOKAHEAD = 32  # 8-hour forward price window fed to the network
    PRICE_HISTORY   = 16  # 4-hour backward price window (context for arbitrage)
    LOAD_LOOKAHEAD  = 16  # 4-hour load forecast fed to the network
    GTI_LOOKAHEAD   = 16  # 4-hour solar irradiance forecast fed to the network

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
        # 17 normalized features + SoC + 8-hour price lookahead + 4-hour price history + load/GTI lookahead
        obs_size = n_features + 1 + self.PRICE_LOOKAHEAD + self.PRICE_HISTORY + self.LOAD_LOOKAHEAD + self.GTI_LOOKAHEAD
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        power_w       = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        energy_kwh_ts = (power_w / 1000.0) / 4
        return max(0.0, energy_kwh_ts)

    # Economic buffer SoC when no outage is forecasted. 0.30 avoids forcing
    # expensive grid charging in winter while still providing an arbitrage reserve.
    DEFAULT_SOC_TARGET = 0.30

    def _calc_target_soc(self, next_outage_h: float, curr_load_kw: float, gti_w_m2: float) -> float:
        """Dynamic target SoC: energy needed to survive the upcoming outage, clipped to [soc_soft_min, soc_soft_max]."""
        if next_outage_h <= 0:
            return self.DEFAULT_SOC_TARGET

        energy_load  = curr_load_kw * next_outage_h
        solar_per_hour = self._calc_solar_generation_ts(gti_w_m2) * 4
        energy_solar   = solar_per_hour * next_outage_h
        net_energy_needed  = max(0.0, energy_load - energy_solar)
        energy_with_buffer = net_energy_needed * 1.10
        target_soc = energy_with_buffer / self.max_batt_capacity
        # DEFAULT_SOC_TARGET is the economic baseline — never let small outages collapse the target to soc_soft_min
        target_soc = max(target_soc, self.DEFAULT_SOC_TARGET)
        target_soc = np.clip(target_soc, self.soc_soft_min, self.soc_soft_max)

        return float(target_soc)

    def get_observe(self) -> np.ndarray:
        idx = min(self.curr_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        base = np.append(row.values, self.soc)

        # Forward lookahead: 8-hour price, 4-hour load/GTI; zero-pad beyond dataset end.
        price_fwd = np.zeros(self.PRICE_LOOKAHEAD, dtype=np.float32)
        load_vec  = np.zeros(self.LOAD_LOOKAHEAD,  dtype=np.float32)
        gti_vec   = np.zeros(self.GTI_LOOKAHEAD,   dtype=np.float32)

        p = self.df['DAM_Price'].iloc[idx:idx + self.PRICE_LOOKAHEAD].values
        l = self.df['Load'].iloc[idx:idx + self.LOAD_LOOKAHEAD].values
        g = self.df['Global_tilted_irradiance_instant'].iloc[idx:idx + self.GTI_LOOKAHEAD].values

        price_fwd[:len(p)] = p
        load_vec[:len(l)]  = l
        gti_vec[:len(g)]   = g

        # Backward price history: most recent at the end, zero-padded at the start.
        price_hist = np.zeros(self.PRICE_HISTORY, dtype=np.float32)
        hist_start = max(0, idx - self.PRICE_HISTORY)
        h = self.df['DAM_Price'].iloc[hist_start:idx].values
        price_hist[self.PRICE_HISTORY - len(h):] = h

        return np.concatenate([base, price_fwd, price_hist, load_vec, gti_vec]).astype(np.float32)

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

        # Day-average price for price-timing reward: mean of the current episode day's prices.
        day_start = (self.curr_step // 96) * 96
        day_end   = min(day_start + 96, len(self.df_raw))
        day_avg_price = float(self.df_raw['DAM_Price'].iloc[day_start:day_end].mean()) / 1000

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

            # During outage the grid can't supply charging energy — cap to solar only.
            # Without this the battery gains phantom SoC from a non-existent grid source
            # and grid_needed_for_batt inflates unmet_load with demand that can't be real.
            if grid_status == 0:
                energy_drawn_for_batt = solar_used_for_batt
                actual_chem_in        = solar_used_for_batt * self.batt_efficiency
                grid_needed_for_batt  = 0.0

            self.soc = min(1.0, self.soc + actual_chem_in / self.max_batt_capacity)

            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)
            batt_contribution_ts    = -energy_drawn_for_batt
            actual_batt_energy_abs  = actual_chem_in

        else:
            energy_to_draw = abs(battery_energy_delta)
            # Convert requested DC output to chemical kWh needed (mirrors charge-side logic).
            # Without this, energy_to_draw (DC) is compared to max_drawable (chemical),
            # causing underdelivery and incorrect SoC drain in the non-limiting case.
            chem_needed = energy_to_draw / self.batt_efficiency
            # Grid up: strategic floor (min_reserve). Grid down: physical BMS floor only.
            if grid_status == 1:
                max_drawable = max(0.0, (self.soc - self.soc_soft_min) * self.max_batt_capacity)
            else:
                max_drawable = max(0.0, (self.soc - self.soc_hard_min) * self.max_batt_capacity)
            actual_draw    = min(chem_needed, max_drawable)   # both in chemical kWh

            # Island mode: battery can only discharge as much as load demands — no grid to absorb excess.
            # Without this cap the full requested discharge drains SoC even though only load_kwh reaches demand.
            if grid_status == 0:
                load_chem   = residual_demand_ts / self.batt_efficiency
                actual_draw = min(actual_draw, load_chem)

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

        r_market         = 0.0
        r_unmet          = 0.0
        r_soc_soft       = 0.0
        r_reserve        = 0.0
        r_preparation    = 0.0
        r_soc_target     = 0.0
        r_curtail        = 0.0
        r_solar_priority = 0.0

        if actual_grid_ts > 0:
            r_market = -actual_grid_ts * buy_price
        else:
            r_market = abs(actual_grid_ts) * curr_price

        if actual_grid_ts < 0:
            money_earned_ts = abs(actual_grid_ts) * curr_price   # revenue from selling
        else:
            money_earned_ts = -actual_grid_ts * buy_price         # cost of buying (negative)

        lcos_cost = self.lcos * actual_batt_energy_abs
        r_lcos    = -2.5 * lcos_cost

        if unmet_load > 0:
            r_unmet = -unmet_load * buy_price * 5

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
        # Coefficient 100.0: strong enough that daily gap penalty (-64 norm) exceeds one
        # grid-charge step cost (-90 norm), giving SAC a clear margin to learn winter charging.
        if grid_status == 1 and self.soc < target_soc:
            r_soc_soft -= 100.0 * ((target_soc - self.soc) ** 2)

        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, target_soc - self.soc)
            if soc_deficit > 0:
                r_reserve = -50.0 * soc_deficit * np.log1p(outage_remaining_h)

        if grid_status == 1 and 0 < hours_until_outage <= 6.0:
            urgency       = np.exp(-0.5 * hours_until_outage)
            soc_ready     = min(self.soc, target_soc)
            r_preparation = 20.0 * urgency * soc_ready

        # Reward solar-sourced charging toward the reserve target at avoided-purchase value.
        # Grid-sourced charging is intentionally excluded: r_market already penalises the
        # import cost, and adding r_soc_target on top creates a near-zero net signal that
        # teaches the agent to freely grid-charge (r_market + r_soc_target ≈ 0). The
        # r_unmet penalty alone is sufficient to drive pre-outage grid charging when needed.
        if battery_energy_delta > 0 and solar_used_for_batt > 0:
            pre_charge_soc = self.soc - actual_chem_in / self.max_batt_capacity
            if pre_charge_soc < target_soc:
                solar_chem_stored = solar_used_for_batt * self.batt_efficiency
                r_soc_target = min(actual_chem_in, solar_chem_stored) * buy_price

        # Wasted-discharge: battery energy that can't reach demand OR be exported to grid.
        # Grid headroom = capacity remaining after solar takes its share.
        r_waste = 0.0
        if battery_energy_delta < 0 and batt_contribution_ts > 0:
            demand_covered  = min(batt_contribution_ts, residual_demand_ts)
            grid_headroom   = 0.0 if grid_status == 0 else max(0.0, self.max_grid_capacity_ts - remaining_solar_surplus)
            exportable_batt = min(batt_export_possible, grid_headroom)
            wasted          = max(0.0, batt_contribution_ts - demand_covered - exportable_batt)
            r_waste         = -10.0 * self.lcos * wasted

        # Solar curtailment: penalise free surplus that is neither stored nor exported.
        # The environment requires an explicit export action; a passive action[1]≤0 silently
        # discards the surplus. 3× curr_price: solar lost now is gone forever — the agent
        # cannot store it for a better price later, so the penalty must exceed r_price_timing.
        curtailed_kwh = 0.0
        if grid_status == 1 and total_export_possible > 0.01:
            actually_exported = max(0.0, -actual_grid_ts)
            curtailed_kwh = max(0.0, total_export_possible - actually_exported)
            if curtailed_kwh > 0.01:
                r_curtail = -curtailed_kwh * curr_price * 3.0

        # Price-timing bonus: reward selling above the day's average price, penalise buying above it.
        # Coefficient 1.0 (down from 2.0): less aggressive holding reduces curtailment in summer
        # while keeping the buy-cheap/sell-expensive gradient intact.
        # Suppressed when outage is imminent (≤3h) and battery is underprepared — at that point
        # charging is mandatory regardless of price.
        r_price_timing = 0.0
        outage_imminent = grid_status == 1 and 0 < hours_until_outage <= 3.0 and self.soc < target_soc
        if grid_status == 1 and not outage_imminent:
            price_dev = curr_price - day_avg_price
            if actual_grid_ts < 0:   # selling to grid
                r_price_timing =  price_dev * abs(actual_grid_ts) * 1.0   # bonus for selling high
            elif actual_grid_ts > 0: # buying from grid
                r_price_timing = -price_dev * actual_grid_ts * 1.0        # penalty for buying high

        # Solar-priority penalty: penalise supplementing battery charging from the grid while
        # solar is simultaneously available. Only fires when grid_needed_for_batt > 0 (the agent
        # chose a charge rate that solar surplus alone can't cover). Load-driven grid imports
        # (unavoidable when load > solar + battery discharge) are excluded — the agent can't
        # avoid those and penalising them creates unlearnable gradient noise.
        # Bypassed when an outage is imminent and emergency pre-charging is mandatory.
        if grid_status == 1 and grid_needed_for_batt > 0.01 and solar_gen_ts > 0.1 and not outage_imminent:
            solar_fraction    = min(1.0, solar_gen_ts / (solar_gen_ts + grid_needed_for_batt))
            r_solar_priority  = -grid_needed_for_batt * solar_fraction * curr_price * 2.0

        # Normalize by capacity so episodes with different hardware produce comparable gradient scales.
        # Without this, a 250 kWh episode dominates a 50 kWh one by 5× in the replay buffer.
        reward = (r_market + r_lcos + r_unmet + r_mismatch + r_soc_soft + r_reserve + r_preparation + r_soc_target + r_waste + r_curtail + r_price_timing + r_solar_priority) / (self.max_batt_capacity / 100.0)

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
            'curtailed_kwh':         curtailed_kwh,
            'reward_waste':          r_waste,
            'reward_curtail':        r_curtail,
            'reward_price_timing':   r_price_timing,
            'reward_solar_priority': r_solar_priority,
        }

        return observation, reward, terminated, truncated, info

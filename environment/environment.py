import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class Environment(gym.Env):

    PRICE_LOOKAHEAD = 96
    PRICE_HISTORY   = 16
    LOAD_LOOKAHEAD  = 16
    GTI_LOOKAHEAD   = 16

    DEFAULT_LOAD_CONFIG = {'peak_kw': 60.0, 'profile': 'office'}

    def __init__(self, df_raw: pd.DataFrame, df: pd.DataFrame, system_config: dict, episode_len: int = 96,
                 load_kw: np.ndarray | None = None):
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

        self.soc_soft_min         = batt['min_reserve'] / 100
        self.soc_soft_max         = 0.80
        self.soc_hard_min         = 0.05

        self.max_grid_capacity    = min(inv['max_power'], grid['capacity'])
        self.max_grid_capacity_ts = self.max_grid_capacity / 4

        self.solar_peak_power_kw  = solar['peak_power']
        self.solar_efficiency     = solar['efficiency']

        self.panel_area_m2        = self.solar_peak_power_kw / self.solar_efficiency

        load_cfg = system_config.get('load') or self.DEFAULT_LOAD_CONFIG
        self.load_peak_kw = float(load_cfg['peak_kw'])
        self.load_kw = np.asarray(load_kw if load_kw is not None else df_raw['Load'].values, dtype=np.float64)
        assert len(self.load_kw) == len(df_raw) == len(df), "load_kw must align with df/df_raw rows"
        self.rel_load = (self.load_kw / self.load_peak_kw).astype(np.float32)
        self._load_cap_ratio  = self.load_peak_kw / self.max_batt_capacity
        self._load_grid_ratio = self.load_peak_kw / self.max_grid_capacity

        self.soc            = 0.0
        self.curr_step      = 0
        self.episode_start  = 0

        # Hardware context scalars: under domain randomization the agent cannot infer
        # capacity / lcos / reserve floor / efficiency from the data stream alone.
        self._hw_obs = np.array([
            self.max_batt_capacity / 250.0,
            self.lcos / 1.25,
            self.soc_soft_min,
            (self.batt_efficiency - 0.90) / 0.08,
        ], dtype=np.float32)
        self._day_avg_price = 0.0
        self._tomorrow_solar_cached = np.zeros(1, dtype=np.float32)

        self._n_starts = max(1, len(df) // episode_len)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        n_features = df.shape[1]
        # Observation layout (169 with the 16-col normalized df):
        #   [0:16]    normalized feature row (Load column removed from df)
        #   [16]      current relative load = load_kw / peak_kw
        #   [17]      SoC
        #   [18:114]  DAM_Price lookahead (96, zero-padded past episode end)
        #   [114:130] DAM_Price history (16, zero-padded before episode start)
        #   [130:146] relative-load lookahead (16, zero-padded past episode end)
        #   [146:162] GTI lookahead (16, zero-padded past episode end)
        #   [162]     tomorrow-solar mean GTI (next day's mean, cached at reset)
        #   [163:165] load_cap_ratio, load_grid_ratio (constant per episode)
        #   [165:169] capacity/250, lcos/1.25, soc_soft_min, (efficiency-0.90)/0.08
        obs_size = n_features + 2 + self.PRICE_LOOKAHEAD + self.PRICE_HISTORY + self.LOAD_LOOKAHEAD + self.GTI_LOOKAHEAD + 1 + 2 + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        power_w       = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        energy_kwh_ts = (power_w / 1000.0) / 4
        return max(0.0, energy_kwh_ts)

    DEFAULT_SOC_TARGET = 0.30

    def _calc_target_soc(self, next_outage_h: float, curr_load_kw: float, gti_w_m2: float,
                         hours_until_outage: float = 0.0) -> float:
        """Dynamic target SoC: energy needed to survive the upcoming outage, clipped to [soc_soft_min, soc_soft_max]."""
        if next_outage_h <= 0:
            return self.DEFAULT_SOC_TARGET

        energy_load = curr_load_kw * next_outage_h
        # Only credit solar when the outage starts within 3 h — beyond that current GTI is not
        # representative (midday GTI would underestimate the reserve needed for a night outage).
        if hours_until_outage <= 3.0:
            solar_per_hour    = self._calc_solar_generation_ts(gti_w_m2) * 4
            net_energy_needed = max(0.0, energy_load - solar_per_hour * next_outage_h)
        else:
            net_energy_needed = energy_load
        energy_with_buffer = net_energy_needed * 1.10
        target_soc = energy_with_buffer / self.max_batt_capacity
        target_soc = max(target_soc, self.DEFAULT_SOC_TARGET)
        target_soc = np.clip(target_soc, self.soc_soft_min, self.soc_soft_max)

        return float(target_soc)

    def get_observe(self) -> np.ndarray:
        idx = min(self.curr_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        base = np.append(row.values, [self.rel_load[idx], self.soc])

        price_fwd = np.zeros(self.PRICE_LOOKAHEAD, dtype=np.float32)
        load_vec  = np.zeros(self.LOAD_LOOKAHEAD,  dtype=np.float32)
        gti_vec   = np.zeros(self.GTI_LOOKAHEAD,   dtype=np.float32)

        # Lookaheads are clipped to the episode's day: at inference the df is exactly 96 rows
        # and everything past the end is zero-padded — training must match that distribution.
        ep_end = self.episode_start + self.episode_len
        p = self.df['DAM_Price'].iloc[idx:min(idx + self.PRICE_LOOKAHEAD, ep_end)].values
        l = self.rel_load[idx:min(idx + self.LOAD_LOOKAHEAD, ep_end)]
        g = self.df['Global_tilted_irradiance_instant'].iloc[idx:min(idx + self.GTI_LOOKAHEAD, ep_end)].values

        price_fwd[:len(p)] = p
        load_vec[:len(l)]  = l
        gti_vec[:len(g)]   = g

        price_hist = np.zeros(self.PRICE_HISTORY, dtype=np.float32)
        hist_start = max(self.episode_start, idx - self.PRICE_HISTORY)
        h = self.df['DAM_Price'].iloc[hist_start:idx].values
        price_hist[self.PRICE_HISTORY - len(h):] = h

        load_context = np.array([self._load_cap_ratio, self._load_grid_ratio], dtype=np.float32)

        return np.concatenate([base, price_fwd, price_hist, load_vec, gti_vec,
                               self._tomorrow_solar_cached, load_context, self._hw_obs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc           = float(np.random.uniform(0.0, 1.0))
        day_idx            = int(np.random.randint(0, self._n_starts))
        self.episode_start = day_idx * self.episode_len
        self.curr_step     = self.episode_start

        self._day_avg_price = float(
            self.df_raw['DAM_Price'].iloc[self.episode_start:self.episode_start + self.episode_len].mean()
        ) / 1000
        tomorrow_start = self.episode_start + self.episode_len
        next_day_gti   = self.df['Global_tilted_irradiance_instant'].iloc[tomorrow_start:tomorrow_start + self.episode_len].values
        self._tomorrow_solar_cached = np.array(
            [next_day_gti.mean() if len(next_day_gti) > 0 else 0.0], dtype=np.float32
        )

        return self.get_observe(), {}

    def step(self, action):

        row = self.df_raw.iloc[self.curr_step]

        curr_price         = row['DAM_Price'] / 1000
        buy_price          = curr_price + 3.0
        curr_load_kw       = float(self.load_kw[self.curr_step])
        curr_load_ts       = curr_load_kw / 4
        grid_status        = int(row['Grid'])
        gti                = row['Global_tilted_irradiance_instant']
        hours_until_outage = row['hours_until_outage']
        outage_remaining_h = row['outage_remaining_h']
        next_outage_h      = row['next_outage_duration']

        target_soc = self._calc_target_soc(next_outage_h, curr_load_kw, gti, hours_until_outage)

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

            if grid_status == 0:
                energy_drawn_for_batt = solar_used_for_batt
                actual_chem_in        = solar_used_for_batt * self.batt_efficiency
                grid_needed_for_batt  = 0.0
            else:
                # Demand has first priority on grid capacity; cap battery's grid draw to what remains.
                available_grid_for_batt = max(0.0, self.max_grid_capacity_ts - residual_demand_ts)
                if grid_needed_for_batt > available_grid_for_batt:
                    grid_needed_for_batt  = available_grid_for_batt
                    energy_drawn_for_batt = solar_used_for_batt + grid_needed_for_batt
                    actual_chem_in        = energy_drawn_for_batt * self.batt_efficiency

            self.soc = min(1.0, self.soc + actual_chem_in / self.max_batt_capacity)

            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)
            batt_contribution_ts    = -energy_drawn_for_batt
            actual_batt_energy_abs  = actual_chem_in

        else:
            energy_to_draw = abs(battery_energy_delta)
            chem_needed = energy_to_draw / self.batt_efficiency
            if grid_status == 1:
                max_drawable = max(0.0, (self.soc - self.soc_soft_min) * self.max_batt_capacity)
            else:
                max_drawable = max(0.0, (self.soc - self.soc_hard_min) * self.max_batt_capacity)
            actual_draw    = min(chem_needed, max_drawable)

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
            batt_export_possible  = max(0.0, batt_contribution_ts - residual_demand_ts)

        total_export_possible = remaining_solar_surplus + batt_export_possible

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
            money_earned_ts = abs(actual_grid_ts) * curr_price
        else:
            money_earned_ts = -actual_grid_ts * buy_price

        lcos_cost = self.lcos * actual_batt_energy_abs
        r_lcos    = -2.0 * lcos_cost

        if unmet_load > 0:
            r_unmet = -unmet_load * buy_price * 5

        mismatch   = 0.0
        r_mismatch = 0.0

        if self.soc < self.soc_soft_min:
            r_soc_soft -= 50.0 * ((self.soc_soft_min - self.soc) ** 2)
        if self.soc > self.soc_soft_max:
            r_soc_soft -= 50.0 * ((self.soc - self.soc_soft_max) ** 2)

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

        if battery_energy_delta > 0 and solar_used_for_batt > 0:
            pre_charge_soc = self.soc - actual_chem_in / self.max_batt_capacity
            if pre_charge_soc < target_soc:
                solar_chem_stored = solar_used_for_batt * self.batt_efficiency
                r_soc_target = min(actual_chem_in, solar_chem_stored) * buy_price

        r_waste = 0.0
        if battery_energy_delta < 0 and batt_contribution_ts > 0:
            demand_covered  = min(batt_contribution_ts, residual_demand_ts)
            grid_headroom   = 0.0 if grid_status == 0 else max(0.0, self.max_grid_capacity_ts - remaining_solar_surplus)
            exportable_batt = min(batt_export_possible, grid_headroom)
            wasted          = max(0.0, batt_contribution_ts - demand_covered - exportable_batt)
            r_waste         = -10.0 * self.lcos * wasted

        curtailed_kwh = 0.0
        if grid_status == 1:
            actually_exported = max(0.0, -actual_grid_ts)
            # total curtailment for reporting (solar + battery surplus not exported)
            curtailed_kwh = max(0.0, total_export_possible - actually_exported)
            # r_curtail only fires for SOLAR curtailment — r_waste already penalises wasted battery discharge
            solar_curtailed = max(0.0, remaining_solar_surplus - actually_exported)
            if solar_curtailed > 0.01:
                r_curtail = -solar_curtailed * curr_price * 3.0
        elif grid_status == 0 and remaining_solar_surplus > 0.01:
            # during outage surplus solar is physically lost — track for reporting but don't penalise
            curtailed_kwh = remaining_solar_surplus

        r_price_timing = 0.0
        outage_imminent = grid_status == 1 and 0 < hours_until_outage <= 6.0 and self.soc < target_soc
        if grid_status == 1 and not outage_imminent:
            price_dev = curr_price - self._day_avg_price
            if actual_grid_ts < 0:
                # don't penalise selling surplus SOLAR below average — curtailing is always worse.
                # battery discharge keeps full price-timing signal so the agent learns sell-high timing.
                if not (remaining_solar_surplus > 0.1 and price_dev < 0):
                    r_price_timing = price_dev * abs(actual_grid_ts) * 1.0
            elif actual_grid_ts > 0:
                r_price_timing = -price_dev * actual_grid_ts * 1.0

        if grid_status == 1 and grid_needed_for_batt > 0.01 and solar_gen_ts > 0.1 and not outage_imminent:
            solar_fraction = min(1.0, solar_gen_ts / (solar_gen_ts + grid_needed_for_batt))
            # scale penalty continuously with soc/target — no hard cliff at target_soc
            priority_scale = float(np.clip(self.soc / target_soc, 0.5, 1.0)) if target_soc > 0 else 1.0
            r_solar_priority = -grid_needed_for_batt * solar_fraction * curr_price * 4.0 * priority_scale

        r_eod_soc = 0.0
        step_in_ep = self.curr_step - self.episode_start
        if step_in_ep == 95:
            soc_carried = max(0.0, self.soc - self.DEFAULT_SOC_TARGET)
            # use avg day buy price — midnight buy_price is artificially low and weakens the incentive
            r_eod_soc = soc_carried * self.max_batt_capacity * (self._day_avg_price + 3.0) * 0.15

        reward = (r_market + r_lcos + r_unmet + r_mismatch + r_soc_soft + r_reserve + r_preparation + r_soc_target + r_waste + r_curtail + r_price_timing + r_solar_priority + r_eod_soc) / (self.max_batt_capacity / 100.0)

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
            'battery_kwh':       -batt_contribution_ts,
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
            'reward_eod_soc':        r_eod_soc,
        }

        return observation, reward, terminated, truncated, info

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
 
 
class Environment(gym.Env):
    """
    EMS RL-середовище. Пріоритет покриття навантаження:
        1. Сонячні панелі  — безкоштовно, завжди першими
        2. Батарея         — накопичений заряд
        3. Мережа          — платний імпорт / канал продажу
 
    Два "м'які" обмеження через штрафи (не clip):
        - SoC поза діапазоном 20%–80% → невеликий штраф (агент може порушити якщо треба)
        - SoC нижче динамічного резерву під час/перед аутажем → великий штраф
    """
 
    def __init__(self, df: pd.DataFrame):
        super().__init__()
 
        self.df = df
 
        # ── Батарея ───────────────────────────────────────────────
        self.max_batt_capacity    = 2.0    # кВт·год
        self.max_batt_power       = 1.0    # кВт/год
        self.max_batt_power_ts    = self.max_batt_power / 4   # кВт·год за таймстеп
        self.batt_efficiency      = 0.95
        self.lcos                 = 1.5    # UAH/кВт·год деградації
 
        # М'які межі SoC — агент штрафується за їх порушення,
        # але НЕ обмежується clip(). Може порушити якщо вигідно.
        self.soc_soft_min         = 0.20   # нижче → м'який штраф
        self.soc_soft_max         = 0.80   # вище  → м'який штраф
 
        # ── Мережа ───────────────────────────────────────────────
        self.max_grid_capacity    = 5.0
        self.max_grid_capacity_ts = self.max_grid_capacity / 4
 
        # ── Сонячні панелі ────────────────────────────────────────
        self.solar_peak_power_kw  = 3.0
        self.solar_efficiency     = 0.18
        self.panel_area_m2        = (self.solar_peak_power_kw * 1000) / (1000 * self.solar_efficiency)
 
        # ── Початковий стан ───────────────────────────────────────
        self.soc       = 0.5
        self.curr_step = 0
 
        # ── Простори ─────────────────────────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 1,), dtype=np.float32
        )
 
    # ─────────────────────────────────────────────────────────────
    # Генерація сонця за таймстеп
    # ─────────────────────────────────────────────────────────────
    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        power_w       = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        energy_kwh_ts = (power_w / 1000.0) / 4
        return max(0.0, energy_kwh_ts)
 
    # ─────────────────────────────────────────────────────────────
    # Динамічний цільовий SoC на основі прогнозу відключення
    # ─────────────────────────────────────────────────────────────
    def _calc_target_soc(self, next_outage_h: float, curr_load_kw: float, gti_w_m2: float) -> float:
        """
        Рахує скільки енергії треба запасти щоб пережити наступне відключення.
 
        Логіка:
            energy_needed  = споживання за час відключення
                           - прогнозована генерація сонця за той самий час
            target_soc     = (energy_needed × буфер_10%) / ємність_батареї
 
        Результат завжди в межах [soc_soft_min, soc_soft_max] —
        навіть якщо відключення коротке і треба зовсім мало,
        нижче 20% не опускаємось (абсолютний мінімум безпеки).
        Навіть якщо відключення дуже довге — не більше 80%
        (захист від деградації при повному заряді).
        """
 
        if next_outage_h <= 0:
            # Відключення не прогнозується — тримаємо тільки абсолютний мінімум
            return self.soc_soft_min
 
        # Скільки кВт·год споживе будинок за час відключення
        energy_load = curr_load_kw * next_outage_h   # кВт × год = кВт·год
 
        # Груба оцінка генерації сонця під час відключення:
        # беремо поточний GTI як proxy і множимо на кількість годин.
        # Ділимо на 4 щоб перевести з "за таймстеп" в "за годину", потім × next_outage_h.
        solar_per_hour = self._calc_solar_generation_ts(gti_w_m2) * 4   # кВт·год/год
        energy_solar   = solar_per_hour * next_outage_h                  # кВт·год за відключення
 
        # Чистий дефіцит що треба покрити батареєю
        net_energy_needed = max(0.0, energy_load - energy_solar)
 
        # +10% буфер на непередбачені піки споживання
        energy_with_buffer = net_energy_needed * 1.10
 
        # Переводимо в частку від ємності батареї
        target_soc = energy_with_buffer / self.max_batt_capacity
 
        # Обмежуємо в безпечний діапазон:
        # мінімум = soc_soft_min (20%) — навіть якщо відключення коротке
        # максимум = soc_soft_max (80%) — навіть якщо відключення дуже довге
        target_soc = np.clip(target_soc, self.soc_soft_min, self.soc_soft_max)
 
        return target_soc
 
    # ─────────────────────────────────────────────────────────────
    def get_observe(self) -> np.ndarray:
        row = self.df.iloc[self.curr_step]
        return np.append(row.values, self.soc).astype(np.float32)
 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc       = 0.5
        self.curr_step = 0
        return self.get_observe(), {}
 
    # ─────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
 
        row = self.df.iloc[self.curr_step]
 
        # ── Дані поточного таймстепу ──────────────────────────────
        curr_price         = row['DAM_Price'] / 1000       # UAH/кВт·год
        curr_load_ts       = row['Load'] / 1000 / 4        # кВт·год за 15 хв
        curr_load_kw       = row['Load'] / 1000            # кВт (для розрахунку резерву)
        grid_status        = int(row['Grid'])
        gti                = row['Global_tilted_irradiance_instant']
        hours_until_outage = row['hours_until_outage']
        outage_remaining_h = row['outage_remaining_h']
        next_outage_h      = row['next_outage_duration']   # тривалість НАСТУПНОГО відключення
 
        # ════════════════════════════════════════════════════════
        # БЛОК 0: ДИНАМІЧНИЙ ЦІЛЬОВИЙ SoC
        #
        # Рахуємо ПЕРЕД усіма діями — це та ціль до якої агент
        # повинен прагнути ДО наступного відключення.
        # Використовується в reward для штрафу/бонусу підготовки.
        # ════════════════════════════════════════════════════════
 
        target_soc = self._calc_target_soc(next_outage_h, curr_load_kw, gti)
        # Приклад: відключення 4 год, споживання 0.3 кВт, сонця мало →
        # energy_needed = 0.3×4 = 1.2 кВт·год × 1.1 = 1.32 кВт·год
        # target_soc = 1.32 / 2.0 = 0.66 (66%)
 
        # ════════════════════════════════════════════════════════
        # БЛОК 1: СОНЯЧНА ГЕНЕРАЦІЯ
        # ════════════════════════════════════════════════════════
 
        solar_gen_ts = self._calc_solar_generation_ts(gti)
 
        demand_after_solar = curr_load_ts - solar_gen_ts
        solar_surplus_ts   = max(0.0, -demand_after_solar)  # надлишок сонця
        residual_demand_ts = max(0.0,  demand_after_solar)  # попит після сонця
 
        # ════════════════════════════════════════════════════════
        # БЛОК 2: БАТАРЕЯ
        # ════════════════════════════════════════════════════════
 
        battery_action       = float(action[0])
        battery_energy_delta = battery_action * self.max_batt_power_ts
 
        if battery_energy_delta > 0:
            # ── ЗАРЯДКА ───────────────────────────────────────────
            chem_to_add    = battery_energy_delta * self.batt_efficiency
            room_in_batt   = (1.0 - self.soc) * self.max_batt_capacity
            actual_chem_in = min(chem_to_add, room_in_batt)
 
            energy_drawn_for_batt = actual_chem_in / self.batt_efficiency
            solar_used_for_batt   = min(solar_surplus_ts, energy_drawn_for_batt)
            grid_needed_for_batt  = max(0.0, energy_drawn_for_batt - solar_used_for_batt)
 
            self.soc = min(1.0, self.soc + actual_chem_in / self.max_batt_capacity)
 
            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)
            batt_contribution_ts    = -energy_drawn_for_batt
            actual_batt_energy_abs  = actual_chem_in
 
        else:
            # ── РОЗРЯД ────────────────────────────────────────────
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
                if grid_power_ts < 0:
                    actual_grid_ts = max(
                        -solar_export_possible,
                        -self.max_grid_capacity_ts,
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
 
        # ── 5.1 Ринковий P&L ─────────────────────────────────────
        # actual_grid_ts > 0 = купуємо (мінус), < 0 = продаємо (плюс)
        reward -= actual_grid_ts * curr_price
 
        # ── 5.2 Деградація батареї (LCOS) ────────────────────────
        lcos_cost = self.lcos * actual_batt_energy_abs
        reward -= lcos_cost
 
        # ── 5.3 Непокрите навантаження ────────────────────────────
        # 50 UAH >> ринкова ціна → агент завжди покриє навантаження
        if unmet_load > 0:
            reward -= unmet_load * 50.0
 
        # ── 5.4 Штраф за нереалістичну дію (mismatch) ────────────
        mismatch = abs(grid_power_ts - actual_grid_ts)
        reward -= 2.0 * mismatch
 
        # ── 5.5 М'який штраф за SoC поза діапазоном 20%–80% ──────
        #
        # Квадратичний штраф — чим далі від межі тим боліше,
        # але загалом маленький щоб не блокувати агента.
        #
        # Чому квадратичний а не лінійний?
        # Лінійний дає однаковий "тиск" і при SoC=19% і при SoC=5%.
        # Квадратичний сильніше карає екстремальні значення (0%, 100%)
        # і майже не помітний при незначному порушенні (18%, 82%).
        # Це саме та поведінка що нам потрібна — "бажано але не заборонено".
        #
        # Коефіцієнт 3.0 підібраний так що при SoC=0%:
        # штраф = 3 × 0.20² = 0.12 UAH/крок — менше ніж LCOS за мінімальну дію.
        # Агент точно не буде ігнорувати арбітраж заради уникнення цього штрафу.
 
        if self.soc < self.soc_soft_min:
            soc_below = self.soc_soft_min - self.soc   # наскільки нижче 20%
            reward -= 3.0 * (soc_below ** 2)
 
        if self.soc > self.soc_soft_max:
            soc_above = self.soc - self.soc_soft_max   # наскільки вище 80%
            reward -= 3.0 * (soc_above ** 2)
 
        # ── 5.6 Штраф за недостатній резерв під час відключення ───
        #
        # target_soc — динамічна ціль розрахована під КОНКРЕТНЕ відключення.
        # Якщо SoC нижче цілі під час відключення — реальний ризик що
        # батарея скінчиться раніше ніж прийде світло.
        #
        # time_weight = log1p(remaining) щоб штраф зростав з часом
        # але не вибухав при дуже довгих відключеннях (6+ год зимою).
 
        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, target_soc - self.soc)
            if soc_deficit > 0:
                time_weight = np.log1p(outage_remaining_h)
                reward -= 30.0 * soc_deficit * time_weight
 
        # ── 5.7 Бонус за підготовку ДО відключення ───────────────
        #
        # Спрацьовує тільки в передаутажному вікні ≤3 год.
        # Чому не завжди? Якщо нагороджувати за заряд весь час —
        # агент тримав би батарею на 80% постійно і не робив арбітраж.
        # А так він навчиться: "до відключення зарядись, між ними — торгуй".
        #
        # soc_ready = min(soc, target_soc) — враховуємо тільки ту частину
        # SoC що реально "призначена" під резерв, не весь заряд.
 
        if grid_status == 1 and 0 < hours_until_outage <= 3.0:
            urgency   = np.exp(-0.5 * hours_until_outage)
            soc_ready = min(self.soc, target_soc)
            reward   += 5.0 * urgency * soc_ready
 
        # ════════════════════════════════════════════════════════
        # ЗАВЕРШЕННЯ КРОКУ
        # ════════════════════════════════════════════════════════
 
        self.curr_step += 1
        terminated  = self.curr_step >= len(self.df) - 1
        truncated   = False
        observation = self.get_observe()
 
        info = {
            "soc":                  self.soc,
            "target_soc":           target_soc,
            "reward":               reward,
            "solar_gen_ts_kwh":     solar_gen_ts,
            "solar_surplus_kwh":    solar_export_possible,
            "residual_demand_kwh":  residual_demand_ts,
            "net_demand_kwh":       net_demand_after_batt,
            "unmet_load_kwh":       unmet_load,
            "actual_grid_kwh":      actual_grid_ts,
            "lcos_cost":            lcos_cost,
            "mismatch":             mismatch,
        }
 
        return observation, reward, terminated, truncated, info
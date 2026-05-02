import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


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

    # ──────────────────────────────────────────────────────────────
    # Ініціалізація
    # ──────────────────────────────────────────────────────────────

    def __init__(self, df: pd.DataFrame):

        super().__init__()

        self.df = df

        # ── Параметри батареї ──────────────────────────────────────
        self.max_batt_capacity    = 2.0   # кВт·год — загальна ємність
        self.max_batt_power       = 1.0   # кВт     — макс. потужність заряду/розряду за годину
        self.max_batt_power_ts    = self.max_batt_power / 4   # кВт·год за таймстеп (15 хв)
        self.batt_efficiency      = 0.95  # ККД інвертора батареї (однонаправлений)
        self.lcos                 = 1.5   # UAH/кВт·год — вартість деградації батареї
        self.min_soc_reserve      = 0.20  # 20% — мінімальний SoC під час відключення

        # ── Параметри мережі ──────────────────────────────────────
        self.max_grid_capacity    = 5.0   # кВт     — макс. потужність обміну з мережею
        self.max_grid_capacity_ts = self.max_grid_capacity / 4  # кВт·год за таймстеп

        # ── Параметри сонячних панелей ────────────────────────────
        # peak_power — номінальна потужність при GTI=1000 Вт/м² і STC-температурі
        self.solar_peak_power_kw  = 3.0   # кВт (наприклад, 10 × 300 Вт панелей)
        self.solar_efficiency     = 0.18  # ККД панелей (типово 17–20% для полікристалу)
        self.panel_area_m2        = self.solar_peak_power_kw * 1000 / (
            1000 * self.solar_efficiency
        )
        # Формула площі: P_peak [Вт] = GTI_stc [Вт/м²] × η × Area [м²]
        # Звідси: Area = P_peak / (GTI_stc × η)

        # ── Початковий стан ───────────────────────────────────────
        self.soc      = 0.5
        self.curr_step = 0

        # ── Простори дій та спостережень ─────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        n_features = df.shape[1]  # кількість колонок датасету
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 1,), dtype=np.float32
            # +1 — це SoC, який додається динамічно в get_observe()
        )

    # ──────────────────────────────────────────────────────────────
    # Допоміжні методи
    # ──────────────────────────────────────────────────────────────

    def _calc_solar_generation_ts(self, gti_w_m2: float) -> float:
        """
        Розраховує генерацію панелей за один таймстеп (15 хв) в кВт·год.

        Формула: P [Вт] = GTI [Вт/м²] × η × Area [м²]
        Потім переводимо в кВт·год за 15 хв: E = P[кВт] × (15/60)

        GTI (Global Tilted Irradiance) — це вже скоригована на кут нахилу
        і азимут панелі інсоляція, тобто саме те що реально падає на панель.
        """
        power_w  = gti_w_m2 * self.solar_efficiency * self.panel_area_m2
        power_kw = power_w / 1000.0
        energy_kwh_ts = power_kw / 4  # 15 хв = 1/4 години
        return max(0.0, energy_kwh_ts)  # не може бути від'ємною

    def get_observe(self) -> np.ndarray:
        row = self.df.iloc[self.curr_step]
        observation = np.append(row.values, self.soc).astype(np.float32)
        return observation

    # ──────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc       = 0.5
        self.curr_step = 0
        observation    = self.get_observe()
        return observation, {}

    # ──────────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────────

    def step(self, action):

        row = self.df.iloc[self.curr_step]

        # ── Зчитуємо дані поточного таймстепу ────────────────────
        curr_price           = row['DAM_Price'] / 1000    # UAH/МВт·год → UAH/кВт·год
        curr_load_ts         = row['Load'] / 1000 / 4     # Вт → кВт·год за 15 хв
        grid_status          = int(row['Grid'])            # 1 = є мережа, 0 = відключення
        gti                  = row['Global_tilted_irradiance_instant']  # Вт/м²
        hours_until_outage   = row['hours_until_outage']  # год до наступного відключення
        outage_remaining_h   = row['outage_remaining_h']  # год що залишилось у відключенні

        # ════════════════════════════════════════════════════════
        # БЛОК 1: СОНЯЧНА ГЕНЕРАЦІЯ
        # Сонце — це завжди пріоритет №1: безкоштовна енергія,
        # яку ми беремо першою, ще до батареї та мережі.
        # ════════════════════════════════════════════════════════

        solar_gen_ts = self._calc_solar_generation_ts(gti)
        # solar_gen_ts — скільки кВт·год згенерували панелі за цей таймстеп

        # Навантаження після сонця: якщо генерація покриває споживання —
        # demand_after_solar буде 0 або від'ємним (надлишок).
        # Від'ємне значення = є зайва сонячна енергія яку треба кудись дівати.
        demand_after_solar = curr_load_ts - solar_gen_ts
        solar_surplus_ts   = max(0.0, -demand_after_solar)   # надлишок сонця (кВт·год)
        residual_demand_ts = max(0.0,  demand_after_solar)   # попит що залишився після сонця

        # ════════════════════════════════════════════════════════
        # БЛОК 2: БАТАРЕЯ
        # Агент вирішує скільки заряджати/розряджати батарею.
        # Але ми обмежуємо дію фізичними можливостями і поточним SoC.
        # ════════════════════════════════════════════════════════

        battery_action       = float(action[0])
        battery_power_delta  = battery_action * self.max_batt_power          # кВт за годину
        battery_energy_delta = battery_power_delta / 4                        # кВт·год за таймстеп

        if battery_energy_delta > 0:
            # ── ЗАРЯДКА батареї ───────────────────────────────────
            # Спочатку намагаємося зарядити надлишком сонця,
            # решту — з мережі (якщо агент так вирішив).

            # Скільки хімічної енергії реально ввійде в батарею
            # (враховуємо ККД інвертора: частина енергії губиться у тепло)
            chem_to_add    = battery_energy_delta * self.batt_efficiency

            # Скільки є вільного місця в батареї
            room_in_batt   = (1.0 - self.soc) * self.max_batt_capacity

            # Реальна хімічна енергія що увійде (не більше ніж є місце)
            actual_chem_in = min(chem_to_add, room_in_batt)

            # Скільки треба взяти ззовні (з мережі або сонця) з урахуванням ККД
            energy_drawn_for_batt = actual_chem_in / self.batt_efficiency

            # Пріоритет: спочатку беремо надлишок сонця, решта — з мережі
            solar_used_for_batt   = min(solar_surplus_ts, energy_drawn_for_batt)
            grid_needed_for_batt  = max(0.0, energy_drawn_for_batt - solar_used_for_batt)

            # Оновлюємо SoC
            self.soc = min(1.0, self.soc + actual_chem_in / self.max_batt_capacity)

            # Скільки сонячного надлишку лишилось після заряджання батареї
            remaining_solar_surplus = max(0.0, solar_surplus_ts - solar_used_for_batt)

            # Реальна зміна енергії в системі з боку батареї (витрата)
            batt_contribution_ts    = -energy_drawn_for_batt  # батарея "споживає"
            actual_batt_energy_abs  = actual_chem_in          # для LCOS: скільки кВт·год пройшло

        else:
            # ── РОЗРЯД батареї ────────────────────────────────────
            # Батарея допомагає покрити залишковий попит після сонця.

            # Скільки хочемо отримати з батареї (без ККД)
            energy_to_draw = abs(battery_energy_delta)

            # Фізичне обмеження: не більше ніж є в батареї
            max_drawable   = self.soc * self.max_batt_capacity
            actual_draw    = min(energy_to_draw, max_drawable)

            # На виході батареї — з урахуванням ккд (частина губиться)
            batt_output_ts = actual_draw * self.batt_efficiency

            # SoC зменшується на те що взяли з хімії
            self.soc = max(0.0, self.soc - actual_draw / self.max_batt_capacity)

            # Батарея "дає" енергію в систему
            batt_contribution_ts   = batt_output_ts    # позитивне = дає енергію
            grid_needed_for_batt   = 0.0
            remaining_solar_surplus = solar_surplus_ts # батарея не бере сонце при розряді
            actual_batt_energy_abs = actual_draw       # для LCOS

        # ════════════════════════════════════════════════════════
        # БЛОК 3: БАЛАНС ПІСЛЯ СОНЦЯ І БАТАРЕЇ
        # Рахуємо скільки ще треба/лишилось після роботи панелей і батареї.
        # ════════════════════════════════════════════════════════

        if battery_energy_delta >= 0:
            # При зарядці батареї: залишковий попит = початковий попит після сонця
            # плюс те що йде на заряд батареї з мережі
            net_demand_after_batt = residual_demand_ts + grid_needed_for_batt
            solar_export_possible = remaining_solar_surplus
        else:
            # При розряді: батарея допомогла покрити попит
            net_demand_after_batt = max(0.0, residual_demand_ts - batt_contribution_ts)
            solar_export_possible = remaining_solar_surplus

        # ════════════════════════════════════════════════════════
        # БЛОК 4: МЕРЕЖА
        # Мережа — це останній ресурс або канал продажу.
        # action[1]: +1 = продати в мережу (експорт), -1 = купити з мережі (імпорт)
        # Знак інвертований: позитивна дія агента = продаж = прибуток.
        # ════════════════════════════════════════════════════════

        grid_action     = float(action[1])
        # Від'ємний знак: +1 від агента → від'ємний grid_power → продаж
        grid_power_ts   = -grid_action * self.max_grid_capacity_ts

        unmet_load      = 0.0  # кВт·год що не вдалося покрити (штраф)
        actual_grid_ts  = 0.0  # реальний обмін з мережею за таймстеп

        if grid_status == 1:
            # ── Мережа є ─────────────────────────────────────────
            # Реальний обмін — це те що реально потрібно системі,
            # але не більше за фізичне обмеження мережі.
            #
            # Позитивне actual_grid_ts = купуємо (імпорт, платимо)
            # Від'ємне actual_grid_ts  = продаємо (експорт, отримуємо)

            if net_demand_after_batt > 0:
                # Системі не вистачає енергії — докупляємо з мережі
                actual_grid_ts = min(net_demand_after_batt, self.max_grid_capacity_ts)
                if actual_grid_ts < net_demand_after_batt:
                    # Навіть мережа не може покрити — це критична ситуація
                    unmet_load = net_demand_after_batt - actual_grid_ts
            else:
                # Є надлишок (від сонця або батареї) — можна продати в мережу
                # Але тільки якщо агент дійсно хоче продавати (grid_power_ts < 0)
                exportable = solar_export_possible
                if grid_power_ts < 0:
                    # Агент хоче продавати — продаємо надлишок
                    actual_grid_ts = max(
                        -exportable,                    # не більше ніж є надлишок
                        -self.max_grid_capacity_ts,     # не більше фізичного ліміту
                        grid_power_ts                   # не більше ніж хоче агент
                    )
                else:
                    actual_grid_ts = 0.0  # надлишок є, але агент не продає

        else:
            # ── Мережа відсутня (відключення) ─────────────────────
            actual_grid_ts = 0.0
            if net_demand_after_batt > 0:
                # Навантаження не покрите — критичний штраф
                unmet_load = net_demand_after_batt

        # ════════════════════════════════════════════════════════
        # БЛОК 5: ФУНКЦІЯ ВИНАГОРОДИ
        #
        # reward = (дохід від продажу) - (витрати на покупку)
        #        - (деградація батареї)
        #        - (штраф за непокрите навантаження)
        #        - (штраф за неузгоджену дію)
        #        - (штраф за відключення без резерву)
        #        + (бонус за підготовку до відключення)
        # ════════════════════════════════════════════════════════

        reward = 0.0

        # ── 5.1 Ринковий P&L ─────────────────────────────────────
        # actual_grid_ts > 0 = купуємо (витрата), < 0 = продаємо (дохід)
        reward -= actual_grid_ts * curr_price
        # Приклад: купили 0.5 кВт·год по 3 UAH → reward -= 1.5 UAH
        # Приклад: продали 0.3 кВт·год по 3 UAH → reward += 0.9 UAH

        # ── 5.2 Вартість деградації батареї (LCOS) ────────────────
        # Кожен кВт·год що пройшов через батарею коштує LCOS UAH
        # Це не штраф — це реальна вартість зносу акумулятора
        lcos_cost = self.lcos * actual_batt_energy_abs
        reward -= lcos_cost

        # ── 5.3 Штраф за непокрите навантаження ──────────────────
        # Найсерйозніший штраф — навантаження будинку не покрите.
        # 50 UAH/кВт·год >> curr_price (~3–15 UAH) → агент завжди
        # буде намагатися покрити навантаження навіть ціною збитків.
        if unmet_load > 0:
            reward -= unmet_load * 50.0

        # ── 5.4 Штраф за "брехливу" дію агента ───────────────────
        # Якщо агент сказав "хочу взяти X з мережі" але реально взяв Y —
        # штрафуємо різницю. Це навчає агента діяти реалістично.
        mismatch = abs(grid_power_ts - actual_grid_ts)
        reward -= 2.0 * mismatch

        # ── 5.5 Штраф за відключення без резерву ─────────────────
        # Якщо зараз відключення і SoC нижче мінімального резерву —
        # штраф тим більший, чим більше часу відключення ще попереду.
        if grid_status == 0 and outage_remaining_h > 0:
            soc_deficit = max(0.0, self.min_soc_reserve - self.soc)
            if soc_deficit > 0:
                # log1p щоб штраф зростав повільніше при дуже довгих відключеннях
                time_weight = np.log1p(outage_remaining_h)
                reward -= 30.0 * soc_deficit * time_weight

        # ── 5.6 Бонус за підготовку до відключення ───────────────
        # Якщо відключення наближається (≤3 год) і ми вже маємо
        # SoC вище мінімального резерву — бонус за передбачливість.
        if grid_status == 1 and 0 < hours_until_outage <= 3.0:
            # urgency: чим ближче — тим сильніший сигнал (exp decay)
            urgency     = np.exp(-0.5 * hours_until_outage)
            soc_surplus = max(0.0, self.soc - self.min_soc_reserve)
            reward += 5.0 * urgency * soc_surplus

        # ════════════════════════════════════════════════════════
        # ЗАВЕРШЕННЯ КРОКУ
        # ════════════════════════════════════════════════════════

        self.curr_step += 1
        terminated  = self.curr_step >= len(self.df) - 1
        truncated   = False
        observation = self.get_observe()

        info = {
            "soc":                  self.soc,
            "reward":               reward,
            "solar_gen_ts_kwh":     solar_gen_ts,           # генерація сонця за таймстеп
            "solar_surplus_kwh":    solar_export_possible,  # надлишок сонця
            "residual_demand_kwh":  residual_demand_ts,     # попит після сонця
            "net_demand_kwh":       net_demand_after_batt,  # попит після сонця + батареї
            "unmet_load_kwh":       unmet_load,             # непокрите навантаження
            "actual_grid_kwh":      actual_grid_ts,         # реальний обмін з мережею
            "lcos_cost":            lcos_cost,              # вартість деградації за крок
            "mismatch":             mismatch,               # різниця між дією і реальністю
        }

        return observation, reward, terminated, truncated, info

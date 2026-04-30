import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd





class Envoriment(gym.Env):

    def __init__(self, df):

        super().__init__()

        self.df = df

        self.max_batt_capacity = 2.0

        #Це скільки батарея пропускає через себе за годину і за таймстеп
        self.max_batt_power = 1
        self.max_batt_power_ts = self.max_batt_power / 4

        self.soc = 0.5

        self.curr_batt_capacity = self.max_batt_capacity * self.soc


        #Це скільки мережа може пропустити через себе за годину і за таймстеп
        self.max_grid_capacity = 5.0
        self.max_grid_capacity_ts = self.max_grid_capacity / 4

        self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (2, ), dtype = np.float32)

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (31, ), dtype = np.float32)

        self.curr_step = 0
    
    def get_observe(self):
        
        row = self.df.iloc[self.curr_step]

        observation = np.append(row, self.soc).astype(np.float32)

        return observation
    
    def reset(self, seed= None, options=None):

        super().reset(seed=seed)

        self.soc = 0.5

        self.curr_step = 0

        observation = self.get_observe()

        info = {}

        return observation, info
    
    def step(self, action):

        row = self.df.iloc[self.curr_step]

        #ціна за кіловат годину
        curr_price = row['DAM_Price'] / 1000
        
        #Нагрузка підприємства за годину і за таймстеп
        curr_load = row['Load']
        curr_load_ts = curr_load / 4

        grid_status = row['Grid']
        

        #Блок обрахунку змін для батареї
        battery_power_delta = action[0] * self.max_batt_power
        battery_power_delta_ts = battery_power_delta / 4

        if battery_power_delta > 0:
            # Скільки енергії агент хоче взяти ззовні
            energy_needed_from_outside = battery_power_delta_ts

            # Скільки ХІМІЇ він хоче засунути в батарею (після втрат інвертора)
            chem_to_add = energy_needed_from_outside * 0.95

            # Скільки є вільного місця
            max_to_charge = (1.0 - self.soc) * self.max_batt_capacity

            # Скільки РЕАЛЬНО енегрії буде додано 
            max_can_charge = min(chem_to_add, max_to_charge)

            # Оце вже скільки треба взяти з мережі З ВРАХУВАННЯМ ККД
            actual_energy_needed_from_outside = max_can_charge / 0.95

            # Це наскільки енергія змінилась в системі будинку (+)
            actual_energy_delta_ts = actual_energy_needed_from_outside

            # РЕАЛЬНИЙ заряд батареї після заряджання
            self.soc = self.soc + max_can_charge / self.max_batt_capacity

        else:
            #Скільки треба віддати(без ККД)
            energy_to_give = battery_power_delta_ts

            #Рахує чи ми взагалі можемо стільки віддати скільки треба
            max_to_give = min(self.max_batt_capacity * self.soc, abs(energy_to_give))

            #Скільки РЕАЛЬНО батарея віддасть в мережу
            actual_energy_to_give = -max_to_give 
            actual_energy_to_give_ts = min(actual_energy_to_give, self.max_batt_power_ts)

            #Наскільки енергія змінилась в Батареї
            actual_energy_delta_ts = -(max_to_give * 0.95)


            #РЕАЛЬНИЙ заряд батареї пілся розряд
            self.soc = self.soc - abs(max_to_give) / self.max_batt_capacity



        #Перед цим блоком треба чотко розуміти різницю між -1 і +1 

        #-1 для батареї це продаж +1 - зарядка і це впливає на внутрішній
        #баланс підприємства і на функцію винагороди, коли ми з підприємства
        #віднімаємо actual_energy_delta_ts в випадку +1 це значить що ми з підприємства
        #забираємо енергію і тому там actual_energy_delta_ts ДОДАТНЄ
        #коли ми віднімаємо actual_energy_delta_ts в випадку -1 це значить що ми
        #даємо енергію підприємству і тому actual_energy_delta_ts ВІДЄМНЕ

        

        reward = 0
    
        #1 пріоритет - підриємство, рахується за таймстеп
        net_demand_ts = curr_load_ts + actual_energy_delta_ts

        #скільки не хватає щоб задовільнити нагрузку підприємства
        unmet_load = 0



        #Що ми делаєм з мережею за таймстеп
        grid_power_delta = -action[1] * self.max_grid_capacity
        grid_power_delta_ts = grid_power_delta / 4

        
        if grid_status == 1:
            actual_grid_energy_ts = np.clip(net_demand_ts, -self.max_grid_capacity_ts, self.max_grid_capacity_ts)

            if actual_grid_energy_ts < net_demand_ts:
                unmet_load = net_demand_ts - actual_grid_energy_ts

        else:
            actual_grid_energy_ts = 0
            if net_demand_ts > 0:
                unmet_load = net_demand_ts



        reward -= actual_grid_energy_ts * curr_price

        if unmet_load > 0:
            reward -= unmet_load * 50

        mismatch = abs(grid_power_delta_ts - actual_grid_energy_ts)

        reward -= 2*mismatch

        self.curr_step += 1

        terminated = self.curr_step >= len(self.df) - 1
        truncated = False

        observation = self.get_observe()

        info = {
            "soc": self.soc,
            "reward": reward,
            "unmet_load": unmet_load,
            "mismatch": mismatch,
            "actual_grid_energy_ts": actual_grid_energy_ts
        }

        return observation, reward, terminated, truncated, info

        





        


        

        





    
    
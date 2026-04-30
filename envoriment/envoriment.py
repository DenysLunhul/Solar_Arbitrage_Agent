import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd




class Envoriment(gym.Env):

    def __init__(self, df):

        super().init()

        self.df = df

        self.max_batt_capacity = 2.0

        self.max_batt_power = 1

        self.soc = 0.5

        self.curr_batt_capacity = self.max_batt_capacity * self.soc

        self.max_grid_capacity = 5.0

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

        curr_price = row['DAM_Price'] / 4000

        curr_load = row['Load'] / 4

        grid_status = row['Grid']
        

        #Блок обрахунку змін для батареї
        battery_power_delta = action[0] * self.max_batt_power / 4 

        if(battery_power_delta > 0):
            #Скільки енергії треба ззвоні(без врахування ККД)
            energy_needed_from_outside = battery_power_delta
            #Скільки взагалі максимально ми можемо втиснути в батарею
            max_to_charge = (1.0 - self.soc) * self.max_batt_capacity
            #Це скільки в теорії можна засунути в батарею
            max_can_charge = min(self.max_batt_capacity, max_to_charge)
            #Оце вже скільки треба взяти з мережі З ВРАХУВАННЯМ ККД
            energy_needed_from_outside = max_can_charge / 0.95
            #Це наскільки енергія змінилась в Батареї
            actual_energy_delta = energy_needed_from_outside
            #РЕАЛЬНИЙ заряд батареї після заряджання
            self.soc = self.soc + max_can_charge / self.max_batt_capacity

        elif(battery_power_delta < 0):
            #Скільки треба віддати(без ККД)
            energy_to_give = battery_power_delta
            #Рахує чи ми взагалі можемо стільки віддати скільки треба
            max_to_give = min(self.max_batt_capacity * self.soc, abs(energy_to_give))
            #Скільки РЕАЛЬНО батарея віддасть в мережу
            actual_energy_to_give = max_to_give * 0.95
            #Наскільки енергія змінилась в Батареї
            actual_energy_delta = -max_to_give
            #РЕАЛЬНИЙ заряд батареї пілся розряд
            self.soc = self.soc - abs(max_to_give) / self.max_batt_capacity



    

        #Блок обрахунку змін для мережі
        grid_delta = action[1] * self.max_grid_capacity / 4
        





    
    
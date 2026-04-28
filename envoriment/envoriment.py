import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd



class Envoriment(gym.Env):

    def init(self, df):

        super().init()

        self.df = df

        self.max_batt_capacity = 2.0

        self.max_batt_power = 1

        self.soc = 0.5

        self.curr_batt_capacity = self.max_batt_capacity * self.soc

        self.action_space = spaces.Box(low = -1.0, high= 1.0, shape = (1, ), dtype = np.float32)

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (31, ), dtype = np.float32)

        self.curr_step = 0
    
    def get_observe(self):
        
        row = self.df.iloc[self.curr_step].drop(columns = ['Date']).values

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

        curr_price = row['DAM_Price']

        curr_load = row['Load']

        grid_status = row['Grid']

        power = action[0] * self.max_batt_power

        energy_change = power / 4

        if(energy_change > 0):
            energy_change *= 0.95
        else:
            energy_change /= 0.95

        oldSoc = self.soc

        self.soc = np.clip(self.soc + (energy_change / self.max_batt_capacity), 0.0, 1.0)

        actual_energy_delta = (self.soc - oldSoc) * self.max_batt_capacity
# An implementation of a simple Energy mix problem using the OpenAI Gym environment
# Author : Sharath Ram Kumar
# Date   : 27 Jan 2022

import gym
import numpy as np
#from matplotlib.pyplot import plt

class EnergyEnv(gym.Env):
    def __init__(self):
        # The observation space is a set of 3 floating point numbers
        # [energy_demand, solar_generation, coal_generation]
        # In arbitrary units (max value is 100)
        obs_low = np.array([0.0, 0.0, 0.0], dtype = np.float32)
        obs_high = np.array([100.0, 100.0, 100.0], dtype = np.float32)
        
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = gym.spaces.Box(high = obs_high, low = obs_low)
        
        self.state = np.array([0.0, 0.0, 50.0])
        self.action = None
        self.time_of_day = 0
        
        self.energy_curves = self.create_energy_curves()
        
    def step(self, action):
        # Calculate the reward
        self.last_action = action
        reward = self.calculate_reward(action)
        
        # Update the state
        self.state[0] = np.abs(np.random.normal(self.energy_curves[0][self.time_of_day], 0.5))
        self.state[1] = np.abs(np.random.normal(self.energy_curves[1][self.time_of_day], 0.5))
        
        # Update the time of day
        if self.time_of_day == 23:
            self.time_of_day = 0
        else:
            self.time_of_day += 1
        
        done = False 
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.array([0.0, 0.0, 50.0])
        self.action = 0
        self.time_of_day = 0
        
        self.energy_curves = self.create_energy_curves()
        
        return self.state
    
    def create_energy_curves(self):
        # Create a random demand curve representing 24 data points for 24 hours
        # Base demand is always 20 units, with peaks occuring in the evening at 6 PM
        demand_curve = np.array([20.0] * 5 + [22.0, 24.0, 26.0, 29.0, 32.0] + [34.0] * 8 + [36.0, 40.0, 35.0, 30.0, 23.0, 20.0], dtype = np.float32) + np.random.normal(0, 0.5, 24)
        
        # Around 35 units of solar energy is available at noon
        solar_curve = np.array([0.0] * 6 + [2.0, 4.0, 8.0, 15.0, 25.0, 35.0, 33.0, 30.0, 25.0, 15.0, 8.0, 2.0, 1.0] + [0.0] * 5)
        return (demand_curve, solar_curve)
    
    def render(self, mode='human'):
        # Plot the instantaneous energy demand and the policy taken by the agent
        print(f"Demand:{self.state[0]}, Time:{self.time_of_day}, Action:{self.last_action}")
    
    def calculate_reward(self, action):
        reward = 0
        # The action is a discrete set of 11 items
        # Action 0 translates to 0% solar, 100% coal
        # Action 1 translates to 10% solar, 90% coal
        # etc..
        self.solar_used = min((action/10.0) * self.state[0], self.state[1]) 
        self.coal_used = min((1-(action/10.0)) * self.state[0], self.state[2])
        #print(f"Solar Used = {solar_used}, Coal Used = {coal_used}")
        
        # If the energy demand is met, then there is a reward
        balance = self.state[0] - (self.solar_used + self.coal_used)
        if balance <= 0:
            reward += 15
        
        # Any energy surplus or deficit is penalized
        reward -= 2 * abs(balance)
        
        # Using coal is penalized slightly to represent green initiatives
        reward -= 0.5 * self.coal_used
        
        #print(f"Demand = {self.state[0]}, Solar Used = {solar_used}, Coal Used = {coal_used}, Reward = {reward}")
        return reward
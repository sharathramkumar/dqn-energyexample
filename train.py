'''
A short programming project demonstrating the use of a DQN in a simple energy optimization problem

Author : Sharath Ram Kumar
Date   : 28/01/2022
'''

from gym_energy.envs.EnergyEnv import EnergyEnv

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Activation, Flatten
from keras.optimizers import Adam

import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import matplotlib.pyplot as plt
#%%
env = EnergyEnv()
env.reset()

print(f"Action Space : {env.action_space}\nObservation Space : {env.observation_space}")
#%%
def create_model(input_shape, output_shape):
    '''
    The model accepts a vector of 3 parameters representing the energy demand, solar availability and coal energy availability
    It returns one of 11 output choices representing the action to be taken
    '''
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    print(model.summary())
    return model

# Create and train the model using a DQN Agent
model = create_model(env.observation_space.shape, env.action_space.n)
policy = EpsGreedyQPolicy()
memory = SequentialMemory(50000, window_length = 1)
dqn = DQNAgent(model = model, nb_actions = env.action_space.n, memory = memory, nb_steps_warmup = 100, target_model_update = 1e-2, policy = policy)
dqn.compile(Adam(lr=0.01), metrics=['mae'])

hh = dqn.fit(env, nb_steps=20000, verbose=2)
dqn.test(env, nb_episodes=5, nb_max_episode_steps = 240, visualize = True)

#%% Evaluate its performance
pred_model = Sequential()
pred_model.add(InputLayer(input_shape = (3,)))
pred_model.add(Dense(16))
pred_model.add(Activation('relu'))
pred_model.add(Dense(env.action_space.n))
pred_model.add(Activation('linear'))

pred_model.set_weights(dqn.model.get_weights())
#%% Generate plots to evaluate the model's performance
env = EnergyEnv()
state, reward, done, info = env.step(env.action_space.sample())
action = np.argmax(pred_model.predict(state.reshape(1, -1)))
demands = []
solar_available = []
actions = []
solars = []
coals = []
rewards = []
for ts in range(10000):
    state, reward, done, info = env.step(action)
    demands += [state[0]]
    solar_available += [state[1]]
    actions += [action]
    rewards += [reward]
    solars += [env.solar_used]
    coals += [env.coal_used]
    #action = env.action_space.sample()
    action = np.argmax(pred_model.predict(state.reshape(1, -1)))
#%% Plot the results
fig, ax = plt.subplots(5,1, figsize = (6,8))

ax[0].plot(demands[2000:2072])
ax[0].set(title = 'Hourly Electric Energy Demand')

ax[1].plot(solar_available[2000:2072])
ax[1].set(title = 'Solar Energy Available')

ax[2].plot(actions[2000:2072])
ax[2].set(title = 'Action Policy Taken by the Agent')

ax[3].plot(solars[2000:2072])
ax[3].set(title = 'Solar Energy Used')

ax[4].plot(rewards[2000:2072])
ax[4].set(title = 'Reward Obtained by the Agent')

plt.tight_layout()
#plt.savefig('plots\\trained_agent.png')
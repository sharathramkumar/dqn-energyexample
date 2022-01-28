This is a short programming project demonstrating the following

1. Creation of a custom OpenAI Gym environment for a simple energy balance problem
2. Creation and training of a DQN Reinforcement Learning agent that learns the optimal strategy to solve the problem
3. Comparison of the trained agent against a random policy maker

The custom gym environment (from gym_energy.envs.EnergyEnv import EnergyEnv) implements the following problem:

Given the hourly energy demand and solar energy availability for a hypothetical city, the agent is required to choose the policy which ensures that the energy demand is met while minimizing the use of coal energy.

The states are described as a set of 3 floating point numbers:
- Current Energy Demand
- Available Solar Energy
- Available Coal Energy
The state is obtained as a function of the time of day every hour, by imposing a small random variation on a template profile.

The actions which can be taken by the agent are represented by a set of 11 discrete numbers.
Action ID 0 : Use 0% solar, 100% coal
Action ID 1 : Use 10% solar, 90% coal
..
Action ID 10 : Use 100% solar, 0% coal 

The percentages are calculated with respect to the energy demand. 
For example, suppose the energy demand is 40 units, with 25 units of availble solar capacity and 50 units of available coal power capacity. Then,
Action ID 0 : 0 units of solar, 40 units of coal energy
Action ID 1 : 4 units of solar, 36 units of coal energy
..
Action ID 10 : 25 units of solar (since that is the max capacity), 0 units of coal energy

As such, it is possible for actions to lead to energy deficits, which the agent should learn to avoid.

The rewards are structured as follows:
- +15 reward if the energy requirement is met
- Penalty of (-2 * balance), where balance is the magnitude of deficit (or surplus) production
- Penalty of (0.5 * coal_used), where coal_used is the amount of energy generated using coal power plants

The following plots show the results of training the agent

Random agent:
![Random Agent](plots\\random_agent.png)

Trained agent:
![Trained Agent](plots\\trained_agent.png)
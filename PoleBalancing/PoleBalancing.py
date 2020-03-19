from math import sqrt, tanh

import gym
import numpy as np

from CheeseGame.QLearn import Agent
from NeuralNetworks.Net import Layer, Net
from NeuralNetworks.Utils import Sigmoid, Linear, ReLu

env = gym.make('CartPole-v0')
env2 = gym.make('CartPole-v0')
"""
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
Source:
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
Observation: 
    Type: Box(4)
    Num	Observation                 Min         Max
    0	Cart Position             -4.8            4.8
    1	Cart Velocity             -Inf            Inf
    2	Pole Angle                 -24 deg        24 deg
    3	Pole Velocity At Tip      -Inf            Inf

Actions:
    Type: Discrete(2)
    Num	Action
    0	Push cart to the left
    1	Push cart to the right

    Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
Reward:
    Reward is 1 for every step taken, including the termination step
Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
Episode Termination:
    Pole Angle is more than 12 degrees
    Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
    Episode length is greater than 200
    Solved Requirements
    Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""


def check_bound(value, *break_points):
    """
    Converts to range (lower, in range, higher)
    """
    r = break_points
    if not isinstance(break_points[0], int) and not isinstance(break_points[0], float):
        r = break_points[0]
    for i, p in enumerate(r):
        if value <= p:
            return i

    return len(break_points)

def round_observation(observation):
    return [
        #int(check_bound(observation[0], np.arange(-4.8, 4.8, 0.1))),            # 0	Cart Position             -4.8            4.8
        int(check_bound(observation[1], np.arange(-0.88, 0.88, 0.08))),          # 1	Cart Velocity             -Inf            Inf
        int(check_bound(np.degrees(observation[2]), np.arange(-11, 11, 1))),     # 2	Pole Angle                 -24 deg        24 deg
        int(check_bound(observation[3], np.arange(-0.88, 0.88, 0.08)))           # 3	Pole Velocity At Tip      -Inf            Inf
    ]

# Create Agent
actions = range(env.action_space.n)
agent = Agent(None, (100, 100, 100), actions)
net_size = 32
net = Net([
    Layer(4, net_size, Sigmoid),
    Layer(net_size, 2, Sigmoid)
])

i_episode = 0
guts = 0
guts_required = 100
temp_agent = agent.__copy__()
while guts < guts_required:
#for i_episode in range(1000):
    observation = env.reset()
    temp_agent.Position = agent.Position = round_observation(observation)
    score = 0
    t = 0
    done = False
    observations = []
    while not done:
    #for t in range(200):
        #env.render()
        i, action = temp_agent.decide()
        observation, reward, done, info = env.step(action)
        #reward -= abs(tanh(observation[0]))
        score += reward

        # learn
        observations.append([observation, round_observation(observation), i])

        observation = round_observation(observation)
        # print(observation, action, reward, done, info)
        temp_agent.remember(observation, i, reward)
        if done and score >= 200 and guts % (guts_required/100) == 0:
            print("{:3d} %: Episode {} finished: Score {} after {} timesteps".format(int(guts/guts_required*100), i_episode, score, t+1))
        t += 1

    # learn
    gut = score >= 200
    if gut:
        for observation in observations:
            agent.remember(observation[1], observation[2], 1)

        for observation in observations:
            real = [1.0, 0.0] if observation[2] == 0 else [0.0, 1.0]
            result = net.forward(observation[0])
            net.backwards(real, result, 0.1)

        guts += 1
    else:
        temp_agent = agent.__copy__()

    i_episode += 1

# Network showcase
i_episode = 0
while True:
    observation = env.reset()
    observation2 = env2.reset()
    score = 0
    t = 0
    done2 = done = False
    while not done:
        env.render()
        result = net.forward(observation)
        observation, reward, done, info = env.step(np.argmin(result))
        score += reward

        if not done2:
            env2.render()
            agent.Position = round_observation(observation2)
            i, a = agent.decide()
            observation2, reward2, done2, info2 = env2.step(a)
        if done:
            print("Episode {} finished: Score {} after {} timesteps".format(i_episode, score, t + 1))
            break
        t += 1

    i_episode += 1

env.close()
env2.close()
